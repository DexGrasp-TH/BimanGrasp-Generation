import math
import os
import torch
import pytorch_kinematics as pk
import trimesh as tm
from scipy.spatial.transform import Rotation as sciR
import mujoco
import numpy as np
import json
from torchsdf import index_vertices_by_faces, compute_sdf
import pytorch3d.structures
import pytorch3d.ops
import plotly.graph_objects as go
import logging

from mr_utils.utils_calc import sciR, transformPositions, posQuat2Isometry3d, quatWXYZ2XYZW
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix

try:
    from utils.common import robust_compute_rotation_matrix_from_ortho6d
except Exception:
    from common import robust_compute_rotation_matrix_from_ortho6d


def load_mujoco_model(mjcf_path, load_mode="xml_string"):
    """
    Args:
        load_mode: use 'xml_string' to be consistent with the inner functions of pk build_chain_from_mjcf().
    Return:
        mujoco_model
    """
    if load_mode == "xml_path":
        model = mujoco.MjModel.from_xml_path(mjcf_path)
    elif load_mode == "xml_string":
        data = open(mjcf_path).read()
        model = mujoco.MjModel.from_xml_string(data)

    return model


def extract_trimesh_from_mjcf(model, use_chamfer_box=True):
    """
    Args:
        model: mujoco model.
        use_chamfer_box: use mesh-based chamfer box (with smoother box edges) if True; otherwise, use primitive box.
    Return:
        link_geom_dict: a dict {body_name: [geom_info_dict, ...], ...}
    """

    def get_trimesh_from_mjmodel_mesh(model, mesh_id):
        """
        Convert a MuJoCo mesh (by mesh_id) to a trimesh.Trimesh object.
        """
        v_start = model.mesh_vertadr[mesh_id]
        v_count = model.mesh_vertnum[mesh_id]

        f_start = model.mesh_faceadr[mesh_id]
        f_count = model.mesh_facenum[mesh_id]

        vertices = model.mesh_vert[v_start : v_start + v_count, :]
        faces = model.mesh_face[f_start : f_start + f_count, :]

        mesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
        return mesh

    link_geom_dict = {}
    for geom_id in range(model.ngeom):
        # Skip geoms with `contype="0" conaffinity="0"``; Only consider the collision meshes.
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        pos = model.geom_pos[geom_id].tolist()
        quat = model.geom_quat[geom_id]  # (w, x, y, z)
        geom_type = model.geom_type[geom_id].copy()

        # geometry in trimesh format
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            # mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            mesh_scale = model.mesh_scale[mesh_id].tolist()
            trimesh = get_trimesh_from_mjmodel_mesh(model, mesh_id)
            trimesh.apply_scale(mesh_scale)
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            size = model.geom_size[geom_id]  # 3-dim box size
            if use_chamfer_box:
                trimesh = tm.load_mesh(
                    os.path.join("mjcf/box.obj"), process=False
                )  # load the chamfer box (with smoother edge)
                trimesh.apply_scale(size)
                geom_type = mujoco.mjtGeom.mjGEOM_MESH  # change the geom type to 'MESH'
            else:  # use primitive box
                trimesh = tm.primitives.Box(extents=2 * size)
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
            size = model.geom_size[geom_id]  # 1-dim radius
            trimesh = tm.primitives.Sphere(radius=size[0])
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            size = model.geom_size[geom_id]  # 2-dim: radius, height
            # trimesh = tm.primitives.Capsule(radius=size[0], height=size[1] * 2).apply_translation((0, 0, -size[1]))
            trimesh = tm.primitives.Capsule(radius=size[0], height=size[1] * 2)
        else:
            raise NotImplementedError(f"Unsupported geom type: {geom_type}!")

        geom_info_dict = {
            "geom_id": geom_id,
            "geom_type": geom_type,
            "geom_size": model.geom_size[geom_id],
            "geom_pos": pos,
            "geom_quat": quat,  # (w, x, y, z)
            "mesh": trimesh,
        }

        if body_name not in link_geom_dict:
            link_geom_dict[body_name] = []
        link_geom_dict[body_name].append(geom_info_dict)

    return link_geom_dict


class HandModel:
    def __init__(
        self,
        mjcf_path,
        contact_points_path,
        penetration_points_path,
        n_surface_points=0,
        device="cpu",
        handedness=None,
    ):
        """
        Create a Hand Model for a MJCF robot

        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of hand, use fps
        device: str | torch.Devicet
            device for torch tensors
        """

        """
        We need different joint initialization for left hand and right hand

        Parameter
        ----------
        handedness: str
            initialize left or right hand
        """

        self.device = device
        self.handedness = handedness
        self.n_surface_points = n_surface_points

        self.logger = logging.getLogger(__name__)

        # load articulation
        if not mjcf_path.endswith(".xml"):
            raise Exception("Only support .xml robot file.")

        # Build pytorch kinematics chain
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # load mujoco model
        self.mj_model = load_mujoco_model(mjcf_path, load_mode="xml_string")

        self._build_mesh(
            mjcf_path,
            contact_points_path,
            penetration_points_path,
        )
        self._build_link_collision_mask()

        # parameters
        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None
        self.surface_points_in_base = None
        self.surface_point = None

    def _build_mesh(
        self,
        mjcf_path,
        contact_points_path,
        penetration_points_path,
    ):
        """
        Build mesh informations, contact point candidates, and penetration points for each link.
        """
        # load contact points and penetration points
        contact_points = json.load(open(contact_points_path, "r")) if contact_points_path is not None else None
        penetration_points = (
            json.load(open(penetration_points_path, "r")) if penetration_points_path is not None else None
        )

        # build mesh
        self.mesh = {}
        areas = {}

        # extract links' trimeshes from the xml file via mujoco API
        self.link_geom_dict = link_geom_dict = extract_trimesh_from_mjcf(self.mj_model)

        for link_name, geoms in link_geom_dict.items():
            self.mesh[link_name] = {"geoms": []}
            link_vertices = []
            link_faces = []
            n_link_vertices = 0
            for geom_dict in geoms:
                mesh = geom_dict["mesh"]
                geom_type = geom_dict["geom_type"]

                # get the mesh defined in the link's local frame
                pos = geom_dict["geom_pos"]
                quat = geom_dict["geom_quat"]
                transformed_mesh = mesh.copy()
                transform = posQuat2Isometry3d(pos, quatWXYZ2XYZW(quat))
                transformed_mesh.apply_transform(transform)

                geom_dict["geom_pos"] = torch.tensor(pos, device=self.device)  # to tensor
                geom_dict["geom_quat"] = torch.tensor(quat, device=self.device)
                geom_dict["geom_size"] = torch.tensor(geom_dict["geom_size"], device=self.device)

                # get the vertices and faces of the transformed mesh defined in the link's local frame
                vertices = torch.tensor(transformed_mesh.vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(transformed_mesh.faces, dtype=torch.long, device=self.device)
                link_vertices.append(vertices)
                link_faces.append(faces + n_link_vertices)
                n_link_vertices += len(vertices)

                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    v = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)  # defined in geom frame
                    f = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
                    geom_dict.update({"face_verts": index_vertices_by_faces(v, f)})
                self.mesh[link_name]["geoms"].append(geom_dict)

            # The total vertices and faces of this link. Seems only used for visualization.
            link_vertices = torch.cat(link_vertices, dim=0)
            link_faces = torch.cat(link_faces, dim=0)

            # the contact point candidates and penetration keypoints of this link
            if link_name not in contact_points:
                contact_points[link_name] = []
            contact_candidates = (
                torch.tensor(contact_points[link_name], dtype=torch.float32, device=self.device).reshape(-1, 3)
                if contact_points is not None
                else None
            )
            penetration_keypoints = (
                torch.tensor(penetration_points[link_name], dtype=torch.float32, device=self.device).reshape(-1, 3)
                if penetration_points is not None
                else None
            )

            self.mesh[link_name].update(
                {
                    "vertices": link_vertices,
                    "faces": link_faces,
                    "contact_candidates": contact_candidates,
                    "penetration_keypoints": penetration_keypoints,
                }
            )
            areas[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()

        ############## Set joint limits ##############
        self.joints_names = self.chain.get_joint_parameter_names()
        self.joints_lower, self.joints_upper = self.chain.get_joint_limits()
        self.joints_lower = torch.tensor(self.joints_lower).float().to(self.device)
        self.joints_upper = torch.tensor(self.joints_upper).float().to(self.device)
        # TODO: this is just a temp workaround. The xml for left and right hand should be self-contained.
        if self.handedness == "left_hand":
            self.joints_lower, self.joints_upper = -self.joints_upper, -self.joints_lower

        ############## Sample surface points ##############
        # uniformly sample points from the hand surface, according to each link's area
        total_area = sum(areas.values())
        n_surface_points = self.n_surface_points
        num_samples = dict(
            [(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh]
        )
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=self.device
                ).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(
                self.mesh[link_name]["vertices"].unsqueeze(0), self.mesh[link_name]["faces"].unsqueeze(0)
            )
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=self.device)
            self.mesh[link_name]["surface_points"] = surface_points

        # indexing
        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))
        self.surface_points_link_indices = torch.cat(
            [
                self.link_name_to_link_index[link_name]
                * torch.ones(self.mesh[link_name]["surface_points"].shape[0], dtype=torch.long, device=self.device)
                for link_name in self.mesh
            ]
        )  # specify that each surface point belongs to which link

        self.contact_candidates = [self.mesh[link_name]["contact_candidates"] for link_name in self.mesh]
        self.global_index_to_link_index = sum(
            [[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)], []
        )
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=self.device
        )
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = [self.mesh[link_name]["penetration_keypoints"] for link_name in self.mesh]
        self.global_index_to_link_index_penetration = sum(
            [[i] * len(penetration_keypoints) for i, penetration_keypoints in enumerate(self.penetration_keypoints)], []
        )
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(
            self.global_index_to_link_index_penetration, dtype=torch.long, device=self.device
        )
        self.n_keypoints = self.penetration_keypoints.shape[0]

    # def _build_adjacency_mask(self):
    #     # TODO: make it more flexible to contact pairs or exluded contacts
    #     self.adjacency_mask = torch.zeros([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=self.device)

    #     def build_mask_recurse(body):
    #         for children in body.children:
    #             link_name = body.link.name
    #             child_link_name = children.link.name
    #             if link_name in self.mesh.keys() and child_link_name in self.mesh.keys():
    #                 parent_id = self.link_name_to_link_index[link_name]
    #                 child_id = self.link_name_to_link_index[child_link_name]
    #                 self.adjacency_mask[parent_id, child_id] = True
    #                 self.adjacency_mask[child_id, parent_id] = True

    #             build_mask_recurse(children)

    #     build_mask_recurse(self.chain._root)

    def _build_link_collision_mask(self):
        """
        Build the collision mask based on xml's contact pair information.
        Returns:
            collision_mask: a matrix mask specifiying whether requiring collision check between two links.
                False: no need for collision check; True: need collision check.
        """
        # TODO: support reading excluded contacts from xml.

        self.collision_mask = torch.zeros([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=self.device)

        model = self.mj_model
        n_contact_pairs = model.npair  # the contact pair specified in mujoco xml

        if n_contact_pairs == 0:
            raise NotImplementedError("Currently only support contact pair information in xml.")

        for i in range(n_contact_pairs):
            geom1_id = model.pair_geom1[i]
            geom2_id = model.pair_geom2[i]

            body1_id = model.geom_bodyid[geom1_id]
            body2_id = model.geom_bodyid[geom2_id]
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            parent_id = self.link_name_to_link_index[body1_name]
            child_id = self.link_name_to_link_index[body2_name]

            self.collision_mask[parent_id, child_id] = self.collision_mask[child_id, parent_id] = True

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """

        """
        You will need surface points to calculate the penetration energy between the left hand and the right hand 
        """

        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(self.hand_pose[:, 3:9])
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, n_contact, 4, 4)
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat(
                [self.contact_points, torch.ones(batch_size, n_contact, 1, dtype=torch.float, device=self.device)],
                dim=2,
            )
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[:, :, :3, 0]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(
                1, 2
            ) + self.global_translation.unsqueeze(1)

        # get surface points in world frame
        self.surface_points_in_base = points = self.get_surface_points()  # in base
        self.surface_point = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

    def calculate_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes
        Interiors are positive, exteriors are negative
        Use analytical method and our modified Kaolin package

        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            points in world frame
        """
        # x in hand base frame
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation

        dis_all = []
        for link_name in self.mesh.keys():
            dis = self.calculate_dis_to_link(x_in_base=x, link_name=link_name)
            dis_all.append(dis)

        dis_max = torch.max(torch.stack(dis_all, dim=0), dim=0)[0]
        return dis_max

    def calculate_dis_to_link(self, x_in_base, link_name):
        """
        Calculate the signed distance between the points and the link geometry.
        Interiors are positive, exteriors are negative.
        """
        dis_all = []
        matrix = self.current_status[link_name].get_matrix()
        x_in_link = (x_in_base - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]

        for geom in self.mesh[link_name]["geoms"]:
            geom_type = geom["geom_type"]
            geom_pos = geom["geom_pos"]
            geom_rotmat = quaternion_to_matrix(geom["geom_quat"].unsqueeze(0)).float()
            x_in_geom = (x_in_link - geom_pos.reshape(1, 1, 3)) @ geom_rotmat.reshape(1, 3, 3)
            x_in_geom = x_in_geom.reshape(-1, 3)  # (total_batch_size * num_samples, 3)

            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                face_verts = geom["face_verts"]
                dis_local, dis_signs, _, _ = compute_sdf(x_in_geom, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_height = geom["geom_size"][1]
                radius = geom["geom_size"][0]
                nearest_point = x_in_geom.detach().clone()
                nearest_point[:, :2] = 0
                nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], -half_height, half_height)
                dis_local = radius - (x_in_geom - nearest_point).norm(dim=1)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                box_size = geom["geom_size"]
                q = torch.abs(x_in_geom) - box_size.unsqueeze(0)
                q_clamped = torch.clamp(q, min=0)
                outside_distance = torch.norm(q_clamped, dim=1)
                inside_distance = torch.clamp(torch.max(q, dim=1)[0], max=0)
                dis_local = -inside_distance + outside_distance
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = geom["geom_size"][0]
                dis_local = self.mesh[link_name]["radius"] - x_in_geom.norm(dim=1)
            else:
                raise NotImplementedError(f"Unsupported geom type in calculate_distance(): {geom_type}!")

            self.logger.debug(f"link_name: {link_name}, geom_type: {geom_type}, dis_local: {dis_local}")
            dis_all.append(dis_local.reshape(x_in_base.shape[0], x_in_base.shape[1]))

        dis_max = torch.max(torch.stack(dis_all, dim=0), dim=0)[0]  # max distance to all geoms of this link
        return dis_max

    def cal_distance(self, x):
        """
        Backward compatibility wrapper for calculate_distance.
        """
        return self.calculate_distance(x)

    def cal_self_distance(self):
        """
        Calculate the max distance between self links,
        by calcuating the distances between sampled surface points and link geometries.
        """
        # get surface points in base frame
        if self.surface_points_in_base is None:
            raise NameError("self.surface_points_in_base is None !")
        x = self.surface_points_in_base.clone()
        if len(x.shape) == 2:
            x = x.expand(1, x.shape[0], x.shape[1])

        # cal distance
        dis = []
        for link_name in self.mesh:
            dis_local = self.calculate_dis_to_link(x_in_base=x, link_name=link_name)
            dis_local = dis_local.reshape(x.shape[0], x.shape[1])  # (total_batch_size, n_surface_points)

            need_collision_check = self.collision_mask[
                self.link_name_to_link_index[link_name], self.surface_points_link_indices
            ]
            dis_local[:, ~need_collision_check] = -float("inf")  # no need for collision check
            dis.append(dis_local)

        dis_max = torch.max(torch.stack(dis, dim=0), dim=0)[0]  # the max distance to other links of each surface point

        # # check whether the link names are correct
        # sp_idx = torch.argmax(dis_max)
        # max_dis = dis_max[:, sp_idx]
        # link_index = self.surface_points_link_indices[sp_idx].item()

        # link_index_to_link_name = {v: k for k, v in self.link_name_to_link_index.items()}
        # link_name = link_index_to_link_name[link_index]
        # print(f"link 1 name: {link_name}, sp_idx: {sp_idx}")

        # for i, link_name in enumerate(self.mesh):
        #     d = dis[i]
        #     print(f"link 2 name: {link_name}, d[:, sp_idx]: {d[:, sp_idx]}")

        return dis_max

    def self_penetration(self, mode="surface_points"):
        if mode == "surface_points":
            dis = self.cal_self_distance()
            dis[dis <= 0] = 0
            E_spen = dis.sum(-1)
            return E_spen

        elif mode == "penetration_points":
            if not self.penetration_keypoints:
                raise NameError("No 'self.penetration_keypoints' exists!")

            batch_size = self.global_translation.shape[0]
            points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
            link_indices = self.global_index_to_link_index_penetration.clone().repeat(batch_size, 1)

            # Corresponding link poses of each keypoint
            transforms = torch.zeros(batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = (
                    self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, self.n_keypoints, 4, 4)
                )
                transforms[mask] = cur[mask]

            # Get the point position in world frame
            points = torch.cat(
                [points, torch.ones(batch_size, self.n_keypoints, 1, dtype=torch.float, device=self.device)], dim=2
            )  # extend to [x, y, z, 1]
            points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
            points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

            # Compute the distance of each pair of keypoints
            dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
            dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)  # avoid self-keypoint-computation
            dis = 0.02 - dis
            E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
            return E_spen.sum((1, 2))

    def get_surface_points(self):
        """
        Get surface points in base frame.

        Returns:
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]["surface_points"]))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)

        return points

    def get_contact_candidates(self):
        """
        Get all contact candidates

        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["contact_candidates"].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]["contact_candidates"]))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_plotly_data(self, i, opacity=0.5, color="lightblue", with_contact_points=False, pose=None, with_axes=True):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity
                )
            )
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                )
            )

        # Optional: visualize coordinate axes
        axis_length = 0.1
        if with_axes:
            origin = np.zeros(3)
            axes = np.eye(3) * axis_length
            colors = ["red", "green", "blue"]
            names = ["x", "y", "z"]

            for j in range(3):
                p1 = origin
                p2 = axes[j]
                if pose is not None:
                    p1 = p1 @ pose[:3, :3].T + pose[:3, 3]
                    p2 = p2 @ pose[:3, :3].T + pose[:3, 3]
                data.append(
                    go.Scatter3d(
                        x=[p1[0], p2[0]],
                        y=[p1[1], p2[1]],
                        z=[p1[2], p2[2]],
                        mode="lines",
                        line=dict(color=colors[j], width=6),
                        name=f"{names[j]}-axis",
                    )
                )

        return data
