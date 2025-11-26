import math
import os
import torch
import pytorch_kinematics as pk
import trimesh as tm
from scipy.spatial.transform import Rotation as sciR
import mujoco
import numpy as np
import json
import kaolin
import torchsdf
import pytorch3d.structures
import pytorch3d.ops
import plotly.graph_objects as go
import logging
import copy
from omegaconf import DictConfig, OmegaConf
from typing import Optional, List, Union, Dict
import re
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

        # # replace 'meshdir' relative to the mjcf file with 'meshdir' relative to the workspace folder.
        # match = re.search(r'meshdir="([^"]*)"', data)
        # if match:
        #     old_meshdir = match.group(1)
        #     prefix = os.path.basename(mjcf_path)
        #     new_meshdir = os.path.join(prefix, old_meshdir)
        #     data = re.sub(r'meshdir="[^"]*"', f'meshdir="{new_meshdir}"', data)
        # else:
        #     raise ValueError("No meshdir found in the XML file")

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
            trimesh = get_trimesh_from_mjmodel_mesh(model, mesh_id)
            # mesh_scale = model.mesh_scale[mesh_id].tolist()  # only get the sign
            # trimesh.apply_scale(mesh_scale)  # should not apply scale
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


class DualArmHandModel:
    def __init__(
        self,
        n_surface_points=0,
        device="cpu",
        sdf_tool="torchsdf",
        cfg: DictConfig = None,
    ):
        self.device = device
        self.sdf_tool = sdf_tool
        self.cfg = cfg
        self.n_surface_points = n_surface_points

        mjcf_path = cfg.mjcf_path
        urdf_path = cfg.urdf_path

        # load articulation
        if not mjcf_path.endswith(".xml"):
            raise Exception("Only support .xml robot file.")

        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)

        # load mujoco model
        self.mj_model = load_mujoco_model(mjcf_path, load_mode="xml_string")

        self._build_mesh()
        self._build_link_collision_mask()

        self.qpos = None
        self.surface_points_in_base = None
        self.surface_point = None

    def _build_mesh(self):
        """
        Build mesh informations for each link.
        """
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
                    v = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)  # in geom frame
                    f = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
                    if self.sdf_tool == "kaolin":
                        geom_dict.update({"vertices": v})
                        geom_dict.update({"faces": f})
                        geom_dict.update(
                            {"face_verts": kaolin.ops.mesh.index_vertices_by_faces(v.unsqueeze(0), f).unsqueeze(0)}
                        )
                    elif self.sdf_tool == "torchsdf":
                        geom_dict.update({"face_verts": torchsdf.index_vertices_by_faces(v, f)})

                self.mesh[link_name]["geoms"].append(geom_dict)

            # The total vertices and faces of this link. Seems only used for visualization.
            link_vertices = torch.cat(link_vertices, dim=0)
            link_faces = torch.cat(link_faces, dim=0)

            self.mesh[link_name].update(
                {
                    "vertices": link_vertices,
                    "faces": link_faces,
                }
            )
            areas[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()

        ############## Set joint limits ##############
        self.joints_names = self.chain.get_joint_parameter_names()
        self.joints_lower, self.joints_upper = self.chain.get_joint_limits()
        self.joints_lower = torch.tensor(self.joints_lower).float().to(self.device)
        self.joints_upper = torch.tensor(self.joints_upper).float().to(self.device)

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

    def _build_link_collision_mask(self):
        """
        Build the collision mask based on xml's contact pair information.
        Returns:
            collision_mask: a matrix mask specifiying whether requiring collision check between two links.
                False: no need for collision check; True: need collision check.
        """

        model = self.mj_model
        n_contact_pairs = model.npair  # the contact pair specified in mujoco xml

        if n_contact_pairs > 0:
            self.collision_mask = torch.zeros([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=self.device)
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
        else:
            self.collision_mask = torch.ones([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=self.device)
            # exclude self-collision-detection
            self.collision_mask.fill_diagonal_(False)
            # exclude all adjacent bodies
            body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
            for i in range(1, model.nbody):
                body1_id = model.body_parentid[i]
                body2_id = i
                body1_name = body_names[body1_id]
                body2_name = body_names[body2_id]
                if body1_name in self.link_name_to_link_index and body2_name in self.link_name_to_link_index:
                    parent_id = self.link_name_to_link_index[body1_name]
                    child_id = self.link_name_to_link_index[body2_name]
                    self.collision_mask[parent_id, child_id] = self.collision_mask[child_id, parent_id] = False
            # exclude body-pairs specified in the xml
            if model.nexclude > 0:
                for i in range(model.nexclude):
                    # reference: https://github.com/google-deepmind/mujoco/blob/4e46db89037de9a2e388dfbb830b97ec37c4326c/src/engine/engine_io.c#L2032
                    body1_id = model.exclude_signature[i] & 0xFFFF
                    body2_id = model.exclude_signature[i] >> 16
                    body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                    body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                    # print(f"body1_name: {body1_name}, body2_name: {body2_name}")
                    if body1_name in self.link_name_to_link_index and body2_name in self.link_name_to_link_index:
                        parent_id = self.link_name_to_link_index[body1_name]
                        child_id = self.link_name_to_link_index[body2_name]
                        self.collision_mask[parent_id, child_id] = self.collision_mask[child_id, parent_id] = False

    def set_parameters(self, qpos):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        qpos: (B, n_dofs) torch.FloatTensor
            joint angles
        """

        self.qpos = qpos.clone()
        if self.qpos.requires_grad:
            self.qpos.retain_grad()
        self.current_status = self.chain.forward_kinematics(self.qpos)

        # get surface points in world frame
        self.surface_points_in_base = self.get_surface_points()  # in base
        self.surface_point = self.surface_points_in_base

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

                if self.sdf_tool == "kaolin":
                    # SDF computation based on kaolin, instead of TorchSDF
                    verts = geom["vertices"]
                    faces = geom["faces"]
                    dis_local, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                        x_in_geom.unsqueeze(0), face_verts.unsqueeze(0)
                    )
                    dis_signs = kaolin.ops.mesh.check_sign(
                        verts.unsqueeze(0), faces, x_in_geom.unsqueeze(0)
                    )  # True if inside mesh
                    dis_local = dis_local.squeeze(0)  # square distances
                    dis_signs = torch.where(dis_signs, -1.0, 1.0).squeeze(0)

                elif self.sdf_tool == "torchsdf":
                    dis_local, dis_signs, _, _ = torchsdf.compute_sdf(x_in_geom, face_verts)

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
                dis_local = radius - x_in_geom.norm(dim=1)
            else:
                raise NotImplementedError(f"Unsupported geom type in calculate_distance(): {geom_type}!")

            # self.logger.debug(f"link_name: {link_name}, geom_type: {geom_type}, dis_local: {dis_local}")
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

        return dis_max

    def self_penetration(self):
        dis = self.cal_self_distance()
        dis[dis <= 0] = 0
        E_spen = dis.sum(-1)
        return E_spen

    def get_surface_points(self):
        """
        Get surface points in base frame.

        Returns:
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.qpos.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]["surface_points"]))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)

        if batch_size == 1:  # ensure the shape is (N_batch, N_points, 3)
            points = points.unsqueeze(0)

        return points

    def get_global_surface_points(self):
        """
        Get surface points in global frame.
        """
        return self.surface_point.clone()

    def get_plotly_data(self, i, opacity=0.5, color="lightblue", with_axes=True):
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

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """

        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            data.append(
                go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity
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

    def get_trimesh_data(self, i, rgba, pose=None):
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)

        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]

            mesh = tm.Trimesh(vertices=v, faces=f, process=False)
            mesh.visual.vertex_colors = rgba

            data.append(mesh)

        return data

    def create_serial_chain(
        self,
        handedness,
    ):
        hand_base_link = self.cfg.rh_base_link if handedness == "right_hand" else self.cfg.lh_base_link
        s_chain = pk.SerialChain(self.urdf_chain, hand_base_link).to(device=self.device, dtype=torch.float32)

        serial_joint_names = s_chain.get_joint_parameter_names()
        serial_indices = [self.joints_names.index(name) for name in serial_joint_names]

        return s_chain, serial_indices

    def create_ik_solver(
        self,
        s_chain,
        num_retries=10,
        regularlization=1e-3,
        initial_noise_std: Optional[torch.tensor] = None,
    ):
        """
        Args:
            initial_noise_std: (B, N_full_joints)
        """
        self.n_ik_retries = num_retries
        self.initial_noise_std = initial_noise_std

        if initial_noise_std is not None:
            assert self.initial_noise_std.shape[0] == len(self.joints_names), (
                "initial_noise_std must match n_robot_joints"
            )

        joint_lims = torch.tensor(s_chain.get_joint_limits(), device=self.device, dtype=s_chain.dtype)
        ik = pk.PseudoInverseIK(
            s_chain,
            max_iterations=50,
            retry_configs=None,
            num_retries=num_retries,
            joint_limits=joint_lims.T,
            early_stopping_any_converged=False,
            early_stopping_no_improvement="any",
            debug=False,
            lr=0.2,
            regularlization=regularlization,
        )
        return ik

    def create_dual_serial_chains(self):
        # The urdf is only used for IK solving
        self.urdf_chain = pk.build_chain_from_urdf(open(self.cfg.urdf_path).read()).to(
            dtype=torch.float, device=self.device
        )

        if self.urdf_chain.get_joint_parameter_names() != self.chain.get_joint_parameter_names():
            raise ValueError("The joints in URDF and MJCF are inconsistent!")

        self.ra_s_chain, self.ra_s_indices = self.create_serial_chain("right_hand")
        self.la_s_chain, self.la_s_indices = self.create_serial_chain("left_hand")

    def create_dual_ik_solvers(
        self,
        num_retries=10,
        regularlization=1e-3,
        initial_noise_std: Optional[torch.tensor] = None,
    ):
        self.ra_ik = self.create_ik_solver(self.ra_s_chain, num_retries, regularlization, initial_noise_std)
        self.la_ik = self.create_ik_solver(self.la_s_chain, num_retries, regularlization, initial_noise_std)

    def solve_ik_batch(
        self,
        handedness,
        matrix: torch.Tensor,
        ref_configs: Optional[torch.Tensor] = None,
        use_ref_as_init: bool = True,
    ):
        """
        Solve a batch of inverse kinematics problems.

        Args:
            matrix (torch.Tensor): target poses of shape (B, 4, 4), defined in the world frame
            ref_configs (Optional[torch.Tensor]): reference full joint configs (B, DOF), optional

        Returns:
            Dict[str, torch.Tensor]: dictionary with keys:
                - "q": (B, DOF) full joint configuration
                - "success": (B, n_seeds) convergence flags
                - "err_pos": (B, n_seeds) position error
                - "err_rot": (B, n_seeds) rotation error
        """

        s_chain = self.ra_s_chain if handedness == "right_hand" else self.la_s_chain
        ik = self.ra_ik if handedness == "right_hand" else self.la_ik

        B = matrix.shape[0]
        serial_joint_names = s_chain.get_joint_parameter_names()
        serial_indices = [self.joints_names.index(name) for name in serial_joint_names]
        num_joints = len(self.joints_names)
        joint_lims = torch.tensor(s_chain.get_joint_limits(), device=self.device)

        # Prepare full_q and serial_ref_configs if needed
        noised_serial_ref_configs = None
        if ref_configs is not None:
            assert ref_configs.shape[0] == B, "ref_configs must match batch size"
            full_q = ref_configs.clone()
            serial_ref_configs = ref_configs[:, serial_indices]
            serial_ref_configs = serial_ref_configs.unsqueeze(1).repeat(1, self.n_ik_retries, 1)

            if use_ref_as_init:
                noise_std = (
                    self.initial_noise_std[serial_indices].view(1, 1, -1) if self.initial_noise_std is not None else 0
                )  # add some noises to the initial values
                noised_serial_ref_configs = serial_ref_configs + torch.randn_like(serial_ref_configs) * noise_std
                # clamp into the joint limits
                joint_mins = joint_lims[0, :].view(1, 1, -1)  # shape (1, 1, D)
                joint_maxs = joint_lims[1, :].view(1, 1, -1)  # shape (1, 1, D)
                noised_serial_ref_configs = torch.clamp(noised_serial_ref_configs, min=joint_mins, max=joint_maxs)
        else:
            full_q = torch.zeros((B, num_joints), dtype=matrix.dtype, device=matrix.device)
            # re-sample initial configs (default: uniform sampling)
            ik.sample_configs(num_configs=self.n_ik_retries)

        # Solve IK
        goal_tf_in_world = pk.Transform3d(matrix=matrix, device=self.device)
        goal_tf_in_base = self.tf_world_to_base(goal_tf_in_world)
        sol = ik.solve(goal_tf_in_base, ref_configs=noised_serial_ref_configs)

        # check if the solutions are within joint limits
        within_lims = self.check_in_joint_limits(sol.solutions[sol.converged, :], joint_lims)
        if not within_lims.all():
            raise RuntimeError("Some converged IK solutions are out of joint limits.")

        # Select best IK solution
        if ref_configs is not None:
            # Use reference config to find the closest solution (among successful ones)
            ref_q = serial_ref_configs[:, 0:1, :]  # shape: (B, 1, D)
            diff = sol.solutions - ref_q
            dist = torch.norm(diff, dim=-1)  # (B, n_seeds)
            dist[~sol.converged] = 1e6
            best_idx = torch.argmin(dist, dim=1)  # (B,)
        else:
            # Use first converged solution (fallback if no ref config)
            best_idx = torch.argmax(sol.converged.to(torch.int), dim=1)  # (B,)

        batch_idx = torch.arange(B, device=matrix.device)
        serial_q = sol.solutions[batch_idx, best_idx, :].to(matrix.dtype)  # (B, serial_dof)
        full_q[:, serial_indices] = serial_q  # Fill in full joint vector

        # return the best IK solution among the retries for each item
        return {
            "q": full_q,
            "success": sol.converged[batch_idx, best_idx],
            "err_pos": sol.err_pos[batch_idx, best_idx],
            "err_rot": sol.err_rot[batch_idx, best_idx],
        }

    def tf_world_to_base(self, tf_in_w: pk.Transform3d):
        """
        Note: The current implementation assumes base == world.
        """
        return tf_in_w

    def check_in_joint_limits(self, qpos: torch.Tensor, joint_lims: torch.Tensor):
        """
        Args:
            qpos: shape (B, n_dof)
            joint_lims: shape (2, n_dof)
        """
        lower = joint_lims[0]  # shape: (D,)
        upper = joint_lims[1]  # shape: (D,)

        # Reshape limits to match solution shape
        while lower.dim() < qpos.dim():
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)

        # Check if each solution is within the limits
        within_lower = qpos >= lower
        within_upper = qpos <= upper
        within_limits = within_lower & within_upper  # shape: (B, D) or (B, N, D)

        # All joints in each solution must be within limits
        is_valid = within_limits.all(dim=-1)  # shape: (B,) or (B, N)

        return is_valid
