import os
import sys
import glob
import logging
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from omegaconf import DictConfig, OmegaConf
import cProfile
from typing import List
import json
import copy

from mr_utils.pytorch3d.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.bimanual_energy import BimanualEnergyComputer
from utils.common import robust_compute_rotation_matrix_from_ortho6d
from utils.bimanual_handler import BimanualPair, build_bimanual_three_poses, qpos_to_dict
from utils.common import setup_device, set_random_seeds, TRANSLATION_NAMES, ROTATION_NAMES
from utils.dual_arm_hand_model import DualArmHandModel


def poses9d_to_matrix(poses):
    """
    Args:
        poses: (B, 9)
    """
    pos = poses[:, :3]
    rot = robust_compute_rotation_matrix_from_ortho6d(poses[:, 3:])
    return pos_rot_to_matrix(pos, rot)


def matrix_to_poses9d(matrix):
    pos = matrix[:, :3, 3]
    rot = matrix[:, :3, :3]
    rot6d = rot[:, :, :2].transpose(1, 2).reshape(-1, 6)
    return torch.cat([pos, rot6d], dim=-1)


def pos_rot_to_matrix(pos, rot):
    matrix = torch.zeros((pos.shape[0], 4, 4), dtype=pos.dtype, device=pos.device)
    matrix[:, :3, :3] = rot
    matrix[:, :3, 3] = pos
    matrix[:, 3, 3] = 1.0
    return matrix


def split_by_max_size(obj_list, max_object_per_batch):
    """
    Split obj_list into batches with at most max_object_per_batch items each.
    """
    return [obj_list[i : i + max_object_per_batch] for i in range(0, len(obj_list), max_object_per_batch)]


class GraspExperiment:
    """
    Main experiment class for bimanual grasp generation.
    """

    def __init__(self, cfg: DictConfig):
        self.config: DictConfig = cfg  # hydra
        self.device = None
        self.bimanual_pair = None
        self.object_model = None
        self.energy_computer = None

        # Profiling
        self.profiler = cProfile.Profile()

    def setup_environment(self):
        """Setup device, random seeds, and environment variables."""
        self.device = setup_device(self.config.gpu)
        set_random_seeds(self.config.seed)
        np.seterr(all="raise")

        print(f"Using device: {self.device}")

    def setup_models(self):
        """Initialize hand and object models."""
        print("Setting up models...")

        self.object_model = ObjectModel(
            data_root_path=self.config.paths.data_root_path,
            batch_size_each=self.config.model.batch_size * 3 * 4,  # (3 grasp types) * (4 object poses)
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size,
            sdf_tool=self.config.model.object_sdf_tool,
            bodex_format=True,
        )

        right_hand_model = HandModel(
            handedness="right_hand",
            mjcf_path=self.config.hand.paths.right_hand_mjcf,
            contact_points_path=self.config.hand.paths.right_contact_points,
            device=self.device,
            n_surface_points=self.config.model.n_surface_points,
            sdf_tool=self.config.model.hand_sdf_tool,
            thumb_links=self.config.hand.hand_params.right_thumb_links,
        )
        left_hand_model = HandModel(
            handedness="left_hand",
            mjcf_path=self.config.hand.paths.left_hand_mjcf,
            contact_points_path=self.config.hand.paths.left_contact_points,
            device=self.device,
            n_surface_points=self.config.model.n_surface_points,
            sdf_tool=self.config.model.hand_sdf_tool,
            thumb_links=self.config.hand.hand_params.left_thumb_links,
        )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)

        self.dual_arm_hand_model = DualArmHandModel(
            n_surface_points=self.config.model.n_surface_points,
            device=self.device,
            cfg=self.config.dual_arm_hand,
        )
        self.dual_arm_hand_model.create_dual_serial_chains()  # for IK solving
        self.dual_arm_hand_model.create_dual_ik_solvers(
            num_retries=self.config.task.ik.num_retries,
            regularlization=self.config.task.ik.regularlization,
        )

    def setup_energy(self):
        """Initialize optimizer and energy computer."""
        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)

    @torch.no_grad()
    def run_task(self, all_object_code_list):
        """
        Filtering synthesized grasps with energy-based metrics.
        Consider pregrasp, grasp, and squeeze poses.
        """

        exp_path = os.path.join(self.config.paths.experiments_base, self.config.name)
        result_path = os.path.join(exp_path, self.config.task.source_dir)

        ############################ Object list preparation ############################

        # split all objects into batches
        n_samples_per_obj = self.config.model.batch_size
        max_object_per_batch = self.config.model.max_total_batch_size // n_samples_per_obj
        batched_object_code_list = split_by_max_size(all_object_code_list, max_object_per_batch)

        ############################ Pre-calculated values ############################

        right_joint_names = self.bimanual_pair.right.get_joint_names()
        left_joint_names = self.bimanual_pair.left.get_joint_names()
        ra_ref_qpos = torch.tensor(self.config.dual_arm_hand.ra_ref_qpos, device=self.device).unsqueeze(0)
        la_ref_qpos = torch.tensor(self.config.dual_arm_hand.la_ref_qpos, device=self.device).unsqueeze(0)

        obj_transforms = []
        trans = self.config.task.obj_trans
        for z_rot in self.config.task.obj_z_rot_lst:
            transform = pos_rot_to_matrix(
                pos=torch.tensor(trans, device=self.device).unsqueeze(0),
                rot=euler_angles_to_matrix(
                    torch.tensor([0, 0, z_rot], device=self.device).unsqueeze(0),
                    "XYZ",
                ),
            )
            obj_transforms.append(transform)
        obj_transforms = torch.cat(obj_transforms, dim=0).unsqueeze(0)  # shape (1, 4, 4, 4)

        n_valid = 0
        n_all = 0

        ############################ Process the objects in batch ############################

        for i_batch, object_code_list in enumerate(batched_object_code_list):
            logging.info(f"Batch: {i_batch + 1} / {len(batched_object_code_list)}")

            self.object_model.initialize(object_code_list)
            n_obj = len(object_code_list)
            right_hand_poses = torch.zeros(
                (n_obj, n_samples_per_obj, 3, 9 + len(right_joint_names)), device=self.device
            )
            left_hand_poses = torch.zeros((n_obj, n_samples_per_obj, 3, 9 + len(left_joint_names)), device=self.device)

            ############################ Load grasp data of the batch ############################
            data_dict_lst_all_obj = []
            for i_obj, object_code in enumerate(object_code_list):
                # load synthesized grasps
                data_dict_lst = np.load(os.path.join(result_path, f"{object_code}.npy"), allow_pickle=True)[
                    :n_samples_per_obj
                ]
                data_dict_lst_all_obj.append(data_dict_lst)

                for i_grasp, data_dict in enumerate(data_dict_lst):
                    # Set object scale
                    obj_scale = data_dict["scale"]
                    self.object_model.object_scale_tensor[i_obj, 3 * i_grasp : 3 * i_grasp + 3] = obj_scale

                    (
                        right_pregrasp_poses,
                        right_grasp_poses,
                        right_squeeze_poses,
                        left_pregrasp_poses,
                        left_grasp_poses,
                        left_squeeze_poses,
                    ) = build_bimanual_three_poses(
                        data_dict["pregrasp_qpos_right"],
                        data_dict["qpos_right"],
                        data_dict["squeeze_qpos_right"],
                        data_dict["pregrasp_qpos_left"],
                        data_dict["qpos_left"],
                        data_dict["squeeze_qpos_left"],
                        TRANSLATION_NAMES,
                        ROTATION_NAMES,
                        right_joint_names,
                        left_joint_names,
                        self.device,
                    )
                    right_hand_poses[i_obj, i_grasp, 0, :] = right_pregrasp_poses
                    left_hand_poses[i_obj, i_grasp, 0, :] = left_pregrasp_poses
                    right_hand_poses[i_obj, i_grasp, 1, :] = right_grasp_poses
                    left_hand_poses[i_obj, i_grasp, 1, :] = left_grasp_poses
                    right_hand_poses[i_obj, i_grasp, 2, :] = right_squeeze_poses
                    left_hand_poses[i_obj, i_grasp, 2, :] = left_squeeze_poses

            right_hand_poses = right_hand_poses.reshape(n_obj * n_samples_per_obj, 3, -1)
            left_hand_poses = left_hand_poses.reshape(n_obj * n_samples_per_obj, 3, -1)

            ############################ Augment the object poses ############################

            # Transform the objects to 4 poses to better find the object poses reachable by the dual arms.
            dim0 = n_obj * n_samples_per_obj * 4
            right_hand_poses = right_hand_poses.unsqueeze(1).repeat(1, 4, 1, 1).reshape(dim0, 3, -1)
            left_hand_poses = left_hand_poses.unsqueeze(1).repeat(1, 4, 1, 1).reshape(dim0, 3, -1)

            rh_base_poses = right_hand_poses[:, 1, :9]  # use the grasp pose for IK
            lh_base_poses = left_hand_poses[:, 1, :9]
            transform = obj_transforms.repeat(n_obj * n_samples_per_obj, 1, 1, 1).reshape(dim0, 4, 4)
            rh_matrix = transform @ poses9d_to_matrix(rh_base_poses)
            lh_matrix = transform @ poses9d_to_matrix(lh_base_poses)

            # Transform the object poses accordingly.
            obj_poses = pos_rot_to_matrix(self.object_model.global_translation, self.object_model.global_rotation)
            transform = transform.unsqueeze(1).repeat(1, 3, 1, 1).reshape(-1, 4, 4)
            transformed_obj_poses = transform @ obj_poses
            self.object_model.global_translation = transformed_obj_poses[:, :3, 3]
            self.object_model.global_rotation = transformed_obj_poses[:, :3, :3]

            # Modify the original hand poses
            rh_base_poses = matrix_to_poses9d(rh_matrix)
            lh_base_poses = matrix_to_poses9d(lh_matrix)
            right_hand_poses[:, :, :9] = rh_base_poses.unsqueeze(1)
            left_hand_poses[:, :, :9] = lh_base_poses.unsqueeze(1)

            ############################ Compute IK in batch ############################

            ref_qpos = torch.zeros((rh_matrix.shape[0], len(self.dual_arm_hand_model.joints_names)), device=self.device)
            ra_s_indices = self.dual_arm_hand_model.ra_s_indices
            la_s_indices = self.dual_arm_hand_model.la_s_indices
            ref_qpos[:, ra_s_indices] = ra_ref_qpos
            ref_qpos[:, la_s_indices] = la_ref_qpos

            # IK solving
            ra_res = self.dual_arm_hand_model.solve_ik_batch("right_hand", matrix=rh_matrix, ref_configs=ref_qpos)
            la_res = self.dual_arm_hand_model.solve_ik_batch("left_hand", matrix=lh_matrix, ref_configs=ref_qpos)

            # Assemble to get dual_arm_hand qpos (pregrasp, grasp, and squeeze)
            qpos = torch.zeros((rh_matrix.shape[0], 3, len(self.dual_arm_hand_model.joints_names)), device=self.device)
            # arm qpos
            qpos[:, :, ra_s_indices] = ra_res["q"][:, ra_s_indices].unsqueeze(1)
            qpos[:, :, la_s_indices] = la_res["q"][:, la_s_indices].unsqueeze(1)
            # hand qpos
            rh_indices = [self.dual_arm_hand_model.joints_names.index(name) for name in right_joint_names]
            lh_indices = [self.dual_arm_hand_model.joints_names.index(name) for name in left_joint_names]
            qpos[:, :, rh_indices] = right_hand_poses[:, :, 9:]
            qpos[:, :, lh_indices] = left_hand_poses[:, :, 9:]

            # Set qpos to robot model
            self.dual_arm_hand_model.set_parameters(qpos.reshape(-1, qpos.shape[-1]))
            self.bimanual_pair.right.set_parameters(right_hand_poses.reshape(-1, right_hand_poses.shape[-1]))
            self.bimanual_pair.left.set_parameters(left_hand_poses.reshape(-1, left_hand_poses.shape[-1]))

            ############################ DEBUG Visualization ############################
            if self.config.task.debug:
                for grasp_idx in range(3):
                    grasp_type = 1  # 0: pregrasp; 1: grasp; 2: squeeze

                    plot_idx = 3 * grasp_idx + grasp_type
                    plot_dual_arm_hand = self.dual_arm_hand_model.get_plotly_data(plot_idx, opacity=0.3)
                    plot_right_hand = self.bimanual_pair.right.get_plotly_data(
                        plot_idx, opacity=0.8, color="lightslategray"
                    )
                    plot_left_hand = self.bimanual_pair.left.get_plotly_data(
                        plot_idx, opacity=0.8, color="lightslategray"
                    )
                    plot_obj = self.object_model.get_plotly_data(plot_idx, opacity=0.7, color="coral")

                    # robot surface poitns
                    robot_surface_points = self.dual_arm_hand_model.get_global_surface_points()
                    robot_surface_points = robot_surface_points[plot_idx].cpu().detach().numpy()
                    robot_surface_points_plot = go.Scatter3d(
                        x=robot_surface_points[:, 0],
                        y=robot_surface_points[:, 1],
                        z=robot_surface_points[:, 2],
                        mode="markers",
                        marker=dict(color="lightpink", size=2),
                    )

                    plot_lst = (
                        plot_dual_arm_hand + plot_obj + plot_right_hand + plot_left_hand + [robot_surface_points_plot]
                    )
                    fig = go.Figure(plot_lst)

                    fig.update_layout(
                        paper_bgcolor="#E2F0D9",
                        plot_bgcolor="#E2F0D9",
                        scene_aspectmode="data",
                        scene=dict(
                            xaxis=dict(
                                visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False
                            ),
                            yaxis=dict(
                                visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False
                            ),
                            zaxis=dict(
                                visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False
                            ),
                        ),
                    )
                    fig.show()

            ############################ Filtering ############################
            # Dual arm IK
            ik_valid = ra_res["success"] & la_res["success"]

            # Bimanual hand self penetration
            energy_terms = self.energy_computer.compute_all_energies(
                self.bimanual_pair, self.object_model, verbose=True
            )
            max_joint_violation = self.bimanual_pair.compute_joint_limits_violation()

            thres_pen = self.config.task.thres.penetration
            thres_spen = self.config.task.thres.self_penetration
            thres_joint_limit = self.config.task.thres.joint_limit

            hand_valid = (
                (energy_terms.penetration < thres_pen)
                & (energy_terms.self_penetration < thres_spen)
                & (max_joint_violation < thres_joint_limit)
            )
            hand_valid = hand_valid.reshape(-1, 3)
            hand_valid = hand_valid[:, 0] & hand_valid[:, 1]  # consider only the pregrasp and grasp poses

            # Robot-object peneration & robot's self-peneration (robot: dual_arm_hand).
            # This cannot replace the above hand energy, since the surface point sampling
            # on the dual_arm_hand is very coarse, which can only be used for arm-related check.
            robot_surface_points = self.dual_arm_hand_model.get_global_surface_points()
            distances, _ = self.object_model.cal_distance(robot_surface_points)
            distances = torch.clamp(distances, min=0)
            pen_robot = distances.sum(-1)
            spen_robot = self.dual_arm_hand_model.self_penetration()
            robot_valid = (pen_robot < thres_pen) & (spen_robot < thres_spen)
            robot_valid = robot_valid.reshape(-1, 3)[:, 1]  # consider only the grasp poses

            # Combine them
            valid = ik_valid & hand_valid & robot_valid
            n_valid += valid.sum().item()
            n_all += valid.numel()

            ############################ Save arm-filtered grasp data ############################
            save_dir = os.path.join(exp_path, self.config.task.save_dir)

            valid = valid.reshape(n_obj, n_samples_per_obj, 4)
            obj_pos, obj_quat = transformed_obj_poses[:, :3, 3], matrix_to_quaternion(transformed_obj_poses[:, :3, :3])
            obj_poses = (
                torch.cat([obj_pos, obj_quat], dim=-1).reshape(n_obj, n_samples_per_obj, 4, 3, 7).detach().cpu().numpy()
            )
            qpos = qpos.reshape(n_obj, n_samples_per_obj, 4, 3, -1).detach().cpu().numpy()

            for i_obj, object_code in enumerate(object_code_list):
                for i_grasp in range(len(data_dict_lst)):
                    for i_p in range(4):
                        if valid[i_obj, i_grasp, i_p]:
                            save_path = os.path.join(save_dir, object_code, f"grasp_{i_grasp}_pose_{i_p}.npy")
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)

                            d = {}
                            d["obj_pose"] = obj_poses[i_obj, i_grasp, i_p, 0]  # 7d vector (x, y, z, qw, qx, qy, qz)
                            q = qpos[i_obj, i_grasp, i_p]
                            d["pregrasp_qpos"] = qpos_to_dict(q[0, :], self.dual_arm_hand_model.joints_names)
                            d["grasp_qpos"] = qpos_to_dict(q[1, :], self.dual_arm_hand_model.joints_names)
                            d["squeeze_qpos"] = qpos_to_dict(q[2, :], self.dual_arm_hand_model.joints_names)

                            data_dict = copy.deepcopy(data_dict_lst_all_obj[i_obj][i_grasp])
                            data_dict["dual_arm_hand"] = d
                            data_dict["obj_path"] = os.path.join(self.config.paths.data_root_path, object_code).replace(
                                "../", ""
                            )
                            # TODO: scene_path

                            np.save(save_path, data_dict)
                            logging.info(f"Save filtered grasp data to {save_path}.")

        logging.info("===============================================")
        logging.info(f"Passed grasp ratio (all): {n_valid}/{n_all} = {n_valid / n_all}.")
        logging.info("===============================================")

    def run_full_experiment(self, object_code_list: List[str]):
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")

        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_energy()

        self.run_task(object_code_list)


def task_arm_filter(cfg: DictConfig):
    # Object code list (keep as python list)
    if "object_code_list" in cfg:
        object_code_list = OmegaConf.to_object(cfg.object_code_list)
    else:
        with open(cfg.object_code_path, "r") as f:
            object_code_list = sorted(json.load(f))

    experiment = GraspExperiment(cfg)
    experiment.run_full_experiment(object_code_list)

    # Check the saved files
    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    save_dir = os.path.join(exp_path, cfg.task.save_dir)
    file_lst = glob.glob(os.path.join(save_dir, "**/*.npy"), recursive=True)
    logging.info(f"Save {len(file_lst)} files (grasps) in {save_dir}.")
    logging.info("Finish the filtering of bimanual grasps (no arms).")

    return
