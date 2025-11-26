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
import pytorch_kinematics as pk

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.common import TRANSLATION_NAMES, ROTATION_NAMES
from utils.bimanual_handler import BimanualPair, hand_pose_to_dict, build_bimanual_pose
from utils.common import setup_device, set_random_seeds, ensure_directory


def damped_inverse(J, damping=1e-4):
    """
    Compute damped pseudoinverse for a batch of matrices.

    Args:
        J: tensor of shape (B, M, N)
        damping: scalar λ (lambda)

    Returns:
        J_plus: tensor of shape (B, N, M)
    """
    B, M, N = J.shape
    I = torch.eye(M, device=J.device).unsqueeze(0).expand(B, M, M)

    # Compute JJ^T
    JJt = J @ J.transpose(1, 2)

    # Add damping term λ^2 I
    A = JJt + (damping**2) * I

    # Solve A X = I  (rather than taking inverse)
    # J^+ = J^T A^{-1}
    # => A^{-1} = solution of A X = I
    A_inv = torch.linalg.solve(A, I)  # shape (B, M, N)

    # J^+ = J^T * X
    J_plus = J.transpose(1, 2) @ A_inv  # shape (B, N, M)
    return J_plus


def split_by_max_size(obj_list, max_object_per_batch):
    """
    Split obj_list into batches with at most max_object_per_batch items each.
    """
    return [obj_list[i : i + max_object_per_batch] for i in range(0, len(obj_list), max_object_per_batch)]


class GraspExperiment:
    """
    Main experiment class for bimanual grasp generation.
    """

    def __init__(self, config: DictConfig):
        self.config: DictConfig = config  # hydra
        self.device = None
        self.bimanual_pair = None
        self.object_model = None
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
        self.object_model = ObjectModel(
            data_root_path=self.config.paths.data_root_path,
            batch_size_each=self.config.model.batch_size,
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size,
            sdf_tool=self.config.model.object_sdf_tool,
            bodex_format=True,
        )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)

    def hand_prepare(self, hand_model: HandModel):
        """
        Pre-compute fingertips' contact point and SerialChain.
        """
        tip_links = (
            self.config.hand_params.right_tip_links
            if hand_model.handedness == "right_hand"
            else self.config.hand_params.left_tip_links
        )

        contact_point_dict = {}
        chain_dict = {}  # PK's SerialChain

        for link_name in tip_links:
            contact_candidates = hand_model.mesh[link_name]["contact_candidates"]

            # Set the contact point as the average of the tip's contact candidates
            contact_point_dict[link_name] = contact_candidates.mean(dim=0, keepdim=True)

            chain_dict[link_name] = pk.SerialChain(hand_model.chain, link_name).to(
                device=self.device, dtype=torch.float32
            )

        return contact_point_dict, chain_dict

    def compute_three_hand_poses_jaco(
        self,
        object_model: ObjectModel,
        hand_model: HandModel,
        contact_point_dict,
        chain_dict,
    ):
        raise NotImplementedError("Not fully supported.")

        hand_qpos = hand_model.hand_pose[:, 9:]
        n_batch = hand_qpos.shape[0]
        n_link = len(chain_dict.keys())
        n_dof = hand_qpos.shape[1]
        all_joint_names = hand_model.chain.get_joint_parameter_names()
        device = hand_qpos.device
        dtype = hand_qpos.dtype

        # hand base pose in global frame
        base_pose = torch.zeros((n_batch, 4, 4), dtype=dtype, device=device)
        base_pose[:, :3, :3] = hand_model.global_rotation
        base_pose[:, :3, 3] = hand_model.global_translation

        jaco_dict = {}
        contact_pos_dict = {}

        for link_name in chain_dict.keys():
            s_chain = chain_dict[link_name]
            contact_point = contact_point_dict[link_name].repeat(n_batch, 1)

            s_chain_joint_names = s_chain.get_joint_parameter_names()
            s_chain_joint_indices = [all_joint_names.index(name) for name in s_chain_joint_names]
            s_hand_qpos = hand_qpos[:, s_chain_joint_indices]
            jaco = torch.zeros((n_batch, 3, n_dof), dtype=dtype, device=device)
            jaco[:, :, s_chain_joint_indices] = s_chain.jacobian(th=s_hand_qpos, locations=contact_point)[
                :, :3, :
            ]  # only extract the translation part
            jaco_dict[link_name] = jaco

            # compute (virtual) contact point in global frame
            link_pose = base_pose @ hand_model.current_status[link_name].get_matrix()
            link_rot, link_pos = link_pose[:, :3, :3], link_pose[:, :3, 3]
            contact_pos = link_rot @ contact_point.unsqueeze(2) + link_pos.unsqueeze(2)
            contact_pos_dict[link_name] = contact_pos.reshape(-1, 3)

        all_jaco = torch.cat([v.unsqueeze(1) for _, v in jaco_dict.items()], dim=1)  # shape (B, N_link, 3, N_dof)
        all_contact_pos = torch.cat([v.unsqueeze(1) for _, v in contact_pos_dict.items()], dim=1)  # (B, N_link, 3)

        distances, normals = object_model.cal_distance(all_contact_pos, with_closest_points=False)

        spead_dis = 0.05
        spread_movements = (
            spead_dis * normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
        )  # shape (B, N_link, 3)
        spread_movements = spread_movements.reshape(n_batch, n_link * 3, 1)
        all_jaco = all_jaco.reshape(n_batch, n_link * 3, n_dof)

        spread_delta_q = damped_inverse(all_jaco, damping=5e-2) @ spread_movements

        pregrasp_qpos = hand_qpos + spread_delta_q.squeeze(-1)

        pregrasp_poses = hand_model.hand_pose.clone()
        pregrasp_poses[:, 9:] = pregrasp_qpos

        raise NotImplementedError("Unfinished.")

        return pregrasp_poses

    def compute_three_hand_poses_simple(self, hand_model: HandModel):
        """
        Return:
            The grasp poses consisting of the pregrasp poses, grasp poses, and squeeze poses.
        """
        squeeze_joint_magnitude = self.config.task.squeeze_joint_magnitude
        spread_joint_magnitude = self.config.task.spread_joint_magnitude

        hand_qpos = hand_model.hand_pose[:, 9:]
        all_joint_names = hand_model.chain.get_joint_parameter_names()

        # Get the joint indices those are allowed to spread/squeeze
        squeeze_joint_names = (
            self.config.hand.hand_params.right_squeeze_joints
            if hand_model.handedness == "right_hand"
            else self.config.hand.hand_params.left_squeeze_joints
        )
        squeeze_joint_indices = [all_joint_names.index(name) for name in squeeze_joint_names]

        spread_delta_q = torch.zeros_like(hand_qpos)
        spread_delta_q[:, squeeze_joint_indices] = spread_joint_magnitude

        squeeze_delta_q = torch.zeros_like(hand_qpos)
        squeeze_delta_q[:, squeeze_joint_indices] = squeeze_joint_magnitude

        pregrasp_poses = hand_model.hand_pose.clone()
        grasp_poses = hand_model.hand_pose.clone()
        squeeze_poses = hand_model.hand_pose.clone()
        pregrasp_poses[:, 9:] = torch.clamp(
            hand_qpos + spread_delta_q, hand_model.joints_lower, hand_model.joints_upper
        )
        squeeze_poses[:, 9:] = torch.clamp(
            hand_qpos + squeeze_delta_q, hand_model.joints_lower, hand_model.joints_upper
        )

        return torch.stack([pregrasp_poses, grasp_poses, squeeze_poses], dim=1)

    def run_task(self, all_object_code_list):
        """
        Filtering synthesized grasps with energy-based metrics.
        """

        exp_path = os.path.join(self.config.paths.experiments_base, self.config.name)
        source_path = os.path.join(exp_path, self.config.task.source_dir)

        right_joint_names = self.bimanual_pair.right.get_joint_names()
        left_joint_names = self.bimanual_pair.left.get_joint_names()

        ############# Object list preparation #############

        # split all objects into batches
        n_samples_per_obj = self.config.model.batch_size
        max_object_per_batch = self.config.model.max_total_batch_size // n_samples_per_obj
        batched_object_code_list = split_by_max_size(all_object_code_list, max_object_per_batch)

        ############# Process the objects in batch #############

        for i_batch, object_code_list_ in enumerate(batched_object_code_list):
            # Check if the grasp files of the objects exist
            object_code_list = []
            for object_code in object_code_list_:
                data_path = os.path.join(source_path, f"{object_code}.npy")
                if os.path.exists(data_path):
                    object_code_list.append(object_code)
                else:
                    logging.warning(f"Grasp file of {object_code} does not exist.")
            if len(object_code_list) == 0:
                continue

            self.object_model.initialize(object_code_list)
            n_obj = len(object_code_list)
            right_hand_poses = torch.zeros((n_obj, n_samples_per_obj, 9 + len(right_joint_names)), device=self.device)
            left_hand_poses = torch.zeros((n_obj, n_samples_per_obj, 9 + len(left_joint_names)), device=self.device)

            ############# Load grasp data files #############

            data_dict_lst_all_obj = []
            for i_obj, object_code in enumerate(object_code_list):
                # load synthesized grasps
                data_path = os.path.join(source_path, f"{object_code}.npy")
                data_dict_lst = np.load(data_path, allow_pickle=True)[:n_samples_per_obj]
                data_dict_lst_all_obj.append(data_dict_lst)

                for i_grasp, data_dict in enumerate(data_dict_lst):
                    # set object scale
                    self.object_model.object_scale_tensor[i_obj, i_grasp] = data_dict["scale"]

                    right_hand_pose, left_hand_pose = build_bimanual_pose(
                        data_dict["qpos_right"],
                        data_dict["qpos_left"],
                        TRANSLATION_NAMES,
                        ROTATION_NAMES,
                        right_joint_names,
                        left_joint_names,
                        self.device,
                    )
                    right_hand_poses[i_obj, i_grasp, :] = right_hand_pose
                    left_hand_poses[i_obj, i_grasp, :] = left_hand_pose

            right_hand_poses = right_hand_poses.reshape(n_obj * n_samples_per_obj, -1)
            left_hand_poses = left_hand_poses.reshape(n_obj * n_samples_per_obj, -1)

            self.bimanual_pair.right.set_parameters(right_hand_poses)
            self.bimanual_pair.left.set_parameters(left_hand_poses)

            ############# Compute pregrasp and squeeze poses #############

            rh_grasp_poses = self.compute_three_hand_poses_simple(self.bimanual_pair.right)
            lh_grasp_poses = self.compute_three_hand_poses_simple(self.bimanual_pair.left)

            ################# Save the data #################
            rh_grasp_poses = rh_grasp_poses.reshape(n_obj, n_samples_per_obj, 3, -1)
            lh_grasp_poses = lh_grasp_poses.reshape(n_obj, n_samples_per_obj, 3, -1)

            save_dir = os.path.join(exp_path, self.config.task.save_dir)
            os.makedirs(save_dir, exist_ok=True)

            for i_obj, object_code in enumerate(object_code_list):
                data_dict_lst = []
                for i_grasp in range(n_samples_per_obj):
                    data_dict = data_dict_lst_all_obj[i_obj][i_grasp]

                    data_dict["pregrasp_qpos_right"] = hand_pose_to_dict(
                        rh_grasp_poses[i_obj, i_grasp, 0, :], right_joint_names
                    )
                    data_dict["squeeze_qpos_right"] = hand_pose_to_dict(
                        rh_grasp_poses[i_obj, i_grasp, 2, :], right_joint_names
                    )
                    data_dict["pregrasp_qpos_left"] = hand_pose_to_dict(
                        lh_grasp_poses[i_obj, i_grasp, 0, :], left_joint_names
                    )
                    data_dict["squeeze_qpos_left"] = hand_pose_to_dict(
                        lh_grasp_poses[i_obj, i_grasp, 2, :], left_joint_names
                    )

                    data_dict_lst.append(data_dict)

                save_path = os.path.join(save_dir, f"{object_code}.npy")
                np.save(save_path, data_dict_lst)
                logging.info(f"Batch {i_batch}. Compute pregrasp/squeeze poses and save to {save_path}.")

        if self.config.task.debug_visualize:
            grasp_type = 2  # 0: pregrasp; 1: grasp; 2: squeeze
            self.bimanual_pair.right.set_parameters(rh_grasp_poses[:, grasp_type, :])
            right_plot = self.bimanual_pair.right.get_plotly_data(
                i=0, opacity=1.0, color="lightslategray", with_contact_points=False
            )
            object_plot = self.object_model.get_plotly_data(i=0, color="seashell", opacity=1.0)

            # Combine everything
            plot_lst = right_plot + object_plot
            fig = go.Figure(plot_lst)

            fig.update_layout(
                paper_bgcolor="#E2F0D9",
                plot_bgcolor="#E2F0D9",
                scene_aspectmode="data",
                scene=dict(
                    xaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
                    yaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
                    zaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
                ),
            )
            fig.show()

    def run_full_experiment(self, object_code_list: List[str]):
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")

        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.run_task(object_code_list)


def task_compute_three_poses(cfg: DictConfig):
    """
    Computing pregrasp and squeeze poses based on the synthesized grasp poses.
    Currently using a very simple strategy.
    """

    # Object code list (keep as python list)
    if "object_code_list" in cfg:
        object_code_list = OmegaConf.to_object(cfg.object_code_list)
    else:
        with open(cfg.object_code_path, "r") as f:
            object_code_list = sorted(json.load(f))

    experiment = GraspExperiment(cfg)
    experiment.run_full_experiment(object_code_list)

    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    save_dir = os.path.join(exp_path, cfg.task.save_dir)
    file_lst = glob.glob(os.path.join(save_dir, "**/*.npy"), recursive=True)
    logging.info(f"Save {len(file_lst)} files (objects) in {save_dir}.")
    logging.info("Finish the calculation of pregrasp and squeeze poses.")

    return
