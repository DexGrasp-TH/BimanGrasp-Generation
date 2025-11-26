import os
import sys
import glob
import logging
import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
import plotly.io as pio
from omegaconf import DictConfig, OmegaConf
import hydra
import re
import imageio
import trimesh as tm
import multiprocessing
import trimesh
import cProfile
from typing import List
import json
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.bimanual_energy import BimanualEnergyComputer
from utils.common import TRANSLATION_NAMES, ROTATION_NAMES
from utils.bimanual_handler import BimanualPair, build_bimanual_pose
from utils.common import setup_device, set_random_seeds
from utils.dual_arm_hand_model import DualArmHandModel


def build_qpos(qpos_dict, joint_names, device):
    """Build a torch tensor for hand pose given qpos dict."""
    qpos = torch.tensor(
        [qpos_dict[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    return qpos


class GraspExperiment:
    """
    Main experiment class for bimanual grasp generation.
    """

    def __init__(self, config: DictConfig):
        self.config: DictConfig = config  # hydra
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

        self.object_model = ObjectModel(
            data_root_path=self.config.paths.data_root_path,
            batch_size_each=1,
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size,
        )

        self.dual_arm_hand_model = DualArmHandModel(
            n_surface_points=self.config.model.n_surface_points,
            device=self.device,
            cfg=self.config.dual_arm_hand,
        )

    def setup_energy(self):
        """Initialize optimizer and energy computer."""

        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)

    def run_vis_hand_before_filter(self):
        """
        Visualiing bimanual hand grasps before filtering.
        No arms.
        """
        exp_path = os.path.join(self.config.paths.experiments_base, self.config.name)
        source_path = os.path.join(exp_path, self.config.task.source_dir)

        right_joint_names = self.bimanual_pair.right.get_joint_names()
        left_joint_names = self.bimanual_pair.left.get_joint_names()

        object_code_list = self.config.task.object_code_list
        grasp_indices = self.config.task.grasp_indices
        opt_steps = self.config.task.opt_steps if source_path.endswith("intermediate") else [-1]

        for object_code in object_code_list:
            self.object_model.initialize([object_code])

            for grasp_idx in grasp_indices:
                for opt_step in opt_steps:
                    #################### Load grasp data ####################

                    if opt_step != -1:
                        path = os.path.join(source_path, f"{object_code}_{opt_step}.npy")
                    else:
                        path = os.path.join(source_path, f"{object_code}.npy")

                    data_dict = np.load(path, allow_pickle=True)[grasp_idx]

                    # Set object scale
                    self.object_model.object_scale_tensor[0] = data_dict["scale"]

                    right_hand_pose, left_hand_pose = build_bimanual_pose(
                        data_dict["qpos_right"],
                        data_dict["qpos_left"],
                        TRANSLATION_NAMES,
                        ROTATION_NAMES,
                        right_joint_names,
                        left_joint_names,
                        self.device,
                    )

                    right_contact_point_indices = data_dict["contact_point_indices_right"]
                    left_contact_point_indices = data_dict["contact_point_indices_left"]
                    right_contact_point_indices = torch.tensor(right_contact_point_indices, device=self.device)
                    left_contact_point_indices = torch.tensor(left_contact_point_indices, device=self.device)

                    if "pregrasp_qpos_right" in data_dict.keys():
                        right_pregrasp_pose, left_pregrasp_pose = build_bimanual_pose(
                            data_dict["pregrasp_qpos_right"],
                            data_dict["pregrasp_qpos_left"],
                            TRANSLATION_NAMES,
                            ROTATION_NAMES,
                            right_joint_names,
                            left_joint_names,
                            self.device,
                        )
                        right_squeeze_pose, left_squeeze_pose = build_bimanual_pose(
                            data_dict["squeeze_qpos_right"],
                            data_dict["squeeze_qpos_left"],
                            TRANSLATION_NAMES,
                            ROTATION_NAMES,
                            right_joint_names,
                            left_joint_names,
                            self.device,
                        )

                    #################### Energy ####################

                    # Set hand qpos & contact points (batch size = 1)
                    self.bimanual_pair.right.set_parameters(
                        right_hand_pose.unsqueeze(0), right_contact_point_indices.unsqueeze(0)
                    )
                    self.bimanual_pair.left.set_parameters(
                        left_hand_pose.unsqueeze(0), left_contact_point_indices.unsqueeze(0)
                    )

                    energy_terms = self.energy_computer.compute_all_energies(
                        self.bimanual_pair, self.object_model, verbose=True
                    )
                    keys = [
                        "total",
                        "force_closure",
                        "distance",
                        "penetration",
                        "self_penetration",
                        "joint_limits",
                        "wrench_volume",
                    ]
                    print(f"======== Energy of {object_code}:grasp_{grasp_idx} ========")
                    for key in keys:
                        val = getattr(energy_terms, key).clone()
                        print(f"{key}: {val}")

                    #################### Visualization ####################

                    # Hand grasp poses
                    self.bimanual_pair.right.set_parameters(
                        right_hand_pose.unsqueeze(0), right_contact_point_indices.unsqueeze(0)
                    )
                    self.bimanual_pair.left.set_parameters(
                        left_hand_pose.unsqueeze(0), left_contact_point_indices.unsqueeze(0)
                    )
                    right_plot = self.bimanual_pair.right.get_plotly_data(
                        i=0, opacity=1.0, color="lightslategray", with_contact_points=True
                    )
                    left_plot = self.bimanual_pair.left.get_plotly_data(
                        i=0, opacity=1.0, color="lightslategray", with_contact_points=True
                    )

                    # Object mesh
                    object_plot = self.object_model.get_plotly_data(i=0, color="seashell", opacity=1.0)

                    # Object surface points
                    obj_surface_points = (
                        self.object_model.object_scale_tensor[0] * self.object_model.surface_points_tensor[0]
                    )
                    obj_surface_points = obj_surface_points.cpu().detach().numpy()
                    obj_surface_points_plot = go.Scatter3d(
                        x=obj_surface_points[:, 0],
                        y=obj_surface_points[:, 1],
                        z=obj_surface_points[:, 2],
                        mode="markers",
                        marker=dict(color="blue", size=2),
                    )

                    # Hand surface poitns
                    hand_surface_points_right = self.bimanual_pair.right.get_global_surface_points()
                    hand_surface_points_left = self.bimanual_pair.left.get_global_surface_points()
                    hand_surface_points = torch.cat([hand_surface_points_right, hand_surface_points_left], dim=1)
                    hand_surface_points = hand_surface_points[0].cpu().detach().numpy()
                    hand_surface_points_plot = go.Scatter3d(
                        x=hand_surface_points[:, 0],
                        y=hand_surface_points[:, 1],
                        z=hand_surface_points[:, 2],
                        mode="markers",
                        marker=dict(color="lightpink", size=2),
                    )

                    # Combine everything
                    plot_lst = (
                        right_plot + left_plot + object_plot + [obj_surface_points_plot, hand_surface_points_plot]
                    )

                    # Pregrasp and squeeze hand poses
                    if "pregrasp_qpos_right" in data_dict.keys():
                        self.bimanual_pair.right.set_parameters(right_pregrasp_pose.unsqueeze(0))
                        self.bimanual_pair.left.set_parameters(left_pregrasp_pose.unsqueeze(0))
                        right_pregrasp_plot = self.bimanual_pair.right.get_plotly_data(
                            i=0, opacity=0.5, color="#FFB74D", with_contact_points=False
                        )
                        left_pregrasp_plot = self.bimanual_pair.left.get_plotly_data(
                            i=0, opacity=0.5, color="#FFB74D", with_contact_points=False
                        )

                        self.bimanual_pair.right.set_parameters(right_squeeze_pose.unsqueeze(0))
                        self.bimanual_pair.left.set_parameters(left_squeeze_pose.unsqueeze(0))
                        right_squeeze_plot = self.bimanual_pair.right.get_plotly_data(
                            i=0, opacity=0.5, color="#81C784", with_contact_points=False
                        )
                        left_squeeze_plot = self.bimanual_pair.left.get_plotly_data(
                            i=0, opacity=0.5, color="#81C784", with_contact_points=False
                        )

                        plot_lst += right_pregrasp_plot + left_pregrasp_plot + right_squeeze_plot + left_squeeze_plot

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

    def run_vis_arm_hand_after_filter(self):
        """
        Visualiing dual-arm-hand grasps before filtering.
        """
        exp_path = os.path.join(self.config.paths.experiments_base, self.config.name)
        source_path = os.path.join(exp_path, self.config.task.source_dir)

        joint_names = self.dual_arm_hand_model.joints_names

        object_code_list = self.config.task.object_code_list
        grasp_indices = self.config.task.grasp_indices

        for object_code in object_code_list:
            self.object_model.initialize([object_code])

            for grasp_id in grasp_indices:
                #################### Load grasp data ####################

                path = os.path.join(source_path, f"{object_code}/{grasp_id}.npy")
                if not os.path.exists(path):
                    logging.warning(f"File {path} does not exist.")
                    continue

                data_dict = np.load(path, allow_pickle=True).item()

                # Set object scale and pose
                self.object_model.object_scale_tensor[0] = data_dict["scale"]
                obj_pose7d = data_dict["dual_arm_hand"]["obj_pose"]
                obj_pose = posQuat2Isometry3d(obj_pose7d[:3], quatWXYZ2XYZW(obj_pose7d[3:]))
                self.object_model.set_parameters(
                    poses=torch.tensor(obj_pose, device=self.device, dtype=torch.float32).unsqueeze(0)
                )

                grasp_qpos_dict = data_dict["dual_arm_hand"]["grasp_qpos"]
                grasp_qpos = build_qpos(grasp_qpos_dict, joint_names, device=self.device)
                self.dual_arm_hand_model.set_parameters(grasp_qpos.unsqueeze(0))

                #################### Visualization ####################

                # Dual-arm-hand mesh
                plot_dual_arm_hand = self.dual_arm_hand_model.get_plotly_data(0, opacity=1.0, color="lightslategray")

                # Object mesh
                plot_obj = self.object_model.get_plotly_data(i=0, opacity=1.0, color="coral")

                # Object surface points
                obj_surface_points = self.object_model.get_global_surface_points()
                obj_surface_points = obj_surface_points[0].cpu().detach().numpy()
                obj_surface_points_plot = go.Scatter3d(
                    x=obj_surface_points[:, 0],
                    y=obj_surface_points[:, 1],
                    z=obj_surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="blue", size=2),
                )

                # Robot surface poitns
                robot_surface_points = self.dual_arm_hand_model.get_global_surface_points()
                robot_surface_points = robot_surface_points[0].cpu().detach().numpy()
                robot_surface_points_plot = go.Scatter3d(
                    x=robot_surface_points[:, 0],
                    y=robot_surface_points[:, 1],
                    z=robot_surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="lightpink", size=2),
                )

                plot_lst = plot_dual_arm_hand + plot_obj + [robot_surface_points_plot] + [obj_surface_points_plot]
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

    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")

        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_energy()

        source_dir = self.config.task.source_dir

        if source_dir == "arm_filtered":
            self.run_vis_arm_hand_after_filter()
        else:
            self.run_vis_hand_before_filter()


def task_vis_single_grasp(cfg: DictConfig):
    experiment = GraspExperiment(cfg)
    experiment.run_full_experiment()

    return
