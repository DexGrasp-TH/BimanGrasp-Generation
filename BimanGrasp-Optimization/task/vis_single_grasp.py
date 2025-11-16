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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_dual_hand
from utils.bimanual_energy import calculate_energy, cal_energy, BimanualEnergyComputer
from utils.common import robust_compute_rotation_matrix_from_ortho6d
from utils.common import Logger
from utils.config import ExperimentConfig, create_config_from_args, TRANSLATION_NAMES, ROTATION_NAMES
from utils.bimanual_handler import BimanualPair, save_grasp_results, EnergyTerms
from utils.common import setup_device, set_random_seeds, ensure_directory


def get_scene(plot_lst):
    # --- 从plotly数据中提取顶点 ---
    xs, ys, zs = [], [], []
    for trace in plot_lst:
        if hasattr(trace, "x") and hasattr(trace, "y") and hasattr(trace, "z"):
            xs.extend(trace.x)
            ys.extend(trace.y)
            zs.extend(trace.z)

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # --- 计算范围 ---
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()

    scene_fixed = dict(
        xaxis=dict(range=[xmin, xmax], visible=False, showgrid=False),
        yaxis=dict(range=[ymin, ymax], visible=False, showgrid=False),
        zaxis=dict(range=[zmin, zmax], visible=False, showgrid=False),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene_fixed


def experiment_config_from_dict(cfg: DictConfig) -> ExperimentConfig:
    """Convert a Hydra DictConfig to ExperimentConfig dataclass."""
    exp = ExperimentConfig()

    # Top level simple fields
    for key in ("name", "seed", "gpu"):
        if key in cfg:
            setattr(exp, key, cfg.get(key))

    # Object code list (keep as python list)
    if "object_code_list" in cfg:
        if cfg.object_code_list:
            exp.object_code_list = OmegaConf.to_object(cfg.object_code_list)
        else:
            with open(cfg.object_code_path, "r") as f:
                exp.object_code_list = sorted(json.load(f))

    # Helper to apply nested dict to dataclass-like object
    def apply_section(section_name, target_obj):
        if section_name in cfg:
            sec = cfg.get(section_name)
            for k, v in sec.items():
                if hasattr(target_obj, k):
                    # convert lists/dicts to native Python
                    val = OmegaConf.to_object(v) if isinstance(v, (dict, list)) else v
                    setattr(target_obj, k, val)

    apply_section("hand_params", exp.hand)
    apply_section("paths", exp.paths)
    apply_section("energy", exp.energy)
    apply_section("optimizer", exp.optimizer)
    apply_section("initialization", exp.initialization)
    apply_section("model", exp.model)

    return exp


# --- Utility to build hand pose tensor ---
def build_hand_pose(qpos, translation_names, rot_names, joint_names, device):
    """Build a torch tensor for hand pose given qpos dict."""
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()  # flatten first two rotation columns
    hand_pose = torch.tensor(
        [qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    return hand_pose


class GraspExperiment:
    """
    Main experiment class for bimanual grasp generation.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg  # hydra
        self.device = None
        self.bimanual_pair = None
        self.object_model = None
        self.optimizer = None
        self.energy_computer = None
        self.logger = None

        # Convert to ExperimentConfig dataclass
        self.config: ExperimentConfig = experiment_config_from_dict(cfg)

        # Profiling
        self.profiler = cProfile.Profile()

        # State tracking
        self.left_hand_pose_st = None
        self.right_hand_pose_st = None

    def setup_environment(self):
        """Setup device, random seeds, and environment variables."""
        self.device = setup_device(self.config.gpu)
        set_random_seeds(self.config.seed)
        np.seterr(all="raise")

        print(f"Using device: {self.device}")

    def setup_models(self):
        """Initialize hand and object models."""
        print("Setting up models...")

        # Create right hand model
        right_hand_model = HandModel(
            mjcf_path=self.config.paths.right_hand_mjcf,
            contact_points_path=self.config.paths.right_contact_points,
            penetration_points_path=self.config.paths.penetration_points,
            device=self.device,
            n_surface_points=self.config.model.n_surface_points,
            handedness="right_hand",
        )
        left_hand_model = HandModel(
            mjcf_path=self.config.paths.left_hand_mjcf,
            contact_points_path=self.config.paths.left_contact_points,
            penetration_points_path=self.config.paths.penetration_points,
            device=self.device,
            n_surface_points=self.config.model.n_surface_points,
            handedness="left_hand",
        )

        # Create object model
        self.object_model = ObjectModel(
            data_root_path=self.config.paths.data_root_path,
            batch_size_each=1,
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size,
        )

        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)

    def setup_energy(self):
        """Initialize optimizer and energy computer."""

        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)

    def run_vis(self):
        """
        Filtering synthesized grasps with energy-based metrics.
        """

        exp_path = os.path.join(self.config.paths.experiments_base, self.config.name)
        result_path = os.path.join(exp_path, "results")

        right_joint_names = self.bimanual_pair.right.get_joint_names()
        left_joint_names = self.bimanual_pair.left.get_joint_names()

        object_code_list = self.cfg.task.object_code_list
        grasp_indices = self.cfg.task.grasp_indices

        for object_code in object_code_list:
            for grasp_idx in grasp_indices:
                self.object_model.initialize([object_code])

                # load synthesized grasps
                data_dict_lst = np.load(os.path.join(result_path, f"{object_code}.npy"), allow_pickle=True)
                data_dict = data_dict_lst[grasp_idx]

                right_qpos = data_dict["qpos_right"]
                left_qpos = data_dict["qpos_left"]
                obj_scale = data_dict["scale"]

                # set object scale
                self.object_model.object_scale_tensor[0] = obj_scale

                right_hand_pose = build_hand_pose(
                    right_qpos, TRANSLATION_NAMES, ROTATION_NAMES, right_joint_names, self.device
                )
                left_hand_pose = build_hand_pose(
                    left_qpos, TRANSLATION_NAMES, ROTATION_NAMES, left_joint_names, self.device
                )

                self.bimanual_pair.right.set_parameters(right_hand_pose.unsqueeze(0))  # batch size = 1
                self.bimanual_pair.left.set_parameters(left_hand_pose.unsqueeze(0))

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

                # --- Visualization ---
                # Final poses (solid colors)
                right_plot = self.bimanual_pair.right.get_plotly_data(
                    i=0, opacity=1.0, color="lightslategray", with_contact_points=False
                )
                left_plot = self.bimanual_pair.left.get_plotly_data(
                    i=0, opacity=1.0, color="lightslategray", with_contact_points=False
                )
                object_plot = self.object_model.get_plotly_data(i=0, color="seashell", opacity=1.0)

                # object surface points
                obj_surface_points = (
                    self.object_model.object_scale_tensor[0] * self.object_model.surface_points_tensor[0]
                )
                obj_surface_points = obj_surface_points.cpu().detach().numpy()
                obj_surface_points_plot = go.Scatter3d(
                    x=obj_surface_points[:, 0],
                    y=obj_surface_points[:, 1],
                    z=obj_surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="blue", size=5),
                )

                # hand surface poitns
                hand_surface_points_right = self.bimanual_pair.right.get_global_surface_points()
                hand_surface_points_left = self.bimanual_pair.left.get_global_surface_points()
                hand_surface_points = torch.cat([hand_surface_points_right, hand_surface_points_left], dim=1)
                hand_surface_points = hand_surface_points[0].cpu().detach().numpy()
                hand_surface_points_plot = go.Scatter3d(
                    x=hand_surface_points[:, 0],
                    y=hand_surface_points[:, 1],
                    z=hand_surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                )

                # Combine everything
                plot_lst = right_plot + left_plot + object_plot + [obj_surface_points_plot, hand_surface_points_plot]
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

        self.run_vis()


def task_vis_single_grasp(cfg: DictConfig):
    # merge cfg.hand.paths into cfg.paths
    cfg.paths = OmegaConf.merge(cfg.paths, cfg.hand.paths)
    cfg.hand_params = OmegaConf.merge(cfg.hand_params, cfg.hand.hand_params)
    cfg.initialization = OmegaConf.merge(cfg.initialization, cfg.hand.initialization)

    experiment = GraspExperiment(cfg)
    experiment.run_full_experiment()

    return
