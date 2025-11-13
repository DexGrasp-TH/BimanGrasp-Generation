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
import pyrender
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

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = None
        self.bimanual_pair = None
        self.object_model = None
        self.optimizer = None
        self.energy_computer = None
        self.logger = None

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
            batch_size_each=self.config.model.batch_size,
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size,
        )

        # Initialize dual hands
        # left_hand_model, right_hand_model = initialize_dual_hand(
        #     right_hand_model, left_hand_model, self.object_model, self.config.initialization
        # )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)

    def setup_energy(self):
        """Initialize optimizer and energy computer."""

        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)

    def setup_logging(self):
        """Setup experiment logging and result directories."""

        # Create directories
        logs_path = self.config.paths.get_experiment_logs_path(self.config.name)
        results_path = self.config.paths.get_experiment_results_path(self.config.name)

        ensure_directory(logs_path, clean=False)
        ensure_directory(results_path, clean=False)

        # Create logger
        self.logger = Logger(
            log_dir=logs_path,
            thres_fc=self.config.energy.thres_fc,
            thres_dis=self.config.energy.thres_dis,
            thres_pen=self.config.energy.thres_pen,
        )

        # Save experiment configuration
        config_path = os.path.join(results_path, "config.txt")
        with open(config_path, "w") as f:
            f.write(str(self.config))

    def run_filter(self, all_object_code_list):
        """
        Filtering synthesized grasps with energy-based metrics.
        """

        exp_path = os.path.join(self.config.paths.experiments_base, self.config.name)
        result_path = os.path.join(exp_path, "results")

        right_joint_names = self.bimanual_pair.right.get_joint_names()
        left_joint_names = self.bimanual_pair.left.get_joint_names()

        # split all objects into batches
        n_samples_per_obj = self.config.model.batch_size
        max_object_per_batch = self.config.model.max_total_batch_size // n_samples_per_obj

        def split_by_max_size(obj_list, max_object_per_batch):
            """
            Split obj_list into batches with at most max_object_per_batch items each.
            """
            return [obj_list[i : i + max_object_per_batch] for i in range(0, len(obj_list), max_object_per_batch)]

        batched_object_code_list = split_by_max_size(all_object_code_list, max_object_per_batch)

        # Process the objects in batch
        for i_batch, object_code_list in enumerate(batched_object_code_list):
            self.object_model.initialize(object_code_list)
            n_obj = len(object_code_list)
            right_hand_poses = torch.zeros((n_obj, n_samples_per_obj, 
                                            9 + len(right_joint_names)), device=self.device)
            left_hand_poses = torch.zeros((n_obj, n_samples_per_obj, 
                                            9 + len(left_joint_names)), device=self.device)

            for i_obj, object_code in enumerate(object_code_list):
                # load synthesized grasps
                data_dict_lst = np.load(os.path.join(result_path, f"{object_code}.npy"), allow_pickle=True)

                for i_grasp, data_dict in enumerate(data_dict_lst):
                    right_qpos = data_dict["qpos_right"]
                    left_qpos = data_dict["qpos_left"]
                    obj_scale = data_dict["scale"]

                    # set object scale
                    self.object_model.object_scale_tensor[i_obj, i_grasp] = obj_scale

                    right_hand_pose = build_hand_pose(
                        right_qpos, TRANSLATION_NAMES, ROTATION_NAMES, right_joint_names, self.device
                    )
                    left_hand_pose = build_hand_pose(
                        left_qpos, TRANSLATION_NAMES, ROTATION_NAMES, left_joint_names, self.device
                    )
                    right_hand_poses[i_obj, i_grasp, :] = right_hand_pose
                    left_hand_poses[i_obj, i_grasp, :] = left_hand_pose

            self.bimanual_pair.right.set_parameters(right_hand_poses.reshape(n_obj * n_samples_per_obj, -1))
            self.bimanual_pair.left.set_parameters(left_hand_poses.reshape(n_obj * n_samples_per_obj, -1))

            energy_terms = self.energy_computer.compute_all_energies(self.bimanual_pair, self.object_model, verbose=True)

            
            obj_idx = 0
            grasp_idx = 0
            print(f"obj_name: {object_code_list[obj_idx]}, grasp_idx: {grasp_idx}")

            keys = ["total", "force_closure", "distance", "penetration", "self_penetration", "joint_limits", "wrench_volume"]
            for key in keys:
                val = getattr(energy_terms, key).clone()
                val = val.reshape(n_obj, n_samples_per_obj)
                print(f"{key}: {val[obj_idx, grasp_idx]}")

            a = 1

    def run_full_experiment(self, object_code_list: List[str]):
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")

        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_energy()
        self.setup_logging()

        self.run_filter(object_code_list)


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



def task_filter(cfg: DictConfig):

    # merge cfg.hand.paths into cfg.paths
    cfg.paths = OmegaConf.merge(cfg.paths, cfg.hand.paths)
    cfg.hand_params = OmegaConf.merge(cfg.hand_params, cfg.hand.hand_params)
    cfg.initialization = OmegaConf.merge(cfg.initialization, cfg.hand.initialization)

    # Convert to ExperimentConfig dataclass
    config = experiment_config_from_dict(cfg)

    experiment = GraspExperiment(config)

    experiment.run_full_experiment(config.object_code_list)

    return
