import os
import sys

sys.path.append(os.path.realpath("."))

import cProfile

try:
    import memory_profiler

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import random
import transforms3d
import shutil
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_dual_hand
from utils.bimanual_energy import calculate_energy, cal_energy, BimanualEnergyComputer
from utils.bimanual_optimizer import MALAOptimizer
from utils.common import robust_compute_rotation_matrix_from_ortho6d
from torch.multiprocessing import set_start_method
import plotly.graph_objects as go
from utils.common import Logger
from utils.config import ExperimentConfig, create_config_from_args
from utils.bimanual_handler import BimanualPair, save_grasp_results, EnergyTerms
from utils.common import setup_device, set_random_seeds, ensure_directory
from omegaconf import DictConfig, OmegaConf
import hydra


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
        self.object_model.initialize(self.config.object_code_list)

        # Initialize dual hands
        left_hand_model, right_hand_model = initialize_dual_hand(
            right_hand_model, left_hand_model, self.object_model, self.config.initialization
        )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)

        # Save initial poses for optional debugging
        self.left_hand_pose_st = left_hand_model.hand_pose.detach()
        self.right_hand_pose_st = right_hand_model.hand_pose.detach()

        print(f"Left hand contact candidates: {left_hand_model.n_contact_candidates}")
        print(f"Right hand contact candidates: {right_hand_model.n_contact_candidates}")
        print(f"Total batch size: {self.config.total_batch_size}")

    def setup_optimization(self):
        """Initialize optimizer and energy computer."""

        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)

        # Create optimizer
        self.optimizer = MALAOptimizer(
            self.bimanual_pair.left, self.bimanual_pair.right, config=self.config.optimizer, device=self.device
        )

    def run_full_experiment(self) -> EnergyTerms:
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")

        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_optimization()

        left_hand_model = self.bimanual_pair.left
        right_hand_model = self.bimanual_pair.right
        object_model = self.object_model
        device = self.device

        # load results
        result_path = "../data/experiments/debug/results"
        object_code = "core_bottle_1a7ba1f4c892e2da30711cdbdbc73924"
        step = 10000
        idx = 3
        data_dict = np.load(os.path.join(result_path, object_code + f"_{step}.npy"), allow_pickle=True)[idx]

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

        # --- Load qpos and construct hand poses ---
        translation_names = ["WRJTx", "WRJTy", "WRJTz"]
        rot_names = ["WRJRx", "WRJRy", "WRJRz"]

        # Right hand
        right_qpos = data_dict["qpos_right"]
        right_hand_pose = build_hand_pose(
            right_qpos, translation_names, rot_names, right_hand_model.get_joint_names(), device
        )
        # Left hand
        left_qpos = data_dict["qpos_left"]
        left_hand_pose = build_hand_pose(
            left_qpos, translation_names, rot_names, left_hand_model.get_joint_names(), device
        )

        right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
        left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))

        self.bimanual_pair.compute_joint_limits_energy()

        a = 1


def experiment_config_from_dict(cfg: DictConfig) -> ExperimentConfig:
    """Convert a Hydra DictConfig to ExperimentConfig dataclass."""
    exp = ExperimentConfig()

    # Top level simple fields
    for key in ("name", "seed", "gpu"):
        if key in cfg:
            setattr(exp, key, cfg.get(key))

    # Object code list (keep as python list)
    if "object_code_list" in cfg:
        exp.object_code_list = OmegaConf.to_object(cfg.object_code_list)

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


@hydra.main(config_path="../cfg", config_name="base", version_base=None)  # must use version_base=None for compatibility
def main(cfg: DictConfig):
    """Hydra entrypoint. Builds ExperimentConfig from config.yaml and runs the experiment.

    The optional `args` parameter is accepted for compatibility but not required or used by
    the function. This allows callers to pass a second positional argument without breaking
    the Hydra-decorated entrypoint.
    """

    # merge cfg.hand.paths into cfg.paths
    cfg.paths = OmegaConf.merge(cfg.paths, cfg.hand.paths)
    cfg.hand_params = OmegaConf.merge(cfg.hand_params, cfg.hand.hand_params)
    cfg.initialization = OmegaConf.merge(cfg.initialization, cfg.hand.initialization)

    # Convert to ExperimentConfig dataclass
    config = experiment_config_from_dict(cfg)

    # Print configuration summary
    print("=== Experiment Configuration ===")
    print(f"Name: {config.name}")
    print(f"Objects: {len(config.object_code_list)} objects")
    print(f"Batch size: {config.model.batch_size} per object")
    print(f"Total batch: {config.total_batch_size}")
    print(f"iterations: {config.optimizer.num_iterations}")
    print(f"Energy weights: dis={config.energy.w_dis}, pen={config.energy.w_pen}, vew={config.energy.w_vew}")
    print(f"temperature: {config.optimizer.initial_temperature}")
    print(f"Langevin noise: {config.optimizer.langevin_noise_factor}")
    print("=" * 45)

    # Run experiment
    experiment = GraspExperiment(config)
    experiment.run_full_experiment()


if __name__ == "__main__":
    main()
