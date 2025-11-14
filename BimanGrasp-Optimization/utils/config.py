"""
Configuration management for bimanual grasp generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import os


# Global Constants - Transform Names
TRANSLATION_NAMES = ["WRJTx", "WRJTy", "WRJTz"]
ROTATION_NAMES = ["WRJRx", "WRJRy", "WRJRz"]


@dataclass
class HandConfig:
    name = None


@dataclass
class PathConfig:
    """File and directory path configuration."""

    # MJCF and mesh paths
    right_hand_mjcf: str = ""
    right_hand_vis_mjcf: str = ""
    left_hand_mjcf: str = ""
    left_hand_vis_mjcf: str = ""
    mesh_path: str = ""
    right_contact_points: str = ""
    left_contact_points: str = ""
    penetration_points: str = ""

    # Data paths
    # data_root_path: str = '../data/meshdata'
    data_root_path: str = "../data/object/DGN_2k/processed_data"  # by mingrui
    experiments_base: str = "../data/experiments"
    results_base: str = "data/graspdata"

    @property
    def experiment_path(self) -> str:
        return self.experiments_base

    def get_experiment_logs_path(self, exp_name: str) -> str:
        return os.path.join(self.experiments_base, exp_name, "logs")

    def get_experiment_results_path(self, exp_name: str) -> str:
        return os.path.join(self.experiments_base, exp_name, "results")


@dataclass
class EnergyConfig:
    """Energy function weights and thresholds."""

    # Energy weights - the "magic" parameters
    w_dis: float = 100.0  # Contact distance weight
    w_pen: float = 125.0  # Penetration penalty weight
    w_spen: float = 10.0  # Self-penetration weight
    w_joints: float = 1.0  # Joint limit weight
    w_vew: float = 0.5  # Wrench ellipse volume weight

    # Energy thresholds for filtering
    thres_fc: float = 0.45  # Force closure threshold
    thres_dis: float = 0.005  # Distance threshold
    thres_pen: float = 0.001  # Penetration threshold

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for energy function calls."""
        return {
            "w_dis": self.w_dis,
            "w_pen": self.w_pen,
            "w_spen": self.w_spen,
            "w_joints": self.w_joints,
            "w_vew": self.w_vew,
        }


@dataclass
class OptimizerConfig:
    """Optimization algorithm configuration."""

    # Annealing parameters
    switch_possibility: float = 0.5  # Contact point switching probability
    starting_temperature: float = 18  # Initial annealing temperature
    temperature_decay: float = 0.95  # Temperature decay rate
    annealing_period: int = 30  # Annealing period steps

    # Step size parameters
    step_size: float = 0.005  # Base step size
    stepsize_period: int = 50  # Step size decay period
    momentum: float = 0.98  # RMSProp momentum parameter

    # Iteration settings
    num_iterations: int = 10000  # Number of optimization iterations

    joint_limit_clamp = False
    individual_ema_grad = False
    mean_ema_grad_weight = 1.0

    # Compatibility properties for MALAOptimizer
    @property
    def initial_temperature(self) -> float:
        """Alias for starting_temperature (MALA compatibility)."""
        return self.starting_temperature

    @property
    def cooling_schedule(self) -> float:
        """Alias for temperature_decay (MALA compatibility)."""
        return self.temperature_decay

    @property
    def preconditioning_decay(self) -> float:
        """Alias for momentum (MALA compatibility)."""
        return self.momentum

    @property
    def langevin_noise_factor(self) -> float:
        """Langevin noise factor (default for MALA)."""
        return 0.1


@dataclass
class InitializationConfig:
    """Hand initialization parameters."""

    # Spatial constraints
    distance_lower: float = 0.2  # Minimum initial distance from object
    distance_upper: float = 0.3  # Maximum initial distance from object
    theta_lower: float = -math.pi / 6  # Minimum rotation angle
    theta_upper: float = math.pi / 6  # Maximum rotation angle

    # Joint initialization
    jitter_strength: float = 0.1  # Joint angle randomization strength

    # Contact points
    num_contacts: int = 4  # Number of contact points per hand

    left_hand_joint_mu = None
    right_hand_joint_mu = None


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    # Hand model parameters
    n_surface_points: int = 2000  # Number of surface points for sampling

    # Object model parameters
    num_samples: int = 2000  # Number of object surface samples
    size: str = "large"  # Object size setting

    # Batch processing
    batch_size: int = 128  # Batch size for single experiments
    # batch_size_each: int = 5  # Batch size per object for large-scale
    max_total_batch_size: int = 100  # Maximum total batch size for multi-GPU

    sdf_tool: str = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Basic settings
    name: str = "exp_2025"  # Experiment name
    seed: int = 1  # Random seed
    gpu: str = "0"  # GPU device ID

    # Object selection
    object_code_list: List[str] = field(
        default_factory=lambda: [
            "Cole_Hardware_Dishtowel_Multicolors",
            "Curver_Storage_Bin_Black_Small",
            "Hasbro_Monopoly_Hotels_Game",
            "Breyer_Horse_Of_The_Year_2015",
            "Schleich_S_Bayala_Unicorn_70432",
        ]
    )

    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hand: HandConfig = field(default_factory=HandConfig)

    # Derived properties
    @property
    def total_batch_size(self) -> int:
        return len(self.object_code_list) * self.model.batch_size

    @property
    def device_str(self) -> str:
        return f"cuda:{self.gpu}" if self.gpu != "cpu" else "cpu"

    def update_from_args(self, args) -> None:
        """Update configuration from argparse Namespace."""
        # Update basic settings
        for attr in ["name", "seed", "gpu"]:
            if hasattr(args, attr):
                setattr(self, attr, getattr(args, attr))

        # Update object list
        if hasattr(args, "object_code_list"):
            self.object_code_list = args.object_code_list

        # Update energy weights
        energy_attrs = ["w_dis", "w_pen", "w_spen", "w_joints"]
        for attr in energy_attrs:
            if hasattr(args, attr):
                setattr(self.energy, attr, getattr(args, attr))

        # Update optimizer parameters
        opt_attrs = [
            "switch_possibility",
            "starting_temperature",
            "temperature_decay",
            "annealing_period",
            "step_size",
            "stepsize_period",
            "momentum",
            "num_iterations",
        ]
        for attr in opt_attrs:
            if hasattr(args, attr):
                setattr(self.optimizer, attr, getattr(args, attr))

        # Update initialization parameters
        init_attrs = [
            "distance_lower",
            "distance_upper",
            "theta_lower",
            "theta_upper",
            "jitter_strength",
            "num_contacts",
        ]
        for attr in init_attrs:
            if hasattr(args, attr):
                setattr(self.initialization, attr, getattr(args, attr))

        # Update model parameters
        # model_attrs = ["batch_size", "batch_size_each", "max_total_batch_size"]
        model_attrs = ["batch_size", "max_total_batch_size"]
        for attr in model_attrs:
            if hasattr(args, attr):
                setattr(self.model, attr, getattr(args, attr))

        # Update energy thresholds
        thresh_attrs = ["thres_fc", "thres_dis", "thres_pen"]
        for attr in thresh_attrs:
            if hasattr(args, attr):
                setattr(self.energy, attr, getattr(args, attr))


DEFAULT_CONFIG = ExperimentConfig()


def get_config(config_override: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """
    Get configuration with optional overrides.

    Args:
        config_override: Dictionary of configuration overrides

    Returns:
        ExperimentConfig: Configuration instance
    """
    config = ExperimentConfig()

    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Try to set nested attributes
                parts = key.split(".")
                if len(parts) == 2 and hasattr(config, parts[0]):
                    sub_config = getattr(config, parts[0])
                    if hasattr(sub_config, parts[1]):
                        setattr(sub_config, parts[1], value)

    return config


def create_config_from_args(args) -> ExperimentConfig:
    """
    Create configuration from command line arguments.

    Args:
        args: argparse.Namespace from command line arguments

    Returns:
        ExperimentConfig: Configuration instance
    """
    config = ExperimentConfig()
    config.update_from_args(args)
    return config
