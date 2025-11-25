import os
import sys

sys.path.append(os.path.realpath("."))

import cProfile

try:
    import memory_profiler

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

import numpy as np
import torch
from tqdm import tqdm
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_dual_hand
from utils.bimanual_energy import BimanualEnergyComputer
from utils.bimanual_optimizer import MALAOptimizer
from utils.common import robust_compute_rotation_matrix_from_ortho6d
from torch.multiprocessing import set_start_method
import plotly.graph_objects as go
from utils.common import Logger

from utils.bimanual_handler import BimanualPair, save_grasp_results, EnergyTerms
from utils.common import setup_device, set_random_seeds, ensure_directory
from omegaconf import DictConfig, OmegaConf
from typing import List
import json


class GraspExperiment:
    """
    Main experiment class for bimanual grasp generation.
    """

    def __init__(self, config: DictConfig):
        # preprocess the config
        config.task.initialization = OmegaConf.merge(config.task.initialization, config.hand.initialization)

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
        self.object_code_list = None

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
        self.object_model.initialize(self.object_code_list)

        # Initialize dual hands
        left_hand_model, right_hand_model = initialize_dual_hand(
            right_hand_model, left_hand_model, self.object_model, self.config.task.initialization
        )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)

        # Save initial poses for optional debugging
        self.left_hand_pose_st = left_hand_model.hand_pose.detach()
        self.right_hand_pose_st = right_hand_model.hand_pose.detach()

        print(f"Left hand contact candidates: {left_hand_model.n_contact_candidates}")
        print(f"Right hand contact candidates: {right_hand_model.n_contact_candidates}")

    def setup_optimization(self):
        """Initialize optimizer and energy computer."""

        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)

        # Create optimizer
        self.optimizer = MALAOptimizer(
            self.bimanual_pair.left,
            self.bimanual_pair.right,
            config=self.config.task.optimizer,
            device=self.device,
            total_batch_size=len(self.object_code_list) * self.config.model.batch_size,
        )

    def setup_logging(self):
        """Setup experiment logging and result directories."""

        # Create directories
        self.logs_path = os.path.join(self.config.paths.experiments_base, self.config.name, "logs")
        self.results_path = os.path.join(self.config.paths.experiments_base, self.config.name, "results")
        ensure_directory(self.logs_path, clean=False)
        ensure_directory(self.results_path, clean=False)

        # Create logger
        self.logger = Logger(
            log_dir=self.logs_path,
            thres_fc=self.config.energy.thres_fc,
            thres_dis=self.config.energy.thres_dis,
            thres_pen=self.config.energy.thres_pen,
        )

        # Save experiment configuration
        config_path = os.path.join(self.results_path, "config.txt")
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.config))

    def run_optimization(self):
        """Run the optimization loop."""
        print("Starting optimization...")

        self.profiler.enable()

        # Initial energy computation
        energy_terms = self.energy_computer.compute_all_energies(self.bimanual_pair, self.object_model, verbose=True)

        energy_terms.total.sum().backward(retain_graph=True)
        self.logger.log(
            energy_terms.total,
            energy_terms.force_closure,
            energy_terms.distance,
            energy_terms.penetration,
            energy_terms.self_penetration,
            energy_terms.joint_limits,
            0,
            show=False,
        )

        # Main optimization loop
        for step in tqdm(range(1, self.config.task.optimizer.num_iterations + 1), desc="optimizing", miniters=1):
            # MALA proposal step with Langevin dynamics
            step_size = self.optimizer.langevin_proposal()

            # Zero gradients and compute new energy
            self.optimizer.zero_grad()
            new_energy_terms = self.energy_computer.compute_all_energies(
                self.bimanual_pair, self.object_model, verbose=True
            )

            new_energy_terms.total.sum().backward(retain_graph=True)

            # Metropolis-Hastings acceptance step
            with torch.no_grad():
                accept, temperature = self.optimizer.metropolis_hastings_step(
                    energy_terms.total, new_energy_terms.total
                )

                # Update energies for accepted samples
                energy_terms.total[accept] = new_energy_terms.total[accept]
                energy_terms.distance[accept] = new_energy_terms.distance[accept]
                energy_terms.force_closure[accept] = new_energy_terms.force_closure[accept]
                energy_terms.penetration[accept] = new_energy_terms.penetration[accept]
                energy_terms.self_penetration[accept] = new_energy_terms.self_penetration[accept]
                energy_terms.joint_limits[accept] = new_energy_terms.joint_limits[accept]
                energy_terms.wrench_volume[accept] = new_energy_terms.wrench_volume[accept]

                # Log progress
                self.logger.log(
                    energy_terms.total,
                    energy_terms.force_closure,
                    energy_terms.distance,
                    energy_terms.penetration,
                    energy_terms.self_penetration,
                    energy_terms.joint_limits,
                    step,
                    show=False,
                )

                if (step) % self.config.task.optimizer.intermediate_save_step == 0:
                    self.save_intermediate_results(step=step, energy_terms=energy_terms)

        self.profiler.disable()
        return energy_terms

    def save_intermediate_results(self, step: int, energy_terms: EnergyTerms):
        """Save intermediate results during optimization."""
        results_path = os.path.join(self.results_path, "intermediate")
        os.makedirs(results_path, exist_ok=True)

        save_grasp_results(
            results_path,
            self.object_code_list,
            self.config.model.batch_size,
            self.object_model,
            self.bimanual_pair,
            self.left_hand_pose_st,
            self.right_hand_pose_st,
            energy_terms,
            step=step,
        )

    def save_final_results(self, energy_terms: EnergyTerms):
        """Save final optimization results."""
        print("Saving final results...")
        results_path = os.path.join(self.results_path, "final")
        os.makedirs(results_path, exist_ok=True)

        save_grasp_results(
            results_path,
            self.object_code_list,
            self.config.model.batch_size,
            self.object_model,
            self.bimanual_pair,
            self.left_hand_pose_st,
            self.right_hand_pose_st,
            energy_terms,
            step=None,
        )

    def print_performance_stats(self):
        """Print performance and profiling statistics."""
        print("\n=== Performance Statistics ===")
        # self.profiler.print_stats()

        if HAS_MEMORY_PROFILER:
            try:
                memory_usage = memory_profiler.memory_usage(-1, interval=1)
                print(f"Peak memory usage: {max(memory_usage):.2f} MB")
            except (RuntimeError, OSError) as e:
                print(f"Memory profiling error: {e}")
        else:
            print("Memory profiling unavailable (memory_profiler not installed)")

    def run_full_experiment(self, object_code_list: List[str]) -> EnergyTerms:
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")

        self.object_code_list = object_code_list  # objects in this batch

        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_optimization()
        self.setup_logging()

        # Run optimization
        final_energy_terms = self.run_optimization()

        # Save results
        self.save_final_results(final_energy_terms)

        # Print statistics
        self.print_performance_stats()

        print(f"Experiment completed: {self.config.name}")
        return final_energy_terms


def task_synthesize(config: DictConfig):
    """
    Synthesizing bimanual grasps using the the DexGraspNet-pipeline method.
    This program runs on a single GPU.
    """

    # Print configuration summary
    print("=== Experiment Configuration ===")
    print(f"Name: {config.name}")
    print(f"Objects: {len(config.object_code_list)} objects")
    print(f"Batch size: {config.model.batch_size} per object")
    print(f"iterations: {config.task.optimizer.num_iterations}")
    print(
        f"Energy weights: dis={config.energy.w_dis}, pen={config.energy.w_pen}, vew={config.energy.w_vew}",
        f", joint={config.energy.w_joints}, spen={config.energy.w_spen}",
    )
    print(f"temperature: {config.task.optimizer.initial_temperature}")
    print(f"Langevin noise: {config.task.optimizer.langevin_noise_factor}")
    print("=" * 45)

    # Run experiment
    experiment = GraspExperiment(config)

    n_samples_per_obj = config.model.batch_size
    max_object_per_batch = config.model.max_total_batch_size // n_samples_per_obj

    def split_by_max_size(obj_list, max_object_per_batch):
        """
        Split obj_list into batches with at most max_object_per_batch items each.
        """
        return [obj_list[i : i + max_object_per_batch] for i in range(0, len(obj_list), max_object_per_batch)]

    if "object_code_list" in config:
        object_code_list = OmegaConf.to_object(config.object_code_list)
    else:
        with open(config.object_code_path, "r") as f:
            object_code_list = sorted(json.load(f))

    all_object_code_list = config.task.object_code_list
    batched_object_code_list = split_by_max_size(all_object_code_list, max_object_per_batch)

    for i_batch, object_code_list in enumerate(batched_object_code_list):
        print("\n=========================================")
        print(f"Batch id: {i_batch}")
        print("object_code_list: ", object_code_list)
        final_energy_terms = experiment.run_full_experiment(object_code_list)

    return
