import os
import sys
import glob
import logging
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from omegaconf import DictConfig, OmegaConf
import trimesh as tm
import cProfile
from typing import List
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.bimanual_energy import BimanualEnergyComputer
from utils.bimanual_handler import BimanualPair, EnergyTerms, build_bimanual_three_poses
from utils.common import setup_device, set_random_seeds, TRANSLATION_NAMES, ROTATION_NAMES


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
        self.object_model = ObjectModel(
            data_root_path=self.config.paths.data_root_path,
            batch_size_each=self.config.model.batch_size * 3,  # consider pregrasp, grasp, and squeeze poses
            num_samples=self.config.model.num_samples,
            device=self.device,
            size=self.config.model.size,
            sdf_tool=self.config.model.object_sdf_tool,
            bodex_format=True,
        )
        self.bimanual_pair = BimanualPair(left_hand_model, right_hand_model, self.device)

    def setup_energy(self):
        """Initialize optimizer and energy computer."""

        # Create energy computer with optimized FC+VEW computation
        self.energy_computer = BimanualEnergyComputer(self.config.energy, self.device)

    @torch.no_grad()
    def run_filter(self, all_object_code_list):
        """
        Filtering synthesized grasps with energy-based metrics.
        Consider pregrasp, grasp, and squeeze poses.
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

        n_valid = 0
        n_all = 0
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
            right_hand_poses = torch.zeros(
                (n_obj, n_samples_per_obj, 3, 9 + len(right_joint_names)), device=self.device
            )
            left_hand_poses = torch.zeros((n_obj, n_samples_per_obj, 3, 9 + len(left_joint_names)), device=self.device)

            ############# Load grasp data files #############

            data_dict_lst_all_obj = []
            for i_obj, object_code in enumerate(object_code_list):
                # load synthesized grasps
                data_dict_lst = np.load(os.path.join(source_path, f"{object_code}.npy"), allow_pickle=True)[
                    :n_samples_per_obj
                ]
                data_dict_lst_all_obj.append(data_dict_lst)

                for i_grasp, data_dict in enumerate(data_dict_lst):
                    # set object scale
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

            self.bimanual_pair.right.set_parameters(right_hand_poses.reshape(n_obj * n_samples_per_obj * 3, -1))
            self.bimanual_pair.left.set_parameters(left_hand_poses.reshape(n_obj * n_samples_per_obj * 3, -1))

            ############# Filter based on energy #############

            energy_terms = self.energy_computer.compute_all_energies(
                self.bimanual_pair, self.object_model, verbose=True
            )
            max_joint_violation = self.bimanual_pair.compute_joint_limits_violation()

            # # DEBUG check
            # obj_idx = 0
            # grasp_idx = 2
            # print(f"obj_name: {object_code_list[obj_idx]}, grasp_idx: {grasp_idx}")

            # keys = ["total", "force_closure", "distance", "penetration", "self_penetration", "joint_limits", "wrench_volume"]
            # for key in keys:
            #     val = getattr(energy_terms, key).clone()
            #     val = val.reshape(n_obj, n_samples_per_obj)
            #     print(f"{key}: {val[obj_idx, grasp_idx]}")
            # a = 1

            thres_pen = self.config.task.thres.penetration
            thres_spen = self.config.task.thres.self_penetration
            thres_joint_limit = self.config.task.thres.joint_limit
            valid = (
                (energy_terms.penetration < thres_pen)  # hand-object peneration
                & (energy_terms.self_penetration < thres_spen)  # hand self peneration
                & (max_joint_violation < thres_joint_limit)  # joint bound
            )
            valid = valid.reshape(n_obj, n_samples_per_obj, 3)

            valid = valid[:, :, 0] & valid[:, :, 1]  # consider only the pregrasp and gras poses

            n_valid += valid.sum().item()
            n_all += valid.numel()

            # Saving
            save_dir = os.path.join(exp_path, self.config.task.save_dir)
            for i_obj, object_code in enumerate(object_code_list):
                for i_grasp in range(len(data_dict_lst)):
                    if valid[i_obj, i_grasp] and valid[i_obj, i_grasp]:
                        save_path = os.path.join(save_dir, object_code, f"grasp_{i_grasp}.npy")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        np.save(save_path, data_dict_lst_all_obj[i_obj][i_grasp])
                        logging.info(f"Batch {i_batch} - Save filtered grasp data to {save_path}.")

        logging.info("===============================================")
        logging.info(f"Passed grasp ratio (all): {n_valid / n_all}.")
        logging.info("===============================================")

    def run_full_experiment(self, object_code_list: List[str]):
        """Run the complete experiment pipeline."""
        print(f"Starting experiment: {self.config.name}")

        # Setup pipeline
        self.setup_environment()
        self.setup_models()
        self.setup_energy()

        self.run_filter(object_code_list)


def task_filter(cfg: DictConfig):
    """
    Filtering the synthesized bimanual grasps via energy.
    Considering the peneration and self-peneration of the pregrasp and grasp poses.
    No arms.
    """

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
