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
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Retrieve raw CLI args (before Hydra parsing)
    original_args = HydraConfig.get().args

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_code", type=str, default="core_bottle_1a7ba1f4c892e2da30711cdbdbc73924")
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--no_st", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--result_path", type=str, default="../data/experiments/debug/results")

    args = parser.parse_args(original_args)

    print("Parsed CLI args:", args)
    print("Hydra config:", OmegaConf.to_yaml(cfg))


main()
