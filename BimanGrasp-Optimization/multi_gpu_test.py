import os
import numpy as np
from glob import glob
import argparse
import subprocess
import multiprocessing
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import json


def worker(output_path, command):
    with open(output_path, "w") as output_file:
        subprocess.call(
            command,
            shell=True,
            stdout=output_file,
            stderr=output_file,
        )


@hydra.main(config_path="cfg", config_name="base", version_base=None)  # must use version_base=None for compatibility
def main(cfg: DictConfig):
    """Hydra entrypoint. Builds ExperimentConfig from config.yaml and runs the experiment.

    The optional `args` parameter is accepted for compatibility but not required or used by
    the function. This allows callers to pass a second positional argument without breaking
    the Hydra-decorated entrypoint.
    """

    commands = [
        "python main_batch.py name=server_30 gpu=0 optimizer.mean_ema_grad_weight=1.0 optimizer.joint_limit_clamp=True",
        "python main_batch.py name=server_31 gpu=1 optimizer.mean_ema_grad_weight=0.5 optimizer.joint_limit_clamp=True",
        "python main_batch.py name=server_32 gpu=2 optimizer.mean_ema_grad_weight=0.0 optimizer.joint_limit_clamp=True",
        "python main_batch.py name=server_33 gpu=3 optimizer.mean_ema_grad_weight=1.0 optimizer.joint_limit_clamp=True energy.w_joints=5.0",
        "python main_batch.py name=server_34 gpu=4 optimizer.mean_ema_grad_weight=1.0 optimizer.joint_limit_clamp=True energy.w_joints=10.0",
    ]

    multi_gpu_cfg_path = os.path.join(cfg.paths.experiments_base, cfg.name, "multi_gpu")
    os.makedirs(multi_gpu_cfg_path, exist_ok=True)

    # run separated synthesis on multiple GPU
    p_list = []
    for i, command in enumerate(commands):
        output_path = os.path.join(multi_gpu_cfg_path, f"output_{i}.txt")
        p = multiprocessing.Process(
            target=worker,
            args=(
                output_path,
                command,
            ),
        )
        p.start()
        print(f"create process :{p.pid}")
        p_list.append(p)

    for p in p_list:
        p.join()


if __name__ == "__main__":
    main()
