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


def worker(gpu_id, output_path, object_code_path):
    with open(output_path, "w") as output_file:
        subprocess.call(
            f"CUDA_VISIBLE_DEVICES={gpu_id} python main_batch.py object_code_path={object_code_path}",
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

    # get the cfg
    multi_gpu_lst = cfg.model.multi_gpu_lst
    n_gpus = len(multi_gpu_lst)
    multi_gpu_cfg_path = os.path.join(cfg.paths.experiments_base, cfg.name, "multi_gpu_cfgs")
    os.makedirs(multi_gpu_cfg_path, exist_ok=True)

    def split_list(lst, n_batch):
        """
        Split lst into n_batch sublists as evenly as possible.
        """
        k, m = divmod(len(lst), n_batch)
        return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_batch)]

    # load the full object code list
    with open(cfg.object_code_path, "r") as f:
        all_object_code_list = json.load(f)

    # split the full object code list for each GPU
    batched_object_code_list = split_list(all_object_code_list, n_gpus)
    object_code_list_paths = []
    for i, object_code_list in enumerate(batched_object_code_list):
        path = os.path.join(multi_gpu_cfg_path, f"gpu_{multi_gpu_lst[i]}.json")
        object_code_list_paths.append(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(object_code_list, f, indent=4, ensure_ascii=False)

    # run separated synthesis on multiple GPU
    p_list = []
    for i, gpu_id in enumerate(multi_gpu_lst):
        output_path = os.path.join(multi_gpu_cfg_path, f"output_{i}.txt")
        p = multiprocessing.Process(
            target=worker,
            args=(
                gpu_id,
                output_path,
                object_code_list_paths[i],
            ),
        )
        p.start()
        print(f"create process :{p.pid}")
        p_list.append(p)

    for p in p_list:
        p.join()


if __name__ == "__main__":
    main()
