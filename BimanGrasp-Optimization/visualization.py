import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
import plotly.io as pio
from omegaconf import DictConfig, OmegaConf
import hydra

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.config import ExperimentConfig, create_config_from_args

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]
joint_names = [
    "robot0:FFJ3",
    "robot0:FFJ2",
    "robot0:FFJ1",
    "robot0:FFJ0",
    "robot0:MFJ3",
    "robot0:MFJ2",
    "robot0:MFJ1",
    "robot0:MFJ0",
    "robot0:RFJ3",
    "robot0:RFJ2",
    "robot0:RFJ1",
    "robot0:RFJ0",
    "robot0:LFJ4",
    "robot0:LFJ3",
    "robot0:LFJ2",
    "robot0:LFJ1",
    "robot0:LFJ0",
    "robot0:THJ4",
    "robot0:THJ3",
    "robot0:THJ2",
    "robot0:THJ1",
    "robot0:THJ0",
]


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

    apply_section("paths", exp.paths)
    apply_section("energy", exp.energy)
    apply_section("optimizer", exp.optimizer)
    apply_section("initialization", exp.initialization)
    apply_section("model", exp.model)

    return exp


@hydra.main(config_path="cfg", config_name="base", version_base=None)  # must use version_base=None for compatibility
def main(cfg: DictConfig):
    """Hydra entrypoint. Builds ExperimentConfig from config.yaml and runs the experiment.

    The optional `args` parameter is accepted for compatibility but not required or used by
    the function. This allows callers to pass a second positional argument without breaking
    the Hydra-decorated entrypoint.
    """

    # merge cfg.hand.paths into cfg.paths
    cfg.paths = OmegaConf.merge(cfg.paths, cfg.hand.paths)

    # Convert to ExperimentConfig dataclass
    config = experiment_config_from_dict(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_code", type=str, default="core_bottle_1a7ba1f4c892e2da30711cdbdbc73924")
    parser.add_argument("--num", type=int, default=2)
    parser.add_argument("--no_st", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--result_path", type=str, default="../data/experiments/debug/results")

    args = parser.parse_args()

    # hyper-parameters
    object_code = args.object_code
    grasp_idx = args.num
    result_path = args.result_path
    device = "cpu"
    load_intermediate_results = True
    step = 10000

    right_hand_model = HandModel(
        mjcf_path=config.paths.right_hand_vis_mjcf,
        mesh_path=config.paths.mesh_path,
        contact_points_path=config.paths.right_contact_points,
        penetration_points_path=config.paths.penetration_points,
        device=device,
        n_surface_points=config.model.n_surface_points,
        handedness="right_hand",
    )
    left_hand_model = HandModel(
        mjcf_path=config.paths.left_hand_vis_mjcf,
        mesh_path=config.paths.mesh_path,
        contact_points_path=config.paths.left_contact_points,
        penetration_points_path=config.paths.penetration_points,
        device=device,
        n_surface_points=config.model.n_surface_points,
        handedness="left_hand",
    )

    object_model = ObjectModel(
        data_root_path=config.paths.data_root_path,
        batch_size_each=1,
        num_samples=2000,
        device=device,
        size="large",
        bodex_format=True,
    )
    # load results

    if load_intermediate_results:
        data_dict = np.load(os.path.join(result_path, object_code + f"_{step}.npy"), allow_pickle=True)[grasp_idx]
    else:
        data_dict = np.load(os.path.join(result_path, object_code + ".npy"), allow_pickle=True)[grasp_idx]

    right_qpos = data_dict["qpos_right"]
    right_rot = np.array(transforms3d.euler.euler2mat(*[right_qpos[name] for name in rot_names]))
    right_rot = right_rot[:, :2].T.ravel().tolist()
    right_hand_pose = torch.tensor(
        [right_qpos[name] for name in translation_names] + right_rot + [right_qpos[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    if "qpos_right_st" in data_dict:
        right_qpos_st = data_dict["qpos_right_st"]
        right_rot = np.array(transforms3d.euler.euler2mat(*[right_qpos_st[name] for name in rot_names]))
        right_rot = right_rot[:, :2].T.ravel().tolist()
        right_hand_pose_st = torch.tensor(
            [right_qpos_st[name] for name in translation_names]
            + right_rot
            + [right_qpos_st[name] for name in joint_names],
            dtype=torch.float,
            device=device,
        )

    # load left results
    left_qpos = data_dict["qpos_left"]
    left_rot = np.array(transforms3d.euler.euler2mat(*[left_qpos[name] for name in rot_names]))
    left_rot = left_rot[:, :2].T.ravel().tolist()
    left_hand_pose = torch.tensor(
        [left_qpos[name] for name in translation_names] + left_rot + [left_qpos[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    if "qpos_left_st" in data_dict:
        left_qpos_st = data_dict["qpos_left_st"]
        left_rot = np.array(transforms3d.euler.euler2mat(*[left_qpos_st[name] for name in rot_names]))
        left_rot = left_rot[:, :2].T.ravel().tolist()
        left_hand_pose_st = torch.tensor(
            [left_qpos_st[name] for name in translation_names]
            + left_rot
            + [left_qpos_st[name] for name in joint_names],
            dtype=torch.float,
            device=device,
        )

    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict["scale"], dtype=torch.float, device=device).reshape(1, 1)

    right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
    right_hand_en_plotly = right_hand_model.get_plotly_data(
        i=0, opacity=1, color="lightslategray", with_contact_points=False
    )

    left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))
    left_hand_en_plotly = left_hand_model.get_plotly_data(
        i=0, opacity=1, color="lightslategray", with_contact_points=False
    )
    object_plotly = object_model.get_plotly_data(i=0, color="seashell", opacity=1)

    fig = go.Figure(right_hand_en_plotly + object_plotly + left_hand_en_plotly)

    fig.update_layout(paper_bgcolor="#E2F0D9", plot_bgcolor="#E2F0D9")

    fig.update_layout(scene_aspectmode="data")
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            yaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            zaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
        )
    )

    fig.show()


if __name__ == "__main__":
    main()
