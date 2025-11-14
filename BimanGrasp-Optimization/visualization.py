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
import logging

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.config import ExperimentConfig

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]


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
    """Hydra entrypoint. Builds ExperimentConfig from config.yaml and runs the experiment."""

    # merge cfg.hand.paths into cfg.paths
    cfg.paths = OmegaConf.merge(cfg.paths, cfg.hand.paths)

    # Convert to ExperimentConfig dataclass
    config = experiment_config_from_dict(cfg)

    # hyper-parameters
    # object_code = "core_bottle_10dff3c43200a7a7119862dbccbaa609"
    object_code = "core_bottle_1a7ba1f4c892e2da30711cdbdbc73924"
    grasp_idx_lst = [1]
    result_path = f"../data/experiments/{cfg.name}/results"
    device = "cuda:0"
    load_intermediate_results = True
    step = 1800

    right_hand_model = HandModel(
        mjcf_path=config.paths.right_hand_mjcf,
        contact_points_path=config.paths.right_contact_points,
        penetration_points_path=config.paths.penetration_points,
        device=device,
        n_surface_points=config.model.n_surface_points,
        handedness="right_hand",
    )
    left_hand_model = HandModel(
        mjcf_path=config.paths.left_hand_mjcf,
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

    for idx in grasp_idx_lst:
        print(f"grasp_idx: {idx}")

        # load results
        if load_intermediate_results:
            data_dict = np.load(
                os.path.join(result_path, "intermediate", object_code + f"_{step}.npy"), allow_pickle=True
            )[idx]
        else:
            data_dict = np.load(os.path.join(result_path, object_code + ".npy"), allow_pickle=True)[idx]

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

        # Right hand
        right_qpos = data_dict["qpos_right"]
        right_hand_pose = build_hand_pose(
            right_qpos, translation_names, rot_names, right_hand_model.get_joint_names(), device
        )

        right_hand_pose_st = None
        if "qpos_right_st" in data_dict:
            right_qpos_st = data_dict["qpos_right_st"]
            right_hand_pose_st = build_hand_pose(
                right_qpos_st, translation_names, rot_names, right_hand_model.get_joint_names(), device
            )

        # Left hand
        left_qpos = data_dict["qpos_left"]
        left_hand_pose = build_hand_pose(
            left_qpos, translation_names, rot_names, left_hand_model.get_joint_names(), device
        )
        # print(f"left_qpos: {left_qpos}")
        # print(f"right_qpos: {right_qpos}")

        left_hand_pose_st = None
        if "qpos_left_st" in data_dict:
            left_qpos_st = data_dict["qpos_left_st"]
            left_hand_pose_st = build_hand_pose(
                left_qpos_st, translation_names, rot_names, left_hand_model.get_joint_names(), device
            )

        # --- Initialize models ---
        object_model.initialize(object_code)
        object_model.object_scale_tensor = torch.tensor(data_dict["scale"], dtype=torch.float, device=device).reshape(
            1, 1
        )

        # --- Visualization ---
        # Final poses (solid colors)
        right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
        right_plot = right_hand_model.get_plotly_data(
            i=0, opacity=1.0, color="lightslategray", with_contact_points=False
        )

        left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))
        left_plot = left_hand_model.get_plotly_data(i=0, opacity=1.0, color="lightslategray", with_contact_points=False)

        object_plot = object_model.get_plotly_data(i=0, color="seashell", opacity=1.0)

        # Starting poses (semi-transparent, distinct color)
        start_plots = []
        if right_hand_pose_st is not None:
            right_hand_model.set_parameters(right_hand_pose_st.unsqueeze(0))
            start_plots += right_hand_model.get_plotly_data(
                i=0, opacity=0.3, color="deepskyblue", with_contact_points=False
            )

        if left_hand_pose_st is not None:
            left_hand_model.set_parameters(left_hand_pose_st.unsqueeze(0))
            start_plots += left_hand_model.get_plotly_data(
                i=0, opacity=0.3, color="palevioletred", with_contact_points=False
            )

        # Combine everything
        plot_lst = right_plot + left_plot + object_plot + start_plots
        fig = go.Figure(right_plot + left_plot + object_plot + start_plots)

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

        # 屏蔽 kaleido 日志
        logging.getLogger("kaleido").setLevel(logging.WARNING)
        # 屏蔽 choreographer 日志
        logging.getLogger("choreographer").setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
