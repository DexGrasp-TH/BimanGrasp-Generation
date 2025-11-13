import os
import sys
import glob
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.config import ExperimentConfig

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]


def natural_key(filename):
    """Extract numeric part from the filename for natural sorting."""
    match = re.search(r"_(\d+)\.npy$", filename)
    return int(match.group(1)) if match else float("inf")


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


def get_scene(plot_lst):
    xs, ys, zs = [], [], []
    for trace in plot_lst:
        if hasattr(trace, "x") and hasattr(trace, "y") and hasattr(trace, "z"):
            xs.extend(trace.x)
            ys.extend(trace.y)
            zs.extend(trace.z)

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()

    # 计算各轴范围
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    # 计算比例（归一化到最长轴为1）
    max_range = max(x_range, y_range, z_range)
    aspectratio = dict(
        x=x_range / max_range,
        y=y_range / max_range,
        z=z_range / max_range,
    )

    scene_fixed = dict(
        xaxis=dict(range=[xmin, xmax], visible=False, showgrid=False),
        yaxis=dict(range=[ymin, ymax], visible=False, showgrid=False),
        zaxis=dict(range=[zmin, zmax], visible=False, showgrid=False),
        aspectmode="manual",
        aspectratio=aspectratio,
    )
    return scene_fixed


@hydra.main(config_path="../cfg", config_name="base", version_base=None)  # must use version_base=None for compatibility
def main(cfg: DictConfig):
    """Hydra entrypoint. Builds ExperimentConfig from config.yaml and runs the experiment."""

    # merge cfg.hand.paths into cfg.paths
    cfg.paths = OmegaConf.merge(cfg.paths, cfg.hand.paths)

    # Convert to ExperimentConfig dataclass
    config = experiment_config_from_dict(cfg)

    # hyper-parameters
    object_code_list = list(cfg.task.object_code_list)
    grasp_idx_lst = cfg.task.grasp_idx_lst
    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    result_path = os.path.join(exp_path, "results/intermediate")
    device = "cuda:0"

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
    object_model.initialize(object_code_list)

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

    for i_obj, object_code in enumerate(object_code_list):
        for idx in grasp_idx_lst:
            pattern = os.path.join(result_path, f"{object_code}_*.npy")
            files = sorted(glob.glob(pattern), key=natural_key)
            # files = files[1:][::4]

            right_hand_pose_lst = []
            left_hand_pose_lst = []
            opt_step_lst = []

            for file in files:
                data_dict = np.load(file, allow_pickle=True)[idx]
                opt_step = int(re.search(r"_(\d+)\.npy$", file).group(1))
                print(f"opt_step: {opt_step}")

                # --- Initialize models ---
                object_model.object_scale_tensor = torch.tensor(
                    data_dict["scale"], dtype=torch.float, device=device
                ).reshape(1, 1)

                # --- Load qpos and construct hand poses ---
                if file == files[0] and "qpos_right_st" in data_dict:
                    right_qpos_st = data_dict["qpos_right_st"]
                    right_hand_pose_st = build_hand_pose(
                        right_qpos_st, translation_names, rot_names, right_hand_model.get_joint_names(), device
                    )
                    right_hand_pose_lst.append(right_hand_pose_st)
                if file == files[0] and "qpos_left_st" in data_dict:
                    left_qpos_st = data_dict["qpos_left_st"]
                    left_hand_pose_st = build_hand_pose(
                        left_qpos_st, translation_names, rot_names, left_hand_model.get_joint_names(), device
                    )
                    left_hand_pose_lst.append(left_hand_pose_st)
                    opt_step_lst.append(0)

                # Right hand
                right_qpos = data_dict["qpos_right"]
                right_hand_pose = build_hand_pose(
                    right_qpos, translation_names, rot_names, right_hand_model.get_joint_names(), device
                )
                right_hand_pose_lst.append(right_hand_pose)

                # Left hand
                left_qpos = data_dict["qpos_left"]
                left_hand_pose = build_hand_pose(
                    left_qpos, translation_names, rot_names, left_hand_model.get_joint_names(), device
                )
                left_hand_pose_lst.append(left_hand_pose)
                opt_step_lst.append(opt_step)

            # --- Visualization ---
            # Final poses (solid colors)
            right_hand_model.set_parameters(torch.stack(right_hand_pose_lst, dim=0).to(device))
            left_hand_model.set_parameters(torch.stack(left_hand_pose_lst, dim=0).to(device))

            # # Example color choices
            right_color = "lightslategray"
            left_color = "lightcoral"
            # Transparency gradient: from 0.2 (transparent) → 1.0 (opaque)
            num_steps = len(right_hand_pose_lst)
            opacities = torch.linspace(0.2, 1.0, num_steps).tolist()

            plot_lst = []
            for i_plot, opacity in enumerate(opacities):
                right_plot = right_hand_model.get_plotly_data(
                    i=i_plot, opacity=opacity, color=right_color, with_contact_points=False
                )
                left_plot = left_hand_model.get_plotly_data(
                    i=i_plot, opacity=opacity, color=left_color, with_contact_points=False
                )
                plot_lst += right_plot
                plot_lst += left_plot

            object_plot = object_model.get_plotly_data(i=i_obj, color="seashell", opacity=1.0)
            plot_lst += object_plot

            # # Show a figure with the whole optimization process
            # fig = go.Figure(plot_lst)
            # fig.update_layout(
            #     paper_bgcolor="#E2F0D9",
            #     plot_bgcolor="#E2F0D9",
            #     scene_aspectmode="data",
            #     scene=dict(
            #         xaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            #         yaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            #         zaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            #     ),
            # )
            # fig.show()

            save_animation = True
            if save_animation:
                # 屏蔽 kaleido 日志
                logging.getLogger("kaleido").setLevel(logging.WARNING)
                # 屏蔽 choreographer 日志
                logging.getLogger("choreographer").setLevel(logging.WARNING)

                scene_fixed = get_scene(plot_lst)
                camera = dict(
                    eye=dict(x=0.8, y=0.8, z=0.8),  # fixed view angle
                    center=dict(x=0, y=0, z=0),
                )

                filenames = []
                save_dir = os.path.join(exp_path, f"visualizations/opt_process/{object_code}/grasp_{idx}")
                for i_plot in range(num_steps):
                    fig = go.Figure(
                        right_hand_model.get_plotly_data(i=i_plot, opacity=1.0, color=right_color)
                        + left_hand_model.get_plotly_data(i=i_plot, color=left_color)
                        + object_model.get_plotly_data(i=0, color="seashell")
                    )
                    fig.update_layout(
                        scene=scene_fixed,
                        scene_camera=camera,
                        paper_bgcolor="#E2F0D9",
                        plot_bgcolor="#E2F0D9",
                    )
                    path = os.path.join(save_dir, f"step_{opt_step_lst[i_plot]}.jpg")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    fig.write_image(path, width=1920, height=1080)  # engine: "kaleido"
                    filenames.append(path)
                    print(f"Save image {path}.")

                # 合成为 gif
                images = [imageio.v2.imread(f) for f in filenames]
                imageio.mimsave(os.path.join(save_dir, "ani.gif"), images, fps=1)


if __name__ == "__main__":
    main()
