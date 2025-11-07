import math
import os
import sys
import torch
import pytorch_kinematics as pk
import trimesh as tm
from scipy.spatial.transform import Rotation as sciR
import mujoco
import numpy as np
import json
from torchsdf import index_vertices_by_faces, compute_sdf
import pytorch3d.structures
import pytorch3d.ops
import plotly.graph_objects as go
import logging

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.hand_model import HandModel


def test():
    # building the robot from a MJCF file
    mjcf_path = "mjcf/shadow/right_shadow_hand.xml"
    contact_points_path = "mjcf/shadow/right_hand_contact_points.json"
    penetration_points_path = "mjcf/shadow/penetration_points.json"
    device = "cuda:0"

    hand_model = HandModel(
        mjcf_path=mjcf_path,
        contact_points_path=contact_points_path,
        penetration_points_path=penetration_points_path,
        n_surface_points=2000,
        device=device,
        handedness="right_hand",
    )

    hand_pos = torch.zeros((3,), device=device)
    hand_rot = torch.eye(3, device=device)[:, :2].T.reshape(-1)
    hand_q = (hand_model.joints_lower + hand_model.joints_upper) / 2
    hand_pose = torch.cat([hand_pos, hand_rot, hand_q]).reshape(1, -1)
    contact_point_indices = torch.arange(0, hand_model.contact_candidates.shape[0]).to(device).reshape(1, -1)

    hand_model.set_parameters(hand_pose, contact_point_indices)

    ########### Visualize via plotly ###########
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color="lightslategray", with_contact_points=True)

    fig = go.Figure(hand_en_plotly)
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


def test_cal_distance():
    # building the robot from a MJCF file
    mjcf_path = "mjcf/shadow/right_shadow_hand.xml"
    contact_points_path = "mjcf/shadow/right_hand_contact_points.json"
    penetration_points_path = "mjcf/shadow/penetration_points.json"
    device = "cuda:0"

    hand_model = HandModel(
        mjcf_path=mjcf_path,
        contact_points_path=contact_points_path,
        penetration_points_path=penetration_points_path,
        n_surface_points=2000,
        device=device,
        handedness="right_hand",
    )

    hand_pos = torch.zeros((3,), device=device)
    hand_rot = torch.eye(3, device=device)[:, :2].T.reshape(-1)
    hand_q = (hand_model.joints_lower + hand_model.joints_upper) / 2
    hand_pose = torch.cat([hand_pos, hand_rot, hand_q]).reshape(1, -1)
    contact_point_indices = torch.arange(0, hand_model.contact_candidates.shape[0]).to(device).reshape(1, -1)

    hand_model.set_parameters(hand_pose, contact_point_indices)

    xp = torch.tensor([0.0, 0.0, 0.1]).float().to(device).reshape(1, -1, 3)
    xp = xp.repeat(3, 2, 1)  # (B, N, 3)
    dis = hand_model.cal_distance(xp)
    print(f"dis: {dis}.")

    ########### Visualize via plotly ###########
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color="lightslategray", with_contact_points=True)

    # Visualize the query point
    xp = xp.detach().cpu().numpy()
    hand_en_plotly.append(
        go.Scatter3d(x=xp[0, :, 0], y=xp[0, :, 1], z=xp[0, :, 2], mode="markers", marker=dict(color="blue", size=10))
    )

    fig = go.Figure(hand_en_plotly)
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


def test_self_penetration():
    # building the robot from a MJCF file
    mjcf_path = "mjcf/shadow/left_shadow_hand.xml"
    contact_points_path = "mjcf/shadow/left_hand_contact_points.json"
    penetration_points_path = "mjcf/shadow/penetration_points.json"
    device = "cuda:0"

    hand_model = HandModel(
        mjcf_path=mjcf_path,
        contact_points_path=contact_points_path,
        penetration_points_path=penetration_points_path,
        n_surface_points=2000,
        device=device,
        handedness="left_hand",
    )

    hand_pos = torch.zeros((3,), device=device)
    hand_rot = torch.eye(3, device=device)[:, :2].T.reshape(-1)
    hand_q = (hand_model.joints_lower + hand_model.joints_upper) / 2
    hand_pose = torch.cat([hand_pos, hand_rot, hand_q]).reshape(1, -1)
    contact_point_indices = torch.arange(0, hand_model.contact_candidates.shape[0]).to(device).reshape(1, -1)

    hand_model.set_parameters(hand_pose, contact_point_indices)

    E_spen = hand_model.self_penetration()
    print(f"E_spen: {E_spen}.")

    ########### Visualize via plotly ###########
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color="lightslategray", with_contact_points=True)

    fig = go.Figure(hand_en_plotly)
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
    # 一般放在文件开头，只执行一次
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别
        format="[%(asctime)s]-[%(name)s]-[%(levelname)s]: %(message)s",  # 日志格式
        datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式
    )

    test_self_penetration()
