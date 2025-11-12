import pyvista as pv
import sys
import os
import numpy as np
import mujoco
import torch
import json
import plotly.graph_objects as go

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel


def main():
    # building the robot from a MJCF file
    mjcf_path = "mjcf/shadow2/right_hand.xml"
    contact_points_path = "mjcf/shadow2/right_hand_contact_points.json"
    penetration_points_path = None
    device = "cuda:0"

    # save an empty json file if no file exists
    if not os.path.exists(contact_points_path):
        with open(contact_points_path, "w") as f:
            candidate_dict = {}
            json.dump(candidate_dict, f, indent=4)  # indent=4 makes it more readable

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
    # hand_q = torch.zeros_like(hand_model.joints_lower)
    hand_pose = torch.cat([hand_pos, hand_rot, hand_q]).reshape(1, -1)
    contact_point_indices = torch.arange(0, hand_model.contact_candidates.shape[0]).to(device).reshape(1, -1)

    hand_model.set_parameters(hand_pose, contact_point_indices)

    ########### Visualize via plotly ###########
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=0.6, color="lightslategray", with_contact_points=True)

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

    ########### Pick contact candidates via GUI ###########
    target_link_name = "rh_thmiddle"

    picked_points = hand_model.pick_contact_candidates(i=0, target_link_name=target_link_name)

    # load the contact points file
    with open(contact_points_path, "r") as f:
        candidate_dict = json.load(f)

    # add/replace the key
    candidate_dict[target_link_name] = picked_points

    # save the contact points file
    if len(picked_points) > 0:
        with open(contact_points_path, "w") as f:
            json.dump(candidate_dict, f, indent=4)  # indent=4 makes it more readable


if __name__ == "__main__":
    main()
