"""
Bimanual hand initialization utilities for grasp generation.
Provides functions to initialize hand poses and orientations around target objects.
"""

import math
import numpy as np
import torch
import transforms3d
import trimesh as tm
import pytorch3d.structures
import pytorch3d.ops

from utils.hand_model import HandModel


def sample_contact_points(hand_model: HandModel, total_batch_size: int, args):
    # Handle both old and new parameter names for backward compatibility
    n_contact = getattr(args, "num_contacts", getattr(args, "n_contact", 4))
    device = hand_model.device

    if not args.keep_thumb_contact:
        contact_indices = torch.randint(
            hand_model.n_contact_candidates, size=[total_batch_size, n_contact], device=device
        )
    else:
        # non-thumb fingers
        candidates = hand_model.non_thumb_contact_candi_indices
        contact_indices = candidates[torch.randint(len(candidates), (total_batch_size, n_contact), device=device)]
        # thumb
        candidates = hand_model.thumb_contact_candi_indices
        contact_indices[:, -1] = candidates[torch.randint(len(candidates), (total_batch_size,), device=device)]

    return contact_indices


def sample_base_poses(
    hand_model: HandModel,
    p: torch.tensor,
    n: torch.tensor,
    batch_per_obj: int,
    args,
):
    device = hand_model.device

    # Sample initialization parameters for the left hand
    rand_vals = torch.rand([4, batch_per_obj], dtype=torch.float, device=device)
    distance = args.distance_lower + (args.distance_upper - args.distance_lower) * rand_vals[0]
    cone_angle = args.theta_lower + (args.theta_upper - args.theta_lower) * rand_vals[1]
    azimuth = 2 * math.pi * rand_vals[2]
    roll = 2 * math.pi * rand_vals[3]

    # Solve transformation matrices for the left hand
    # hand_rot: rotate the hand to align its grasping direction with the +z axis
    # cone_rot: jitter the hand's orientation in a cone
    # world_rot and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull
    cone_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
    world_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
    for j in range(batch_per_obj):
        cone_rot[j] = torch.tensor(
            transforms3d.euler.euler2mat(azimuth[j], cone_angle[j], roll[j], axes="rzxz"),
            dtype=torch.float,
            device=device,
        )
        world_rot[j] = torch.tensor(
            transforms3d.euler.euler2mat(math.atan2(n[j, 1], n[j, 0]) - np.pi / 2, -math.acos(n[j, 2]), 0, axes="rzxz"),
            dtype=torch.float,
            device=device,
        )

    # The palm frame is defined as: located at the palm center and Z-axis towards object
    z_vec = torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
    palm_trans = p - distance.unsqueeze(1) * (world_rot @ z_vec).squeeze(2)
    palm_rot = world_rot @ cone_rot

    # Hand-dependent defination of palm frame in the hand base
    hand_rot = args.left_hand_rot if hand_model.handedness == "left_hand" else args.right_hand_rot
    hand_trans = args.left_hand_trans if hand_model.handedness == "left_hand" else args.right_hand_trans
    palm_rot_in_base = torch.tensor(
        transforms3d.quaternions.quat2mat(hand_rot), dtype=torch.float, device=device
    ).reshape(1, 3, 3)
    palm_trans_in_base = torch.tensor(hand_trans, dtype=torch.float, device=device).reshape(1, 3, 1)

    # Compute corresponding global hand base frame
    base_rot = palm_rot @ palm_rot_in_base.transpose(1, 2)
    base_trans = palm_trans.reshape(-1, 3, 1) - base_rot @ palm_trans_in_base

    return base_trans.reshape(-1, 3), base_rot.reshape(-1, 3, 3)


def sample_finger_joint_angles(hand_model: HandModel, total_batch_size: int, args):
    device = hand_model.device

    # joint_angles_mu: hand-crafted canonicalized hand articulation
    joint_angles_mu = args.left_hand_joint_mu if hand_model.handedness == "left_hand" else args.right_hand_joint_mu
    joint_angles_mu = torch.tensor(joint_angles_mu, dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)

    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            joint_angles_mu[i],
            joint_angles_sigma[i],
            hand_model.joints_lower[i] - 1e-6,
            hand_model.joints_upper[i] + 1e-6,
        )

    return joint_angles


def initialize_dual_hand(right_hand_model, left_hand_model, object_model, args):
    """
    Initialize both hands' positions and rotations to grasp an object symmetrically.

    Args:
        right_hand_model: HandModel instance for right hand
        object_model: ObjectModel instance containing target objects
        args: Configuration namespace with initialization parameters

    Returns:
        tuple: (left_hand_model, right_hand_model) with initialized poses
    """

    assert left_hand_model.handedness == "left_hand"
    assert right_hand_model.handedness == "right_hand"

    device = right_hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_per_obj = object_model.batch_size_each
    total_batch_size = n_objects * batch_per_obj

    # Initialize translation and rotation tensors
    left_translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    left_rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)
    right_translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    right_rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        ################ Object mesh processing ################

        # Get inflated convex hull
        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]
        vertices += args.convex_expand_dis * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # Sample points from mesh surface
        dense_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_per_obj)
        p = pytorch3d.ops.sample_farthest_points(dense_cloud, K=batch_per_obj)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        ################ Left hand global pose ################

        base_trans, base_rot = sample_base_poses(left_hand_model, p, n, batch_per_obj, args)
        start_idx = i * batch_per_obj
        end_idx = start_idx + batch_per_obj
        left_translation[start_idx:end_idx] = base_trans
        left_rotation[start_idx:end_idx] = base_rot

        ################ Right hand global pose ################

        p = -p  # Mirror the normal vectors and points for symmetric grasp
        n = -n
        base_trans, base_rot = sample_base_poses(right_hand_model, p, n, batch_per_obj, args)
        right_translation[start_idx:end_idx] = base_trans
        right_rotation[start_idx:end_idx] = base_rot

    ################ Left hand finger joint angles ################

    joint_angles = sample_finger_joint_angles(left_hand_model, total_batch_size, args)

    hand_pose = torch.cat([left_translation, left_rotation.transpose(1, 2)[:, :2].reshape(-1, 6), joint_angles], dim=1)
    hand_pose.requires_grad_()
    # Initialize contact point indices
    contact_indices = sample_contact_points(left_hand_model, total_batch_size, args)
    left_hand_model.set_parameters(hand_pose, contact_indices)

    ################ Right hand finger joint angles ################

    joint_angles = sample_finger_joint_angles(right_hand_model, total_batch_size, args)

    hand_pose = torch.cat(
        [right_translation, right_rotation.transpose(1, 2)[:, :2].reshape(-1, 6), joint_angles], dim=1
    )
    hand_pose.requires_grad_()
    # Initialize contact point indices
    contact_indices = sample_contact_points(right_hand_model, total_batch_size, args)
    right_hand_model.set_parameters(hand_pose, contact_indices)

    return left_hand_model, right_hand_model
