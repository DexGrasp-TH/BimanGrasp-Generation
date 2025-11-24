import os
from timeit import default_timer as timer

import numpy as np
import torch

import pytorch_kinematics as pk
import pytorch_seed


def create_test_chain(file_type, device="cpu"):
    if file_type == "urdf":
        full_urdf = "mjcf/ur10e/ur10e.urdf"
        chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "wrist_3_link").to(
            device=device, dtype=torch.float64
        )
    elif file_type == "mjcf":
        full_urdf = "mjcf/ur10e/ur10e.xml"
        chain = pk.build_serial_chain_from_mjcf(open(full_urdf).read(), "wrist_3_link").to(
            device=device, dtype=torch.float64
        )

    return chain


def test_ik_in_place_no_err(file_type):
    pytorch_seed.seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    chain = create_test_chain(file_type, device=device)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # goal equal to current configuration
    lim = torch.tensor(chain.get_joint_limits(), device=device, dtype=torch.float64)
    # cur_q = torch.rand(lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]
    M = 10
    cur_q = torch.rand(M, lim.shape[1], device=device, dtype=torch.float64) * (lim[1] - lim[0]) + lim[0]
    goal_q = cur_q + 0.01 * torch.randn_like(cur_q)

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

    # transform to world frame for visualization
    goal_tf = rob_tf.compose(goal_in_rob_frame_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    ik = pk.PseudoInverseIK(
        chain,
        max_iterations=300,
        num_retries=10,
        joint_limits=lim.T,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="any",
        retry_configs=cur_q,
        # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
        debug=False,
        lr=0.2,
        regularlization=1e-3,
    )

    # do IK
    timer_start = timer()
    sol = ik.solve(goal_in_rob_frame_tf)
    timer_end = timer()
    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.numel()))
    print("IK took %d iterations" % sol.iterations)
    print("IK solved %d / %d goals" % (sol.converged_any.sum(), M))

    # check that solving again produces the same solutions
    sol_again = ik.solve(goal_in_rob_frame_tf)
    assert torch.allclose(sol.solutions, sol_again.solutions)
    assert torch.allclose(sol.converged, sol_again.converged)


def test_jacobian():
    pytorch_seed.seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    chain_mjcf = create_test_chain("mjcf", device=device)
    chain_urdf = create_test_chain("urdf", device=device)

    # goal equal to current configuration
    lim = torch.tensor(chain_urdf.get_joint_limits(), device=device, dtype=torch.float64)
    # cur_q = torch.rand(lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]
    M = 2
    cur_q = torch.rand(M, lim.shape[1], device=device, dtype=torch.float64) * (lim[1] - lim[0]) + lim[0]

    jaco_urdf = chain_urdf.jacobian(cur_q)

    print("-------------------------")

    jaco_mjcf = chain_mjcf.jacobian(cur_q)

    a = 1

    return jaco


if __name__ == "__main__":
    print("Testing my robot")

    # print("------------- URDF -------------")
    # test_ik_in_place_no_err(file_type="urdf")

    # print("------------- MJCF -------------")
    # test_ik_in_place_no_err(file_type="mjcf")

    test_jacobian()
