import pytorch_kinematics as pk
import torch
import math

device = "cuda:0"
full_urdf = "mjcf/dual_ur5_shadow/dual_ur5_shadow.urdf"
chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "rh_palm").to(device=device, dtype=torch.float64)

th = [0.0, -math.pi / 4.0, math.pi / 4.0, math.pi / 2.0, 0.0, 0.0]
ee_tf = chain.forward_kinematics(th, end_only=True)


# goals are specified as Transform3d poses in the **robot frame**
# so if you have the goals specified in the world frame, you also need the robot frame in the world frame
# pos = torch.tensor([0.5, -0.35, 0.22], device=device, dtype=torch.float64)
# rot = torch.tensor([0, 0, 0, 1.0], device=device, dtype=torch.float64)
# goal_in_rob_frame_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

goal_in_rob_frame_tf = ee_tf

# get robot joint limits
lim = torch.tensor(chain.get_joint_limits(), device=device)

# create the IK object
# see the constructor for more options and their explanations, such as convergence tolerances
ik = pk.PseudoInverseIK(
    chain,
    max_iterations=50,
    num_retries=10,
    joint_limits=lim.T,
    early_stopping_any_converged=False,
    early_stopping_no_improvement="any",
    debug=True,
    lr=0.2,
    regularlization=1e-3,
)
# solve IK
sol = ik.solve(goal_in_rob_frame_tf)
# num goals x num retries x DOF tensor of joint angles; if not converged, best solution found so far
print(sol.solutions)
# num goals x num retries can check for the convergence of each run
print(sol.converged)
# num goals x num retries can look at errors directly
print(sol.err_pos)
print(sol.err_rot)
