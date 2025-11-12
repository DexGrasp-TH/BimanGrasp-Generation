import time
import os
import mujoco
import mujoco.viewer

spec = mujoco.MjSpec()
# spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
spec.option.timestep = 0.004
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY

xml_path = "mjcf/shadow2/right_hand.xml"

# Read hand xml
child_spec = mujoco.MjSpec.from_file(xml_path)
for m in child_spec.meshes:
    m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
child_spec.meshdir = "asset"

for g in child_spec.geoms:
    # This solimp and solref comes from the Shadow Hand xml
    # They can generate larger force with smaller penetration
    # The body will be more "rigid" and less "soft"
    g.solimp[:3] = [0.5, 0.99, 0.0001]
    g.solref[:2] = [0.005, 1]

attach_frame = spec.worldbody.add_frame()
hand_prefix = "prefix_"
child_world = attach_frame.attach_body(child_spec.worldbody, hand_prefix, "")

m = spec.compile()
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
