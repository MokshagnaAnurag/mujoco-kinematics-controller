"""
visualize.py
============

OPTIONAL, LOCAL-ONLY dev tool -- opens an interactive MuJoCo viewer
window so you can *watch* the arm reach targets in real time. This is
not part of the graded pipeline (experiments.py doesn't import it and
run.sh doesn't call it), it needs a display (won't work over SSH without
X forwarding / won't work in a headless CI environment), and it is safe
to delete before submitting the repo if you don't want it in there.

Usage (from the project root, with the venv active):

    python3 src/visualize.py                    # baseline controller, random targets forever
    python3 src/visualize.py --broken            # inject the Part 4 faults
    python3 src/visualize.py --fixed             # Part 5 hierarchical fix (implies --broken)
    python3 src/visualize.py --seconds-per-target 4

Press Ctrl+C in the terminal to quit.

Implementation note
--------------------
Target-switch timing and convergence tracking are driven by *simulated*
time (step_count * sim.dt), never by wall-clock time. If wall-clock
rendering falls behind simulated time (slow compositor, remote display,
etc.) the visual playback will simply look slower than real-time, but
the underlying physics/control experiment is always run for the full
intended number of simulated seconds -- so what you see on screen is
never cut short before the controller had a fair chance to converge.
Physics is stepped in small batches between viewer redraws (rather than
syncing the viewer on every single 2 ms physics step) purely to reduce
render overhead; this does not change the control loop's behavior at
all, only how often the picture on screen is refreshed.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from simulation import ArmSim
from controller import DLSReachingController, HierarchicalDLSController
from kinematics import TwoLinkKinematics

L1, L2 = 0.30, 0.25
SUCCESS_THRESHOLD = 0.01  # meters, matches experiments.py
TARGET_RENDER_HZ = 60.0   # how often we redraw, independent of physics rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broken", action="store_true",
                         help="Inject the Part 4 faults (sensor noise + 3x damping + 50Hz control).")
    parser.add_argument("--fixed", action="store_true",
                         help="Use the Part 5 hierarchical controller (implies --broken).")
    parser.add_argument("--seconds-per-target", type=float, default=6.0,
                         help="Simulated seconds to chase each target before picking a new one "
                              "(matches experiments.py's MAX_STEPS budget of 6s by default).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for target generation (default: random each run).")
    args = parser.parse_args()

    broken = args.broken or args.fixed
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**32))

    sim = ArmSim(seed=seed)
    if broken:
        sim.enable_sensor_noise = True
        sim.sensor_noise_std = 0.05
        sim.set_incorrect_joint_damping(3.0)
        control_hz = 50
        print("Faults ON: sensor noise (std=0.05 rad), 3x joint damping, 50 Hz control rate.")
    else:
        control_hz = None
        print("Faults OFF: nominal system.")

    if args.fixed:
        controller = HierarchicalDLSController(L1, L2)
        print("Controller: Part 5 hierarchical (outer IK @50Hz / inner torque PD @full rate).")
    else:
        controller = DLSReachingController(L1, L2)
        print("Controller: Part 3 damped-least-squares.")

    kin = TwoLinkKinematics(L1, L2)
    rng = np.random.default_rng(seed + 1)

    sim.reset()

    physics_hz = 1.0 / sim.dt
    control_decimation = 1 if control_hz is None else max(1, int(round(physics_hz / control_hz)))
    steps_per_render = max(1, int(round(physics_hz / TARGET_RENDER_HZ)))
    switch_after_steps = max(1, int(round(args.seconds_per_target * physics_hz)))

    def new_target():
        t = kin.sample_reachable_target(rng)
        # Reset to the same q=(0,0) start pose experiments.py uses for every
        # trial. Without this, letting the arm chain continuously from
        # wherever the previous target left it can occasionally park a
        # joint right at its +-pi range limit (these are range-limited
        # hinges, not full continuous-rotation joints), from which some
        # next targets become genuinely unreachable without crossing the
        # limit -- a real mechanical constraint, but one that has nothing
        # to do with the controller and isn't part of the graded
        # experiment. Resetting here keeps this demo an honest, faithful
        # picture of the same trials reported in results/summary.txt.
        sim.reset()
        sim.set_target(t)
        if args.fixed:
            controller.reset()
        return t

    target = new_target()
    step_in_episode = 0
    global_step = 0
    tau = np.zeros(2)
    best_err_this_episode = np.inf
    converged_at = None

    print("Opening viewer window... (Ctrl+C in this terminal to quit)")
    print(f"Target: ({target[0]:+.3f}, {target[1]:+.3f})")

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                batch_wall_start = time.time()

                for _ in range(steps_per_render):
                    if args.fixed:
                        if global_step % control_decimation == 0:
                            controller.update_outer(sim.get_measured_q(), target)
                        tau = controller.compute_inner_tau(sim.get_true_qdot())
                    else:
                        if global_step % control_decimation == 0:
                            tau = controller.compute_torque(
                                sim.get_measured_q(), sim.get_true_qdot(), target
                            )
                    sim.apply_control_and_step(tau)

                    err = np.linalg.norm(sim.get_ee_pos_mujoco() - target)
                    if err < best_err_this_episode:
                        best_err_this_episode = err
                    if converged_at is None and err < SUCCESS_THRESHOLD:
                        converged_at = step_in_episode * sim.dt

                    global_step += 1
                    step_in_episode += 1

                    if step_in_episode >= switch_after_steps:
                        break  # finish this render batch, then switch below

                viewer.sync()

                # Pace real time to roughly match simulated time for this
                # batch; if we're running behind (slow renderer), don't
                # sleep -- just keep going as fast as we can. This never
                # affects switch_after_steps, which counts physics steps.
                sim_seconds_this_batch = steps_per_render * sim.dt
                wall_elapsed = time.time() - batch_wall_start
                sleep_time = sim_seconds_this_batch - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step_in_episode >= switch_after_steps:
                    final_err = np.linalg.norm(sim.get_ee_pos_mujoco() - target)
                    if converged_at is not None:
                        print(f"  -> converged at t={converged_at:.2f}s, "
                              f"final error {final_err*1000:.1f} mm")
                    else:
                        print(f"  -> did NOT converge in {args.seconds_per_target:.1f}s "
                              f"(best error {best_err_this_episode*1000:.1f} mm, "
                              f"final error {final_err*1000:.1f} mm)")

                    target = new_target()
                    step_in_episode = 0
                    best_err_this_episode = np.inf
                    converged_at = None
                    print(f"Target: ({target[0]:+.3f}, {target[1]:+.3f})")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
