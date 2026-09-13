"""
experiments.py
==============

Runs Parts 3, 4 and 5 of the assignment and writes plots/CSVs to
../results/. Run from the project root via ./run.sh, or directly:

    cd src && python3 experiments.py

Sections:
  Part 3 - baseline controller on 20 random reachable targets
  Part 4 - "break your robot": inject 2 disturbances (sensor noise +
           lower control frequency + incorrect joint damping are all
           implemented; sensor noise & low control-rate are used as the
           two primary faults, incorrect damping is used again in Part 5),
           rerun the same 20 targets, compare against baseline.
  Part 5 - debug one failure in depth: noisy-sensor case causes jittery,
           sometimes-diverging torque commands. Root-caused to the
           controller differentiating noisy *position* into a velocity
           command every control step with no filtering. Fix: add a
           low-pass filter on the measured joint angle before it is used
           by the controller. Re-run and show quantitative improvement.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import ArmSim
from controller import DLSReachingController, HierarchicalDLSController
from kinematics import TwoLinkKinematics

L1, L2 = 0.30, 0.25
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TARGETS = 20
MAX_STEPS = 3000          # 3000 * 0.002s = 6s per episode, plenty to converge or fail
SUCCESS_THRESHOLD = 0.01  # 1 cm


def make_targets(n=N_TARGETS, seed=42):
    kin = TwoLinkKinematics(L1, L2)
    rng = np.random.default_rng(seed)
    return [kin.sample_reachable_target(rng) for _ in range(n)]


def run_batch(sim: ArmSim, ctrl, targets, control_hz=None, label=""):
    """Run the controller on a list of targets, return a metrics dict."""
    results = {
        "target": [], "final_err": [], "success": [],
        "convergence_time": [], "trajectories": [], "err_curves": [],
    }
    for target in targets:
        sim.reset()
        log = sim.run_control_loop(ctrl, target, MAX_STEPS, control_hz=control_hz)
        final_err = log["err"][-1]
        success = bool(final_err < SUCCESS_THRESHOLD)

        conv_idx = np.argmax(log["err"] < SUCCESS_THRESHOLD) if success else -1
        conv_time = log["t"][conv_idx] if success else np.nan

        results["target"].append(target)
        results["final_err"].append(final_err)
        results["success"].append(success)
        results["convergence_time"].append(conv_time)
        results["trajectories"].append(log["ee"])
        results["err_curves"].append(log["err"])

    results["final_err"] = np.array(results["final_err"])
    results["success"] = np.array(results["success"])
    results["convergence_time"] = np.array(results["convergence_time"])

    print(f"[{label}] success rate: {results['success'].mean()*100:.1f}% "
          f"({results['success'].sum()}/{len(targets)})")
    print(f"[{label}] mean final error: {results['final_err'].mean()*1000:.2f} mm "
          f"(median {np.median(results['final_err'])*1000:.2f} mm)")
    conv = results["convergence_time"][results["success"]]
    if len(conv) > 0:
        print(f"[{label}] mean convergence time (successes only): {conv.mean():.2f} s")
    return results


def plot_trajectories(results, title, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot((L1 + L2) * np.cos(theta), (L1 + L2) * np.sin(theta), "k--", lw=0.5, alpha=0.4)
    ax.plot(abs(L1 - L2) * np.cos(theta), abs(L1 - L2) * np.sin(theta), "k--", lw=0.5, alpha=0.4)

    for traj, target, success in zip(results["trajectories"], results["target"], results["success"]):
        color = "tab:green" if success else "tab:red"
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, lw=1)
        ax.plot(target[0], target[1], "x", color="black", ms=8, mew=2)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_error_curves(results, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for err_curve, success in zip(results["err_curves"], results["success"]):
        color = "tab:green" if success else "tab:red"
        ax.plot(np.arange(len(err_curve)) * 0.002, err_curve, color=color, alpha=0.5, lw=1)
    ax.axhline(SUCCESS_THRESHOLD, color="black", ls="--", lw=1, label="success threshold (1cm)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("end-effector position error (m)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_comparison_bar(baseline, degraded, fixed, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = ["Baseline", "Degraded\n(noise+low-rate)", "Fixed\n(hierarchical)"]
    success_rates = [baseline["success"].mean() * 100,
                      degraded["success"].mean() * 100,
                      fixed["success"].mean() * 100]
    axes[0].bar(labels, success_rates, color=["tab:blue", "tab:red", "tab:green"])
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Success rate")
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center")

    err_data = [baseline["final_err"] * 1000,
                degraded["final_err"] * 1000,
                fixed["final_err"] * 1000]
    axes[1].boxplot(err_data, tick_labels=labels)
    axes[1].set_ylabel("Final position error (mm)")
    axes[1].set_title("Final error distribution")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def main():
    targets = make_targets()

    # ---------------- Part 3: baseline ----------------
    print("\n=== Part 3: Baseline controller on 20 random reachable targets ===")
    sim = ArmSim(seed=1)
    ctrl = DLSReachingController(L1, L2)
    baseline = run_batch(sim, ctrl, targets, label="baseline")
    plot_trajectories(baseline, "Part 3: Baseline trajectories", "part3_baseline_trajectories.png")
    plot_error_curves(baseline, "Part 3: Baseline error over time", "part3_baseline_error_curves.png")

    # ---------------- Part 4: break the robot ----------------
    # Two+ disturbances injected simultaneously:
    #   1) sensor noise on measured joint angle (std = 0.05 rad ~ 2.9 deg)
    #   2) lower control frequency (50 Hz instead of the 500 Hz physics rate)
    #   3) incorrect joint damping (damping model wrong by 3x) -- a third
    #      fault, included to show the "at least two" requirement is
    #      comfortably exceeded and because it interacts with (2).
    print("\n=== Part 4: Degraded system (sensor noise + low control rate + wrong damping) ===")
    sim2 = ArmSim(seed=1)
    sim2.enable_sensor_noise = True
    sim2.sensor_noise_std = 0.05
    sim2.set_incorrect_joint_damping(3.0)
    ctrl2 = DLSReachingController(L1, L2)
    degraded = run_batch(sim2, ctrl2, targets, control_hz=50, label="degraded")
    plot_trajectories(degraded, "Part 4: Degraded trajectories", "part4_degraded_trajectories.png")
    plot_error_curves(degraded, "Part 4: Degraded error over time", "part4_degraded_error_curves.png")
    sim2.reset_physical_params()

    # ---------------- Part 5: debug + fix ----------------
    # See README.md "Part 5" section for the full 6-step debugging
    # narrative. Short version: isolating each fault individually showed
    # the low control rate (50 Hz outer loop) -- NOT the sensor noise or
    # the wrong damping -- was the dominant cause of failure. The bug was
    # that torque was being held constant (zero-order hold) for the
    # entire 10-physics-step window between controller calls, including
    # the joint-velocity-tracking PD term, so the "damping" part of our
    # torque law could not react to the true instantaneous velocity for
    # up to 20 ms. This turned the loop into a limit cycle that saturated
    # the torque back and forth and settled ~0.1 m away from the target.
    #
    # Fix: split the controller into a slow OUTER kinematic loop (DLS-IK
    # on filtered position, recomputed only at 50 Hz) and a fast INNER
    # torque loop (joint-space PD, recomputed every physics step using
    # the always-current true joint velocity) -- see
    # controller.HierarchicalDLSController. This mirrors how real
    # robot controllers separate a slow planning/kinematics loop from a
    # fast low-level torque/current loop.
    print("\n=== Part 5: Fixed controller (hierarchical outer/inner loop) ===")
    sim3 = ArmSim(seed=1)
    sim3.enable_sensor_noise = True
    sim3.sensor_noise_std = 0.05
    sim3.set_incorrect_joint_damping(3.0)
    hier_ctrl = HierarchicalDLSController(L1, L2)

    results = {"target": [], "final_err": [], "success": [], "convergence_time": [],
               "trajectories": [], "err_curves": []}
    for target in targets:
        sim3.reset()
        log = sim3.run_hierarchical_control_loop(hier_ctrl, target, MAX_STEPS, outer_hz=50)
        final_err = log["err"][-1]
        success = bool(final_err < SUCCESS_THRESHOLD)
        conv_idx = np.argmax(log["err"] < SUCCESS_THRESHOLD) if success else -1
        conv_time = log["t"][conv_idx] if success else np.nan
        results["target"].append(target)
        results["final_err"].append(final_err)
        results["success"].append(success)
        results["convergence_time"].append(conv_time)
        results["trajectories"].append(log["ee"])
        results["err_curves"].append(log["err"])
    results["final_err"] = np.array(results["final_err"])
    results["success"] = np.array(results["success"])
    results["convergence_time"] = np.array(results["convergence_time"])
    fixed = results
    print(f"[fixed] success rate: {fixed['success'].mean()*100:.1f}% "
          f"({fixed['success'].sum()}/{len(targets)})")
    print(f"[fixed] mean final error: {fixed['final_err'].mean()*1000:.2f} mm")

    plot_trajectories(fixed, "Part 5: Fixed (hierarchical control) trajectories", "part5_fixed_trajectories.png")
    plot_error_curves(fixed, "Part 5: Fixed (hierarchical control) error over time", "part5_fixed_error_curves.png")

    plot_comparison_bar(baseline, degraded, fixed, "comparison_baseline_degraded_fixed.png")

    # ---------------- Summary table ----------------
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        for name, res in [("Baseline", baseline), ("Degraded", degraded), ("Fixed", fixed)]:
            f.write(f"{name}:\n")
            f.write(f"  success rate: {res['success'].mean()*100:.1f}%\n")
            f.write(f"  mean final error (mm): {res['final_err'].mean()*1000:.3f}\n")
            f.write(f"  median final error (mm): {np.median(res['final_err'])*1000:.3f}\n")
            conv = res["convergence_time"][res["success"]]
            conv_mean = conv.mean() if len(conv) > 0 else float("nan")
            f.write(f"  mean convergence time (s, successes only): {conv_mean:.3f}\n\n")
    print(f"\nSummary written to {summary_path}")
    print("Plots written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
