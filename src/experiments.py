"""
experiments.py

This script runs the full suite of experiments required by the assignment.
1. Kinematics Verification: Compares manual math vs MuJoCo.
2. Baseline: Runs 20 targets with perfect sensors.
3. Degraded: Runs 20 targets with sensor noise and latency.
4. Fixed: Runs 20 targets with tuned parameters to overcome latency.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from simulation import RobotSimulation
from controller import dls_inverse_kinematics
from kinematics import forward_kinematics, jacobian

def verify_kinematics(sim: RobotSimulation, num_tests: int = 100):
    """Verifies manual forward kinematics and Jacobian against MuJoCo's physics engine."""
    print("\n--- Running Kinematics & Jacobian Verification ---")
    max_pos_error = 0.0
    max_jac_error = 0.0
    
    for _ in range(num_tests):
        q_rand = np.random.uniform(-np.pi, np.pi, size=2)
        sim.data.qpos[sim.model.jnt_qposadr[sim.joint_ids[0]]] = q_rand[0]
        sim.data.qpos[sim.model.jnt_qposadr[sim.joint_ids[1]]] = q_rand[1]
        mujoco.mj_kinematics(sim.model, sim.data)
        mujoco.mj_comPos(sim.model, sim.data)
        
        # 1. Verify Position
        mj_pos = np.array(sim.data.site_xpos[sim.ee_site_id][:2])
        manual_pos = forward_kinematics(q_rand)
        pos_err = np.linalg.norm(mj_pos - manual_pos)
        if pos_err > max_pos_error:
            max_pos_error = pos_err
            
        # 2. Verify Jacobian
        mj_jac_pos = np.zeros((3, sim.model.nv))
        mujoco.mj_jacSite(sim.model, sim.data, mj_jac_pos, None, sim.ee_site_id)
        # We only care about the X and Y rows, and the first 2 joint columns
        mj_jac_2d = mj_jac_pos[:2, :2]
        
        manual_jac = jacobian(q_rand)
        jac_err = np.linalg.norm(mj_jac_2d - manual_jac)
        if jac_err > max_jac_error:
            max_jac_error = jac_err
            
    print(f"Verification complete across {num_tests} random poses.")
    print(f"Max Position Error: {max_pos_error:.8f} meters")
    print(f"Max Jacobian Error: {max_jac_error:.8f}")
    if max_pos_error < 1e-5 and max_jac_error < 1e-5:
        print("-> STATUS: SUCCESS (Both Math and Jacobian perfectly verified!)")

def run_reaching_trial(sim: RobotSimulation, target_pos: np.ndarray, max_steps: int = 2000, 
                       noise_std: float = 0.0, latency_steps: int = 0, 
                       k: float = 2.0, lambda_val: float = 0.1, viewer = None) -> dict:
    """Executes a single reaching trial."""
    sim.set_target_position(target_pos)
    
    sim.data.qpos[sim.model.jnt_qposadr[sim.joint_ids[0]]] = 0.0
    sim.data.qpos[sim.model.jnt_qposadr[sim.joint_ids[1]]] = 0.0
    sim.data.ctrl[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    
    trajectory = []
    errors = []
    sensor_history = []
    converged_step = -1
    
    for step in range(max_steps):
        true_q = sim.get_joint_positions()
        
        noisy_q = true_q + np.random.normal(0, noise_std, size=true_q.shape)
        sensor_history.append(noisy_q)
        
        if step >= latency_steps:
            q_for_control = sensor_history[-1 - latency_steps]
        else:
            q_for_control = sensor_history[-1]
            
        current_ee_pos = sim.get_ee_position()
        trajectory.append(current_ee_pos)
        
        error = np.linalg.norm(target_pos - current_ee_pos)
        errors.append(error)
        
        if error < 0.01 and converged_step == -1:
            converged_step = step
            
        q_dot = dls_inverse_kinematics(q_for_control, target_pos, current_ee_pos, k=k, lambda_val=lambda_val)
        
        sim.set_joint_velocities(q_dot)
        sim.step()
        
        if viewer is not None:
            viewer.sync()
            time.sleep(0.01)
            
        if converged_step != -1 and (step - converged_step) > 50:
            if viewer is not None:
                time.sleep(0.1)
            break
            
    success = errors[-1] < 0.01
    return {
        'trajectory': np.array(trajectory),
        'errors': np.array(errors),
        'success': success,
        'final_error': errors[-1],
        'steps': len(errors),
        'converge_time': converged_step * 0.01 if converged_step != -1 else float('inf')
    }

def generate_random_targets(num_targets: int = 20, seed: int = 42) -> list:
    np.random.seed(seed)
    targets = []
    for _ in range(num_targets):
        r = np.random.uniform(0.2, 1.8)
        theta = np.random.uniform(0, np.pi)
        targets.append(np.array([r * np.cos(theta), r * np.sin(theta)]))
    return targets

def plot_experiment_results(baseline, degraded, fixed, save_dir: str = 'results'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'errors_comparison.png')
    
    plt.figure(figsize=(15, 5))
    
    experiments = [
        (baseline, 'Baseline Errors', '#2196F3'),
        (degraded, 'Degraded Errors (Noise+Latency)', '#F44336'),
        (fixed, 'Fixed Errors (Tuned Controller)', '#4CAF50')
    ]
    
    for i, (data, title, color) in enumerate(experiments):
        plt.subplot(1, 3, i+1)
        for metric in data:
            plt.plot(metric['errors'], color=color, alpha=0.3, linewidth=1.5)
        plt.axhline(y=0.01, color='gray', linestyle='--', label='Success Threshold')
        plt.title(title)
        plt.xlabel('Simulation Steps')
        plt.ylabel('Distance to Target (meters)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def print_summary(label, metrics):
    successes = sum([1 for m in metrics if m['success']])
    avg_steps = np.mean([m['steps'] for m in metrics])
    print(f"[{label}] Success Rate: {successes}/{len(metrics)} | Avg Steps: {avg_steps:.0f}")

def main():
    sim = RobotSimulation()
    
    # Part 2: Math & Jacobian Verification
    verify_kinematics(sim)
    
    # Ensure exactly 20 targets as requested
    targets = generate_random_targets(num_targets=20)
    
    print("\n[INFO] Starting 20-target batch experiments...")
    
    try:
        import mujoco.viewer
        print("Launching Live Simulation Viewer (will show live progress)...")
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    except:
        viewer = None
        
    print("\n--- Running Baseline Experiment ---")
    baseline_metrics = []
    for i, target in enumerate(targets):
        print(f"  > Reaching target {i+1}/20...", end="\r")
        metrics = run_reaching_trial(sim, target, viewer=viewer)
        baseline_metrics.append(metrics)
    print()
    print_summary("Baseline", baseline_metrics)
        
    print("\n--- Running Degraded Experiment (Noise + Latency) ---")
    degraded_metrics = []
    for i, target in enumerate(targets):
        print(f"  > Reaching target {i+1}/20...", end="\r")
        metrics = run_reaching_trial(sim, target, noise_std=0.05, latency_steps=5, viewer=viewer)
        degraded_metrics.append(metrics)
    print()
    print_summary("Degraded", degraded_metrics)
        
    print("\n--- Running Fixed Experiment (Tuned parameters) ---")
    fixed_metrics = []
    for i, target in enumerate(targets):
        print(f"  > Reaching target {i+1}/20...", end="\r")
        # Fix: Lower proportional gain (k=0.8) and higher DLS damping (lambda=0.3)
        metrics = run_reaching_trial(sim, target, noise_std=0.05, latency_steps=5, 
                                     k=0.8, lambda_val=0.3, viewer=viewer)
        fixed_metrics.append(metrics)
    print()
    print_summary("Fixed", fixed_metrics)
    
    # Automatically close the viewer explicitly at the end!
    if viewer is not None:
        viewer.close()
        
    plot_experiment_results(baseline_metrics, degraded_metrics, fixed_metrics)
    print("\n[INFO] All experiments complete! Viewer closed. Plots saved to results/")

if __name__ == '__main__':
    main()
