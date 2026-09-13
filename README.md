# ObliviqLabs Robotics Simulation Take-Home Challenge

This repository contains a complete solution for the ObliviqLabs 2-DOF Robot Arm MuJoCo challenge. It implements a custom physics simulation, manual Forward Kinematics, analytical Jacobians, and a robust Inverse Kinematics controller designed to handle real-world latency and sensor noise.

## Setup & Execution

The project requires a standard Python 3 Linux environment. You can set up the environment and run the experiments using the provided bash script:

```bash
git clone https://github.com/MokshagnaAnurag/mujoco-kinematics-controller.git
cd mujoco-kinematics-controller
chmod +x run.sh
./run.sh
```

Running this script will:
1. Install requirements (`mujoco`, `numpy`, `matplotlib`).
2. Run a mathematical verification to prove the manual Jacobian matches MuJoCo.
3. Launch a live 3D viewer showing the robot reaching 20 targets under Baseline, Degraded, and Fixed conditions.
4. Save a final performance plot to `results/errors_comparison.png`.

## 📁 Repository Structure
* `robot.xml`: MuJoCo MJCF definition of the planar 2-DOF arm and movable target.
* `src/kinematics.py`: Manual implementation of Forward Kinematics and the Analytical Jacobian.
* `src/controller.py`: Implementation of the Damped Least-Squares (DLS) Inverse Kinematics controller.
* `src/simulation.py`: Object-oriented Python wrapper interacting with the MuJoCo C-API.
* `src/experiments.py`: The main runner. Handles mathematical verification, fault injection (noise/latency), and plotting.
* `results/`: Directory for output plots.
* `LEARNING_NOTES.md`: Reflection on challenges, learning, and the Part 5 debugging process.

## Controller Design Choice (Part 3)

I chose to implement **Damped Least-Squares (DLS) Inverse Kinematics** rather than a standard Jacobian Pseudo-Inverse or Jacobian Transpose. 
* **Why?** Standard pseudo-inverse methods become mathematically unstable near singularities (e.g., when the 2-DOF arm is fully extended). This causes joint velocity commands to explode toward infinity. DLS introduces a damping factor ($\lambda$) that safely limits joint velocities near singularities, sacrificing a tiny bit of accuracy for massive gains in real-world stability.

---

## Part 6 - Physical AI Questions

If we were to replace the hand-designed DLS controller with a learned Machine Learning policy, here is how the system would be designed:

### Policy Design
* **Observations:** The policy would consume the current joint angles ($q$), joint velocities ($\dot{q}$), the end effector position ($x_{ee}$), and the relative position vector to the target ($\Delta x$).
* **Actions:** The network would output Joint velocity commands ($\dot{q}_{cmd}$). While direct motor torques ($\tau$) are often preferred for highly dynamic, deep policies, velocity commands are significantly easier and faster to learn for kinematic reaching tasks.
* **Training Data:** Data would be generated via Reinforcement Learning (e.g., PPO or SAC algorithm). We would heavily parallelize the MuJoCo simulation, randomizing the target position across the reachable workspace every episode to gather millions of interaction steps.
* **Evaluation Metrics:** The policy would be evaluated on:
  1. Success Rate (reaching within a 1cm threshold).
  2. Convergence time (average steps to target).
  3. Energy efficiency (integral of squared actions).
  4. Trajectory smoothness (minimizing jerk).

### Sim-to-Real Transfer
* **Randomized Parameters (Domain Randomization):** To prevent the AI from overfitting to a "perfect" simulation, during training I would heavily randomize link masses, link inertias, joint friction (dry and viscous), actuator latency, sensor noise variance, and minor link dimensions.
* **Most Crucial Inaccuracies:** The most dangerous inaccuracies when transferring to a physical robot are **communication latency** and **unmodeled actuator dynamics** (like deadbands, backlash, or non-linear motor friction). If the policy isn't trained to expect a delay between seeing an observation and the motor actually moving, it will violently oscillate on real hardware.

### Data Collection & Recovery
* If the policy repeatedly fails in one particular region of the workspace (e.g., a specific corner), I would collect more data by explicitly initializing episodes with the target in and around that specific region. 
* Crucially, I would also augment this with random start configurations near that region to ensure the policy learns the local recovery dynamics (how to get unstuck) without catastrophically forgetting the rest of the workspace.
