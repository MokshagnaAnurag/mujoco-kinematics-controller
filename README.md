# ObliviqLabs Robotics Simulation Take-Home Challenge

This repository contains a MuJoCo simulation of a 2-DOF robot arm that uses a Damped Least-Squares (DLS) Inverse Kinematics controller to reach randomized target positions.

## Structure
- `robot.xml`: MuJoCo XML definition of the 2-DOF planar robot.
- `src/kinematics.py`: Manual implementation of forward kinematics and analytical Jacobian.
- `src/simulation.py`: Object-oriented wrapper for MuJoCo model, data, and interactions.
- `src/controller.py`: Implementation of the DLS IK controller.
- `src/experiments.py`: Script to run reaching experiments on randomized targets, with options to introduce noise and latency.
- `results/`: Directory for output plots.
- `LEARNING_NOTES.md`: Reflection on challenges, learning, and debugging.

## Setup
The project requires a standard Python 3 Linux environment. You can set up the environment and run the experiments using:

```bash
git clone <repository-url>
cd <repository-directory>
chmod +x run.sh
./run.sh
```

## Part 6 - Short Physical AI question

**Policy design:**
* **Observations**: Current joint angles ($q$), joint velocities ($\dot{q}$), end effector position ($x_{ee}$), and the relative position vector to the target ($\Delta x$).
* **Actions**: Joint velocity commands ($\dot{q}_{cmd}$) or direct motor torques ($\tau$). Using torques is often preferred for deeper policies, but velocities are easier to learn.
* **Training Data**: Generated through reinforcement learning (e.g., PPO or SAC) by randomizing target positions within the workspace and stepping the MuJoCo simulation.
* **Evaluation Metrics**: Success rate (reaching within threshold), average time to target, energy consumption (integral of squared torques), and smoothness of trajectory.

**Sim-to-real:**
* **Randomized Parameters**: Link masses, link inertias, joint friction (dry and viscous), actuator latency, sensor noise variance, and minor dimensional variations.
* **Most crucial inaccuracies**: Actuator dynamics (e.g. non-linearities and deadbands in physical motors) and communication latency often have the largest negative impact when transferring to a real robot.

**Data:**
* If the policy repeatedly fails in one particular region of the workspace, I would collect more data explicitly initialized with targets in and around that specific region. I would also augment this with random start configurations near that region to ensure the policy learns the local recovery dynamics without overfitting.
