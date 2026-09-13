# Learning Notes

This document reflects on the development process, challenges overcome, and the explicit debugging process for Part 5 of the ObliviqLabs Take-Home Challenge.

## What did you have to learn?
Before starting, while I was familiar with the mathematics of robotics (kinematics, Jacobians), I had to quickly learn the specific structure of the modern `mujoco` Python bindings. Specifically, transitioning from the deprecated `mujoco-py` wrapper to the official DeepMind bindings required understanding how to correctly interact with `mjData` arrays (like `qpos`, `ctrl`, and `site_xpos`) and when to explicitly step the physics engine versus just updating the kinematic tree.

## What resources did you use?
* **MuJoCo Official Documentation**: Heavily utilized the XML Reference for defining joints, physical properties, and velocity actuators correctly.
* **NumPy/Matplotlib Documentation**: Standard reference for matrix math and plotting.
* **AI Assistance**: Leveraged an AI coding assistant (like Claude/ChatGPT) to rapidly scaffold the initial MJCF XML syntax and generate the boilerplate for matplotlib graphs, allowing me to focus my time on the core mathematical implementation and debugging logic.

---

## Part 5: Debugging the Most Interesting Failure
As requested in the prompt, here is a detailed breakdown of how I diagnosed and fixed the failure introduced in Part 4.

**1. What was going wrong?**
When I introduced a 5-step actuator/sensor latency and Gaussian sensor noise, the arm stopped moving smoothly. Instead, as it approached the target, it began vibrating and oscillating violently, often repeatedly overshooting the target and taking significantly longer to converge.

**2. What did you initially suspect?**
Because I was using a Damped Least-Squares (DLS) controller, my initial suspicion was that the damping factor ($\lambda$) was too small, allowing the matrix inversion to become unstable under noisy conditions. 

**3. How did you test the hypothesis?**
I tested this by significantly increasing $\lambda$ (from 0.1 to 1.0) and running the simulation again. 

**4. What turned out to be the actual cause?**
Increasing $\lambda$ did not solve the violent overshooting. By logging the error vectors, I realized the actual cause was the **latency**. The arm was moving fast enough that it would cross the target physically, but because the sensors were delayed by 5 steps, the "brain" didn't realize it had crossed the target yet. The controller kept commanding forward velocity, causing a massive overshoot, followed by a violent over-correction backward 5 steps later.

**5. What did you change?**
I implemented two changes to create the "Fixed" controller:
1. I lowered the proportional gain ($k$) from `2.0` to `0.8`.
2. I increased the DLS damping ($\lambda$) to `0.3`.
This combination forced the robot to approach the target much more slowly and cautiously.

**6. Did the change measurably improve performance?**
Yes. As shown in the generated `errors_comparison.png` graph, the red "Degraded" plot shows highly jagged, unstable trajectories. The green "Fixed" plot takes roughly 3x longer to reach the target (sacrificing speed), but the curves are perfectly smooth and stable. In real-world robotics, trading speed to safely handle network latency is exactly the right approach.

---

## What were the hardest bugs? (Aside from the latency issue)
**State Synchronization in MuJoCo:** 
Initially, my controller was receiving stale positions for the end effector (`sim.data.site_xpos`). I realized that if you modify joint positions (`qpos`), the site positions are not updated automatically. I had to learn to explicitly call `mujoco.mj_kinematics(self.model, self.data)` to force the physics engine to calculate the forward kinematics before querying the site position, otherwise the controller would always lag by one frame.

## What would you improve with another day?
* **Operational Space Control (Torque Control):** Instead of using kinematic velocity control (directly setting `q_dot`), I would upgrade the simulation to use torque motors and implement a full Operational Space Controller ($F = \Lambda \ddot{x} + \mu + p$). This would allow the arm to be physically compliant and handle external physical disturbances much better.
* **State Estimation:** I would implement an Extended Kalman Filter (EKF) to fuse the delayed, noisy sensor readings with a forward dynamics model, allowing the robot to estimate its true current state despite the latency, rather than just forcing the robot to move slower.
