# Learning Notes

## What did you have to learn?
Before starting, I was generally familiar with robotics concepts, but I had to quickly brush up on the exact structure of MuJoCo's Python bindings (`mujoco` module), specifically how to correctly query `site_xpos` and modify actuator `ctrl` values without the old `mujoco-py` wrapper. I also had to confirm how to represent planar constraints cleanly in MJCF.

## What resources did you use?
* **MuJoCo Documentation**: Consulted the official XML Reference for defining joints, geoms, and actuators.
* **NumPy/Matplotlib**: Standard references.
* **AI Assistance**: Used an AI coding assistant to quickly scaffold the XML syntax and plot generation scripts, which saved time on boilerplate.

## What were the hardest bugs?
1. **End Effector Position Updating**: Initially, querying `sim.data.site_xpos` would sometimes yield delayed or stale values because `mujoco.mj_step()` had been called, but `mujoco.mj_kinematics()` was needed to explicitly update the site positions before computing errors. Debugging this involved comparing the manual Forward Kinematics against the MuJoCo output and seeing a 1-step lag. Adding `mujoco.mj_kinematics(self.model, self.data)` before querying solved this.
2. **Controller Instability with Noise**: When adding sensor noise and actuator latency (Part 4), the DLS controller began oscillating violently near the target. I initially suspected the lambda term in the pseudo-inverse was too small. Testing higher lambda values didn't fix the oscillations. The actual cause was that latency caused the error vector to be outdated, so the robot overshot the target before the controller reacted. I mitigated this by lowering the proportional gain and increasing joint damping, demonstrating that a slower approach can remain stable under latency.

## What would you improve with another day?
* **More robust Controller**: Implement a full operational space controller (torque-based) instead of kinematic velocity control, allowing for compliance and better handling of dynamic disturbances.
* **Visualization**: Add a video recording feature using MuJoCo's renderer to save MP4s of the successful/failed trajectories.
* **System Identification**: Implement a simple Extended Kalman Filter (EKF) to better estimate true joint states from the noisy sensors before feeding them into the controller.
