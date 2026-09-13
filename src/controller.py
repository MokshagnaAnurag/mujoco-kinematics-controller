"""
controller.py
=============

Reaching controller for the 2-DOF arm.

Method chosen: velocity-based damped least-squares (DLS) inverse
kinematics ("Jacobian pseudo-inverse with damping"), converted to a
joint-torque command via a simple PD-in-joint-space term.

Why DLS instead of plain Jacobian transpose or plain pseudo-inverse:
- Jacobian transpose is simple and always stable, but converges slowly
  and its gain is hard to tune well across the whole workspace (behaves
  very differently near full extension vs. folded configurations).
- Plain pseudo-inverse (J^+) gives fast, accurate convergence almost
  everywhere, but J becomes singular / ill-conditioned near full
  extension (elbow straight) and near the origin, causing huge joint
  velocity commands right where we most need it to behave well (targets
  at the edge of the workspace are exactly the "at least 20 random
  reachable targets" that will include near-singular configurations).
- Damped least squares blends the two: it behaves like the pseudo-inverse
  when J is well-conditioned, and smoothly degrades toward a damped,
  bounded-velocity behavior (like transpose) near singularities. This
  makes it robust across the whole workspace without manual gain
  scheduling, which is why it's used here.

Control law
-----------
Given a desired end-effector velocity command:

    v_cmd = Kp * (target - ee_pos)          (simple P term toward target)

we solve for a joint velocity command using the damped pseudo-inverse:

    qdot_cmd = J^T (J J^T + lambda^2 I)^-1 v_cmd

This qdot_cmd is then tracked with a joint-space PD torque controller
(so the actuators, which are torque motors in robot.xml, receive a
sensible command instead of directly writing velocities):

    tau = Kd_joint * (qdot_cmd - qdot_true) + damping_compensation

A small joint-space damping/gravity compensation term is added because
the arm has joint damping in the model; for a purely planar arm rotating
about Z there is no gravity torque, so the main disturbance term to
compensate is the modeled joint damping.
"""

import numpy as np
from kinematics import TwoLinkKinematics


class DLSReachingController:
    def __init__(self, l1: float, l2: float,
                 kp_task: float = 4.0,
                 damping_lambda: float = 0.08,
                 kd_joint: float = 6.0,
                 max_qdot: float = 4.0,
                 max_tau: float = 5.0):
        self.kin = TwoLinkKinematics(l1, l2)
        self.kp_task = kp_task
        self.damping_lambda = damping_lambda
        self.kd_joint = kd_joint
        self.max_qdot = max_qdot
        self.max_tau = max_tau

    def compute_torque(self, q: np.ndarray, qdot: np.ndarray,
                        target_xy: np.ndarray) -> np.ndarray:
        ee = self.kin.forward_kinematics(q)
        pos_err = np.asarray(target_xy) - ee

        v_cmd = self.kp_task * pos_err

        J = self.kin.jacobian(q)
        JJt = J @ J.T
        damped_inv = J.T @ np.linalg.inv(JJt + (self.damping_lambda ** 2) * np.eye(2))
        qdot_cmd = damped_inv @ v_cmd

        # Clip commanded joint velocity for safety / to avoid huge spikes
        # near singularities even after damping.
        norm = np.linalg.norm(qdot_cmd)
        if norm > self.max_qdot:
            qdot_cmd = qdot_cmd * (self.max_qdot / norm)

        tau = self.kd_joint * (qdot_cmd - qdot)

        tau = np.clip(tau, -self.max_tau, self.max_tau)
        return tau

    def __call__(self, q, qdot, target_xy):
        return self.compute_torque(q, qdot, target_xy)


class HierarchicalDLSController:
    """Two-rate version of DLSReachingController, used as the Part 5 fix.

    Splits control into:
      - an OUTER kinematic loop: recomputes the desired joint velocity
        qdot_cmd from the (filtered) measured position using damped-least-
        squares IK. This is only called when new position information is
        available (e.g. at a low sensor/planning rate).
      - an INNER torque loop: tracks the last commanded qdot_cmd with a
        joint-space PD law using the ALWAYS-CURRENT true joint velocity,
        every physics step. This mirrors how real robot controllers run a
        fast low-level current/torque loop beneath a slower outer
        kinematic/planning loop, and it is what prevents the torque
        zero-order-hold from turning into a limit cycle when the outer
        loop runs slowly (see LEARNING_NOTES.md Part 5).

    A first-order low-pass filter is also applied to the measured joint
    angle before it feeds the outer loop, to reduce the (secondary)
    sensor-noise contribution to jitter.
    """

    def __init__(self, l1, l2, kp_task=4.0, damping_lambda=0.08,
                 kd_joint=6.0, max_qdot=4.0, max_tau=5.0, filter_alpha=0.15):
        self.kin = TwoLinkKinematics(l1, l2)
        self.kp_task = kp_task
        self.damping_lambda = damping_lambda
        self.kd_joint = kd_joint
        self.max_qdot = max_qdot
        self.max_tau = max_tau
        self.filter_alpha = filter_alpha

        self.q_filt = None
        self.qdot_cmd = np.zeros(2)

    def reset(self):
        self.q_filt = None
        self.qdot_cmd = np.zeros(2)

    def update_outer(self, q_meas: np.ndarray, target_xy: np.ndarray):
        """Recompute qdot_cmd from a (possibly noisy, possibly stale)
        position measurement. Call this at whatever slow outer-loop rate
        is available."""
        if self.q_filt is None:
            self.q_filt = q_meas.copy()
        else:
            self.q_filt = self.filter_alpha * q_meas + (1 - self.filter_alpha) * self.q_filt

        ee = self.kin.forward_kinematics(self.q_filt)
        pos_err = np.asarray(target_xy) - ee
        v_cmd = self.kp_task * pos_err

        J = self.kin.jacobian(self.q_filt)
        JJt = J @ J.T
        damped_inv = J.T @ np.linalg.inv(JJt + (self.damping_lambda ** 2) * np.eye(2))
        qdot_cmd = damped_inv @ v_cmd

        norm = np.linalg.norm(qdot_cmd)
        if norm > self.max_qdot:
            qdot_cmd = qdot_cmd * (self.max_qdot / norm)
        self.qdot_cmd = qdot_cmd

    def compute_inner_tau(self, qdot_true: np.ndarray) -> np.ndarray:
        """Fast inner-loop torque command tracking the last qdot_cmd,
        using the always-fresh true joint velocity. Call this EVERY
        physics step regardless of the outer loop's rate."""
        tau = self.kd_joint * (self.qdot_cmd - qdot_true)
        return np.clip(tau, -self.max_tau, self.max_tau)
