"""
simulation.py
=============

Thin wrapper around a MuJoCo model of the 2-DOF planar arm defined in
robot.xml. Provides:

- stepping the physics forward with given controls,
- reading joint state (q, qdot),
- reading the MuJoCo-reported end-effector position (for cross-checking
  our hand-written forward kinematics),
- reading MuJoCo's own Jacobian at the end-effector site (for cross-checking
  our analytical Jacobian),
- setting/reading the (mocap) target position,
- injecting the "break your robot" disturbances used in Part 4:
  actuator/control latency, sensor noise, incorrect link mass, incorrect
  joint damping, actuator noise, lower control frequency, joint friction,
  and external disturbance forces.
"""

import os
import numpy as np
import mujoco

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(THIS_DIR, "..", "robot.xml")


class ArmSim:
    def __init__(self, xml_path: str = DEFAULT_XML, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.link1_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link1"
        )
        self.link2_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link2"
        )

        # Store the "nominal" (correct) physical parameters so Part 4 can
        # temporarily corrupt them and Part 5 can restore/estimate them.
        self.nominal_link_mass = self.model.body_mass[
            [self.link1_body_id, self.link2_body_id]
        ].copy()
        self.nominal_joint_damping = self.model.dof_damping.copy()
        self.nominal_joint_friction = self.model.dof_frictionloss.copy()

        self.dt = self.model.opt.timestep

        # --- "Break your robot" fault-injection settings (all off by default) ---
        self.enable_sensor_noise = False
        self.sensor_noise_std = 0.0          # rad, added to *measured* q
        self.enable_actuator_noise = False
        self.actuator_noise_std = 0.0        # Nm, added to commanded torque
        self.enable_latency = False
        self.latency_steps = 0               # control applied N steps late
        self._ctrl_buffer = []               # FIFO for latency
        self.enable_disturbance = False
        self.disturbance_force = np.zeros(2)  # constant force (x,y) on end effector
        self.control_decimation = 1          # 1 = every physics step gets a new command

    # ------------------------------------------------------------------ #
    # Basic state access
    # ------------------------------------------------------------------ #
    def reset(self, q0=(0.0, 0.0), qdot0=(0.0, 0.0)):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = q0
        self.data.qvel[:2] = qdot0
        self._ctrl_buffer = []
        mujoco.mj_forward(self.model, self.data)

    def get_true_q(self) -> np.ndarray:
        return self.data.qpos[:2].copy()

    def get_true_qdot(self) -> np.ndarray:
        return self.data.qvel[:2].copy()

    def get_measured_q(self) -> np.ndarray:
        """q as the controller would perceive it: true q + optional sensor
        noise (Part 4 fault injection)."""
        q = self.get_true_q()
        if self.enable_sensor_noise:
            q = q + self.rng.normal(0.0, self.sensor_noise_std, size=2)
        return q

    def get_ee_pos_mujoco(self) -> np.ndarray:
        """Ground-truth end-effector (x,y) as reported by MuJoCo's own
        forward kinematics (site position), used to cross-check our
        hand-written forward_kinematics()."""
        return self.data.site_xpos[self.ee_site_id][:2].copy()

    def get_mujoco_jacobian(self) -> np.ndarray:
        """MuJoCo's own translational Jacobian at the end-effector site,
        restricted to the 2 arm joints and the XY plane. Used ONLY to
        verify our analytical Jacobian in kinematics.py -- never used by
        the controller itself."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp[:2, :2].copy()  # rows x,y ; cols joint1,joint2

    def set_target(self, xy: np.ndarray):
        self.model.body_pos[self.target_body_id][:2] = xy
        # mocap position is what actually drives the rendered/mocap body
        mocap_id = self.model.body_mocapid[self.target_body_id]
        if mocap_id >= 0:
            self.data.mocap_pos[mocap_id][:2] = xy
            self.data.mocap_pos[mocap_id][2] = 0.0

    # ------------------------------------------------------------------ #
    # Fault injection configuration (Part 4)
    # ------------------------------------------------------------------ #
    def set_incorrect_link_mass(self, factor: float):
        """Scale link masses by `factor` relative to nominal (simulates a
        model that has wrong mass estimates, e.g. payload change)."""
        self.model.body_mass[self.link1_body_id] = self.nominal_link_mass[0] * factor
        self.model.body_mass[self.link2_body_id] = self.nominal_link_mass[1] * factor

    def set_incorrect_joint_damping(self, factor: float):
        self.model.dof_damping[:2] = self.nominal_joint_damping[:2] * factor

    def set_joint_friction(self, frictionloss: float):
        self.model.dof_frictionloss[:2] = frictionloss

    def reset_physical_params(self):
        self.model.body_mass[self.link1_body_id] = self.nominal_link_mass[0]
        self.model.body_mass[self.link2_body_id] = self.nominal_link_mass[1]
        self.model.dof_damping[:2] = self.nominal_joint_damping[:2]
        self.model.dof_frictionloss[:2] = self.nominal_joint_friction[:2]

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def apply_control_and_step(self, tau: np.ndarray):
        """Apply a commanded joint torque `tau`, subject to whichever fault
        injections are enabled (latency, actuator noise, external
        disturbance), then advance the physics by one timestep.
        """
        tau = np.asarray(tau, dtype=float).copy()

        if self.enable_actuator_noise:
            tau = tau + self.rng.normal(0.0, self.actuator_noise_std, size=2)

        if self.enable_latency and self.latency_steps > 0:
            self._ctrl_buffer.append(tau)
            if len(self._ctrl_buffer) > self.latency_steps:
                tau_applied = self._ctrl_buffer.pop(0)
            else:
                tau_applied = np.zeros(2)
        else:
            tau_applied = tau

        self.data.ctrl[:2] = tau_applied

        if self.enable_disturbance:
            # Apply a constant Cartesian force at the end-effector body,
            # converted to joint torque via J^T (external disturbance,
            # e.g. bumping the arm or an unmodeled payload pulling on it).
            J = self.get_mujoco_jacobian()
            tau_dist = J.T @ self.disturbance_force
            self.data.qfrc_applied[:2] = tau_dist
        else:
            self.data.qfrc_applied[:2] = 0.0

        mujoco.mj_step(self.model, self.data)

    def run_control_loop(self, control_fn, target_xy, max_steps, control_hz=None):
        """Run a closed-loop control episode.

        control_fn(q_measured, qdot_measured, target_xy) -> tau  (2,)
        control_hz: if given (< 1/dt), the controller is only re-evaluated
        every N physics steps (models a lower control frequency); the
        last computed torque is held between updates (zero-order hold).
        """
        self.set_target(target_xy)
        log = {"t": [], "q": [], "ee": [], "err": [], "tau": []}

        physics_hz = 1.0 / self.dt
        if control_hz is None or control_hz >= physics_hz:
            decimation = 1
        else:
            decimation = max(1, int(round(physics_hz / control_hz)))

        tau = np.zeros(2)
        for step in range(max_steps):
            if step % decimation == 0:
                q_meas = self.get_measured_q()
                qdot_meas = self.get_true_qdot()  # velocity noise not modeled separately
                tau = control_fn(q_meas, qdot_meas, target_xy)

            self.apply_control_and_step(tau)

            ee = self.get_ee_pos_mujoco()
            err = np.linalg.norm(ee - target_xy)
            log["t"].append(step * self.dt)
            log["q"].append(self.get_true_q().copy())
            log["ee"].append(ee.copy())
            log["err"].append(err)
            log["tau"].append(tau.copy())

        for k in ("q", "ee", "tau"):
            log[k] = np.array(log[k])
        log["t"] = np.array(log["t"])
        log["err"] = np.array(log["err"])
        return log

    def run_hierarchical_control_loop(self, hier_ctrl, target_xy, max_steps, outer_hz=50):
        """Like run_control_loop, but for a two-rate controller (see
        controller.HierarchicalDLSController): the OUTER kinematic update
        (hier_ctrl.update_outer) only runs every `decimation` physics
        steps, while the INNER torque loop (hier_ctrl.compute_inner_tau)
        runs every physics step using the always-current true joint
        velocity. This is the Part 5 fix for the low-control-rate
        instability -- see LEARNING_NOTES.md.
        """
        self.set_target(target_xy)
        log = {"t": [], "q": [], "ee": [], "err": [], "tau": []}

        physics_hz = 1.0 / self.dt
        decimation = max(1, int(round(physics_hz / outer_hz)))

        hier_ctrl.reset()
        for step in range(max_steps):
            if step % decimation == 0:
                q_meas = self.get_measured_q()
                hier_ctrl.update_outer(q_meas, target_xy)

            qdot_true = self.get_true_qdot()
            tau = hier_ctrl.compute_inner_tau(qdot_true)
            self.apply_control_and_step(tau)

            ee = self.get_ee_pos_mujoco()
            err = np.linalg.norm(ee - target_xy)
            log["t"].append(step * self.dt)
            log["q"].append(self.get_true_q().copy())
            log["ee"].append(ee.copy())
            log["err"].append(err)
            log["tau"].append(tau.copy())

        for k in ("q", "ee", "tau"):
            log[k] = np.array(log[k])
        log["t"] = np.array(log["t"])
        log["err"] = np.array(log["err"])
        return log
