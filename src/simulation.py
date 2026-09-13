import mujoco
import numpy as np

class RobotSimulation:
    def __init__(self, xml_path="robot.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = ['joint1', 'joint2']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target')
        
    def step(self):
        mujoco.mj_step(self.model, self.data)
        
    def get_joint_positions(self):
        return np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids])
        
    def get_ee_position(self):
        # Update kinematics to get latest positions
        mujoco.mj_kinematics(self.model, self.data)
        return np.array(self.data.site_xpos[self.ee_site_id][:2]) # only x, y
        
    def set_target_position(self, pos):
        # Target body has slide joints for x and y
        target_qposadr_x = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'target_x')]
        target_qposadr_y = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'target_y')]
        self.data.qpos[target_qposadr_x] = pos[0]
        self.data.qpos[target_qposadr_y] = pos[1]
        mujoco.mj_kinematics(self.model, self.data)
        
    def get_target_position(self):
        return np.array(self.data.site_xpos[self.target_site_id][:2])
        
    def set_joint_velocities(self, vels):
        actuator_vel1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'vel1')
        actuator_vel2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'vel2')
        self.data.ctrl[actuator_vel1] = vels[0]
        self.data.ctrl[actuator_vel2] = vels[1]
