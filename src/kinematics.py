import numpy as np

def forward_kinematics(q, l1=1.0, l2=1.0):
    """
    Calculate the (x, y) position of the end effector.
    q: [q1, q2] joint angles
    l1, l2: lengths of the two links
    """
    q1, q2 = q
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, y])

def jacobian(q, l1=1.0, l2=1.0):
    """
    Calculate the analytical Jacobian for the 2-DOF arm.
    Returns a 2x2 matrix J such that v = J * q_dot
    """
    q1, q2 = q
    J = np.zeros((2, 2))
    J[0, 0] = -l1 * np.sin(q1) - l2 * np.sin(q1 + q2)
    J[0, 1] = -l2 * np.sin(q1 + q2)
    J[1, 0] = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    J[1, 1] = l2 * np.cos(q1 + q2)
    return J
