import numpy as np
from kinematics import jacobian

def dls_inverse_kinematics(q, target_pos, current_pos, lambda_val=0.1, k=2.0):
    """
    Damped Least-Squares (DLS) Inverse Kinematics.
    Calculates joint velocities to reach the target position.
    """
    J = jacobian(q)
    
    # Error vector (velocity towards target)
    e = target_pos - current_pos
    
    # Gain for the error
    v = k * e
    
    # DLS formula: q_dot = J^T (J J^T + lambda^2 I)^-1 v
    J_T = J.T
    JJ_T = J @ J_T
    I = np.eye(2)
    
    J_pseudo = J_T @ np.linalg.inv(JJ_T + lambda_val**2 * I)
    
    q_dot = J_pseudo @ v
    return q_dot
