"""
kinematics.py
=============

Hand-derived forward kinematics and analytical Jacobian for a planar
2-DOF revolute-joint arm.

Convention
----------
- Both joints rotate about the Z axis (planar arm living in the XY plane).
- q = [q1, q2] are the joint angles in radians.
    q1: angle of link 1 relative to the +X axis (world frame), measured
        at the base joint.
    q2: angle of link 2 relative to link 1 (i.e. the *relative* angle at
        the elbow joint), matching MuJoCo's local/relative joint
        convention for a body chain (link2's hinge is defined in link1's
        frame).
- L1, L2: link lengths (base->elbow, elbow->end-effector).

Forward kinematics
-------------------
Standard 2-link planar arm FK:

    x = L1*cos(q1) + L2*cos(q1 + q2)
    y = L1*sin(q1) + L2*sin(q1 + q2)

Jacobian
--------
J relates joint velocities to end-effector linear velocity:

    [xdot]   [ -L1*sin(q1) - L2*sin(q1+q2)   -L2*sin(q1+q2) ] [q1dot]
    [ydot] = [  L1*cos(q1) + L2*cos(q1+q2)    L2*cos(q1+q2) ] [q2dot]

This is obtained by differentiating x(q), y(q) with respect to q1 and q2.
"""

import numpy as np


class TwoLinkKinematics:
    """Forward kinematics + analytical Jacobian for a 2-link planar arm."""

    def __init__(self, l1: float, l2: float):
        self.l1 = l1
        self.l2 = l2

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector (x, y) position from joint angles q=[q1,q2].

        Derived by hand from the planar 2-link chain geometry (see module
        docstring). This does NOT call MuJoCo -- it is our own
        implementation, used in simulation.py to cross-check against the
        position MuJoCo reports for the end-effector site/body.
        """
        q1, q2 = q[0], q[1]
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return np.array([x, y])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Analytical 2x2 linear-velocity Jacobian d(x,y)/d(q1,q2).

        Implemented directly from the closed-form partial derivatives of
        forward_kinematics -- not obtained by calling a library Jacobian
        function. MuJoCo's own Jacobian (mj_jacSite) is used afterward,
        in simulation.py / experiments.py, purely to *verify* this
        implementation is correct, never to compute it.
        """
        q1, q2 = q[0], q[1]
        l1, l2 = self.l1, self.l2

        j11 = -l1 * np.sin(q1) - l2 * np.sin(q1 + q2)
        j12 = -l2 * np.sin(q1 + q2)
        j21 = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        j22 = l2 * np.cos(q1 + q2)

        return np.array([[j11, j12],
                          [j21, j22]])

    def jacobian_numeric(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Finite-difference Jacobian, used ONLY as an independent numerical
        cross-check for jacobian() (in addition to comparing against
        MuJoCo's Jacobian). Central differences are used for accuracy.
        """
        J = np.zeros((2, 2))
        for i in range(2):
            dq = np.zeros(2)
            dq[i] = eps
            f_plus = self.forward_kinematics(q + dq)
            f_minus = self.forward_kinematics(q - dq)
            J[:, i] = (f_plus - f_minus) / (2 * eps)
        return J

    def is_reachable(self, target_xy: np.ndarray, margin: float = 0.005) -> bool:
        """Check whether target_xy lies within the annular workspace
        [ |L1-L2| + margin , L1+L2 - margin ] of the arm."""
        r = np.linalg.norm(target_xy)
        r_min = abs(self.l1 - self.l2) + margin
        r_max = (self.l1 + self.l2) - margin
        return r_min <= r <= r_max

    def sample_reachable_target(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random reachable (x, y) target uniformly over joint
        angles (a simple and sufficient sampling strategy for this task:
        pick random q1,q2 within joint limits and take the resulting FK
        position -- guarantees reachability by construction)."""
        q1 = rng.uniform(-np.pi, np.pi)
        q2 = rng.uniform(-3.05, 3.05)
        return self.forward_kinematics(np.array([q1, q2]))


if __name__ == "__main__":
    # Small self-test when run directly: python3 src/kinematics.py
    k = TwoLinkKinematics(l1=0.30, l2=0.25)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(1000):
        q = rng.uniform(-np.pi, np.pi, size=2)
        J_analytic = k.jacobian(q)
        J_numeric = k.jacobian_numeric(q)
        err = np.max(np.abs(J_analytic - J_numeric))
        max_err = max(max_err, err)
    print(f"Max |analytic Jacobian - finite-difference Jacobian| over 1000 random q: {max_err:.3e}")
    assert max_err < 1e-5, "Analytical Jacobian does not match finite-difference check!"
    print("Jacobian self-check PASSED.")
