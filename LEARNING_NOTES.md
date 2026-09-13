# Learning Notes

## What did I have to learn?

- **MuJoCo's Python API from scratch** -- I hadn't used `mujoco.MjModel`
  / `mujoco.MjData` before. Specifically: the difference between
  `qpos`/`qvel` (joint state) vs. `site_xpos`/`body_xpos` (Cartesian
  poses derived from forward kinematics), how `mj_jacSite` returns a
  `(3, nv)` translational Jacobian stacked with a rotational one, and
  that you have to call `mj_forward` (or step) before any of the derived
  quantities (site positions, Jacobians) are valid for a given `qpos`.
- **Mocap bodies.** Needed a way to move the target marker around
  without MuJoCo's physics acting on it (no gravity, no collisions). A
  `mocap="true"` body driven via `data.mocap_pos` (not `qpos`) was the
  right tool -- this wasn't obvious at first; my first attempt used a
  free-joint body and then had to fight gravity pulling the target down.
- **Damped least-squares IK**, as a concept -- I knew Jacobian transpose
  and pseudo-inverse IK, but had to look up why plain pseudo-inverse
  Jacobian IK misbehaves near singularities and how the `lambda^2 I`
  damping term in `J^T(JJ^T + lambda^2 I)^-1` trades off accuracy vs.
  stability there.
- **Discrete-time control loop pitfalls** -- I understood in the abstract
  that "low control rate is bad," but hadn't previously run into the
  specific failure mode from Part 5, where holding a *velocity-tracking
  PD torque command* constant across a zero-order-hold window (rather
  than just holding a *kinematic setpoint* constant) turns a normally
  stable loop into a limit cycle. Splitting outer (kinematic) and inner
  (torque) loop rates to fix this is a standard robotics pattern I'd
  heard of by name but never had to actually implement and debug.

## What resources did I use?

- MuJoCo's official Python bindings documentation and the `mujoco`
  PyPI package's docstrings (via `help()`/introspection) for the exact
  signatures of `mj_jacSite`, `mj_forward`, `mj_step`, and the meaning
  of `body_mocapid` / `mocap_pos`.
- General robotics background on Jacobian-based IK (transpose,
  pseudo-inverse, damped least squares) to decide which controller to
  implement and why.
- Claude (this assignment's stated-allowed AI tool) for: scaffolding the
  MuJoCo XML syntax for hinge joints/capsule geoms/mocap bodies,
  reviewing the finite-difference Jacobian check, and pair-debugging the
  Part 5 failure (isolating each of the three faults individually to
  find which one actually mattered was suggested and executed together
  with the model). All actual root-cause investigation (reading the
  logged torque/error time series, forming and testing hypotheses,
  designing the hierarchical-loop fix) is reflected in the step-by-step
  narrative in the README, and I made sure I could reproduce and explain
  every number in `results/summary.txt` myself before writing this up.

## What were the hardest bugs?

1. **The Part 4/5 instability itself** (full narrative in
   `README.md`'s Part 5 section). The short version: my first instinct
   was "sensor noise is causing jitter," and I almost shipped a fix
   (a simple low-pass filter on the measured position, feeding the
   *same* single-rate controller) that only nudged the success rate
   from 0% to 5% -- not the "measurably improved performance" the
   assignment asks for. Isolating each of the three injected faults
   *individually* (rather than trusting my first intuition about which
   one "should" matter most) was the key debugging step: it showed the
   control **rate**, not the noise, was responsible for ~95% of the
   damage, which sent me looking at the torque log instead of the
   position log -- and that's where the saturating, sign-flipping torque
   pattern (the limit cycle) was obvious in hindsight but easy to miss
   without looking at the right signal.
2. **Getting the FK/Jacobian sign and frame conventions to match MuJoCo
   exactly, not just "up to a sign."** My first Jacobian draft used `q2`
   as an absolute world-frame angle (matching a generic 2-link-arm
   formula I remembered) instead of the *relative* angle at the elbow
   joint, which is how MuJoCo's kinematic tree actually defines
   `joint2` (in `link1`'s local frame). The FK numbers were close but
   not identical to MuJoCo's for `q2 != 0`, which was a good reminder to
   check numeric agreement to several decimal places rather than "looks
   about right," and to explicitly write down the joint-angle convention
   in the kinematics module docstring so it can't silently drift again.

## What would I improve with another day?

- **Close the remaining 3/20 failures from Part 5.** They're all near
  the outer edge of the reachable workspace, where the damped-least-
  squares gain and the 3x-wrong damping model leave too little torque
  authority within the fixed 6-second episode. I'd try (a) gain
  scheduling that increases `Kp_task`/`max_tau` as the arm approaches
  the workspace boundary (where the Jacobian's smallest singular value
  is shrinking and DLS is intentionally being conservative), and (b) an
  online damping estimator (fit the effective joint damping from
  observed torque/velocity data during the episode and feed it back
  into the inner-loop PD gain) instead of just tolerating the 3x model
  mismatch, which is closer to the "system identification" option
  mentioned in Part 5's prompt.
- **Add proper unit tests** (e.g. `pytest`) for `kinematics.py` and the
  fault-injection toggles in `simulation.py`, instead of the current
  ad-hoc `__main__` self-check -- would make it much faster to confirm
  nothing regresses while tuning gains.
- **Try a second controller family for comparison**, e.g. an
  operational-space (dynamics-aware) controller that explicitly cancels
  the modeled joint damping term, and see whether it's inherently more
  robust to the "incorrect joint damping" fault than the pure
  kinematic-PD approach used here.
- **Address Part 6 more concretely**: actually train a small RL policy
  (e.g. PPO via `stable-baselines3`) on this exact MuJoCo model as a
  sanity check on the Part 6 answers, and compare its learned behavior
  in the same "break the robot" fault conditions against the
  hand-designed hierarchical controller.
