# Revamp design: pick-and-place under uncertainty, v2

Date: 2026-10-07. Branch: `revamp`. The v1 course submission is preserved at tag `v1-course-submission`.

## 1. Intent

**The user's request:**
- Turn the ASEN 5264 project into a proper, portfolio-grade simulation project.
- Do all five revamp items: fair comparison, localization that matters, real grasping, POMCP done properly, hygiene.
- Produce proper sim footage, then redo `/brag`.
- No hardware.
- Use about 3–4 days of machine time.
- Never overwork the laptop.

**Assumptions (reviewable):**
- The course submission is frozen.
- v2 is a new standalone Python package. The ROS 2 GUI stack stays as v1 legacy, with only the `watch_policy` path bug fixed.

**Success criteria:**
1. Every method sees the identical env, reward and episode seeds; only the observation differs.
2. Grasp success is decided by contact physics: jaw pads, friction, the brick lifted off the table.
3. Results are multi-seed with confidence intervals (IQM + stratified bootstrap; Wilson per cell), and the task's memory gap (oracle − memoryless) is demonstrated before any method claims.
4. One command reproduces each result table and figure from saved runs.
5. Footage shows real pick-and-place from logged episodes.

## 2. Constraints

- **Hardware:** i7-1165G7 (4 cores / 8 threads, ~28 W), 15.7 GB RAM, no GPU.
- **Training load:** ≤ 4 env workers, one training run at a time, `nice 19`, one torch thread per process.
- **Watchdog:** every long job runs under `tools/watchdog.py`. It SIGSTOPs the job when the CPU package goes above 85 °C (resumes below 75 °C), MemAvailable drops under 2 GB, or the laptop is on battery.
- **Compute budget:** ≈ 80 h of training plus evaluation. Run length is fixed after measuring env throughput (see §6).

## 3. Architecture

A new package, `pomdp_pick/`, at the repo root. It's pure Python with no ROS, installable via `pyproject.toml`, and runs in an isolated `.venv` with pinned `requirements.txt`. The isolated venv avoids the system cv2/NumPy 2 crash.

| Module | One job |
|---|---|
| `robot/model.py` | Build the MJCF from `so_arm101_description` URDF + scene: jaw box pads (meshes become visual-only), position actuators, gravity comp, contact excludes, 2 ms `implicitfast`, pyramidal condim 3. Settings come from the verified prototype (24/24 grasps). |
| `robot/kinematics.py` | Geometric IK/FK, vendored from `compute_workspace.py`. Tests check it against the legacy implementation. |
| `robot/controller.py` | Cartesian moves (IK → actuator setpoints, smooth interpolation) and the grasp/place primitives. |
| `sensing/camera.py` | Wrist camera: pose from the `camera_link` body (not aimed at the true target, unlike v1), frustum and occlusion tests (vendored `occlusion.py`), distance-dependent noise. |
| `belief/particle_filter.py` | Vectorised, seeded SIR filter over (x, y, θ mod π, log σ_ep). |
| `envs/pick_env.py` | `PickPlaceEnv` gymnasium env. Observation mode is a constructor argument. |
| `agents/train.py` | One training entry point and method registry, with configs in `configs/*.yaml`. |
| `eval/evaluate.py`, `eval/stats.py`, `eval/figures.py` | Shared-seed evaluation, per-episode CSV, rliable/Wilson statistics, figures. |
| `planning/pomcp.py`, `planning/world_model.py` | POMCP v2 (sub-project 4). |
| `viz/` | Episode logging and footage rendering (sub-project 5). |
| `tools/watchdog.py` | Thermal/memory/battery guarded launcher. |

`runs/<method>/seed<k>/` holds the model, VecNormalize stats, TensorBoard logs and eval CSVs. It's git-ignored.

## 4. Environment v2 (`PickPlaceEnv`)

**Scene:**
- The red 2×4 target brick plus two tall blue distractors, spaced ≥ 50 mm apart.
- Bricks spawn at r ∈ [0.13, 0.23] m, angle ∈ [−1, 1] rad. The goal is ≥ 80 mm from every brick.
- Brick mass is 10 g (heavier than the real 2.3 g, for contact stability; both masses passed the prototype).

**Control:**
- 10 Hz decisions; each step is 0.1 s (50 physics steps).
- The policy controls a point **P**: the centre between the open jaws at grasp height, plus the gripper yaw.
- Action, 5-D in [−1, 1]: Δx, Δy, Δz (±15 mm), Δyaw (±0.3 rad), commit.
- P is clamped to the reachable workspace, and the yaw branch is chosen to keep wrist roll inside its limits (the brick is 180°-symmetric).

**Grasp primitive** (when commit > 0.5 and not holding):
1. Open the jaw to 0.5 rad.
2. Descend so the jaw tips reach 2.5 mm above the table.
3. Close (command −0.17 rad; squeeze comes from position error).
4. Lift 5 cm.

It takes about 2.5 s of sim time, all physical. **Held** means the brick is > 30 mm above the table and between the pads. The jaw clearance is set so the grasp tolerance on P is roughly ±6 mm across the jaw axis and ±12 mm along it, so localization error at σ ≳ 5 mm matters. A miss pushes or bumps the brick for real.

**Place primitive** (scripted, physical): carry to the known goal, lower, open, retreat.
- **Success:** the brick rests on the table within 20 mm of the goal.
- **Failure:** a dropped or misplaced brick.

The goal is known exactly, so scripting the place keeps the learning problem on the uncertain part.

**Budget:**
- 20 s of sim time per episode. Primitives consume their real duration, so a miss costs about 25 decision steps.
- At most 3 grasp attempts.

**Sensing:** a wrist-camera reading each step if the target is inside the frustum and not occluded by a distractor.
- Position noise: σ(d) = clip(σ_ep · d / 0.10 m, 0.3 σ_ep, 3 σ_ep), where d is the camera-to-target distance. Closer is clearer.
- Yaw noise: 0.02 rad per mm of σ(d).
- σ_ep is drawn log-uniform over [2, 20] mm for training. Evaluation sweeps {2, 5, 10, 15, 20, 30} mm; 30 is out of distribution.
- Drift: the untouched brick random-walks 0.5 mm per step.

**Reward** (identical for all methods, computed from the true state):
- Step cost: −0.01 per 0.1 s.
- Potential shaping γΦ(s′) − Φ(s), with Φ = −‖P_xy − brick_xy‖ / 0.05 (Ng et al.; preserves the optimal policy).
- −1 per failed grasp, +5 per successful grasp, +10 per successful placement.
- No method-specific terms; v1's belief-only σ bonus is dropped.

**Observations.** Common to all methods (8 dims): P xyz, sin/cos 2·yaw, jaw state, attempts used, time remaining. On top of that:

| Method | Extra dims | What it sees |
|---|---|---|
| `oracle` | +4 | True brick pose (x, y relative to P; sin/cos 2θ) — the ceiling |
| `plain` | +6 | Last reading (relative x, y; sin/cos 2θ), visible flag, age of reading — the memoryless floor |
| `framestack` | plain × 4 | `VecFrameStack` over plain |
| `recurrent` | plain | sb3-contrib RecurrentPPO, LSTM 128 |
| `belief` | +9 | PF mean (relative x, y; sin/cos 2θ), σ (x, y, θ), E[σ_ep], visible flag |

**Particle filter:** 300 particles over (x, y, θ, log σ_ep).
- Predict: drift-matched process noise. After a grasp attempt touches the brick, it inflates the noise (+10 mm).
- Update: Gaussian likelihood using each particle's own σ_ep through σ(d), with θ residuals wrapped mod π.
- Resampling: systematic, when ESS < N/2, with 5 % reinjection around the latest reading.
- Seeded from the env seed.
- Estimating σ_ep jointly replaces v1's fixed 1.5× likelihood widening, the cause of v1's 30 mm collapse.

## 5. Experiment protocol (sub-project 3)

- **Training:** all PPO variants share hyperparameters.
  - 4 SubprocVecEnv workers, n_steps 1024, batch 256, 10 epochs.
  - Learning rate 3e-4, decaying linearly; entropy coefficient 0.01; network [256, 256]; γ 0.99.
  - VecNormalize on observations.
  - RecurrentPPO uses the same settings plus an LSTM of 128.
  - Seeds {0, 1, 2}. Run order: oracle (sanity and ceiling), plain, belief, framestack, recurrent.
- **Evaluation:** every (method, seed) at each σ_ep level, 200 episodes, deterministic actions.
  - Episode seeds are shared: env reset seed = 10 000 + i, with the PF seeded from it.
  - Every episode is logged to CSV: success, attempts, time to success, P error at commit, belief σ at commit.
- **Statistics:**
  - IQM success with 95 % stratified-bootstrap CIs across seeds × levels (rliable).
  - Wilson 95 % interval per cell.
  - Probability of improvement, belief versus each baseline.
  - The memory gap (oracle − plain) is reported first. If it isn't significant, the task is retuned before methods are compared.
- **Figures:** success vs σ_ep with CI bands, an attempts and time-to-success distribution, σ at commit, and an example belief trace.

## 6. Compute plan

- Measure single-worker and 4-worker steps per second on env v2 before fixing run length.
- **Rule:** 15 runs (5 methods × 3 seeds) must fit in ≈ 60 h, with evaluation ≤ 10 h.
- **Fallbacks, in order:** shorter runs, then dropping `framestack` (keeping `recurrent` as the memory baseline), then 2 seeds for the memory baselines.
- Every job is queued through `tools/run_queue.py` (sequential, under the watchdog) and checkpoints so it can resume after a pause.

## 7. POMCP v2 (sub-project 4)

- **Actions:** discrete macros — ±x, ±y (10 mm), down/up (20 mm), yaw ±0.25 rad, commit.
- **Belief:** the same particle filter. Root states are sampled from the particles.
- **Simulator:** a MuJoCo copy (`mj_copyData`) with the brick pose set from the sampled particle, so primitives run physically.
- **Search:** UCB1, depth 8, heuristic rollouts. Budgets of {32, 128, 512} simulations per step, recording success vs. wall-clock per decision.
- **Learned world-model variant:**
  - Train an MLP on belief-state transitions collected from belief PPO, with VecNormalize loaded correctly.
  - Report held-out per-dimension MSE and grasp-outcome AUC.
  - Use it as a cheap simulator inside POMCP.
- **Evaluation:** σ_ep = 10 mm, 50 episodes per budget, under the watchdog.

## 8. Hygiene (sub-project 1)

- Add `pyproject.toml`, a pinned `requirements.txt` and a `.venv` setup script.
- Add a `.gitignore` for runs, model binaries, ROS build/install/log and brag work dirs.
- Stop tracking binaries with `git rm --cached`. They stay on disk and in the v1 tag; history is not rewritten.
- Legacy: fix `watch_policy.py`'s default model directory, and mark v1 scripts as legacy in the README.
- Tests (pytest): kinematics vs legacy; PF determinism, angle wrapping and convergence; env reset determinism and observation shapes; grasp succeeds at the true pose and fails at a 15 mm offset; camera frustum/occlusion; reward identical across observation modes.
- Keep `docs/training-log.md` updated with every fix and run.

## 9. Footage and brag (sub-project 5)

- **Logging:** episodes are logged at physics resolution (qpos at 30 fps), so renders are actual sim motion with no interpolation.
- **Renderer:** reuses the v1 brag renderer (studio floor, belief cloud) for jaw-closing close-ups and side-by-side method comparisons on shared seeds.
- **Report:** a README rewrite and an updated report (results from v2 only), then `/brag`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| RL fails to learn commit timing | The oracle run is the first sanity gate; reward and primitive tolerance get tuned there before other methods train. |
| Throughput too low | The §6 fallbacks. |
| Memory gap too small | Lengthen occlusion and drift, or tighten the tolerance. This is decided on oracle/plain before investing in the rest. |
| Laptop heat | The watchdog; jobs resume automatically. |
