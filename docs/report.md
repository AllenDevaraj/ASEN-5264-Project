# Picking a brick you can't quite see: memory, belief and planning for a physical pick-and-place POMDP

*SO-ARM101 in MuJoCo · v2 technical report. Main results: `runs_v21/eval/*.csv` (v2.1 protocol, §4.2)*

## 1. Problem

A 5-DoF SO-ARM101 arm must pick a red LEGO 2×4 brick and place it on a goal. It perceives the brick only through a wrist camera, which has three limitations:

1. **Noisy readings:** each reading of the brick's pose (x, y, yaw) is corrupted by Gaussian noise. The noise grows with distance: σ(d) = clip(σ_ep · d / 10 cm, 0.3 σ_ep, 3 σ_ep).
2. **Unknown noise level:** the per-episode level σ_ep is never told to the agent. Yaw noise is 0.02 rad per mm of position noise.
3. **Limited, blockable view:** the camera has a real pose on the gripper, a 100° horizontal field of view and a 2 cm minimum depth. Two taller blue bricks occlude it by ray–box tests, so whether the target is visible at all depends on where the policy moves the arm.

The brick also random-walks 0.5 mm per decision while it rests on the table.

The grasp is physical. The policy chooses where and when to commit; a scripted top-down primitive then runs:
1. open the jaws to 0.12 rad;
2. descend to 8.5 mm;
3. close;
4. lift 5 cm.

Box pads on the jaws squeeze the brick, and it rises only if friction holds it. A held brick is carried to the known goal by a scripted physical place. Success requires the brick to rest on the goal (< 20 mm), upright (tilt < 20°).

A missed grasp physically shoves the brick, so a miss also makes the brick's position less certain.

The measured grasp tolerance makes localization matter: roughly ±6 mm across the jaw axis (§A.1). This is the core change from v1, where a 15 mm proximity check counted as a grasp and belief never mattered.

| quantity | value |
|---|---|
| decision rate | 10 Hz; action = (Δx, Δy, Δz ≤ 15 mm, Δyaw ≤ 0.3 rad, commit) |
| budget | 20 s of sim time, ≤ 3 grasp attempts |
| physics | MuJoCo 3.6, 4 ms, implicitfast, position actuators (kp 50, kv 4.5) + gravity compensation |
| training noise | σ_ep ~ log-uniform [2, 20] mm |
| evaluation noise | σ_ep ∈ {2, 5, 10, 15, 20, 30} mm (30 is out of distribution) |

## 2. Methods compared

Every method sees the same dynamics, the same reward and the same evaluation episodes.

| method | observation | memory |
|---|---|---|
| **Oracle PPO** | true brick pose (privileged upper bound) | — |
| **Plain PPO** | latest reading, visibility flag, age of reading | none |
| **Frame-stack PPO** | last 4 plain observations | fixed window |
| **Recurrent PPO** | plain observation | LSTM |
| **Belief PPO** | particle-filter summary: mean, std (x, y, θ), estimated σ_ep, visibility | Bayesian filter |
| **POMCP** | same particle filter, planned online | belief + search |

**Shared reward** (computed from the true state for every method):
- −0.01 per 0.1 s of sim time;
- −1 per missed grasp, +5 per successful grasp, +10 per successful place;
- potential-based shaping with Φ = −d(P, brick) / 5 cm and Φ(terminal) = 0, which leaves the optimal policy unchanged (Ng et al., 1999).

**Particle filter:** 300 particles over (x, y, θ mod π, log σ_ep).
- *Likelihood:* uses the distance at which the brick was seen, never the true distance.
- *Prediction:* drift noise; after a miss, an extra 10 mm of position spread, because the filter is not told how far the brick was shoved.
- *Resampling:* systematic, when the effective sample size drops below N/2, with 5% of particles reinjected around the reading.

Estimating σ_ep jointly is what keeps the filter calibrated at 30 mm. v1's fixed-width likelihood collapsed there.

**PPO setup:**
- Stable-Baselines3 2.7.1, one env worker;
- n_steps 1024, batch 256, 10 epochs, lr 3e-4, entropy 0.01, γ 0.99, GAE λ 0.95;
- VecNormalize; 1M steps per run.
- Recurrent PPO uses sb3-contrib with the same hyperparameters.

**POMCP (PO-UCT with root sampling):**
- Nine macro actions: ±10 mm in x or y, ±15 mm in z, ±0.25 rad yaw, and commit.
- Each simulation restores an exact copy of the physics, writes a particle sampled from the belief as the true target pose, and runs the real `env.step`. Simulated grasps are therefore physical.
- Search: UCB c = 2, depth 8. Leaf values come from a heuristic rollout: align yaw, centre on a per-axis grid that is always reachable, descend, commit.
- Assumption: the simulator knows the distractor poses and the noise model; only the target pose is uncertain.

**WM-POMCP:** the same search over a learned world model, with no physics.
- Model: an MLP mapping (belief summary, action) to (Δ summary, P(grasp)).
- Data: 200k transitions from belief-PPO rollouts with 30% random macros.
- Training: the Δ head on move transitions only; the grasp head on commits only.
- Accuracy is reported on held-out *episodes*.

## 3. Protocol

| | |
|---|---|
| training seeds | 3 per RL method (0, 1, 2); v2.1 adds the grasp-assist curriculum (§4.2) |
| evaluation | 200 episodes × 6 σ levels per run; episode i uses reset seed 10 000 + i for every method |
| per-cell CI | Wilson 95% |
| aggregate | IQM across seeds × σ levels with stratified bootstrap 95% CIs, and P(belief > X) (rliable, Agarwal et al., 2021) |
| POMCP | σ = 10 mm, 50 episodes (seeds 10 000–10 049) per budget ∈ {32, 128, 512} simulations; CPU time per decision |
| compute | one 4-core laptop CPU, no GPU; every job under a thermal/memory/battery watchdog (`tools/watchdog.py`) |

## 4. Results

### 4.1 Base protocol: training reliability, not information, was the bottleneck

The first protocol (§3, plain PPO from scratch) produced a bimodal outcome. Each run either learned to approach and commit, or fell into a "wander" mode: it avoided the −1 miss penalty by never committing.

| (seeds 0–2, 200 eps × 6 σ) | oracle | plain | belief | frame-stack | recurrent |
|---|---|---|---|---|---|
| mean eval success | 0.83 | 0.28 | 0.18 | 0.08 | 0.00 |
| commit error @ 10 mm (mm) | 6.9 | 14.6 | **10.0** | 24.4 | 35.7 |

Belief PPO localized better than plain at every σ, with lower commit error. Its one committing seed beat plain's best seed at σ ≥ 10 mm. But it collapsed into the wander mode on 2 of 3 seeds, and recurrent PPO collapsed on all three.

Ranking the methods on this protocol would rank their exploration luck, not the value of their observations. The full tables are in [`results.md`](results.md).

### 4.2 Main comparison (v2.1: shared grasp-assist curriculum)

**The fix.** v2.1 adds one training-only curriculum, applied identically to every method. On commit, the gripper is nudged up to `assist` mm toward the true brick before the physical grasp runs. The assist decays linearly from 40 mm to 0 by 50% of training, so the second half and all evaluation run with no assist. Commit error is logged *before* the nudge.

The curriculum makes early commits pay off, which removes the commit-avoidance trap. All 12 runs (4 methods × 3 seeds) trained without collapse. Recurrent PPO was dropped from v2.1 for compute.

**Results** (`runs_v21/eval/*_n200.csv`: 3 seeds × 6 σ × 200 shared episodes; Wilson 95% CIs):

| method | σ=2 | σ=5 | σ=10 | σ=15 | σ=20 | σ=30 (OOD) |
|---|---|---|---|---|---|---|
| oracle | 0.73 [0.69–0.76] | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 |
| **belief** | **0.60** [0.56–0.64] | **0.55** [0.51–0.59] | **0.47** [0.43–0.51] | **0.42** [0.38–0.46] | **0.32** [0.29–0.36] | **0.17** [0.15–0.21] |
| plain | 0.55 [0.50–0.58] | 0.50 [0.46–0.54] | 0.39 [0.35–0.43] | 0.30 [0.27–0.34] | 0.24 [0.21–0.28] | 0.14 [0.11–0.16] |
| frame-stack | 0.56 [0.52–0.60] | 0.46 [0.42–0.50] | 0.37 [0.33–0.41] | 0.30 [0.26–0.33] | 0.20 [0.17–0.24] | 0.09 [0.07–0.12] |

| aggregate (seeds × σ) | IQM | 95% CI | P(belief > X) |
|---|---|---|---|
| oracle | 0.728 | 0.682–0.772 | 0.02 |
| **belief** | **0.435** | 0.420–0.452 | — |
| plain | 0.357 | 0.342–0.370 | 0.89 [0.76–0.99] |
| frame-stack | 0.332 | 0.316–0.349 | 0.94 [0.83–1.00] |

![success vs sigma](figs/v21/success_vs_sigma.png)

- **Belief PPO beats plain and frame-stack PPO at every noise level**, and the IQM confidence intervals do not overlap. The seed spread is the strongest evidence: belief's worst seed (0.39 mean success) is above plain's best (0.37).
- **The gap grows with noise:** +0.05 at 2 mm and +0.12 at 15 mm (0.42 vs 0.30, disjoint CIs). It shrinks again at 30 mm, which is outside the training range and where every method approaches the floor.
- **Mechanism: commit error.** Belief PPO commits closer to the brick at every σ (mean error, mm):

  | σ (mm) | 2 | 5 | 10 | 15 | 20 | 30 |
  |---|---|---|---|---|---|---|
  | belief | 7.6 | 8.3 | 10.5 | 12.5 | 14.2 | 19.5 |
  | plain | 7.9 | 9.4 | 12.4 | 14.7 | 17.2 | 22.4 |
  | frame-stack | 9.2 | 10.8 | 13.6 | 16.7 | 20.2 | 26.4 |
  | oracle | 6.8 | 6.8 | 6.8 | 6.8 | 6.8 | 6.8 |

  With roughly ±6 mm of grasp tolerance (§A.1), 2–5 mm of extra error is the difference between a grasp and a shove.
- **A raw history is not a belief.** Frame-stack PPO sees the last four readings but does worse than plain PPO above 5 mm. Given fixed data, a network learning its own filter from four noisy samples loses to an explicit Bayesian filter that also estimates the noise level.
- **The curriculum is not free.** Oracle PPO fell from 0.83 (base) to 0.73, and its training success dipped when the assist reached zero. Even so, every partially observed method improved substantially:
  - belief 0.18 → 0.42;
  - plain 0.28 → 0.35;
  - frame-stack 0.08 → 0.33.

  All methods are compared under the same curriculum.

### 4.3 Planning: budget vs success vs CPU time

*Pending:* the queue in `configs/queue_pomcp.json` produces it, followed by `figures.success_vs_budget`. The belief-PPO reference line uses the same 50 episodes at 10 mm.

### 4.4 World-model accuracy

*Pending:* from `runs/world_model/report.json`, reporting held-out-episode R² per dimension and grasp AUC.

## 5. Limitations

- **Sim only:** no hardware validation, and the camera is an analytic noise model, not rendered images.
- **Scripted motion:** the grasp and place are scripted primitives. The policy learns *where and when* to grasp, not finger control.
- **Privileged planner simulator:** POMCP's simulator knows the distractor poses and the noise model.
- **WM-POMCP approximations:**
  - a commit is terminal and pays 15·P(grasp) − (1 − P(grasp)), assuming the place always succeeds;
  - the grasp's time cost is ignored;
  - after a miss, the remaining attempts are ignored.
- **Small training budget:** 1M steps per run, 3 seeds, chosen to fit a laptop. Training success at 1M steps (`docs/figs/v21/learning_curves.png`) had flattened after the assist reached zero, but longer training could change the margins.
- **The curriculum uses privileged information during training:** the assist reads the true brick pose. It is gone by mid-training, and evaluation never uses it. It changes what each method learns, though, and it cost the oracle about 0.1 success (§4.2).
- **Recurrent PPO was not retrained under v2.1.** Its base-protocol result (collapse on 3 of 3 seeds) says nothing about LSTM memory with a fixed training process.

## Appendix

### A.1 Grasp tolerance (4 ms physics, 3 poses per cell)

| offset (mm) | -15 | -12 | -9 | -6 | -3 | 0 | +3 | +6 | +9 | +12 | +15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| across jaw axis | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 0/3 | 0/3 |
| along brick axis | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

Yaw error up to ±30° always succeeds.

### A.2 Training success (v2.1, last 1000 training episodes; seeds 0 / 1 / 2)

| method | seed 0 | seed 1 | seed 2 | wall time per run |
|---|---|---|---|---|
| oracle | 64% | 68% | 67% | ~4 h |
| belief | 47% | 52% | 53% | ~2.8 h |
| plain | 45% | 38% | 43% | ~2.9 h |
| frame-stack | 41% | 37% | 39% | ~3.4 h |

Base-protocol seed-0 training was oracle 87.5%, belief 51.8% and plain 13.2%. The per-seed log is in `training-log.md`.

### A.3 Engineering record

The full chronological log of every change, bug and decision is in [`training-log.md`](training-log.md).
