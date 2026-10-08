# Picking a brick you can't quite see: memory, belief and planning for a physical pick-and-place POMDP

*SO-ARM101 in MuJoCo · v2 technical report (draft; results sections are filled in from `runs/eval/*.csv`)*

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
| training seeds | 3 per RL method (0, 1, 2) |
| evaluation | 200 episodes × 6 σ levels per run; episode i uses reset seed 10 000 + i for every method |
| per-cell CI | Wilson 95% |
| aggregate | IQM across seeds × σ levels with stratified bootstrap 95% CIs, and P(belief > X) (rliable, Agarwal et al., 2021) |
| POMCP | σ = 10 mm, 50 episodes (seeds 10 000–10 049) per budget ∈ {32, 128, 512} simulations; CPU time per decision |
| compute | one 4-core laptop CPU, no GPU; every job under a thermal/memory/battery watchdog (`tools/watchdog.py`) |

## 4. Results

### 4.1 Memory gap (seed 0, 100 episodes per σ)

| σ_ep (mm) | 2 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| oracle | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 |
| plain | 0.17 | 0.14 | 0.15 | 0.08 | 0.08 | 0.02 |
| plain commit error (mm) | 10.5 | 13.1 | 15.6 | 19.0 | 23.3 | 28.8 |

At seed 0, a memoryless policy commits roughly 10 mm off even at 2 mm noise. Trained across noise levels, it cannot tell how far to trust one reading. **Seed 0 overstates the gap:** plain PPO's seed 1 learned to approach and commit early and reached 42% training success, against 13% for seed 0. The multi-seed, per-σ comparison in §4.2 is the result to cite.

### 4.2 Main comparison

*Pending:* phase 2 is running. To produce it:

```
./vpy -m pomdp_pick.eval.stats runs/eval/*_n200.csv
./vpy -m pomdp_pick.eval.figures docs/figs/v2/success_vs_sigma.png runs/eval/*_n200.csv
```

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
- **Small training budget:** 1M steps per run, chosen to fit a laptop. The seed-0 learning curves (`docs/figs/v2/learning_curves.png`) show the oracle and plain PPO had plateaued. Belief PPO was still climbing (42% at 450k steps, 52% at 1M), so its numbers are a lower bound.

## Appendix

### A.1 Grasp tolerance (4 ms physics, 3 poses per cell)

| offset (mm) | -15 | -12 | -9 | -6 | -3 | 0 | +3 | +6 | +9 | +12 | +15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| across jaw axis | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 0/3 | 0/3 |
| along brick axis | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

Yaw error up to ±30° always succeeds.

### A.2 Training (seed 0, last 1000 training episodes)

| method | success | attempts | decisions per episode | wall time |
|---|---|---|---|---|
| oracle | 87.5% | 1.03 | 12.5 | 2.7 h |
| belief | 51.8% | 1.52 | 23.5 | 2.0 h |
| plain | 13.2% | 1.56 | 95.5 | 1.35 h |

### A.3 Engineering record

The full chronological log of every change, bug and decision is in [`training-log.md`](training-log.md).
