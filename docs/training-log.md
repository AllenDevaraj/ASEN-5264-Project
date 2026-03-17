# Training Log — SO-ARM101 Pick-and-Place RL

Tracks every training run, reward change, and fix for the final report.

---

## Run 1: Plain PPO — Sparse Reward (FAILED)

**Date:** 2026-03-15
**Config:** `--timesteps 2000000 --n-envs 3`, Plain PPO (belief_mode=False)
**Reward design:**
- -1.0 per step
- +0.5 approach shaping (distance improvement to block)
- -5.0 failed grasp
- +10.0 successful placement (within 15mm of goal)

**Results:**
| Metric | Value |
|---|---|
| Mean reward | -199 |
| Success rate | 0% |
| Episode length | 200 (always timeout) |
| Grasp attempts | 0 |
| log_std | +0.56 (std=1.76, basically random) |
| explained_variance | 0.374 |
| Training time | ~45 min |

**Diagnosis:** Reward too sparse. The agent needed to accidentally chain approach→lower→grasp→carry→place→release perfectly. Probability of this with random actions ≈ 0. The approach shaping (+0.5) was too weak relative to the -1.0 step cost. `log_std` went positive (more random over time), meaning the policy gave up.

**Lesson:** Multi-step manipulation tasks require dense phase-based reward shaping, not sparse terminal rewards.

---

## Run 2: Plain PPO — Dense Per-Step Proximity Bonus (FAILED — Reward Exploit)

**Date:** 2026-03-16
**Config:** `--timesteps 2000000 --n-envs 3`, Plain PPO
**Reward design (changes from Run 1):**
- -0.5 per step (reduced)
- +2.0 distance improvement shaping (4x stronger)
- +1.0/step when within 20mm of block
- +2.0/step when within 10mm of block
- +1.0/step height shaping near block
- +10.0 successful grasp
- -3.0 failed grasp
- +20.0 precise placement, +5.0 close placement

**Results:**
| Metric | Value |
|---|---|
| Mean reward | +307.5 +/- 47.9 |
| Success rate | 0% |
| Episode length | 200 (always timeout) |
| Grasp attempts | 0 |
| log_std | -0.59 (std=0.55, learned something) |
| reward mean (VecNormalize) | 70.5 |

**Diagnosis:** Agent found a reward exploit — hovering near the block collected +2.0 to +3.0 per step for 200 steps = +400 to +600 reward. This was far more profitable than risking a grasp attempt (-3.0 penalty on failure, uncertain +10 on success). The agent learned to park near the block and wait. Zero grasp attempts in 100 eval episodes.

**Lesson:** Per-step proximity bonuses create local optima. Rewards for "being near something" must be one-time milestones, not continuous payments. This is a textbook case of reward hacking (Amodei et al., 2016).

---

## Reward Fix: One-Time Milestones + Potential-Based Shaping

**Date:** 2026-03-16
**Changes applied (commit `f4dfbcb`):**

| Component | Old | New | Rationale |
|---|---|---|---|
| Step cost | -0.5 | -1.0 | Must be meaningful to discourage dawdling |
| Approach shaping | +2.0 improvement | +3.0 improvement | Potential-based only (bounded), no per-step proximity |
| Near-block bonus | +1.0/step + +2.0/step | +5.0 one-time milestone (<15mm) | Fires once, no exploit possible |
| Height shaping | +1.0/step near block | Removed | Was exploitable |
| Grasp success | +10.0 | +15.0 | Must outweigh risk of -2.0 failure penalty |
| Grasp failure | -3.0 | -2.0 | Lower penalty encourages exploration |
| Near-goal bonus | +1.0/step | +5.0 one-time milestone (<30mm) | Fires once |
| Precise placement | +20.0 | +25.0 | Terminal reward must dominate |
| Close placement | +5.0 | +10.0 | Partial success still valuable |

---

## Run 3: Plain PPO — One-Time Milestones (FAILED — Missing Goal in Obs)

**Date:** 2026-03-16
**Config:** `--timesteps 2000000 --n-envs 3`, Plain PPO, 12D obs
**Reward design:** One-time milestones, +3.0 approach shaping, +15 grasp, +25 placement

**Results:**
| Metric | Value |
|---|---|
| Mean reward | -185.5 +/- 6.0 |
| Success rate | 0% |
| Episode length | 200 (always timeout) |
| Grasp attempts | 0 |
| Max reward | -164.1 |

**Diagnosis:** Agent learned to approach block (reward improved from -199 to -185.5) but never attempted a grasp. Root cause: **the goal position was missing from the observation**. The agent had no idea where to place the block, making grasping pointless from the agent's perspective — even if it grasped, it couldn't learn to carry to an invisible goal. Additionally, the EE position was not in the observation, forcing the network to learn forward kinematics from joint angles (a hard nonlinear function).

**Lesson:** Always verify the observation contains all information needed to solve the task. If the agent can't observe the goal, it can't learn the task — no amount of reward shaping fixes a missing observation.

---

## Observation Fix: Added Goal, EE Position, Holding Flag

**Date:** 2026-03-16 (commit `79ec028`)
**Obs changed from 12D to 18D for both modes:**

```
[0:6]   joint angles + gripper
[6:9]   block obs (wrist noisy / PF mu)
[9:12]  block obs (overhead noisy / PF sigma)
[12:15] end-effector position (x, y, z)    ← NEW
[15:17] goal position (x, y)               ← NEW (critical!)
[17]    holding flag (0 or 1)               ← NEW
```

---

## Run 4: Plain PPO — 18D Obs with Goal + EE (FAILED — No Gripper Exploration)

**Date:** 2026-03-16
**Config:** `--timesteps 2000000 --n-envs 3`, Plain PPO, 18D obs

**Results:**
| Metric | Value |
|---|---|
| Mean reward | -183.3 +/- 8.9 |
| Success rate | 0% |
| Episode length | 200 (always timeout) |
| Grasp attempts | 0 |
| Max reward | -144.0 |

**Diagnosis:** Approach improved (max reward -144 vs -164 before — EE position in obs helped), but agent still never closes gripper. The policy converged on "always keep gripper open" because there was no gradient connecting gripper-close to any reward. Even though grasping has positive expected value near the block, the agent never explores it because entropy collapsed early in training.

**Lesson:** In continuous action spaces with discrete-like effects (gripper open/close), the policy can converge away from a critical action dimension before discovering its reward. Need explicit exploration encouragement.

---

## Exploration Fix: Gripper Reward + Higher Entropy

**Date:** 2026-03-16 (commit `9e71be2`)

| Change | Old | New | Rationale |
|---|---|---|---|
| Gripper close near block | No reward | +2.0 within 25mm | Direct gradient for gripper-close |
| Grasp fail penalty | -2.0 | -1.0 | Lower penalty = more willing to try |
| Grasp success reward | +15.0 | +20.0 | Bigger carrot |
| ent_coef (PPO hyperparams) | 0.01 | 0.05 | 5x more exploration pressure |

---

## Run 5: Plain PPO — Gripper Exploration + Higher Entropy (FAILED — Gripper Close Exploit)

**Date:** 2026-03-16
**Config:** `--timesteps 2000000 --n-envs 3`, Plain PPO, 18D obs, ent_coef=0.05

**Results:**
| Metric | Value |
|---|---|
| Mean reward | +167.0 |
| Success rate | 0% |
| Episode length | 200 (always timeout) |
| Grasp successes | 0 |
| Grasp attempts | ~1.0/episode |
| Gripper closed ratio | ~100% near block |

**Diagnosis:** Agent found another exploit — it learned to keep the gripper permanently closed while hovering near the block, collecting the +2.0/step gripper-close-near-block reward continuously. This is the same class of bug as Run 2 (per-step proximity exploit), just applied to the gripper dimension. The agent never actually attempted a real grasp because keeping the gripper closed earned more than risking the grasp outcome.

**Lesson:** Any per-step reward for a binary action (gripper open/close) will be exploited. Gripper rewards must be transition-based (fire only on the open→close transition), not state-based (fire every step gripper is closed).

---

## Reward Fix: Transition-Based Gripper Reward

**Date:** 2026-03-16
**Changes applied:**

| Component | Old | New | Rationale |
|---|---|---|---|
| Gripper close near block | +2.0/step when closed within 25mm | Removed (transition-only) | Per-step was exploitable |
| Grasp success | +20.0 | +20.0 (unchanged) | Fires on close transition only |
| Grasp fail (very close) | -1.0 | +2.0 if <15mm | Reward good attempts even if unlucky |
| Grasp fail (close) | -1.0 | -0.5 if <25mm | Mild penalty, still encourages trying |
| Grasp fail (far) | -1.0 | -2.0 if >25mm | Penalize wasteful attempts |

**Key design change:** All gripper rewards now fire only on the `want_close and not self._gripper_closed` transition. No reward accumulates while gripper stays closed.

---

## Run 6: Plain PPO — Transition-Based Gripper Reward (SUCCESS)

**Date:** 2026-03-17
**Config:** `--timesteps 2000000 --n-envs 2`, Plain PPO, 18D obs, ent_coef=0.05

**Results:**
| Metric | Value |
|---|---|
| Precise placement (<20mm) | **52%** |
| Close placement (<40mm) | **33%** |
| Total task completion | **85%** |
| Grasp success rate | 100% (1.0/ep) |
| Mean reward | 38.1 +/- 17.2 |
| Mean episode length | 21.4 steps |
| Reward range | [-26.4, 66.9] |

**Diagnosis:** First successful training! The transition-based gripper reward eliminated the exploit. The agent learned the full pick-and-place pipeline: approach → grasp → carry → place. 85% of episodes end with successful placement. Reward curve shows learning: -207 (10k) → -119 (400k) → +16 (600k) → +42 plateau (800k+).

**Note:** Initial eval reported 0% success due to missing `success` key in info dict — fixed by adding `placement_success` flag. The agent was succeeding all along from this run.

**Lesson:** Transition-based rewards for binary actions are the correct approach. The full reward structure (approach shaping + one-time milestones + transition-based grasp + carry shaping + placement terminal) works for multi-phase manipulation.

---

## Belief PPO Run 1: Particle Filter + PPO (SUCCESS)

**Date:** 2026-03-17
**Config:** `--timesteps 2000000 --n-envs 3`, Belief PPO, 18D obs (PF mu/sigma), ent_coef=0.05

**Results:**
| Metric | Value |
|---|---|
| Precise placement (<20mm) | **34%** |
| Close placement (<40mm) | **63%** |
| Total task completion | **97%** |
| Grasp success rate | 100% (1.0/ep) |
| Mean reward | 21.5 +/- 41.5 |
| Mean episode length | 37.5 steps |
| Grasp attempts/ep | 11.5 |
| Reward range | [-205.7, 65.6] |

**Learning curve:** -204 (10k) → -189 (400k) → -127 (800k, high variance) → +27 (1.4M) → +36 plateau (1.6M+). Slower to learn than Plain PPO (plateaued at 1.4M vs 600k) but converged to higher completion rate.

**Diagnosis:** Belief PPO achieves higher task completion (97% vs 85%) but lower precision (34% vs 52% within 20mm). The PF belief state (mu, sigma) provides more consistent observations, improving reliability. However, the added uncertainty representation (sigma dimensions) may make fine positioning harder — the policy is more cautious, taking more grasp attempts and longer episodes.

**Lesson:** Belief augmentation improves robustness under observation noise at the cost of speed and precision. This is the expected POMDP trade-off: better state estimation → more reliable completion, but the richer observation space requires more exploration to master.

---

## Head-to-Head: Plain PPO vs Belief PPO

| Metric | Plain PPO | Belief PPO | Winner |
|---|---|---|---|
| Total completion | 85% | **97%** | Belief |
| Precise placement | **52%** | 34% | Plain |
| Close placement | 33% | **63%** | Belief |
| Episode length | **21.4** | 37.5 | Plain |
| Grasp attempts/ep | **7.0** | 11.5 | Plain |
| Reward variance | **17.2** | 41.5 | Plain |

**Summary:** Plain PPO is faster and more precise when it succeeds. Belief PPO is more reliable — it almost always completes the task but takes longer and is less precise. This matches the theoretical prediction: belief-augmented policies are more robust to observation noise but the expanded state space is harder to optimize.

---

## Architecture Changes Log

| Date | Change | Commit | Impact |
|---|---|---|---|
| 2026-03-15 | Fixed EE_Z_MIN from -0.10 to 0.002 | — | Arm no longer goes below ground plane |
| 2026-03-15 | Fixed block fallback overlap | — | Blocks no longer spawn on top of each other |
| 2026-03-15 | Added `mujoco.viewer` import | — | Fixed viewer AttributeError |
| 2026-03-15 | Added overhead camera with top-down occlusion | `9866f52` | 12D obs for both modes, dual PF updates |
| 2026-03-16 | Dense reward v1 (per-step proximity) | `135cb31` | Reward exploit discovered |
| 2026-03-16 | Dense reward v2 (one-time milestones) | `f4dfbcb` | Fixes reward exploit |
| 2026-03-16 | Added goal_xy, ee_pos, holding to obs (12D→18D) | `79ec028` | Agent can now see the goal and its own EE position |
| 2026-03-16 | Gripper exploration reward + ent_coef 0.01→0.05 | `9e71be2` | Direct gradient for gripper close near block |
| 2026-03-16 | Transition-based gripper reward (fix exploit) | — | Gripper reward only on open→close transition, distance-dependent outcomes |

## Dependencies Installed

- `tensorboard` — required by SB3 for logging
- `tqdm`, `rich` — required by SB3 for progress bar
