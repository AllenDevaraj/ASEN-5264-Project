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

## Run 5: Plain PPO — Gripper Exploration + Higher Entropy (PENDING)

**Date:** 2026-03-16
**Config:** `--timesteps 2000000 --n-envs 3`, Plain PPO, 18D obs, ent_coef=0.05
**Status:** Awaiting training...

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

## Dependencies Installed

- `tensorboard` — required by SB3 for logging
- `tqdm`, `rich` — required by SB3 for progress bar
