# Refined Project Proposal
# Robot Pick-and-Place Under Uncertainty (SO-ARM101 + MuJoCo)
# ASEN 5264 — Decision Making under Uncertainty

---

## Summary

Compare three methods for robot pick-and-place under **observation uncertainty** (no camera hardware model by default; uncertainties are injected for the POMDP):

1. **Plain PPO** — learns to tolerate noise blindly (raw noisy obs)
2. **Belief-Augmented PPO** — particle filter outputs (μ, σ); agent learns when to "look" (gather info) vs "grasp" (exploit) — exploration–exploitation / bandit
3. **POMCP with Learned World Model** (stretch) — tree search over belief; same uncertainties

**Uncertainty sources (three; no separate camera noise by default):**
- **Episodic σ** — observation = true pose + N(0, σ_ep²), σ_ep drawn once per episode (easy vs hard episodes)
- **Multi-block occlusion** — one block can occlude the other; occluded block gets no pose (or stale)
- **Cost of looking** — step penalty (and/or explicit look cost) so gathering information trades off with acting (bandit)

**Toggle:** Optional distance-dependent "camera noise" can be enabled for ablations (closer = clearer); default is perfect camera + injected observation uncertainty only.

---

## Existing Infrastructure (reuse these)

| Component | File | What it provides |
|-----------|------|-----------------|
| MuJoCo sim | `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/mujoco_sim.py` | Physics + camera rendering bridge |
| Geometric IK | inside `control_gui.py` (IKSolver class) | EE → joint angles (reuse directly) |
| Lego randomizer | `so_arm101_control/randomize_legos.py` | Random block pose per episode |
| MuJoCo scene | `so_arm101_description/mujoco/so_arm101_scene.xml` | 3 Lego mocap bodies (red 2×4, green 2×3, blue 2×2) |

---

## POMDP Formulation

### State (hidden)
```
s = (ee_pos ∈ R³, block_pose = (x, y, θ), gripper_state ∈ {0,1})
```

### Observation (what agent sees)
```
o = [q1..q5, gripper_joint, x̂, ŷ, θ̂]   # 9D (Plain PPO)
# Belief PPO: same + (μ_x, μ_y, μ_θ, σ_x, σ_y, σ_θ) → 12D
```
Block pose (x̂, ŷ, θ̂) is the only external input about the world (from a "perfect" camera when visible); we do not model camera hardware. Observation uncertainty is injected as below.

### Observation Uncertainty (three sources; camera noise optional)

We **do not** model camera/sensor physics by default. Observation uncertainty comes from:

1. **Episodic σ** — At episode start, draw σ_ep ~ Uniform(σ_low, σ_high). Observation of block pose:
   ```
   (x̂, ŷ, θ̂) = true_pose + N(0, σ_ep²)   # when block is visible
   ```
   So the agent never sees true pose; belief/particle filter and "explore vs exploit" matter. Easy vs hard episodes.

2. **Multi-block occlusion** — Two blocks; one can occlude the other from the camera. When occluded: no pose observation (or stale). When visible: pose + episodic noise (above). Belief tracks both block poses.

3. **Cost of looking (bandit)** — Every step costs −1 (or explicit cost for a LOOK action). So gathering information (move to un-occlude, or get more observations to tighten belief) trades off with acting. Exploration–exploitation.

**Optional: distance-dependent camera noise (toggleable)**  
For ablations, can enable a classic "camera noise" model so that closer = clearer:
```
# Only when use_camera_noise=True (env flag)
σ_xy(t) = 8mm  × (‖ee_pos − block_pos‖ / 0.15m)
σ_θ(t)  = 12°  × (‖ee_pos − block_pos‖ / 0.15m)
```
Then effective obs noise = episodic σ × distance factor. Default: `use_camera_noise=False`.

### Grasp Success Probability
```
p_grasp = sigmoid(−150 × ‖ee_xy − block_xy‖)
# 0mm→95%, 5mm→78%, 10mm→53%, 15mm→30%
```

### Action Space (4D continuous)
```
a = [Δx, Δy, Δz ∈ [−2cm, +2cm],  gripper ∈ {open, close}]
```
Converted to joint angles via existing geometric IK.

### Reward
```
+10.0   successful placement (‖ee_place − goal‖ < 15mm)
 −1.0   per timestep (cost of looking: encourages efficient use of information)
 −5.0   failed grasp attempt
 +0.5   per step of approaching block (shaping, optional)
Max 200 steps per episode.
```

---

## Method 1 — Plain PPO (Baseline)

- **Obs:** 9D raw noisy reading `[q1..q5, gripper, x̂, ŷ, θ̂]`
- **Network:** MLP 2×256, ReLU
- **Library:** Stable-Baselines3 PPO
- **Training:** 2M steps
- **Goal:** >70% success rate. Ceiling of "ignoring uncertainty."

---

## Method 2 — Belief-Augmented PPO (Main contribution)

- **Particle Filter** (N=300) runs alongside sim:
  1. Propagate particles with process noise
  2. Weight by `N(obs; particle_pose, σ_obs²)` (use σ_ep when visible; no update when occluded)
  3. Resample (systematic)
  4. Output: μ = weighted mean, σ = weighted std

- **Obs:** 12D `[q1..q5, gripper, μ_x, μ_y, μ_θ, σ_x, σ_y, σ_θ]`
- **Same PPO + network architecture** as Method 1 (fair comparison)
- **Agent learns:** exploration–exploitation (bandit): when to "look" (move to un-occlude or get more obs to tighten σ) vs. grasp now. Cost of looking makes this trade-off explicit.
- **Key metric:** σ at grasp time — should be lower than Method 1's σ at grasp; fewer wasted steps when using optional camera-noise ablation

---

## Method 3 — POMCP with Learned World Model (Stretch)

### Step 1: Collect data
- Pull 50k transitions from Method 2 training replay buffer

### Step 2: Train world model MLP
```
Input:  (ee_pos[3], block_μ[3], block_σ[3], gripper[1], action[4])  → 14D
Output: (Δee_pos[3], Δblock_μ[3], Δblock_σ[3], grasp_logit[1])
Loss:   MSE + BCE   |   Network: 3×128 ReLU   |   Adam lr=1e-3
```

### Step 3: POMCP
- **Library:** `pomdp_py` (pip install)
- **Actions:** 8 discrete {MOVE±X, MOVE±Y, LOWER, RAISE, CLOSE, OPEN}
- **Search:** UCB1, N=500 rollouts/step, depth=20
- **Belief:** particle set, updated after each real observation

---

## Gym Environment to Build

**New file:** `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/lego_pick_env.py`

```python
class LegoPickEnv(gym.Env):
    def __init__(self, belief_mode=False, use_camera_noise=False, ...):
        # belief_mode=False → 9D obs (Method 1)
        # belief_mode=True  → 12D obs (Method 2)
        # use_camera_noise=False → observation noise = episodic σ + occlusion only (default)
        # use_camera_noise=True  → add distance-dependent σ on top (ablation)
        ...

    def reset():
        # draw σ_ep ~ Uniform(σ_low, σ_high) for this episode
        # randomize block poses (multi-block; use existing randomize_legos.py logic)
        # sample random goal on table
        # reset arm to home
        # reset particle filter (if belief_mode)

    def step(action):
        # clamp action, run geometric IK
        # step MuJoCo physics
        # occlusion: determine which block(s) visible (multi-block occlusion)
        # observation: if visible, pose + N(0, σ_ep²); optionally × distance factor if use_camera_noise
        # update particle filter (if belief_mode)
        # stochastic grasp success
        # reward: +10 success, −1 per step (cost of looking), −5 failed grasp
```

---

## Comparison Metrics

| Metric | What it measures |
|--------|-----------------|
| Success rate (%) | Primary |
| Mean steps to success | Efficiency (cost of looking) |
| Steps before first grasp | Exploration vs exploitation (bandit) |
| Grasp attempts before success | Uncertainty handling |
| σ at grasp time | Did agent wait to be confident? |
| Success vs σ_ep (episodic noise sweep) | Robustness |

---

## Implementation Phases

| Phase | Weeks | Task |
|-------|-------|------|
| 1 | 1-2 | `LegoPickEnv` gym wrapper: episodic σ, multi-block occlusion, cost of looking; particle filter; `use_camera_noise` toggle (default False) |
| 2 | 3-4 | Train Method 1 (plain PPO), then with full uncertainty |
| 3 | 5-6 | Train Method 2 (Belief-Augmented PPO), σ_ep and occlusion sweep |
| 4 | 7-8 | World model training + POMCP integration (stretch) |
| 5 | 9   | Evaluation plots, video demos, optional camera-noise ablation, final report |

---

## Dependencies to Add

```
pip install stable-baselines3 gymnasium torch pomdp-py
```

---

## Deliverables

1. `LegoPickEnv` gym environment (reusable)
2. Trained PPO policy files (`.zip`)
3. Trained Belief-PPO policy files (`.zip`)
4. Learned world model weights (`.pt`) + POMCP solver (stretch)
5. Comparison plots: learning curves, success vs σ_ep sweep, σ at grasp time, steps before first grasp; optional use_camera_noise ablation
6. Video demos: each policy running live in MuJoCo
7. Final report with 3-way comparison narrative

---

## Verification Checklist

- [ ] Geometric IK brings EE to ground-truth block pose, grasp succeeds
- [ ] Episodic σ: easy vs hard episodes (σ_ep sweep); observation = pose + N(0, σ_ep²)
- [ ] Multi-block occlusion: occluded block gets no pose; belief tracks both blocks
- [ ] Cost of looking: step penalty (−1) in reward; metrics: steps to success, σ at grasp time
- [ ] Particle filter: belief converges to truth given repeated noisy obs (when visible)
- [ ] PPO reward curve climbs from ~−150 (random) to positive values
- [ ] Belief PPO: σ_at_grasp_time < Plain PPO σ_at_grasp_time; fewer steps when σ matters
- [ ] Optional `use_camera_noise=True`: distance-dependent σ; plot σ vs distance for ablation
