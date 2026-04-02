# POMCP Direct Simulator — Design Spec

**Date:** 2026-04-02
**Status:** Approved
**Approach:** Direct Simulator POMCP (Approach B) — uses real MuJoCo env for rollouts instead of a learned world model.

---

## 1. Overview

Implement POMCP (Partially Observable Monte Carlo Planning) for the SO-ARM101 pick-and-place task using the real MuJoCo environment for all rollouts. This is the third method in the ASEN 5264 project, alongside Plain PPO and Belief-Augmented PPO.

**Key characteristics:**
- Online planning (no offline training required)
- Full episode rollouts (depth 200)
- Heuristic rollout policy (not random)
- Parallelized across CPU cores via worker pool
- 8 discrete actions at 15mm steps
- Direct comparison with PPO methods using same env + metrics

---

## 2. Architecture

### 2.1 New Files

| File | Purpose |
|---|---|
| `so_arm101_control/pomcp_env_bridge.py` | State serialization/deserialization + worker process loop |
| `so_arm101_control/pomcp_heuristic.py` | Greedy phase-based heuristic rollout policy |

### 2.2 Modified Files

| File | Change |
|---|---|
| `scripts/train_pomcp.py` | Add `DirectPOMCPPlanner` class, `evaluate_pomcp_direct()`, `--direct` CLI flag |

### 2.3 Untouched Files

- `lego_pick_env.py` — env used as-is
- `particle_filter.py` — PF runs normally inside env
- `world_model.py` — preserved for Approach A later
- Existing `POMCPNode`, `POMCPPlanner`, `collect_transitions` — all preserved

### 2.4 Data Flow (per real step)

```
Real env at step N
    │
    ▼
serialize_state(env) → snapshot dict
    │
    ├── Worker 0 ──┐
    ├── Worker 1 ──┤  Each: restore_state(env, snapshot)
    ├── Worker 2 ──┤         step with assigned action_idx
    ├── ...        ┤         then heuristic rollout to completion
    └── Worker N ──┘         return mean discounted return
    │
    ▼
Aggregate Q[action_idx] = mean return across all workers
    │
    ▼
Execute argmax action in real env
```

---

## 3. Discrete Action Space

Reuses the existing mapping from `train_pomcp.py`:

| Index | Action | Delta |
|---|---|---|
| 0 | +X | `[+0.015, 0, 0, -1]` |
| 1 | -X | `[-0.015, 0, 0, -1]` |
| 2 | +Y | `[0, +0.015, 0, -1]` |
| 3 | -Y | `[0, -0.015, 0, -1]` |
| 4 | LOWER | `[0, 0, -0.015, -1]` |
| 5 | RAISE | `[0, 0, +0.015, -1]` |
| 6 | CLOSE | `[0, 0, 0, +1]` |
| 7 | OPEN | `[0, 0, 0, -1]` |

15mm steps match the PPO training scale. Gripper dimension is ±1.0 (matches continuous PPO's gripper command range).

---

## 4. State Serialization

`serialize_state(env) → dict` captures everything needed to reproduce the env mid-episode:

```python
snapshot = {
    # MuJoCo physics
    "qpos":               data.qpos.copy(),
    "qvel":               data.qvel.copy(),

    # Env-level state
    "block_true_poses":   env._block_true_poses.copy(),
    "ee_pos":             env._ee_pos.copy(),
    "gripper_closed":     env._gripper_closed,
    "holding":            env._holding,
    "grasp_offset":       env._grasp_constraint_offset,  # (3,) or None
    "sigma_ep":           env._sigma_ep,
    "goal_xy":            env._goal_xy.copy(),
    "step_count":         env._step_count,

    # Milestone flags (prevent re-firing in rollouts)
    "milestones":         env._milestones_fired.copy(),

    # Particle filter
    "pf_particles":       env.pf.particles.copy(),
    "pf_weights":         env.pf.weights.copy(),
}
```

`restore_state(env, snapshot)` writes all fields back, including `mujoco.mj_forward(model, data)` after setting qpos/qvel to sync derived quantities.

---

## 5. Worker Pool

### 5.1 Lifecycle

- Workers spawned **once** at `DirectPOMCPPlanner.__init__()` via `multiprocessing.Process`
- Each worker owns one `LegoPickEnv(belief_mode=True)` — no sharing, no locks
- Workers persist across all real steps and evaluation episodes
- Shutdown on `planner.close()`

### 5.2 Communication

- Main → Worker: `(snapshot, action_idx, n_rollouts)` via `multiprocessing.Queue`
- Worker → Main: `(action_idx, mean_return)` via result `Queue`
- One broadcast per action × n_workers: 8 actions evaluated in parallel
- `n_rollouts` is **per action** — 100 rollouts means 100 rollouts for each of the 8 actions, split across workers (13 per worker with 8 workers)

### 5.3 Rollout Per Worker

```
for i in range(n_rollouts):
    restore_state(env, snapshot)
    obs, reward, done = env.step(DISCRETE_ACTIONS[action_idx])
    total_return = reward
    discount = gamma

    while not done:
        heuristic_idx = heuristic_action(env state...)
        obs, reward, terminated, truncated, info = env.step(DISCRETE_ACTIONS[heuristic_idx])
        total_return += discount * reward
        discount *= gamma
        done = terminated or truncated

    returns.append(total_return)

return (action_idx, mean(returns))
```

### 5.4 Worker Count

Default `n_workers = 8` (matches number of discrete actions). This allows all 8 action evaluations to run simultaneously — wall time is 1 batch, not 8.

---

## 6. Heuristic Rollout Policy

Greedy phase-based policy mirroring the learned PPO behavior:

```python
def heuristic_action(ee_pos, block_mu, goal_xy, holding, gripper_closed):
    STEP = 0.015

    if not holding:
        dx = block_mu[0] - ee_pos[0]
        dy = block_mu[1] - ee_pos[1]
        dz = ee_pos[2] - TABLE_Z

        xy_dist = sqrt(dx**2 + dy**2)

        if xy_dist > 0.025:              # Phase 1: approach XY
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 1  # +X or -X
            else:
                return 2 if dy > 0 else 3  # +Y or -Y
        elif dz > 0.020:                 # Phase 2: lower Z
            return 4                      # LOWER
        else:                            # Phase 3: grasp
            return 6                      # CLOSE

    else:  # holding
        dx = goal_xy[0] - ee_pos[0]
        dy = goal_xy[1] - ee_pos[1]
        xy_dist = sqrt(dx**2 + dy**2)

        if xy_dist > 0.015:              # Phase 4: carry to goal
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 1
            else:
                return 2 if dy > 0 else 3
        else:                            # Phase 5: place
            return 7                      # OPEN
```

Estimated success rate: ~60-70% in rollouts (vs ~0% for random). This gives POMCP meaningful Q-value signal.

---

## 7. POMCP Reward (Within Rollouts)

Rollouts use the **real env reward** — no separate reward function. Whatever `env.step()` returns as reward is what the rollout accumulates. This includes:

- Step cost (-1.0)
- XY approach shaping
- Z descent shaping
- One-time milestones (milestone flags restored from snapshot)
- Grasp success/fail rewards
- Carry shaping
- Tiered placement rewards (+50/+30/+10)

This is one of the key advantages of Direct Simulator POMCP: the reward function is identical to PPO training, enabling direct comparison.

---

## 8. Evaluation Harness

### 8.1 CLI

```bash
python3 train_pomcp.py --direct \
    --n-episodes 100 \
    --n-rollouts 100 \
    --n-workers 8 \
    --gamma 0.99 \
    --seed 0
```

### 8.2 Metrics

Same format as PPO evaluation for apples-to-apples comparison:

| Metric | Description |
|---|---|
| Success rate | % episodes with successful placement |
| Perfect placement (<10mm) | % |
| Precise placement (<20mm) | % |
| Close placement (<40mm) | % |
| Mean episode length | real steps to completion |
| Mean discounted return | comparable to PPO mean reward |
| Planning time/step | seconds (compute cost) |

### 8.3 Output

Results printed to stdout + saved to `logs/pomcp_direct/eval_results.json`.

### 8.4 Compute Estimate

- 100 rollouts × 200 depth × 8 actions / 8 workers = 100 × 200 = 20,000 MuJoCo steps per real step per worker
- At ~5ms/step: ~100 seconds per real action
- ~30 real steps per episode: ~50 min per episode
- 100 episodes: ~83 hours
- **With heuristic rollouts averaging ~25 steps (early termination on success): ~10 hours**

---

## 9. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Direct sim vs learned model | Direct sim | No dependency on PPO training, exact dynamics, higher accuracy |
| Action discretization | 8 actions × 15mm | Standard for POMCP; matches PPO's EE delta scale |
| Rollout depth | 200 (full episode) | User chosen; heuristic policy makes early termination likely |
| Rollout policy | Heuristic (greedy phase-based) | Random rollouts never reach terminal reward in sparse manipulation tasks |
| Parallelization | Worker pool (multiprocessing) | 8x speedup; one worker per action for maximum parallelism |
| PF in rollouts | Yes, runs normally | Maintains belief consistency; rollout obs go through PF as in real execution |
| UCB1 | Reused from existing POMCPNode | Already implemented and correct |
| Learned model path | Preserved, untouched | Available for Approach A extension later |
