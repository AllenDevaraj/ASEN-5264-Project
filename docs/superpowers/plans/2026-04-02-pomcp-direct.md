# POMCP Direct Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement POMCP online planning using the real MuJoCo environment for rollouts, with heuristic rollout policy and multiprocessing parallelization.

**Architecture:** A `DirectPOMCPPlanner` uses a pool of persistent worker processes, each owning a `LegoPickEnv`. At each real step, the current env state is serialized and broadcast to workers, which run parallel rollouts for each of 8 discrete actions using a heuristic policy. Q-values are aggregated and the best action is executed.

**Tech Stack:** Python multiprocessing, MuJoCo, Gymnasium, numpy. No new dependencies.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `so_arm101_control/pomcp_env_bridge.py` | Create | `serialize_state()`, `restore_state()` — snapshot and restore full env mid-episode |
| `so_arm101_control/pomcp_heuristic.py` | Create | `heuristic_action()` — greedy phase-based rollout policy |
| `scripts/train_pomcp.py` | Modify | Add `DirectPOMCPPlanner`, `evaluate_pomcp_direct()`, `--direct` CLI flag |
| `scripts/test_pomcp_direct.py` | Create | Verification tests for all POMCP components |

---

### Task 1: State Serialization (`pomcp_env_bridge.py`)

**Files:**
- Create: `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pomcp_env_bridge.py`
- Create: `vla_SO-ARM101/src/so_arm101_control/scripts/test_pomcp_direct.py`

- [ ] **Step 1: Write the test for serialize → restore roundtrip**

Create `scripts/test_pomcp_direct.py`:

```python
#!/usr/bin/env python3
"""Verification tests for POMCP Direct Simulator components."""

import math
import sys

import numpy as np

sys.path.insert(0, '/home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control')


def test_serialize_restore_roundtrip():
    """Test 1: serialize → step env → restore → state matches original."""
    from so_arm101_control.lego_pick_env import LegoPickEnv
    from so_arm101_control.pomcp_env_bridge import serialize_state, restore_state

    env = LegoPickEnv(belief_mode=True)
    obs_orig, info = env.reset(seed=42)

    # Take a few steps to get into a non-trivial state
    for _ in range(5):
        action = env.action_space.sample()
        obs_orig, _, _, _, _ = env.step(action)

    # Serialize
    snapshot = serialize_state(env)

    # Record state
    ee_before = env._ee_pos.copy()
    block_before = dict(env._block_true_poses)
    holding_before = env._holding_block
    qpos_before = env.data.qpos.copy()

    # Mutate env with more steps
    for _ in range(10):
        env.step(env.action_space.sample())

    # Confirm state has changed
    assert not np.allclose(env._ee_pos, ee_before, atol=1e-6), \
        "EE should have moved after extra steps"

    # Restore
    restore_state(env, snapshot)

    # Verify all fields match
    assert np.allclose(env._ee_pos, ee_before, atol=1e-8), \
        f"EE mismatch: {env._ee_pos} vs {ee_before}"
    assert np.allclose(env.data.qpos, qpos_before, atol=1e-8), \
        "qpos mismatch after restore"
    assert env._holding_block == holding_before, \
        "holding_block mismatch"
    for name in block_before:
        for i in range(3):
            assert abs(env._block_true_poses[name][i] - block_before[name][i]) < 1e-8, \
                f"Block pose mismatch for {name}"

    # Verify PF state restored
    assert np.allclose(env.pf.particles, snapshot["pf_particles"], atol=1e-8), \
        "PF particles not restored"
    assert np.allclose(env.pf.weights, snapshot["pf_weights"], atol=1e-8), \
        "PF weights not restored"

    env.close()
    print("  PASS: serialize → restore roundtrip")


def test_restore_produces_same_trajectory():
    """Test 2: Two rollouts from same snapshot produce same result with same seed."""
    from so_arm101_control.lego_pick_env import LegoPickEnv
    from so_arm101_control.pomcp_env_bridge import serialize_state, restore_state

    env = LegoPickEnv(belief_mode=True)
    env.reset(seed=42)
    for _ in range(3):
        env.step(env.action_space.sample())

    snapshot = serialize_state(env)

    # Fixed action sequence
    actions = [np.array([0.015, 0.0, 0.0, -1.0], dtype=np.float32)] * 5

    # Rollout 1
    restore_state(env, snapshot)
    rewards_1 = []
    for a in actions:
        _, r, _, _, _ = env.step(a)
        rewards_1.append(r)

    # Rollout 2
    restore_state(env, snapshot)
    rewards_2 = []
    for a in actions:
        _, r, _, _, _ = env.step(a)
        rewards_2.append(r)

    # Rewards should be close (not exact due to PF stochasticity in belief mode,
    # but the deterministic physics parts should match)
    for i, (r1, r2) in enumerate(zip(rewards_1, rewards_2)):
        assert abs(r1 - r2) < 1.0, \
            f"Step {i}: reward diverged too much: {r1} vs {r2}"

    env.close()
    print("  PASS: same snapshot → similar trajectories")


if __name__ == "__main__":
    tests = [
        ("serialize/restore roundtrip", test_serialize_restore_roundtrip),
        ("restore same trajectory", test_restore_produces_same_trajectory),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: `ModuleNotFoundError: No module named 'so_arm101_control.pomcp_env_bridge'`

- [ ] **Step 3: Write the implementation**

Create `so_arm101_control/pomcp_env_bridge.py`:

```python
#!/usr/bin/env python3
"""State serialization and restoration for POMCP Direct Simulator.

Captures the full LegoPickEnv state mid-episode into a picklable dict,
and restores it exactly — enabling MCTS rollouts from arbitrary states.

Usage:
    snapshot = serialize_state(env)
    # ... mutate env ...
    restore_state(env, snapshot)  # env is back to snapshot state
"""

import mujoco
import numpy as np


def serialize_state(env):
    """Capture full env state into a picklable dict.

    Args:
        env: LegoPickEnv instance (belief_mode=True or False).

    Returns:
        dict with all state needed to restore the env mid-episode.
    """
    snapshot = {
        # MuJoCo physics
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),

        # Env-level state
        "block_true_poses": dict(env._block_true_poses),
        "ee_pos": env._ee_pos.copy(),
        "gripper_closed": env._gripper_closed,
        "holding_block": env._holding_block,
        "grasp_offset": env._grasp_offset.copy(),
        "sigma_ep": env._sigma_ep,
        "goal_pos": env._goal_pos.copy(),
        "step_count": env._step_count,

        # Shaping state (for correct reward computation in rollouts)
        "prev_dist_to_block": env._prev_dist_to_block,
        "prev_dist_to_goal": env._prev_dist_to_goal,
        "prev_ee_z": env._prev_ee_z,

        # Milestone flags
        "reached_block": env._reached_block,
        "reached_goal": env._reached_goal,
        "lowered_near_block": env._lowered_near_block,
    }

    # Particle filter (only in belief mode)
    if env.belief_mode:
        snapshot["pf_particles"] = env.pf.particles.copy()
        snapshot["pf_weights"] = env.pf.weights.copy()
        if env.pf._last_obs is not None:
            snapshot["pf_last_obs"] = env.pf._last_obs.copy()
        else:
            snapshot["pf_last_obs"] = None

    return snapshot


def restore_state(env, snapshot):
    """Restore env to a previously serialized state.

    Args:
        env: LegoPickEnv instance (must be same belief_mode as when serialized).
        snapshot: dict from serialize_state().
    """
    # MuJoCo physics
    env.data.qpos[:] = snapshot["qpos"]
    env.data.qvel[:] = snapshot["qvel"]
    mujoco.mj_forward(env.model, env.data)

    # Env-level state
    env._block_true_poses = dict(snapshot["block_true_poses"])
    env._ee_pos = snapshot["ee_pos"].copy()
    env._gripper_closed = snapshot["gripper_closed"]
    env._holding_block = snapshot["holding_block"]
    env._grasp_offset = snapshot["grasp_offset"].copy()
    env._sigma_ep = snapshot["sigma_ep"]
    env._goal_pos = snapshot["goal_pos"].copy()
    env._step_count = snapshot["step_count"]

    # Shaping state
    env._prev_dist_to_block = snapshot["prev_dist_to_block"]
    env._prev_dist_to_goal = snapshot["prev_dist_to_goal"]
    env._prev_ee_z = snapshot["prev_ee_z"]

    # Milestone flags
    env._reached_block = snapshot["reached_block"]
    env._reached_goal = snapshot["reached_goal"]
    env._lowered_near_block = snapshot["lowered_near_block"]

    # Particle filter
    if env.belief_mode and "pf_particles" in snapshot:
        env.pf.particles = snapshot["pf_particles"].copy()
        env.pf.weights = snapshot["pf_weights"].copy()
        env.pf._last_obs = snapshot["pf_last_obs"].copy() if snapshot["pf_last_obs"] is not None else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: `2/2 tests passed`

- [ ] **Step 5: Commit**

```bash
cd /home/the2xman/ASEN-5264-Project
git add vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pomcp_env_bridge.py \
        vla_SO-ARM101/src/so_arm101_control/scripts/test_pomcp_direct.py
git commit -m "feat: add POMCP state serialization/restoration for direct sim rollouts"
```

---

### Task 2: Heuristic Rollout Policy (`pomcp_heuristic.py`)

**Files:**
- Create: `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pomcp_heuristic.py`
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/test_pomcp_direct.py`

- [ ] **Step 1: Write the test for heuristic action selection**

Append to `scripts/test_pomcp_direct.py`:

```python
def test_heuristic_phases():
    """Test 3: Heuristic selects correct action for each task phase."""
    from so_arm101_control.pomcp_heuristic import heuristic_action

    # Phase 1: block is far in +X → should move +X (action 0)
    action = heuristic_action(
        ee_pos=np.array([0.10, 0.0, 0.06]),
        block_mu=np.array([0.20, 0.0, 0.0]),
        goal_xy=np.array([0.15, 0.05]),
        holding=False,
        gripper_closed=False,
    )
    assert action == 0, f"Expected +X (0), got {action}"

    # Phase 1: block is far in -Y → should move -Y (action 3)
    action = heuristic_action(
        ee_pos=np.array([0.15, 0.05, 0.06]),
        block_mu=np.array([0.15, -0.05, 0.0]),
        goal_xy=np.array([0.18, 0.0]),
        holding=False,
        gripper_closed=False,
    )
    assert action == 3, f"Expected -Y (3), got {action}"

    # Phase 2: close in XY, high in Z → should LOWER (action 4)
    action = heuristic_action(
        ee_pos=np.array([0.15, 0.0, 0.05]),
        block_mu=np.array([0.15, 0.0, 0.0]),
        goal_xy=np.array([0.18, 0.0]),
        holding=False,
        gripper_closed=False,
    )
    assert action == 4, f"Expected LOWER (4), got {action}"

    # Phase 3: close in XY, low in Z → should CLOSE (action 6)
    action = heuristic_action(
        ee_pos=np.array([0.15, 0.0, 0.010]),
        block_mu=np.array([0.15, 0.0, 0.0]),
        goal_xy=np.array([0.18, 0.0]),
        holding=False,
        gripper_closed=False,
    )
    assert action == 6, f"Expected CLOSE (6), got {action}"

    # Phase 4: holding, goal in +X → should move +X (action 0)
    action = heuristic_action(
        ee_pos=np.array([0.10, 0.0, 0.02]),
        block_mu=np.array([0.10, 0.0, 0.0]),
        goal_xy=np.array([0.18, 0.0]),
        holding=True,
        gripper_closed=True,
    )
    assert action == 0, f"Expected +X (0), got {action}"

    # Phase 5: holding, at goal → should OPEN (action 7)
    action = heuristic_action(
        ee_pos=np.array([0.18, 0.0, 0.02]),
        block_mu=np.array([0.18, 0.0, 0.0]),
        goal_xy=np.array([0.18, 0.0]),
        holding=True,
        gripper_closed=True,
    )
    assert action == 7, f"Expected OPEN (7), got {action}"

    print("  PASS: heuristic phases")
```

Also add to the `tests` list in `__main__`:
```python
        ("heuristic phases", test_heuristic_phases),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: Test 3 fails with `ModuleNotFoundError: No module named 'so_arm101_control.pomcp_heuristic'`

- [ ] **Step 3: Write the implementation**

Create `so_arm101_control/pomcp_heuristic.py`:

```python
#!/usr/bin/env python3
"""Greedy heuristic rollout policy for POMCP.

Phase-based greedy policy that mimics the learned PPO behavior:
  Phase 1: Approach block in XY (move toward block_mu)
  Phase 2: Lower Z when close in XY
  Phase 3: Close gripper when in grasp zone
  Phase 4: Carry to goal (move toward goal_xy)
  Phase 5: Open gripper at goal

Used as the rollout policy in POMCP tree search to replace random rollouts.
Random rollouts virtually never complete the pick-and-place task, making
Q-value estimates meaningless.

Discrete action mapping:
  0: +X   1: -X   2: +Y   3: -Y   4: LOWER   5: RAISE   6: CLOSE   7: OPEN
"""

import math

TABLE_Z = 0.0055  # matches lego_pick_env.TABLE_Z


def heuristic_action(ee_pos, block_mu, goal_xy, holding, gripper_closed):
    """Select a discrete action based on current state.

    Args:
        ee_pos: (3,) end-effector position [x, y, z].
        block_mu: (3,) belief mean of target block [x, y, theta].
        goal_xy: (2,) goal position [x, y].
        holding: bool, whether block is currently held.
        gripper_closed: bool, whether gripper is closed.

    Returns:
        int: action index (0-7).
    """
    if not holding:
        dx = block_mu[0] - ee_pos[0]
        dy = block_mu[1] - ee_pos[1]
        dz = ee_pos[2] - TABLE_Z  # height above table

        xy_dist = math.sqrt(dx * dx + dy * dy)

        if xy_dist > 0.025:
            # Phase 1: approach block in XY
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 1  # +X or -X
            else:
                return 2 if dy > 0 else 3  # +Y or -Y
        elif dz > 0.020:
            # Phase 2: lower Z
            return 4  # LOWER
        else:
            # Phase 3: close gripper
            return 6  # CLOSE
    else:
        dx = goal_xy[0] - ee_pos[0]
        dy = goal_xy[1] - ee_pos[1]
        xy_dist = math.sqrt(dx * dx + dy * dy)

        if xy_dist > 0.015:
            # Phase 4: carry to goal
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 1
            else:
                return 2 if dy > 0 else 3
        else:
            # Phase 5: release
            return 7  # OPEN
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: `3/3 tests passed`

- [ ] **Step 5: Commit**

```bash
cd /home/the2xman/ASEN-5264-Project
git add vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pomcp_heuristic.py \
        vla_SO-ARM101/src/so_arm101_control/scripts/test_pomcp_direct.py
git commit -m "feat: add heuristic rollout policy for POMCP"
```

---

### Task 3: Direct POMCP Planner (worker pool)

**Files:**
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/train_pomcp.py`
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/test_pomcp_direct.py`

- [ ] **Step 1: Write the test for DirectPOMCPPlanner**

Append to `scripts/test_pomcp_direct.py`:

```python
def test_direct_planner_returns_valid_action():
    """Test 4: DirectPOMCPPlanner.plan() returns action 0-7 and runs without error."""
    from so_arm101_control.lego_pick_env import LegoPickEnv
    from so_arm101_control.pomcp_env_bridge import serialize_state

    # Import from train_pomcp (add scripts to path)
    sys.path.insert(0, '/home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control/scripts')
    from train_pomcp import DirectPOMCPPlanner

    env = LegoPickEnv(belief_mode=True)
    env.reset(seed=42)

    # Small rollout count for fast testing
    planner = DirectPOMCPPlanner(n_rollouts=5, n_workers=2, gamma=0.99)

    snapshot = serialize_state(env)
    action_idx = planner.plan(snapshot)

    assert isinstance(action_idx, int), f"Expected int, got {type(action_idx)}"
    assert 0 <= action_idx <= 7, f"Action {action_idx} out of range 0-7"

    planner.close()
    env.close()
    print("  PASS: DirectPOMCPPlanner returns valid action")


def test_direct_planner_one_episode():
    """Test 5: Run one full episode with DirectPOMCPPlanner, verify no crashes."""
    from so_arm101_control.lego_pick_env import LegoPickEnv
    from so_arm101_control.pomcp_env_bridge import serialize_state

    sys.path.insert(0, '/home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control/scripts')
    from train_pomcp import DirectPOMCPPlanner, DISCRETE_ACTIONS

    env = LegoPickEnv(belief_mode=True)
    env.reset(seed=42)

    planner = DirectPOMCPPlanner(n_rollouts=5, n_workers=2, gamma=0.99)
    total_reward = 0.0
    steps = 0

    done = False
    while not done and steps < 50:  # cap at 50 for test speed
        snapshot = serialize_state(env)
        action_idx = planner.plan(snapshot)
        action = DISCRETE_ACTIONS[action_idx]
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    planner.close()
    env.close()

    print(f"  PASS: 1 episode completed in {steps} steps, reward={total_reward:.1f}")
```

Also add to the `tests` list:
```python
        ("direct planner valid action", test_direct_planner_returns_valid_action),
        ("direct planner one episode", test_direct_planner_one_episode),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: Test 4 fails with `ImportError: cannot import name 'DirectPOMCPPlanner' from 'train_pomcp'`

- [ ] **Step 3: Write the implementation**

Add the following to `scripts/train_pomcp.py`, after the existing `POMCPPlanner` class (around line 244, before `evaluate_pomcp`):

```python
# --- Direct Simulator POMCP ---

import multiprocessing as mp


def _worker_loop(task_queue, result_queue, belief_mode):
    """Persistent worker process: owns one env, runs rollouts on demand.

    Receives: (snapshot, action_idx, n_rollouts, gamma)
    Sends:    (action_idx, mean_return)
    Sentinel: None on task_queue means shutdown.
    """
    from so_arm101_control.lego_pick_env import LegoPickEnv
    from so_arm101_control.pomcp_env_bridge import restore_state
    from so_arm101_control.pomcp_heuristic import heuristic_action

    env = LegoPickEnv(belief_mode=belief_mode)
    env.reset(seed=0)  # initial reset to build model

    while True:
        msg = task_queue.get()
        if msg is None:
            break

        snapshot, action_idx, n_rollouts, gamma = msg
        returns = []

        for _ in range(n_rollouts):
            restore_state(env, snapshot)

            # First step: execute the assigned action
            action = DISCRETE_ACTIONS[action_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            total_return = reward
            discount = gamma

            # Continue with heuristic policy
            while not (terminated or truncated):
                # Extract state for heuristic
                if env.belief_mode:
                    mu, _ = env.pf.get_belief()
                    block_mu = mu[0]
                else:
                    block_mu = np.array([
                        env._block_true_poses["red_lego_2x4"][0],
                        env._block_true_poses["red_lego_2x4"][1],
                        env._block_true_poses["red_lego_2x4"][2],
                    ])

                h_action_idx = heuristic_action(
                    ee_pos=env._ee_pos,
                    block_mu=block_mu,
                    goal_xy=env._goal_pos,
                    holding=env._holding_block,
                    gripper_closed=env._gripper_closed,
                )
                action = DISCRETE_ACTIONS[h_action_idx]
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += discount * reward
                discount *= gamma

            returns.append(total_return)

        result_queue.put((action_idx, float(np.mean(returns))))

    env.close()


class DirectPOMCPPlanner:
    """POMCP planner using real MuJoCo env for rollouts.

    Spawns persistent worker processes, each with their own LegoPickEnv.
    At each planning step, broadcasts a state snapshot to all workers,
    which run parallel rollouts and return Q-value estimates.
    """

    def __init__(self, n_rollouts=100, n_workers=8, gamma=0.99, belief_mode=True):
        """
        Args:
            n_rollouts: Rollouts per action (split across workers).
            n_workers: Number of parallel worker processes.
            gamma: Discount factor for rollout returns.
            belief_mode: Whether envs run in belief (particle filter) mode.
        """
        self.n_rollouts = n_rollouts
        self.n_workers = n_workers
        self.gamma = gamma
        self.n_actions = len(DISCRETE_ACTIONS)

        # Spawn workers
        self._task_queues = []
        self._result_queue = mp.Queue()
        self._workers = []

        for _ in range(n_workers):
            tq = mp.Queue()
            p = mp.Process(target=_worker_loop,
                           args=(tq, self._result_queue, belief_mode),
                           daemon=True)
            p.start()
            self._task_queues.append(tq)
            self._workers.append(p)

    def plan(self, snapshot):
        """Run parallel rollouts for all 8 actions and return best.

        Args:
            snapshot: dict from serialize_state(env).

        Returns:
            int: best action index (0-7).
        """
        # Distribute rollouts across workers for each action
        rollouts_per_worker = max(1, self.n_rollouts // self.n_workers)
        tasks_sent = 0

        for action_idx in range(self.n_actions):
            for w in range(self.n_workers):
                self._task_queues[w].put(
                    (snapshot, action_idx, rollouts_per_worker, self.gamma)
                )
                tasks_sent += 1

        # Collect results
        q_values = {a: [] for a in range(self.n_actions)}
        for _ in range(tasks_sent):
            action_idx, mean_return = self._result_queue.get()
            q_values[action_idx].append(mean_return)

        # Aggregate: mean across workers for each action
        q_means = {a: np.mean(vs) for a, vs in q_values.items()}

        return max(q_means, key=q_means.get)

    def close(self):
        """Shutdown all worker processes."""
        for tq in self._task_queues:
            tq.put(None)
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: `5/5 tests passed` (tests 4 and 5 may take 30-60s due to rollouts)

- [ ] **Step 5: Commit**

```bash
cd /home/the2xman/ASEN-5264-Project
git add vla_SO-ARM101/src/so_arm101_control/scripts/train_pomcp.py \
        vla_SO-ARM101/src/so_arm101_control/scripts/test_pomcp_direct.py
git commit -m "feat: add DirectPOMCPPlanner with parallel worker pool"
```

---

### Task 4: Evaluation Harness + CLI

**Files:**
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/train_pomcp.py`

- [ ] **Step 1: Write the test for evaluation**

Append to `scripts/test_pomcp_direct.py`:

```python
def test_evaluate_direct_runs():
    """Test 6: evaluate_pomcp_direct runs 3 episodes without crashing."""
    sys.path.insert(0, '/home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control/scripts')
    from train_pomcp import evaluate_pomcp_direct

    results = evaluate_pomcp_direct(
        n_episodes=3,
        n_rollouts=5,
        n_workers=2,
        gamma=0.99,
        seed=42,
    )

    assert "success_rate" in results, "Missing success_rate"
    assert "mean_episode_length" in results, "Missing mean_episode_length"
    assert "mean_return" in results, "Missing mean_return"
    assert 0.0 <= results["success_rate"] <= 1.0, "success_rate out of range"

    print(f"  PASS: evaluate_direct — {results['success_rate']*100:.0f}% success, "
          f"mean_len={results['mean_episode_length']:.1f}")
```

Add to `tests` list:
```python
        ("evaluate direct", test_evaluate_direct_runs),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: Test 6 fails with `ImportError: cannot import name 'evaluate_pomcp_direct'`

- [ ] **Step 3: Write evaluate_pomcp_direct and update CLI**

Add to `scripts/train_pomcp.py`, after `DirectPOMCPPlanner`:

```python
def evaluate_pomcp_direct(n_episodes=100, n_rollouts=100, n_workers=8,
                          gamma=0.99, seed=0):
    """Evaluate POMCP Direct Simulator planner.

    Args:
        n_episodes: Number of evaluation episodes.
        n_rollouts: Rollouts per action per planning step.
        n_workers: Number of parallel worker processes.
        gamma: Discount factor.
        seed: Random seed for episode reset.

    Returns:
        dict with evaluation metrics.
    """
    import json
    import time

    env = LegoPickEnv(belief_mode=True)
    planner = DirectPOMCPPlanner(
        n_rollouts=n_rollouts, n_workers=n_workers, gamma=gamma
    )

    successes = 0
    perfect = 0
    precise = 0
    close = 0
    episode_lengths = []
    episode_returns = []
    planning_times = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_return = 0.0
        steps = 0

        while not done:
            step_start = time.time()
            snapshot = serialize_state(env)
            action_idx = planner.plan(snapshot)
            planning_times.append(time.time() - step_start)

            action = DISCRETE_ACTIONS[action_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            steps += 1
            done = terminated or truncated

        episode_lengths.append(steps)
        episode_returns.append(total_return)

        if info.get("success", False):
            successes += 1
            dist = info.get("dist_to_goal", 1.0)
            if dist < 0.01:
                perfect += 1
            elif dist < 0.02:
                precise += 1
            elif dist < 0.04:
                close += 1

        if (ep + 1) % 10 == 0 or (ep + 1) == n_episodes:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"success={successes}/{ep+1} ({successes/(ep+1)*100:.1f}%), "
                  f"avg_steps={np.mean(episode_lengths):.1f}, "
                  f"avg_plan_time={np.mean(planning_times):.1f}s")

    planner.close()
    env.close()

    results = {
        "success_rate": successes / n_episodes,
        "perfect_rate": perfect / n_episodes,
        "precise_rate": precise / n_episodes,
        "close_rate": close / n_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_planning_time_s": float(np.mean(planning_times)),
        "n_episodes": n_episodes,
        "n_rollouts": n_rollouts,
        "n_workers": n_workers,
    }

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "logs", "pomcp_direct")
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print(f"\nPOMCP Direct Results ({n_episodes} episodes):")
    print(f"  Success rate:     {results['success_rate']*100:.1f}%")
    print(f"  Perfect (<10mm):  {results['perfect_rate']*100:.1f}%")
    print(f"  Precise (<20mm):  {results['precise_rate']*100:.1f}%")
    print(f"  Close   (<40mm):  {results['close_rate']*100:.1f}%")
    print(f"  Mean steps:       {results['mean_episode_length']:.1f}")
    print(f"  Mean return:      {results['mean_return']:.1f} ± {results['std_return']:.1f}")
    print(f"  Mean plan time:   {results['mean_planning_time_s']:.1f}s/step")

    return results
```

Also add the import at the top of the `evaluate_pomcp_direct` function's caller scope. Add this line near the top of `train_pomcp.py` imports (after existing imports):

```python
from so_arm101_control.pomcp_env_bridge import serialize_state
```

Update the `main()` function — add `--direct` flag and handler. Replace the existing `main()` with:

```python
def main():
    parser = argparse.ArgumentParser(description="POMCP with Learned World Model")
    parser.add_argument("--collect", action="store_true",
                        help="Collect transitions and train world model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate POMCP planner (learned world model)")
    parser.add_argument("--direct", action="store_true",
                        help="Evaluate POMCP planner (direct simulator)")
    parser.add_argument("--belief-model", type=str,
                        default="models/ppo_belief/best_model")
    parser.add_argument("--world-model", type=str,
                        default="models/pomcp/world_model.pt")
    parser.add_argument("--n-transitions", type=int, default=50000)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./models/pomcp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.collect:
        # Step 1: Collect transitions
        print("Step 1: Collecting transitions from Belief PPO...")
        transitions = collect_transitions(
            args.belief_model, n_transitions=args.n_transitions
        )

        # Save transitions
        trans_path = os.path.join(args.output_dir, "transitions.npz")
        np.savez(trans_path, **transitions)
        print(f"Transitions saved to {trans_path}")

        # Step 2: Train world model
        print("\nStep 2: Training world model...")
        from so_arm101_control.world_model import WorldModel

        wm = WorldModel()
        wm.train_on_buffer(transitions, epochs=args.epochs)
        wm.save(os.path.join(args.output_dir, "world_model.pt"))
        print(f"World model saved to {args.output_dir}/world_model.pt")

    if args.evaluate:
        # Evaluate with learned world model
        print("Evaluating POMCP planner (learned world model)...")
        evaluate_pomcp(
            args.world_model,
            n_episodes=args.n_episodes,
            n_rollouts=args.n_rollouts,
        )

    if args.direct:
        # Evaluate with direct simulator
        print("Evaluating POMCP planner (direct simulator)...")
        evaluate_pomcp_direct(
            n_episodes=args.n_episodes,
            n_rollouts=args.n_rollouts,
            n_workers=args.n_workers,
            gamma=args.gamma,
            seed=args.seed,
        )

    if not args.collect and not args.evaluate and not args.direct:
        print("Specify --collect, --evaluate, and/or --direct. See --help.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: `6/6 tests passed`

- [ ] **Step 5: Run the CLI smoke test**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control/scripts && python3 train_pomcp.py --direct --n-episodes 2 --n-rollouts 5 --n-workers 2`
Expected: Prints 2 episode results and saves `logs/pomcp_direct/eval_results.json`

- [ ] **Step 6: Commit**

```bash
cd /home/the2xman/ASEN-5264-Project
git add vla_SO-ARM101/src/so_arm101_control/scripts/train_pomcp.py \
        vla_SO-ARM101/src/so_arm101_control/scripts/test_pomcp_direct.py
git commit -m "feat: add POMCP direct simulator evaluation harness with CLI"
```

---

### Task 5: Integration Smoke Test (full run)

**Files:**
- No new files — just run the full pipeline

- [ ] **Step 1: Run full test suite**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control && python3 scripts/test_pomcp_direct.py`
Expected: `6/6 tests passed`

- [ ] **Step 2: Run a 5-episode evaluation with realistic settings**

Run: `cd /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control/scripts && python3 train_pomcp.py --direct --n-episodes 5 --n-rollouts 20 --n-workers 4 --seed 42`
Expected: Prints per-episode results with success rate, mean steps, planning time/step. Results saved to `logs/pomcp_direct/eval_results.json`.

- [ ] **Step 3: Verify results file**

Run: `cat /home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control/scripts/logs/pomcp_direct/eval_results.json`
Expected: JSON with `success_rate`, `perfect_rate`, `precise_rate`, `close_rate`, `mean_episode_length`, `mean_return`, `mean_planning_time_s`

- [ ] **Step 4: Commit spec and plan docs**

```bash
cd /home/the2xman/ASEN-5264-Project
git add docs/superpowers/specs/2026-04-02-pomcp-direct-design.md \
        docs/superpowers/plans/2026-04-02-pomcp-direct.md
git commit -m "docs: add POMCP direct simulator design spec and implementation plan"
```
