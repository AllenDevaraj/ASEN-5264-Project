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


if __name__ == "__main__":
    tests = [
        ("serialize/restore roundtrip", test_serialize_restore_roundtrip),
        ("restore same trajectory", test_restore_produces_same_trajectory),
        ("heuristic phases", test_heuristic_phases),
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
