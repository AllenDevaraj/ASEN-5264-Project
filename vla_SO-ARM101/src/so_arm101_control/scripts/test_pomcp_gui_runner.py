#!/usr/bin/env python3
"""Targeted regression tests for GUI POMCP transition semantics."""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from so_arm101_control.pomcp_gui_runner import GRASP_THRESHOLD, WorldModelPOMCPRunner


class _DummyWM:
    """Deterministic world model stub for transition tests."""

    def __init__(self, delta=None, grasp_p=0.0):
        self._delta = np.asarray(delta if delta is not None else np.zeros(9), dtype=np.float32)
        self._grasp_p = float(grasp_p)

    def predict(self, state, action):
        nxt = state.copy()
        nxt[:9] += self._delta
        return nxt, self._grasp_p


def _make_runner(grasp_p=0.0):
    runner = object.__new__(WorldModelPOMCPRunner)
    runner.wm = _DummyWM(grasp_p=grasp_p)
    runner.rng = np.random.default_rng(0)
    return runner


def test_close_sets_holding_when_ee_near_block():
    runner = _make_runner(grasp_p=0.0)
    # [ee(3), mu(3), sigma(3), holding(1)]
    state = np.array([0.18, 0.01, 0.03, 0.18, 0.01, 0.0, 0.01, 0.01, 0.01, 0.0], dtype=np.float32)
    goal_xy = np.array([0.22, 0.01], dtype=np.float32)
    nxt, reward, terminal = runner._transition(state, action_idx=6, goal_pos=goal_xy)
    assert np.linalg.norm(nxt[:2] - state[3:5]) < GRASP_THRESHOLD
    assert nxt[9] > 0.5, "CLOSE near block should set holding state"
    assert reward >= 19.0, f"Expected grasp bonus, got reward={reward}"
    assert not terminal


def test_open_drops_and_terminates_near_goal():
    runner = _make_runner(grasp_p=0.0)
    state = np.array([0.20, 0.00, 0.03, 0.15, -0.01, 0.0, 0.01, 0.01, 0.01, 1.0], dtype=np.float32)
    goal_xy = np.array([0.20, 0.00], dtype=np.float32)
    nxt, reward, terminal = runner._transition(state, action_idx=7, goal_pos=goal_xy)
    assert nxt[9] < 0.5, "OPEN should clear holding state"
    assert terminal, "OPEN near goal while holding should terminate rollout"
    assert reward >= 79.0, f"Expected placement bonus, got reward={reward}"


def test_motion_keeps_holding_flag():
    runner = _make_runner(grasp_p=0.0)
    state = np.array([0.16, -0.02, 0.03, 0.14, -0.02, 0.0, 0.01, 0.01, 0.01, 1.0], dtype=np.float32)
    goal_xy = np.array([0.22, 0.05], dtype=np.float32)
    nxt, reward, terminal = runner._transition(state, action_idx=0, goal_pos=goal_xy)
    assert nxt[9] > 0.5, "Non-gripper move should keep holding state"
    assert abs(reward + 1.0) < 1e-6
    assert not terminal


def test_close_while_holding_is_noop():
    runner = _make_runner(grasp_p=1.0)
    state = np.array([0.18, 0.01, 0.03, 0.18, 0.01, 0.0, 0.01, 0.01, 0.01, 1.0], dtype=np.float32)
    goal_xy = np.array([0.22, 0.01], dtype=np.float32)
    nxt, reward, terminal = runner._transition(state, action_idx=6, goal_pos=goal_xy)
    assert nxt[9] > 0.5, "Holding should remain true"
    assert abs(reward + 1.0) < 1e-6, "CLOSE while holding should be a no-op"
    assert not terminal


def test_close_far_fails_with_penalty():
    runner = _make_runner(grasp_p=0.0)
    state = np.array([0.10, 0.00, 0.03, 0.18, 0.00, 0.0, 0.01, 0.01, 0.01, 0.0], dtype=np.float32)
    goal_xy = np.array([0.22, 0.01], dtype=np.float32)
    nxt, reward, terminal = runner._transition(state, action_idx=6, goal_pos=goal_xy)
    assert nxt[9] < 0.5
    assert abs(reward + 6.0) < 1e-6, f"Expected step + fail penalty, got reward={reward}"
    assert not terminal


if __name__ == "__main__":
    test_close_sets_holding_when_ee_near_block()
    test_open_drops_and_terminates_near_goal()
    test_motion_keeps_holding_flag()
    test_close_while_holding_is_noop()
    test_close_far_fails_with_penalty()
    print("5/5 POMCP GUI runner tests passed")
