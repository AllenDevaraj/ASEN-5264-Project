#!/usr/bin/env python3
"""Greedy heuristic rollout policy for POMCP.

Phase-based greedy policy that mimics the learned PPO behavior:
  Phase 1: Approach block in XY (move toward block_mu)
  Phase 2: Lower Z when close in XY
  Phase 3: Close gripper when in grasp zone
  Phase 4: Carry to goal (move toward goal_xy)
  Phase 5: Open gripper at goal

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
