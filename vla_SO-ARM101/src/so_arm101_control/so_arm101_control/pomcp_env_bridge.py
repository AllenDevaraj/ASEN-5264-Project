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

    # Particle filter
    if env.belief_mode and "pf_particles" in snapshot:
        env.pf.particles = snapshot["pf_particles"].copy()
        env.pf.weights = snapshot["pf_weights"].copy()
        env.pf._last_obs = (
            snapshot["pf_last_obs"].copy()
            if snapshot["pf_last_obs"] is not None
            else None
        )
