"""Exact physics snapshots of a PickPlaceEnv for planning, and the discrete macro-action set."""
import copy

import mujoco
import numpy as np

from pomdp_pick.robot.model import TARGET

# ±x / ±y 10 mm, down / up 15 mm, yaw ±0.25 rad, commit
MACROS = [np.array(a, float) for a in (
    [0.667, 0, 0, 0, -1], [-0.667, 0, 0, 0, -1], [0, 0.667, 0, 0, -1], [0, -0.667, 0, 0, -1],
    [0, 0, -1.0, 0, -1], [0, 0, 1.0, 0, -1], [0, 0, 0, 0.833, -1], [0, 0, 0, -0.833, -1],
    [0, 0, 0, 0, 1.0])]
COMMIT = 8

_ENV_FIELDS = ("attempts", "success", "held", "phi", "t0", "last_reading", "age", "visible", "sigma_ep")
_ARM_FIELDS = ("p", "yaw", "grip", "sol", "sim_time", "_next_frame")


def macro_action(i):
    return MACROS[i].copy()


def snapshot(env):
    d = mujoco.MjData(env.m)
    mujoco.mj_copyData(d, env.m, env.d)
    return {"d": d,
            "env": {k: copy.deepcopy(getattr(env, k)) for k in _ENV_FIELDS},
            "arm": {k: copy.deepcopy(getattr(env.arm, k)) for k in _ARM_FIELDS},
            "rng": copy.deepcopy(env.np_random.bit_generator.state),
            "sense": copy.deepcopy(env.sense_rng.bit_generator.state),
            "pf": (env.pf.particles.copy(), env.pf.weights.copy(), copy.deepcopy(env.pf.rng.bit_generator.state)),
            "lists": (list(env.commit_err), list(env.std_at_commit))}


def restore(env, snap):
    mujoco.mj_copyData(env.d, env.m, snap["d"])
    for k, v in snap["env"].items():
        setattr(env, k, copy.deepcopy(v))
    for k, v in snap["arm"].items():
        setattr(env.arm, k, copy.deepcopy(v))
    env.np_random.bit_generator.state = copy.deepcopy(snap["rng"])
    env.sense_rng.bit_generator.state = copy.deepcopy(snap["sense"])
    env.pf.particles, env.pf.weights = snap["pf"][0].copy(), snap["pf"][1].copy()
    env.pf.rng.bit_generator.state = copy.deepcopy(snap["pf"][2])
    env.commit_err, env.std_at_commit = list(snap["lists"][0]), list(snap["lists"][1])


def set_target(env, x, y, yaw):
    """Write a hypothesised target pose (e.g. a belief particle) into the scratch physics."""
    env.arm.set_brick(x, y, yaw, name=TARGET)
