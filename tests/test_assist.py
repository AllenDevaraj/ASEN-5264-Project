"""v2.1 grasp-assist curriculum (training only; identical for every method)."""
import math

import numpy as np

from pomdp_pick.envs.pick_env import PickPlaceEnv


def _offset_commit(assist_mm, off_mm=15.0, seed=3):
    env = PickPlaceEnv("oracle", sigma_fixed=0.005)
    env.reset(seed=seed)
    env.set_assist(assist_mm)
    x, y, _, yaw = env.arm.brick_pose()
    # hover over a point off_mm across the jaw axis (perpendicular to the brick's long axis)
    p = np.array([x - off_mm / 1000 * math.sin(yaw), y + off_mm / 1000 * math.cos(yaw), 0.03])
    env.arm.move(p, yaw, 1.0)
    _, _, _, _, info = env.step(np.array([0, 0, 0, 0, 1.0]))
    return env, info


def test_no_assist_misses_a_15mm_offset_grasp():
    env, info = _offset_commit(0.0)
    assert not env.held


def test_assist_rescues_the_same_grasp_but_logs_the_policys_own_error():
    env, info = _offset_commit(20.0)
    assert env.held
    assert info["commit_err_mm"][0] > 12.0          # pre-assist error, not the corrected one


def test_assist_schedule_anneals_linearly_to_zero():
    from pomdp_pick.agents.train import assist_at
    sched = {"start_mm": 40.0, "until_frac": 0.5}
    assert assist_at(0, 1000, sched) == 40.0
    assert abs(assist_at(250, 1000, sched) - 20.0) < 1e-9
    assert assist_at(500, 1000, sched) == 0.0 and assist_at(900, 1000, sched) == 0.0
    assert assist_at(100, 1000, None) == 0.0


def test_training_applies_the_assist_to_the_envs(tmp_path):
    import json
    from pomdp_pick.agents.train import train
    hp = {"n_steps": 64, "batch_size": 64, "n_epochs": 1}
    run = train("oracle", 0, 64, tmp_path, n_envs=1, ckpt_every=64, hparams=hp,
                assist={"start_mm": 40.0, "until_frac": 0.5})
    prog = json.loads((run / "progress.json").read_text())
    assert prog["assist"] == {"start_mm": 40.0, "until_frac": 0.5}
    assert 0.0 < prog["last_assist_mm"] <= 40.0
