"""Regression tests for the whole-branch review findings (I1-I3, M1-M6)."""
import importlib.util
import json
import math
import pathlib
import sys

import mujoco
import numpy as np
import pytest

from pomdp_pick.envs.pick_env import PickPlaceEnv
from pomdp_pick.robot.controller import Arm
from pomdp_pick.robot.model import DISTRACTORS, TARGET, build_model

ROOT = pathlib.Path(__file__).parents[1]


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    spec.loader.exec_module(mod)
    return mod


def _far_commit(env, off=0.013):
    """Line P up 13 mm across the jaw axis from the brick, then commit: a near miss that shoves it."""
    for _ in range(40):
        bx, by, _, byaw = env.arm.brick_pose()
        tgt = np.array([bx - off * math.sin(byaw), by + off * math.cos(byaw)])
        d = np.clip((tgt - env.arm.p[:2]) / 0.015, -1, 1)
        dz = np.clip((0.03 - env.arm.p[2]) / 0.015, -1, 1)
        dyaw = np.clip(float(np.angle(np.exp(2j * (byaw - env.arm.yaw)))) / 2 / 0.3, -1, 1)
        if np.hypot(*(tgt - env.arm.p[:2])) < 0.001 and abs(dyaw) < 0.05:
            break
        env.step(np.array([d[0], d[1], dz, dyaw, -1.0]))
    return env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))


# ---------- I1: no true-state leak into the belief filter ----------
def test_failed_grasp_inflation_is_fixed_not_true_displacement():
    env = PickPlaceEnv("belief")
    env.reset(seed=4)
    seen = []
    orig = env.pf.predict
    env.pf.predict = lambda extra_xy=0.0: (seen.append(extra_xy), orig(extra_xy))[1]
    before = np.array(env.arm.brick_pose()[:2])
    _, _, _, _, info = _far_commit(env)
    assert info["attempts"] == 1 and not info["success"]
    moved = np.linalg.norm(np.array(env.arm.brick_pose()[:2]) - before)
    assert moved > 0.001                      # the miss really pushed the brick
    assert seen[-1] == pytest.approx(0.01)    # ...but the filter is not told by how much


def test_filter_likelihood_distance_comes_from_reading_not_truth():
    env = PickPlaceEnv("belief")
    env.reset(seed=4)
    checks = []
    orig = env.pf.update

    def spy(z, dist):
        checks.append((dist, env.cam.distance(env.d, np.asarray(z)[:2]),
                       env.cam.distance(env.d, env.arm.brick_pose()[:2])))
        return orig(z, dist)
    env.pf.update = spy
    for _ in range(6):
        env.step(np.array([0.2, 0.0, -0.2, 0.0, -1.0]))
    assert checks
    for dist, from_reading, from_truth in checks:
        assert dist == pytest.approx(from_reading, abs=1e-12)
    assert any(abs(dist - truth) > 1e-6 for dist, _, truth in checks)


# ---------- I2: no interpenetrating spawns ----------
def _brick_contacts(env):
    ids = {mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in (TARGET, *DISTRACTORS)}
    n = 0
    for i in range(env.d.ncon):
        c = env.d.contact[i]
        b1, b2 = env.m.geom_bodyid[c.geom1], env.m.geom_bodyid[c.geom2]
        if b1 in ids and b2 in ids:
            n += 1
    return n


def test_spawns_never_overlap_and_target_starts_upright():
    env = PickPlaceEnv("oracle")
    for seed in range(40):
        env.reset(seed=seed)
        assert _brick_contacts(env) == 0, seed
        x, y, z, _ = env.arm.brick_pose()
        assert abs(z - env.arm.brick_half_size(TARGET)[2]) < 0.001, seed
        assert env._upright(TARGET), seed


# ---------- I3: queue never stalls / gives up on resumable jobs ----------
def test_low_memory_pause_terminates_after_stall_timeout(tmp_path):
    wd = _load_tool("watchdog")
    rc = wd.run([sys.executable, "-c", "import time; time.sleep(60)"], log_path=tmp_path / "w.log", poll_s=0.1,
                readers=(lambda: 60.0, lambda: 1500.0, lambda: True), mem_stall_s=0.5)
    assert rc != 0
    assert "stalled" in (tmp_path / "w.log").read_text()


def test_queue_retries_failed_job(tmp_path, monkeypatch):
    rq = _load_tool("run_queue")
    marker = tmp_path / "tries"
    script = f"import pathlib,sys; p=pathlib.Path(r'{marker}'); n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); sys.exit(0 if n>=1 else 3)"
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"name": "flaky", "cmd": [sys.executable, "-c", script]}]))
    monkeypatch.chdir(tmp_path)
    rq.main(str(q), retries=2, poll_s=0.1)
    assert (tmp_path / "q.json.done").read_text().split() == ["flaky"]
    assert marker.read_text() == "2"


# ---------- M1: shaping is zeroed at terminal states ----------
def test_terminal_step_drops_future_potential():
    env = PickPlaceEnv("oracle", max_attempts=1)
    env.reset(seed=4)
    for _ in range(12):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, -1.0]))
    phi_prev = env.phi
    t0 = env.arm.sim_time
    _, r, term, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    if info["success"]:
        pytest.skip("brick happened to be under P")
    assert term
    dt = env.arm.sim_time - t0
    assert r == pytest.approx(-1.0 - 0.01 * dt / 0.1 - phi_prev, abs=1e-6)


# ---------- M2: wrist branch hysteresis, bounded yaw ----------
def test_yaw_stays_wrapped_and_xy_keeps_moving():
    m, d = build_model()
    arm = Arm(m, d)
    arm.park_distractors()
    arm.reset_pose(np.array([0.18, 0.0, 0.06]), 0.0)
    for _ in range(25):                       # spin far past ±π
        arm.move(arm.p, arm.yaw + 0.3, 0.1)
    assert -math.pi / 2 <= arm.yaw < math.pi / 2
    p0 = arm.p.copy()
    arm.move(p0 + np.array([0.0, 0.015, 0.0]), arm.yaw, 0.1)
    assert np.linalg.norm(arm.p - p0) > 0.01


def test_wrist_flips_happen_only_with_jaws_lifted():
    """Roll's range is ~1.8π, so a long one-way spin must flip by π at some point. Every actuator
    command that flips the roll must come with P already raised to FLIP_MIN_Z."""
    from pomdp_pick.robot.controller import FLIP_MIN_Z
    from pomdp_pick.robot.kinematics import JOINT_LIMITS
    lo, hi = JOINT_LIMITS["wrist_roll"]
    m, d = build_model()
    arm = Arm(m, d)
    arm.park_distractors()
    arm.reset_pose(np.array([0.18, 0.0, 0.03]), 0.0)
    log = []
    orig = arm._apply
    arm._apply = lambda: (log.append((arm.sol["wrist_roll"], arm.p[2])), orig())[1]
    for _ in range(30):
        arm.move(np.array([*arm.p[:2], 0.03]), arm.yaw + 0.3, 0.1)
    flips = 0
    for (r0, z0), (r1, z1) in zip(log, log[1:]):
        assert lo - 1e-6 <= r1 <= hi + 1e-6
        if abs(r1 - r0) > 0.5:
            flips += 1
            assert z0 >= FLIP_MIN_Z - 1e-9 and z1 >= FLIP_MIN_Z - 1e-9
    assert flips >= 1


# ---------- M3: resume/seed hygiene ----------
def test_env_seed_streams_do_not_overlap_between_runs():
    from pomdp_pick.agents.train import env_seed_base
    a = {env_seed_base(0, 0) + r for r in range(4)}
    b = {env_seed_base(1, 0) + r for r in range(4)}
    assert not (a & b)
    assert env_seed_base(0, 100_000) != env_seed_base(0, 0)   # resume does not replay the stream


def test_resume_keeps_monitor_history_and_exact_target(tmp_path):
    from pomdp_pick.agents.train import train
    hp = {"n_steps": 64, "batch_size": 64, "n_epochs": 1}
    run = train("plain", 0, 128, tmp_path, n_envs=1, ckpt_every=64, hparams=hp)
    rows1 = (run / "monitor" / "w0.monitor.csv").read_text().count("\n")
    run = train("plain", 0, 256, tmp_path, n_envs=1, ckpt_every=64, hparams=hp)
    rows2 = (run / "monitor" / "w0.monitor.csv").read_text().count("\n")
    assert rows2 > rows1
    prog = json.loads((run / "progress.json").read_text())
    assert 256 <= prog["timesteps"] < 256 + 64


# ---------- M4: eval never reuses a CSV with a different episode count ----------
def test_eval_csv_path_encodes_episode_count(tmp_path):
    from pomdp_pick.eval.evaluate import eval_csv_path
    meta = {"method": "plain", "seed": 0}
    assert eval_csv_path(tmp_path, meta, 100) != eval_csv_path(tmp_path, meta, 200)


# ---------- M5: drift keeps orientation and velocity ----------
def test_drift_preserves_orientation():
    env = PickPlaceEnv("oracle")
    env.reset(seed=2)
    q = env.arm.fj[TARGET]
    tilt = np.array([math.cos(0.2), math.sin(0.2), 0.0, 0.0])
    env.d.qpos[q + 3:q + 7] = tilt
    env.d.qvel[env.m.jnt_dofadr[env.m.body_jntadr[env.arm._body(TARGET)]]] = 0.01
    xy0 = env.d.qpos[q:q + 2].copy()
    env._drift_target()
    assert np.allclose(env.d.qpos[q + 3:q + 7], tilt)
    assert not np.allclose(env.d.qpos[q:q + 2], xy0)


# ---------- M6: placement requires a resting, upright brick ----------
def test_hovering_or_tipped_brick_is_not_placed():
    env = PickPlaceEnv("oracle")
    env.reset(seed=7)
    _, _, _, yaw = env.arm.brick_pose()
    env.arm.set_brick(env.goal[0], env.goal[1], yaw, z=0.03)          # hovering above goal
    assert env._placement_ok() is False
    q = env.arm.fj[TARGET]
    env.arm.set_brick(env.goal[0], env.goal[1], 0.0, z=0.008)        # on its side at the goal
    env.d.qpos[q + 3:q + 7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
    mujoco.mj_forward(env.m, env.d)
    assert env._placement_ok() is False


def test_stopping_the_queue_does_not_retry_the_job(tmp_path):
    import os, signal, subprocess, time
    marker = tmp_path / "starts"
    script = f"import pathlib,time; p=pathlib.Path(r'{marker}'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); time.sleep(60)"
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"name": "long", "cmd": [sys.executable, "-c", script]}]))
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools/run_queue.py"), str(q)], cwd=tmp_path)
    for _ in range(100):
        if marker.exists():
            break
        time.sleep(0.1)
    proc.terminate()
    proc.wait(timeout=30)
    time.sleep(1.0)
    assert marker.read_text() == "x"             # started once, never retried
    assert not (tmp_path / "q.json.done").exists()


def test_monitor_csv_has_header_after_fresh_and_resumed_runs(tmp_path):
    from pomdp_pick.agents.train import train
    hp = {"n_steps": 64, "batch_size": 64, "n_epochs": 1}
    run = train("oracle", 0, 64, tmp_path, n_envs=1, ckpt_every=64, hparams=hp)
    train("oracle", 0, 128, tmp_path, n_envs=1, ckpt_every=64, hparams=hp)
    lines = (run / "monitor" / "w0.monitor.csv").read_text().splitlines()
    assert lines[0].startswith("#") and lines[1].startswith("r,l,t")
    assert sum(1 for line in lines if line.startswith("r,l,t")) == 1
