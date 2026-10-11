import numpy as np
import pytest

from pomdp_pick.envs.pick_env import PickPlaceEnv
from pomdp_pick.footage.record import FPS, record_episode


def _scripted(obs, env):
    """Oracle-ish: drive P over the true brick, descend, commit."""
    x, y, _, _ = env.arm.brick_pose()
    p = env.arm.p
    d = np.array([x - p[0], y - p[1]])
    if np.hypot(*d) > 0.003 or p[2] > 0.03:
        return np.array([*np.clip(d / 0.015, -1, 1), -1.0 if p[2] > 0.03 else 0.0, 0.0, -1.0])
    return np.array([0, 0, 0, 0, 1.0])


def test_record_episode_logs_30fps_frames_with_aligned_overlays():
    env = PickPlaceEnv("belief", sigma_fixed=0.005)
    rec = record_episode(env, _scripted, seed=10003, max_decisions=40)
    F = len(rec["t"])
    dt = np.diff(rec["t"])
    assert F > 30 and np.allclose(dt, 1.0 / FPS, atol=0.0045)   # one physics step of slack
    assert rec["qpos"].shape == (F, env.m.nq)
    assert rec["particles"].shape[0] == F and rec["mu"].shape == (F, 3) and rec["reading"].shape == (F, 3)
    assert rec["decision"].shape == (F,) and np.all(np.diff(rec["decision"]) >= 0)
    assert rec["commit"].any() and rec["attempts"][-1] >= 1
    assert rec["goal"].shape == (2,) and "success" in rec


def test_planner_frames_are_not_recorded_during_search():
    from pomdp_pick.footage.record import planner_actor
    from pomdp_pick.planning.pomcp import Planner
    env = PickPlaceEnv("belief", sigma_fixed=0.01)
    rec = record_episode(env, planner_actor(Planner(4, rng=np.random.default_rng(0))), seed=10001,
                         max_decisions=3)
    # Search steps rewind time on restore; leaked frames would break monotonicity and inflate the count.
    elapsed = env.arm.sim_time - env.t0
    assert np.all(np.diff(rec["t"]) > 0) and abs(len(rec["t"]) - (elapsed * FPS + 1)) <= 2


@pytest.mark.skipif(not __import__("os").environ.get("DISPLAY"), reason="MuJoCo rendering needs GLFW/DISPLAY")
def test_render_episode_writes_an_mp4_with_one_frame_per_record(tmp_path):
    import subprocess
    from pomdp_pick.footage.record import save
    from pomdp_pick.footage.render import render_episode
    env = PickPlaceEnv("belief", sigma_fixed=0.005)
    rec = record_episode(env, _scripted, seed=10003, max_decisions=3)
    save(rec, tmp_path / "e.npz")
    out = render_episode(tmp_path / "e.npz", tmp_path / "e.mp4", title="test", w=320, h=180, hold_end_s=0)
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert int(n.stdout.strip()) == len(rec["t"])


def test_episode_cache_name_depends_on_method_run_seed_and_sigma():
    from pomdp_pick.footage.record import episode_name
    names = {episode_name(m, s, sig, 3) for m in ("belief", "plain") for s in (0, 1) for sig in (5, 10)}
    assert len(names) == 8


def test_belief_hidden_only_while_lifted_or_after_the_final_successful_lift():
    from pomdp_pick.footage.render import belief_hidden
    lifted = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0], bool)    # a slipped lift, then the real one
    assert belief_hidden(lifted, True).astype(int).tolist() == [0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert belief_hidden(lifted, False).astype(int).tolist() == [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]


def test_episode_cache_name_distinguishes_runs_roots():
    # The same method/seed/σ/episode from runs/ (base protocol) and runs_v21/ are different policies.
    from pomdp_pick.footage.record import episode_name
    assert episode_name("belief", 0, 10, 3, runs="runs") != episode_name("belief", 0, 10, 3, runs="runs_v21")
    assert episode_name("belief", 0, 10, 3, runs="some/dir/runs_v21") == episode_name("belief", 0, 10, 3, runs="runs_v21")
