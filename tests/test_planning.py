import numpy as np

from pomdp_pick.envs.pick_env import PickPlaceEnv
from pomdp_pick.planning import sim_bridge as sb


def test_snapshot_restore_roundtrip_is_exact():
    env = PickPlaceEnv("belief"); env.reset(seed=3)
    snap = sb.snapshot(env)
    obs_a = [env.step(sb.macro_action(i % 8))[0] for i in range(5)]
    sb.restore(env, snap)
    obs_b = [env.step(sb.macro_action(i % 8))[0] for i in range(5)]
    assert all(np.allclose(a, b) for a, b in zip(obs_a, obs_b))


def test_set_target_moves_only_the_target():
    env = PickPlaceEnv("oracle"); env.reset(seed=3)
    d_before = [env.arm.brick_pose(n) for n in ("blue_lego_2x2", "blue_lego_2x2_b")]
    sb.set_target(env, 0.2, 0.05, 0.3)
    x, y, _, yaw = env.arm.brick_pose()
    assert abs(x - 0.2) < 1e-9 and abs(y - 0.05) < 1e-9
    assert [env.arm.brick_pose(n) for n in ("blue_lego_2x2", "blue_lego_2x2_b")] == d_before


from pomdp_pick.planning.pomcp import Planner  # noqa: E402


def test_plan_does_not_touch_real_env():
    env = PickPlaceEnv("belief"); env.reset(seed=5)
    snap = sb.snapshot(env)
    a, stats = Planner(n_sims=8, rng=np.random.default_rng(0)).plan(env)
    assert 0 <= a < 9 and stats["n_sims"] == 8 and stats["wall_s"] > 0
    after = sb.snapshot(env)
    assert np.array_equal(snap["d"].qpos, after["d"].qpos) and snap["env"]["attempts"] == after["env"]["attempts"]
    assert np.array_equal(snap["pf"][0], after["pf"][0]) and snap["arm"]["sim_time"] == after["arm"]["sim_time"]


def test_planner_handles_prior_only_belief():
    env = PickPlaceEnv("belief", sigma_fixed=0.03); env.reset(seed=6)
    env.pf.reset_from_prior((0.0, 0.0), (0.13, 0.23), (-1.0, 1.0))
    a, _ = Planner(n_sims=4, rng=np.random.default_rng(1)).plan(env)
    assert 0 <= a < 9


def test_heuristic_commits_when_low_and_aligned_and_moves_toward_target_otherwise():
    env = PickPlaceEnv("oracle"); env.reset(seed=3)
    p, yaw = env.arm.p.copy(), env.arm.yaw
    assert Planner.heuristic(env, (p[0] + 0.05, p[1]), yaw) == 0
    assert Planner.heuristic(env, (p[0], p[1] - 0.05), yaw) == 3
    env.arm.p = np.array([p[0], p[1], 0.03])
    assert Planner.heuristic(env, (p[0] + 0.001, p[1]), yaw + np.pi) == sb.COMMIT


def test_pomcp_eval_smoke(tmp_path):
    from pomdp_pick.eval.evaluate_pomcp import evaluate_pomcp
    df = evaluate_pomcp(4, sigma_mm=10, n_episodes=1, out_csv=tmp_path / "p.csv", max_decisions=6)
    assert len(df) == 1 and df.method[0] == "pomcp4" and df.mean_decision_s[0] > 0
    assert (tmp_path / "p.csv").exists()


def test_pomcp_eval_resumes_from_partial_csv(tmp_path):
    import pandas as pd
    from pomdp_pick.eval.evaluate_pomcp import evaluate_pomcp
    out = tmp_path / "p.csv"
    pd.DataFrame([dict(method="pomcp4", seed=0, sigma_mm=10, episode=0, n_sims=4, success=True, attempts=1,
                       sim_time=1.0, commit_err_mm=1.0, belief_std_at_commit=1.0, mean_decision_s=0.5,
                       **{"return": 123.0})]).to_csv(str(out) + ".partial", index=False)
    df = evaluate_pomcp(4, sigma_mm=10, n_episodes=1, out_csv=out, max_decisions=6)
    assert df["return"][0] == 123.0 and out.exists() and not (tmp_path / "p.csv.partial").exists()
