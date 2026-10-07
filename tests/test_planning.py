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


def test_world_model_learns_identity_like_map_and_reports_auc():
    from pomdp_pick.planning.world_model import WorldModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 18)).astype(np.float32)
    Y = np.concatenate([0.1 * X[:, :13], (X[:, :1] > 0).astype(np.float32)], 1)
    rep = WorldModel(hidden=64).fit(X, Y, epochs=30)
    assert np.mean(rep["val_mse_per_dim"]) < 0.01 and rep["grasp_auc"] > 0.95


def test_state_of_picks_the_13_summary_dims_from_a_belief_obs():
    from pomdp_pick.planning.world_model import STATE_DIM, state_of
    env = PickPlaceEnv("belief"); obs, _ = env.reset(seed=2)
    s = state_of(obs)
    assert s.shape == (STATE_DIM,) == (13,)
    assert np.allclose(s[5:7], obs[8:10]) and np.isclose(s[12], obs[15])


def _tiny_model():
    from pomdp_pick.planning.world_model import WorldModel
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 18)).astype(np.float32)
    Y = np.concatenate([5.0 * X[:, :13], (X[:, :1] > 0).astype(np.float32)], 1)
    wm = WorldModel(hidden=16)
    wm.fit(X, Y, epochs=2)
    return wm


def test_world_model_step_keeps_sigmas_nonnegative_and_prob_in_unit_interval():
    wm = _tiny_model()
    s = np.full(13, -50.0, np.float32)
    for a in range(9):
        s2, p = wm.step_state(s, sb.macro_action(a))
        assert np.all(s2[9:13] >= 0) and 0.0 <= p <= 1.0


def test_wm_planner_returns_legal_action_and_leaves_env_untouched(tmp_path):
    from pomdp_pick.planning.pomcp import WMPlanner
    from pomdp_pick.planning.world_model import WorldModel
    wm = _tiny_model()
    wm.save(tmp_path / "wm.pt")
    wm2 = WorldModel.load(tmp_path / "wm.pt")
    env = PickPlaceEnv("belief"); env.reset(seed=4)
    snap = sb.snapshot(env)
    a, stats = WMPlanner(wm2, n_sims=16, rng=np.random.default_rng(0)).plan(env)
    assert 0 <= a < 9 and stats["n_sims"] == 16
    assert np.array_equal(snap["d"].qpos, env.d.qpos) and np.array_equal(snap["pf"][0], env.pf.particles)


def test_collect_logs_state_action_deltas_and_grasp_labels(tmp_path):
    from pomdp_pick.planning.collect import collect

    class Commit:   # always commits: every transition is a grasp attempt
        def reset(self): pass
        def act(self, obs): return np.array([0, 0, 0, 0, 1.0], np.float32)

    out = collect(None, 6, tmp_path / "d.npz", policy=Commit(), eps=0.0, seed=7)
    f = np.load(out)
    X, Y = f["X"], f["Y"]
    assert X.shape == (6, 18) and Y.shape == (6, 14)
    assert np.all(X[:, -1] > 0.5) and set(np.unique(Y[:, -1])) <= {0.0, 1.0}


def test_success_vs_budget_figure(tmp_path):
    import pandas as pd
    from pomdp_pick.eval.figures import budget_table, success_vs_budget
    rows = [dict(method=f"{k}{n}", n_sims=n, success=bool(i % 2), mean_decision_s=0.01 * n)
            for k in ("pomcp", "wmpomcp") for n in (32, 128) for i in range(4)]
    df = pd.DataFrame(rows)
    t = budget_table(df)
    assert set(t.simulator) == {"physics", "learned WM"} and len(t) == 4
    assert all(t.lo <= t.rate) and all(t.rate <= t.hi)
    success_vs_budget(df, tmp_path / "b.png", reference=("belief PPO", 0.45))
    assert (tmp_path / "b.png").stat().st_size > 1000
