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


def test_heuristic_rollout_reaches_commit_from_any_offset_on_the_macro_grid():
    from pomdp_pick.planning.pomcp import _heuristic
    step = 0.667 * 0.015
    rng = np.random.default_rng(0)
    for dx0, dy0 in [(0.025, 0.0)] + [tuple(rng.uniform(-0.06, 0.06, 2)) for _ in range(200)]:
        dx, dy, a = dx0, dy0, None
        for _ in range(30):
            a = _heuristic(dx, dy, 0.03, 0.0)
            if a == sb.COMMIT:
                break
            dx -= step * {0: 1, 1: -1}.get(a, 0)
            dy -= step * {2: 1, 3: -1}.get(a, 0)
        assert a == sb.COMMIT, (dx0, dy0)
        assert max(abs(dx), abs(dy)) <= step / 2 + 1e-9


def test_decision_latency_reports_cpu_time_that_excludes_frozen_time(tmp_path, monkeypatch):
    import time as _t
    from pomdp_pick.eval.evaluate_pomcp import evaluate_pomcp
    from pomdp_pick.planning import pomcp
    real = pomcp.Planner._simulate

    def slow(self, *a):                       # stands in for a SIGSTOP pause: wall time passes, CPU doesn't
        _t.sleep(0.05)
        return real(self, *a)
    monkeypatch.setattr(pomcp.Planner, "_simulate", slow)
    env = PickPlaceEnv("belief"); env.reset(seed=5)
    _, st = pomcp.Planner(n_sims=4, rng=np.random.default_rng(0)).plan(env)
    assert 0 < st["cpu_s"] < st["wall_s"] - 0.15
    df = evaluate_pomcp(4, sigma_mm=10, n_episodes=1, out_csv=tmp_path / "p.csv", max_decisions=2)
    assert 0 < df.mean_decision_cpu_s[0] < df.mean_decision_s[0]


def test_budget_table_prefers_cpu_latency():
    import pandas as pd
    from pomdp_pick.eval.figures import budget_table
    df = pd.DataFrame([dict(method="pomcp32", n_sims=32, success=True, mean_decision_s=9.0, mean_decision_cpu_s=1.0)])
    assert budget_table(df).decision_s[0] == 1.0


def test_world_model_splits_by_episode_when_groups_given():
    from pomdp_pick.planning.world_model import WorldModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 18)).astype(np.float32)
    Y = np.concatenate([0.1 * X[:, :13], (X[:, :1] > 0).astype(np.float32)], 1)
    E = np.repeat(np.arange(60), 10)
    wm = WorldModel(hidden=16)
    wm.fit(X, Y, epochs=1, groups=E)
    tr, va = wm.split_
    assert set(E[tr]).isdisjoint(E[va]) and len(va) > 0


def test_delta_head_ignores_commit_rows():
    from pomdp_pick.planning.world_model import WorldModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 18)).astype(np.float32)
    commit = X[:, -1] > 0.5
    D = 0.1 * X[:, :13]
    D[commit] = 100.0 * rng.normal(size=(commit.sum(), 13))     # e.g. the carry-to-goal jump
    Y = np.concatenate([D, (X[:, :1] > 0).astype(np.float32)], 1)
    rep = WorldModel(hidden=64).fit(X, Y, epochs=30)
    assert np.mean(rep["val_mse_per_dim"]) < 0.01


def test_collect_records_episode_ids(tmp_path):
    from pomdp_pick.planning.collect import collect

    class Commit:
        def reset(self): pass
        def act(self, obs): return np.array([0, 0, 0, 0, 1.0], np.float32)

    f = np.load(collect(None, 7, tmp_path / "d.npz", policy=Commit(), eps=0.0, seed=7))
    assert f["E"].shape == (7,) and np.all(np.diff(f["E"]) >= 0) and f["E"][-1] >= 1   # 3 commits end an episode


def test_evaluate_pomcp_cli_writes_into_out_dir(tmp_path, monkeypatch):
    import sys
    from pomdp_pick.eval import evaluate_pomcp as ep
    monkeypatch.setattr(sys, "argv", ["x", "--sims", "2", "--episodes", "1", "--max-decisions", "2",
                                      "--out-dir", str(tmp_path / "ev")])
    ep.main()
    assert (tmp_path / "ev" / "pomcp2_seed0_n1.csv").exists()


class _ConstP:
    """Stub world model: the state never changes and every commit grasps with probability p."""
    def __init__(self, p):
        self.p = p

    def step_state(self, s, action):
        return np.asarray(s, np.float32).copy(), self.p


def test_wm_commit_only_cashes_out_the_potential_when_the_grasp_succeeds():
    # A missed grasp does not end the real episode, so it must not collect the -Φ terminal bonus;
    # otherwise a hopeless commit far from the brick outscores moving toward it.
    from pomdp_pick.planning.pomcp import WMPlanner
    far = np.zeros(13, np.float32); far[5] = 1.5          # belief mean 15 cm away (obs units: dm)
    _, r_miss, term = WMPlanner(_ConstP(0.0), 1)._step(far, sb.COMMIT)
    assert term and r_miss == -1.0 - 0.01
    _, r_hit, _ = WMPlanner(_ConstP(1.0), 1)._step(far, sb.COMMIT)
    assert r_hit == 15.0 - 0.01 - WMPlanner._phi(far)


def test_mean_heuristic_acts_on_the_belief_mean_without_touching_the_env():
    # No-search ablation: POMCP's rollout policy applied to the particle-filter mean.
    from pomdp_pick.planning.pomcp import MeanHeuristic, _heuristic
    env = PickPlaceEnv("belief"); env.reset(seed=4)
    snap = sb.snapshot(env)
    w = env.pf.weights / env.pf.weights.sum()
    mx, my = w @ env.pf.particles[:, 0], w @ env.pf.particles[:, 1]
    th = np.angle(w @ np.exp(2j * env.pf.particles[:, 2])) / 2
    p = env.arm.p
    a, stats = MeanHeuristic().plan(env)
    assert a == _heuristic(mx - p[0], my - p[1], p[2], th - env.arm.yaw)
    assert stats["n_sims"] == 0 and MeanHeuristic().label == "heuristic"
    assert np.array_equal(snap["d"].qpos, env.d.qpos)


def test_evaluate_pomcp_cli_heuristic_ablation_writes_its_own_csv(tmp_path, monkeypatch):
    import sys
    import pandas as pd
    from pomdp_pick.eval import evaluate_pomcp as ep
    monkeypatch.setattr(sys, "argv", ["x", "--heuristic", "--episodes", "1", "--max-decisions", "3",
                                      "--out-dir", str(tmp_path / "ev")])
    ep.main()
    df = pd.read_csv(tmp_path / "ev" / "heuristic_seed0_n1.csv")
    assert df.method.tolist() == ["heuristic"] and df.n_sims.tolist() == [0]


def test_success_vs_budget_draws_every_reference_line(tmp_path):
    import pandas as pd
    from pomdp_pick.eval.figures import success_vs_budget
    df = pd.DataFrame([dict(method=f"pomcp{n}", n_sims=n, success=bool(i % 2), mean_decision_s=0.01 * n)
                       for n in (32, 128) for i in range(4)])
    fig = success_vs_budget(df, tmp_path / "b.png",
                            reference=[("belief PPO", 0.41), ("no search (rollout rule)", 0.50)])
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "belief PPO" in labels and "no search (rollout rule)" in labels
