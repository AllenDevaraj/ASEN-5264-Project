"""Evaluate POMCP on the shared eval episodes (same seeds as the RL methods).

Episodes are slow (every decision runs n_sims physical simulations), so each finished episode is
appended to `<out>.partial` and a restarted job resumes from it.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pomdp_pick.envs.pick_env import PickPlaceEnv
from pomdp_pick.eval.evaluate import SEED_BASE
from pomdp_pick.planning import sim_bridge as sb
from pomdp_pick.planning.pomcp import Planner


def run_episode(env, planner, seed, max_decisions):
    env.reset(seed=seed)
    ret, times, cpus, info = 0.0, [], [], {}
    for _ in range(max_decisions):
        a, stats = planner.plan(env)
        times.append(stats["wall_s"])
        cpus.append(stats["cpu_s"])
        _, r, term, trunc, info = env.step(sb.macro_action(a))
        ret += r
        if term or trunc:
            break
    return ret, times, cpus, info


def evaluate_pomcp(n_sims, sigma_mm=10, n_episodes=50, out_csv=None, max_decisions=400, make_planner=None):
    make_planner = make_planner or (lambda rng: Planner(n_sims, rng=rng))
    partial = Path(str(out_csv) + ".partial") if out_csv else None
    rows = pd.read_csv(partial).to_dict("records") if partial and partial.exists() else []
    done = {r["episode"] for r in rows}
    env = PickPlaceEnv("belief", sigma_fixed=sigma_mm / 1000.0)
    for ep in range(n_episodes):
        if ep in done:
            continue
        planner = make_planner(np.random.default_rng(SEED_BASE + ep))
        ret, times, cpus, info = run_episode(env, planner, SEED_BASE + ep, max_decisions)
        rows.append(dict(method=getattr(planner, "label", f"pomcp{n_sims}"), seed=0, sigma_mm=sigma_mm,
                         episode=ep, n_sims=n_sims, success=info.get("success", False),
                         attempts=info.get("attempts", 0), sim_time=info.get("sim_time", np.nan),
                         commit_err_mm=np.mean(info["commit_err_mm"]) if info.get("commit_err_mm") else np.nan,
                         belief_std_at_commit=(np.mean(info["belief_std_at_commit"])
                                               if info.get("belief_std_at_commit") else np.nan),
                         mean_decision_s=float(np.mean(times)), mean_decision_cpu_s=float(np.mean(cpus)), **{"return": ret}))
        if partial:
            partial.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(partial, index=False)
    df = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if partial.exists():
            os.remove(partial)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=10)
    ap.add_argument("--wm", default=None, help="world-model .npz: plan with WMPlanner instead of physics")
    ap.add_argument("--out-dir", default="runs/eval")
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--heuristic", action="store_true", help="no-search ablation: rollout policy on the belief mean")
    a = ap.parse_args()
    if a.heuristic:
        from pomdp_pick.planning.pomcp import MeanHeuristic
        out = Path(a.out_dir) / f"heuristic_seed0_n{a.episodes}.csv"
        if not out.exists():
            evaluate_pomcp(0, a.sigma, a.episodes, out, max_decisions=a.max_decisions,
                           make_planner=lambda rng: MeanHeuristic())
            print("wrote", out, flush=True)
        return
    for n in a.sims:
        if a.wm:
            from pomdp_pick.planning.pomcp import WMPlanner
            from pomdp_pick.planning.world_model import WorldModel
            wm = WorldModel.load(a.wm)
            out = Path(a.out_dir) / f"wmpomcp{n}_seed0_n{a.episodes}.csv"
            make = (lambda n_: lambda rng: WMPlanner(wm, n_, rng=rng))(n)
        else:
            out = Path(a.out_dir) / f"pomcp{n}_seed0_n{a.episodes}.csv"
            make = None
        if not out.exists():
            evaluate_pomcp(n, a.sigma, a.episodes, out, max_decisions=a.max_decisions, make_planner=make)
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
