"""Shared-seed deterministic evaluation of trained runs -> one CSV per (method, seed)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pomdp_pick.agents.methods import METHODS, load_policy
from pomdp_pick.envs.pick_env import PickPlaceEnv

EVAL_SIGMAS_MM = (2, 5, 10, 15, 20, 30)
SEED_BASE = 10_000


def evaluate_run(run_dir, sigmas_mm=EVAL_SIGMAS_MM, n_episodes=200, out_csv=None):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "progress.json").read_text())
    pol = load_policy(run_dir)
    rows = []
    for s_mm in sigmas_mm:
        env = PickPlaceEnv(METHODS[meta["method"]].obs_mode, sigma_fixed=s_mm / 1000.0)
        for ep in range(n_episodes):
            obs, info = env.reset(seed=SEED_BASE + ep)
            pol.reset()
            done, ret = False, 0.0
            while not done:
                obs, r, term, trunc, info = env.step(pol.act(obs))
                ret += r
                done = term or trunc
            rows.append(dict(method=meta["method"], seed=meta["seed"], sigma_mm=s_mm, episode=ep,
                             success=info["success"], attempts=info["attempts"], sim_time=info["sim_time"],
                             commit_err_mm=np.mean(info["commit_err_mm"]) if info["commit_err_mm"] else np.nan,
                             belief_std_at_commit=np.mean(info["belief_std_at_commit"]) if info["belief_std_at_commit"] else np.nan,
                             **{"return": ret}))
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for prog in sorted(Path(a.runs).glob("*/seed*/progress.json")):
        meta = json.loads(prog.read_text())
        if not meta.get("done") or (a.only and meta["method"] not in a.only):
            continue
        out = Path(a.runs) / "eval" / f"{meta['method']}_seed{meta['seed']}.csv"
        if out.exists():
            continue
        evaluate_run(prog.parent, n_episodes=a.episodes, out_csv=out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
