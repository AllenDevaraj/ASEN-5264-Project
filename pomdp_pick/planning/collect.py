"""Collect world-model training data: (belief summary, action) -> (Δsummary, grasp held).

Behaviour policy: the trained belief-PPO policy, with probability `eps` replaced by a random
macro action so the model sees every action the planner will query."""
import argparse
from pathlib import Path

import numpy as np

from pomdp_pick.envs.pick_env import PickPlaceEnv
from pomdp_pick.planning import sim_bridge as sb
from pomdp_pick.planning.world_model import state_of

SEED_BASE = 900_000_000   # disjoint from training (seed·10M + ts) and eval (10 000 + i) seeds


def collect(run_dir, n_steps, out_npz, policy=None, eps=0.3, seed=0):
    if policy is None:
        from pomdp_pick.agents.methods import load_policy
        policy = load_policy(run_dir)
    rng = np.random.default_rng(seed)
    env = PickPlaceEnv("belief")
    X, Y, E, ep = [], [], [], 0
    obs, _ = env.reset(seed=SEED_BASE + seed * 1_000_000 + ep)
    policy.reset()
    while len(X) < n_steps:
        a = policy.act(obs)          # always advance the policy's own memory
        if rng.random() < eps:
            a = sb.macro_action(int(rng.integers(len(sb.MACROS))))
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        held_before = env.held
        obs2, _, term, trunc, _ = env.step(a)
        commit = a[4] > 0.5
        X.append(np.concatenate([state_of(obs), sb.MACROS[sb.COMMIT] if commit else a]))
        Y.append(np.concatenate([state_of(obs2) - state_of(obs), [float(commit and env.held and not held_before)]]))
        E.append(ep)
        obs = obs2
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=SEED_BASE + seed * 1_000_000 + ep)
            policy.reset()
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32), E=np.asarray(E, np.int64))
    return out_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/belief/seed0")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--out", default="runs/world_model/data.npz")
    ap.add_argument("--fit-epochs", type=int, default=40)
    a = ap.parse_args()
    from pomdp_pick.planning.world_model import WorldModel, write_report
    out = Path(a.out)
    if not out.exists():
        collect(a.run, a.steps, out)
    f = np.load(out)
    wm = WorldModel()
    rep = wm.fit(f["X"], f["Y"], epochs=a.fit_epochs, groups=f["E"])
    wm.save(out.parent / "wm.npz")
    write_report(rep, out.parent / "report.json")
    print({k: v for k, v in rep.items() if not isinstance(v, list)}, flush=True)


if __name__ == "__main__":
    main()
