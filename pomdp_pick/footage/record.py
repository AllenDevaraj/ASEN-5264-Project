"""Record a real PickPlaceEnv episode at 30 fps of sim time for offline rendering.

Every frame is the physics state itself (no interpolation): the arm's frame callback copies qpos
and the belief overlays at 1/30 s of simulated time, through moves, grasps and the place.
"""
import argparse
from pathlib import Path

import numpy as np

from pomdp_pick.robot.controller import FRAME_DT

FPS = round(1.0 / FRAME_DT)
N_SHOW = 150            # particles drawn per frame (evenly spaced subset)


class _Recorder:
    def __init__(self, env):
        self.env, self.rows, self.decision, self.commit = env, [], 0, False

    def __call__(self, arm):
        env, pf = self.env, self.env.pf
        P = pf.particles
        idx = np.linspace(0, len(P) - 1, N_SHOW).astype(int)
        reading = env.last_reading if env.last_reading is not None else np.full(3, np.nan)
        self.rows.append(dict(t=arm.sim_time - env.t0, qpos=arm.d.qpos.copy(), particles=P[idx, :2].copy(),
                              mu=np.asarray(pf.summary()["mu"], float), reading=np.asarray(reading, float),
                              visible=bool(env.visible), decision=self.decision, commit=self.commit,
                              attempts=env.attempts, p=arm.p.copy()))


def record_episode(env, act, seed, max_decisions=400):
    """act(obs, env) -> 5-D action. Returns per-frame arrays plus episode metadata."""
    env.frame_cb = None
    obs, info = env.reset(seed=seed)
    rec = _Recorder(env)
    env.frame_cb = rec
    env.arm._next_frame = env.arm.sim_time
    rec(env.arm)                                   # opening still, after the belief exists
    env.arm._next_frame = env.arm.sim_time + FRAME_DT
    ret = 0.0
    for k in range(max_decisions):
        env.frame_cb = None                        # a planner's simulated steps are not footage
        a = np.asarray(act(obs, env), float)
        env.frame_cb = rec
        rec.decision, rec.commit = k, bool(a[4] > 0.5)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if term or trunc:
            break
    env.frame_cb = None
    out = {k: np.asarray([row[k] for row in rec.rows]) for k in rec.rows[0]}
    out.update(goal=np.asarray(env.goal, float), success=bool(info["success"]), ret=ret, seed=seed,
               sigma_ep=float(env.sigma_ep), obs_mode=env.obs_mode, commit_err_mm=np.asarray(info["commit_err_mm"]))
    return out


def policy_actor(run_dir):
    from pomdp_pick.agents.methods import load_policy
    pol = load_policy(run_dir)
    state = {"seen": None}

    def act(obs, env):
        if state["seen"] is not env.t0:            # new episode
            pol.reset()
            state["seen"] = env.t0
        return pol.act(obs)
    return pol.spec.obs_mode, act


def planner_actor(planner):
    from pomdp_pick.planning import sim_bridge as sb
    return lambda obs, env: sb.macro_action(planner.plan(env)[0])


def save(rec, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in rec.items()})


def main():
    from pomdp_pick.envs.pick_env import PickPlaceEnv
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="trained run dir (e.g. runs/belief/seed0)")
    ap.add_argument("--pomcp", type=int, help="physics POMCP with this many sims instead of --run")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sigma", type=float, default=10.0, help="camera noise sigma_ep in mm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.pomcp:
        from pomdp_pick.planning.pomcp import Planner
        mode, act = "belief", planner_actor(Planner(a.pomcp, rng=np.random.default_rng(a.seed)))
    else:
        mode, act = policy_actor(a.run)
    rec = record_episode(PickPlaceEnv(mode, sigma_fixed=a.sigma / 1000), act, a.seed)
    save(rec, a.out)
    print(a.out, "success", rec["success"], "frames", len(rec["t"]), flush=True)


if __name__ == "__main__":
    main()
