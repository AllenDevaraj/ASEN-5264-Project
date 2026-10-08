"""Render the showcase clips (1280x720) from recorded episodes; record a POMCP episode if asked.

  ./vpy tools/showcase.py --shot hero      --src runs/footage/preview/belief_s0_sig10_ep6.npz
  ./vpy tools/showcase.py --shot recovery  --src runs/footage/preview/belief_s0_sig10_ep3.npz
  ./vpy tools/showcase.py --shot pomcp --sims 32 --tries 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pomdp_pick.footage import record as R  # noqa: E402
from pomdp_pick.footage.render import render_episode  # noqa: E402

OUT = Path("runs/footage/showcase")
TEXT = {"hero": ("Belief PPO · physical grasp", None),
        "recovery": ("Belief PPO · miss, re-localize, grasp", None),
        "pomcp": ("POMCP · {n} physics simulations per decision", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", choices=list(TEXT), required=True)
    ap.add_argument("--src")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=10.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    title = TEXT[a.shot][0].format(n=a.sims)
    src = Path(a.src) if a.src else None
    if a.shot == "pomcp":
        from pomdp_pick.envs.pick_env import PickPlaceEnv
        from pomdp_pick.eval.evaluate import SEED_BASE
        from pomdp_pick.planning.pomcp import Planner
        env = PickPlaceEnv("belief", sigma_fixed=a.sigma / 1000)
        for i in range(a.tries):
            src = OUT / f"pomcp{a.sims}_sig{a.sigma:g}_ep{i}.npz"
            if not src.exists():
                act = R.planner_actor(Planner(a.sims, rng=np.random.default_rng(SEED_BASE + i)))
                R.save(R.record_episode(env, act, SEED_BASE + i), src)
            ok = bool(np.load(src)["success"])
            print(src.name, "success", ok, flush=True)
            if ok:
                break
    out = OUT / f"{a.shot}.mp4"
    render_episode(src, out, title)
    print("wrote", out, "from", src, flush=True)


if __name__ == "__main__":
    main()
