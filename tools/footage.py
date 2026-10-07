"""Record the same shared eval seeds for several trained methods, then render a contrast pair.

  ./vpy tools/footage.py --methods belief plain --sigma 10 --n 12 --out runs/footage/preview

Picks the first seed where the first method succeeds and the second does not, renders both at
960x540 and stacks them side by side (left = first method).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pomdp_pick.envs.pick_env import PickPlaceEnv  # noqa: E402
from pomdp_pick.eval.evaluate import SEED_BASE  # noqa: E402
from pomdp_pick.footage import record as R  # noqa: E402

TITLES = {"oracle": "Oracle PPO (sees the true pose)", "plain": "Plain PPO (latest noisy reading)",
          "framestack": "Frame-stack PPO (last 4 readings)", "recurrent": "Recurrent PPO (LSTM)",
          "belief": "Belief PPO (particle filter)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs=2, default=["belief", "plain"])
    ap.add_argument("--run-seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="runs/footage/preview")
    a = ap.parse_args()
    out = Path(a.out)
    results = {}
    for m in a.methods:
        mode, act = R.policy_actor(f"runs/{m}/seed{a.run_seed}")
        env = PickPlaceEnv(mode, sigma_fixed=a.sigma / 1000)
        for i in range(a.n):
            p = out / f"{m}_ep{i}.npz"
            if not p.exists():
                R.save(R.record_episode(env, act, SEED_BASE + i), p)
            import numpy as np
            results[(m, i)] = bool(np.load(p)["success"])
        print(m, [int(results[(m, i)]) for i in range(a.n)], flush=True)
    a_m, b_m = a.methods
    pick = next((i for i in range(a.n) if results[(a_m, i)] and not results[(b_m, i)]), None)
    if pick is None:
        print("no contrast seed found", flush=True)
        return
    from pomdp_pick.footage.render import render_episode, side_by_side
    clips = [render_episode(out / f"{m}_ep{pick}.npz", out / f"{m}_ep{pick}.mp4", TITLES[m], w=960, h=540)
             for m in a.methods]
    side_by_side(*clips, out / f"{a_m}_vs_{b_m}_ep{pick}.mp4")
    print("wrote", out / f"{a_m}_vs_{b_m}_ep{pick}.mp4", flush=True)


if __name__ == "__main__":
    main()
