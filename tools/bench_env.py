#!/usr/bin/env python3
"""Measure env steps/s: single process, then N SubprocVecEnv workers (default 4)."""
import sys
import time

import numpy as np
import torch


def _make(mode):
    def f():
        torch.set_num_threads(1)
        from pomdp_pick.envs.pick_env import PickPlaceEnv
        return PickPlaceEnv(mode)
    return f


def single(mode, steps=600):
    env = _make(mode)()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    vis = 0
    t = time.perf_counter()
    for _ in range(steps):
        a = rng.uniform(-1, 1, 5)
        a[4] = -1 if rng.random() > 0.03 else 1
        out = env.step(a)
        vis += out[4]["visible"]
        if out[2] or out[3]:
            env.reset()
    return steps / (time.perf_counter() - t), vis / steps


def vec(mode, n, steps=1200):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([_make(mode) for _ in range(n)], start_method="forkserver")
    venv.reset()
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(steps // n):
        a = rng.uniform(-1, 1, (n, 5))
        a[:, 4] = np.where(rng.random(n) > 0.03, -1, 1)
        venv.step(a)
    r = steps / (time.perf_counter() - t)
    venv.close()
    return r


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for mode in ("plain", "belief"):
        sps, vis = single(mode)
        print(f"{mode}: single {sps:.0f} steps/s (visible {vis:.0%} of steps, random policy), "
              f"{n} workers {vec(mode, n):.0f} steps/s", flush=True)
