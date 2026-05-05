#!/usr/bin/env python3
"""Noise ablation: evaluate plain vs belief PPO across a range of sigma_ep values.

Pins sigma_ep to a fixed value each sweep by setting sigma_low=sigma_high=X,
so every episode in that batch runs at exactly the same noise level.

Usage:
    python3 eval_noise_sweep.py                        # default 8-level sweep
    python3 eval_noise_sweep.py --episodes 100         # more episodes per level
    python3 eval_noise_sweep.py --sigmas 0.003 0.01 0.02 0.04   # custom levels
    python3 eval_noise_sweep.py --save results/noise_sweep.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from so_arm101_control.lego_pick_env import LegoPickEnv


def make_env(belief_mode, sigma, seed):
    def _init():
        env = LegoPickEnv(
            belief_mode=belief_mode,
            use_camera_noise=True,
            sigma_low=sigma,
            sigma_high=sigma,  # pin to exact value
        )
        env.reset(seed=seed)
        return env
    return _init


def evaluate_policy(model, vec_norm, belief_mode, sigma, n_episodes, seed):
    env = DummyVecEnv([make_env(belief_mode, sigma, seed)])
    env = VecNormalize.load(vec_norm, env)
    env.training = False
    env.norm_reward = False

    successes = 0
    wrong_grasps = 0
    timeouts = 0
    grasped_target = 0  # episodes where target was successfully grasped
    ep_rewards = []
    ep_lengths = []

    obs = env.reset()
    ep_r = 0.0
    ep_len = 0
    ep_grabbed_target = False

    while len(ep_rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_r += reward[0]
        ep_len += 1

        # Track if target was grabbed this episode
        if info[0].get("grasp_result") == "red_lego_2x4":
            ep_grabbed_target = True

        if done[0]:
            success = info[0].get("success", False)
            truncated = ep_len >= 200  # MAX_STEPS
            grasp_result = info[0].get("grasp_result")

            successes += int(success)
            timeouts += int(not success and truncated)
            # Wrong grasp: terminated early due to distractor grab
            wrong_grasps += int(not success and not truncated and grasp_result is not None and not ep_grabbed_target)
            grasped_target += int(ep_grabbed_target)

            ep_rewards.append(ep_r)
            ep_lengths.append(ep_len)
            ep_r = 0.0
            ep_len = 0
            ep_grabbed_target = False
            obs = env.reset()

    env.close()
    return {
        "success_rate": successes / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "wrong_grasp_rate": wrong_grasps / n_episodes,
        "grasp_rate": grasped_target / n_episodes,
        "mean_reward": np.mean(ep_rewards),
        "std_reward": np.std(ep_rewards),
        "mean_ep_len": np.mean(ep_lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.003, 0.005, 0.008, 0.010, 0.013, 0.016, 0.020, 0.030],
                        help="Noise levels (sigma_ep) to sweep over")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per noise level per policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save results .npz (optional)")
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.join(scripts_dir, '..')

    def _resolve_model(name):
        # Prefer the package-level models/ (latest training run), fall back to scripts/models/
        candidates = [
            os.path.join(pkg_dir, 'models', name),
            os.path.join(scripts_dir, 'models', name),
        ]
        for c in candidates:
            if os.path.isfile(os.path.join(c, 'best_model.zip')):
                return os.path.normpath(c)
        return os.path.normpath(candidates[0])

    configs = [
        ("plain",  False, _resolve_model("ppo_plain")),
        ("belief", True,  _resolve_model("ppo_belief")),
    ]

    print(f"\nNoise sweep: {len(args.sigmas)} levels × {args.episodes} episodes × 2 policies")
    print(f"Sigmas (mm): {[f'{s*1000:.1f}' for s in args.sigmas]}\n")

    all_results = {}

    for label, belief_mode, model_dir in configs:
        model = PPO.load(os.path.join(model_dir, "best_model"), device="cpu")
        norm_path = os.path.join(model_dir, "vec_normalize.pkl")

        results = []
        print(f"{'─'*80}")
        print(f"  {label.upper()} PPO")
        print(f"{'─'*80}")
        print(f"  {'sigma_ep (mm)':>14} | {'success%':>8} | {'grasped%':>8} | {'timeout%':>8} | {'wronggrsp%':>10} | {'ep_len':>7}")
        print(f"  {'─'*14}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*10}-+-{'─'*7}")

        for sigma in args.sigmas:
            r = evaluate_policy(model, norm_path, belief_mode, sigma, args.episodes, args.seed)
            results.append(r)
            print(f"  {sigma*1000:>12.1f}mm | {r['success_rate']*100:>7.1f}% | "
                  f"{r['grasp_rate']*100:>7.1f}% | {r['timeout_rate']*100:>7.1f}% | "
                  f"{r['wrong_grasp_rate']*100:>9.1f}% | {r['mean_ep_len']:>7.1f}")

        all_results[label] = results
        print()

    # Summary: delta between belief and plain
    print(f"{'─'*80}")
    print(f"  BELIEF - PLAIN  (positive = belief better)")
    print(f"{'─'*80}")
    print(f"  {'sigma_ep (mm)':>14} | {'Δsuccess%':>10} | {'Δgrasp%':>8} | {'Δmean_r':>9}")
    print(f"  {'─'*14}-+-{'─'*10}-+-{'─'*8}-+-{'─'*9}")
    for i, sigma in enumerate(args.sigmas):
        bp = all_results["belief"][i]
        pp = all_results["plain"][i]
        delta_s = (bp["success_rate"] - pp["success_rate"]) * 100
        delta_g = (bp["grasp_rate"] - pp["grasp_rate"]) * 100
        delta_r = bp["mean_reward"] - pp["mean_reward"]
        print(f"  {sigma*1000:>12.1f}mm | {delta_s:>+9.1f}% | {delta_g:>+7.1f}% | {delta_r:>+9.1f}")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        np.savez(
            args.save,
            sigmas=np.array(args.sigmas),
            plain_success=np.array([r["success_rate"] for r in all_results["plain"]]),
            belief_success=np.array([r["success_rate"] for r in all_results["belief"]]),
            plain_reward=np.array([r["mean_reward"] for r in all_results["plain"]]),
            belief_reward=np.array([r["mean_reward"] for r in all_results["belief"]]),
            plain_reward_std=np.array([r["std_reward"] for r in all_results["plain"]]),
            belief_reward_std=np.array([r["std_reward"] for r in all_results["belief"]]),
        )
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
