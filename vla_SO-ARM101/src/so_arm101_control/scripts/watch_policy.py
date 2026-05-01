#!/usr/bin/env python3
"""Watch a trained PPO policy (plain or belief) in the MuJoCo sim viewer.

Usage:
    python3 watch_policy.py --mode plain
    python3 watch_policy.py --mode belief
    python3 watch_policy.py --mode plain --episodes 10 --camera-noise
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from so_arm101_control.lego_pick_env import LegoPickEnv


def make_env(belief_mode, camera_noise, use_overhead_camera, seed=0):
    def _init():
        env = LegoPickEnv(
            belief_mode=belief_mode,
            use_camera_noise=camera_noise,
            use_overhead_camera=use_overhead_camera,
            render_mode="human",
        )
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plain", "belief"], default="plain")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--camera-noise", action="store_true",
                        help="Enable camera noise (match training config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-delay", type=float, default=0.04,
                        help="Seconds between steps (lower = faster)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Override model dir (absolute or relative to scripts/). "
                             "Default: models/ppo_{mode}")
    parser.add_argument("--use-overhead-camera", action="store_true",
                        help="Enable overhead camera. Required for reproducing "
                             "pre-POMDP models from models_old/occlusion/.")
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model_dir:
        model_dir = args.model_dir if os.path.isabs(args.model_dir) \
            else os.path.join(scripts_dir, args.model_dir)
    else:
        model_dir = os.path.join(scripts_dir, f"models/ppo_{args.mode}")
    model_path = os.path.join(model_dir, "best_model.zip")
    norm_path  = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(norm_path):
        print(f"ERROR: vec_normalize.pkl not found at {norm_path}")
        sys.exit(1)

    belief_mode = (args.mode == "belief")

    # Wrap in VecNormalize and load saved stats so obs normalization matches training
    env = DummyVecEnv([make_env(belief_mode, args.camera_noise,
                                 args.use_overhead_camera, seed=args.seed)])
    env = VecNormalize.load(norm_path, env)
    env.training = False   # freeze running stats — eval mode
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    mode_label = "Belief PPO (particle filter)" if belief_mode else "Plain PPO (raw noisy obs)"
    print(f"\n{'='*60}")
    print(f"  Mode:         {mode_label}")
    print(f"  Camera noise: {args.camera_noise}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Model:        {model_path}")
    print(f"{'='*60}\n")

    successes = 0
    rewards = []
    ep_reward = 0.0

    obs = env.reset()
    ep = 0

    try:
        while ep < args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]

            # Access the underlying env for render
            raw_env = env.envs[0]
            raw_env.render()

            if raw_env._viewer is not None and not raw_env._viewer.is_running():
                print("Viewer closed.")
                break

            time.sleep(args.step_delay)

            if done[0]:
                # No Monitor wrapper here — info[0] is the raw env info dict
                success = info[0].get("success", False)
                successes += int(success)
                rewards.append(ep_reward)
                result = "SUCCESS" if success else "fail"
                print(f"  Episode {ep+1}/{args.episodes}: {result}  reward={ep_reward:.1f}")
                ep += 1
                ep_reward = 0.0
                obs = env.reset()

    finally:
        env.close()

    if rewards:
        print(f"\n{'='*60}")
        print(f"  Results: {successes}/{len(rewards)} successes ({100*successes/len(rewards):.0f}%)")
        print(f"  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
