#!/usr/bin/env python3
"""Phase 3: Train Belief-Augmented PPO on LegoPickEnv.

Uses 12D observation (particle filter belief: mu, sigma) instead of raw noisy obs.
Same PPO hyperparameters as Plain PPO for fair comparison.

Usage:
    python3 train_belief_ppo.py
    python3 train_belief_ppo.py --timesteps 500000 --n-envs 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def make_env(rank, seed=0, use_camera_noise=False):
    def _init():
        from so_arm101_control.lego_pick_env import LegoPickEnv
        env = LegoPickEnv(
            belief_mode=True,
            use_camera_noise=use_camera_noise,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Belief PPO on LegoPickEnv")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--camera-noise", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./models/ppo_belief")
    parser.add_argument("--log-dir", type=str, default="./logs/ppo_belief")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create vectorized training env
    env = SubprocVecEnv([
        make_env(i, args.seed, use_camera_noise=args.camera_noise)
        for i in range(args.n_envs)
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create eval env (must also be wrapped in VecNormalize to match training env)
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([make_env(0, args.seed + 100, use_camera_noise=args.camera_noise)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Same PPO hyperparams as Plain PPO for fair comparison
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.output_dir,
        log_path=args.log_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    print(f"Training Belief-Augmented PPO for {args.timesteps} steps...")
    print(f"  Envs: {args.n_envs}, Camera noise: {args.camera_noise}")
    print(f"  Obs: 12D (mu_x, mu_y, mu_theta, sigma_x, sigma_y, sigma_theta)")
    print(f"  Output: {args.output_dir}, Logs: {args.log_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(os.path.join(args.output_dir, "final"))
    env.save(os.path.join(args.output_dir, "vec_normalize.pkl"))
    print(f"Model saved to {args.output_dir}/final.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
