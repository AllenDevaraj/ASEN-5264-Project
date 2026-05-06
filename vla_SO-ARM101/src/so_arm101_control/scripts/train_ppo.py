#!/usr/bin/env python3
"""Phase 2: Train Plain PPO baseline on LegoPickEnv.

Uses 13D observation (joints, noisy block pose, goal, flags) without particle filter.
MLP policy with 2x256 hidden layers.

Usage:
    python3 train_ppo.py
    python3 train_ppo.py --timesteps 500000 --n-envs 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from trajectory_callback import TrajectoryLoggerCallback


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats whenever EvalCallback saves a new best model."""

    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self):
        # Check if EvalCallback just saved a new best model
        best_model_path = os.path.join(self.save_path, "best_model.zip")
        norm_path = os.path.join(self.save_path, "vec_normalize.pkl")
        if os.path.exists(best_model_path):
            model_mtime = os.path.getmtime(best_model_path)
            norm_mtime = os.path.getmtime(norm_path) if os.path.exists(norm_path) else 0
            if model_mtime > norm_mtime:
                self.training_env.save(norm_path)
                if self.verbose:
                    print(f"Saved VecNormalize to {norm_path}")
        return True


def make_env(rank, seed=0, belief_mode=False, use_camera_noise=False, sigma_drift=0.0):
    def _init():
        from so_arm101_control.lego_pick_env import LegoPickEnv
        env = LegoPickEnv(
            belief_mode=belief_mode,
            use_camera_noise=use_camera_noise,
            sigma_drift=sigma_drift,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Plain PPO on LegoPickEnv")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--camera-noise", action="store_true", default=True)
    parser.add_argument("--sigma-drift", type=float, default=0.002,
                        help="Target block drift per step in meters (0=disabled)")
    parser.add_argument("--output-dir", type=str, default="./models/ppo_plain")
    parser.add_argument("--log-dir", type=str, default="./logs/ppo_plain")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create vectorized training env
    env = SubprocVecEnv([
        make_env(i, args.seed, belief_mode=False,
                 use_camera_noise=args.camera_noise, sigma_drift=args.sigma_drift)
        for i in range(args.n_envs)
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create eval env (must also be wrapped in VecNormalize to match training env)
    eval_env = DummyVecEnv([make_env(0, args.seed + 100, belief_mode=False,
                                     use_camera_noise=args.camera_noise, sigma_drift=args.sigma_drift)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Linear decay schedules: start high for exploration, end near zero for fine-tuning
    def linear_schedule(initial, final=0.0):
        def schedule(progress_remaining):
            return final + progress_remaining * (initial - final)
        return schedule

    # PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
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
    save_norm_callback = SaveVecNormalizeCallback(save_path=args.output_dir, verbose=1)

    # Trajectory logger — records per-step uncertainty data during eval
    traj_eval_env = DummyVecEnv([make_env(0, args.seed + 200, belief_mode=False,
                                          use_camera_noise=args.camera_noise, sigma_drift=args.sigma_drift)])
    traj_eval_env = VecNormalize(traj_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    traj_dir = os.path.join(args.log_dir, "trajectories")
    traj_callback = TrajectoryLoggerCallback(
        eval_env=traj_eval_env,
        log_dir=traj_dir,
        eval_freq=200_000,  # Every 200k steps (10 snapshots over 2M)
        n_eval_episodes=20,
        belief_mode=False,
        verbose=1,
    )

    print(f"Training Plain PPO for {args.timesteps} steps...")
    print(f"  Envs: {args.n_envs}, Camera noise: {args.camera_noise}")
    print(f"  Output: {args.output_dir}, Logs: {args.log_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, save_norm_callback, traj_callback],
        progress_bar=True,
    )

    # Save final model (vec_normalize.pkl already saved by callback with best_model)
    model.save(os.path.join(args.output_dir, "final"))
    env.save(os.path.join(args.output_dir, "vec_normalize_final.pkl"))
    print(f"Model saved to {args.output_dir}/final.zip")

    env.close()
    eval_env.close()
    traj_eval_env.close()


if __name__ == "__main__":
    main()
