#!/usr/bin/env python3
"""Custom callback that logs per-episode uncertainty trajectories during eval.

Captures how each of the uncertainty sources evolves over an episode:
  1. Episodic sigma (wrist noise level, constant per episode)
  2. Belief sigma convergence (PF sigma shrinks as agent approaches)
  3. Occlusion events (wrist + overhead camera blocked by distractor)
  4. Effective sigma (distance-dependent wrist noise)

Saves periodic snapshots as .npz files in the log directory.
"""

import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrajectoryLoggerCallback(BaseCallback):
    """Log per-step uncertainty data during periodic evaluation rollouts.

    Runs n_eval_episodes every eval_freq steps using the training env's
    current policy, and records full step-level info dicts.

    Saves to {log_dir}/trajectory_data_{step}.npz at each eval, plus
    a running trajectory_data_latest.npz that always has the most recent.
    """

    def __init__(
        self,
        eval_env,
        log_dir,
        eval_freq=50_000,
        n_eval_episodes=20,
        belief_mode=False,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.belief_mode = belief_mode
        self._last_eval_step = -1

    def _on_step(self):
        # Check if it's time to evaluate
        step = self.num_timesteps
        if step - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = step

        episodes = self._run_eval()
        self._save(episodes, step)
        return True

    def _sync_vec_normalize(self):
        """Sync VecNormalize stats from training env to trajectory eval env."""
        if hasattr(self.training_env, "obs_rms"):
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms

    def _run_eval(self):
        """Run evaluation episodes and collect trajectory data."""
        self._sync_vec_normalize()
        episodes = []

        for ep_idx in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            episode = {
                "steps": [],
                "sigma_ep": None,
                "total_reward": 0.0,
                "success": False,
                "ep_length": 0,
            }
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done_arr, infos = self.eval_env.step(action)
                done = done_arr[0]
                info = infos[0]

                step_data = {
                    "step": info.get("step", 0),
                    "reward": info.get("reward", float(reward[0])),
                    "ee_pos": info.get("ee_pos", np.zeros(3)),
                    "true_block_pos": info.get("true_block_pos", np.zeros(3)),
                    "dist_to_block": info.get("dist_to_block", 0.0),
                    "dist_to_goal": info.get("dist_to_goal", 0.0),
                    "holding": info.get("holding", False),
                    "grasp_result": info.get("grasp_result", None),
                    "wrist_occluded": info.get("wrist_occluded", False),
                    "overhead_occluded": info.get("overhead_occluded", False),
                    "effective_sigma": info.get("effective_sigma", 0.0),
                }

                if self.belief_mode:
                    step_data["belief_mu"] = info.get(
                        "belief_mu", np.zeros(3)
                    )
                    step_data["belief_sigma"] = info.get(
                        "belief_sigma", np.zeros(3)
                    )

                episode["steps"].append(step_data)
                episode["total_reward"] += float(reward[0])
                episode["sigma_ep"] = info.get("sigma_ep", 0.0)
                if info.get("success", False):
                    episode["success"] = True

            episode["ep_length"] = len(episode["steps"])
            episodes.append(episode)

        return episodes

    def _save(self, episodes, step):
        """Save trajectory data as .npz with structured arrays."""
        # Flatten into arrays for efficient storage
        n_episodes = len(episodes)
        max_steps = max(ep["ep_length"] for ep in episodes)

        # Per-episode scalars
        sigma_eps = np.array([ep["sigma_ep"] for ep in episodes])
        successes = np.array([ep["success"] for ep in episodes])
        total_rewards = np.array([ep["total_reward"] for ep in episodes])
        ep_lengths = np.array([ep["ep_length"] for ep in episodes])

        # Per-step arrays (padded to max_steps, NaN for padding)
        rewards = np.full((n_episodes, max_steps), np.nan)
        dist_to_block = np.full((n_episodes, max_steps), np.nan)
        dist_to_goal = np.full((n_episodes, max_steps), np.nan)
        ee_pos = np.full((n_episodes, max_steps, 3), np.nan)
        true_block_pos = np.full((n_episodes, max_steps, 3), np.nan)
        holding = np.full((n_episodes, max_steps), False)
        wrist_occluded = np.full((n_episodes, max_steps), False)
        overhead_occluded = np.full((n_episodes, max_steps), False)
        effective_sigma = np.full((n_episodes, max_steps), np.nan)
        grasp_events = np.full((n_episodes, max_steps), 0, dtype=np.int8)
        # 0=none, 1=success, -1=fail

        if self.belief_mode:
            belief_mu = np.full((n_episodes, max_steps, 3), np.nan)
            belief_sigma = np.full((n_episodes, max_steps, 3), np.nan)

        for i, ep in enumerate(episodes):
            for j, s in enumerate(ep["steps"]):
                rewards[i, j] = s["reward"]
                dist_to_block[i, j] = s["dist_to_block"]
                dist_to_goal[i, j] = s["dist_to_goal"]
                ee_pos[i, j] = s["ee_pos"]
                true_block_pos[i, j] = s["true_block_pos"]
                holding[i, j] = s["holding"]
                wrist_occluded[i, j] = s["wrist_occluded"]
                overhead_occluded[i, j] = s["overhead_occluded"]
                effective_sigma[i, j] = s["effective_sigma"]
                if s["grasp_result"] == "success":
                    grasp_events[i, j] = 1
                elif s["grasp_result"] == "fail":
                    grasp_events[i, j] = -1

                if self.belief_mode:
                    belief_mu[i, j] = s["belief_mu"]
                    belief_sigma[i, j] = s["belief_sigma"]

        save_dict = {
            "step": np.array(step),
            "sigma_eps": sigma_eps,
            "successes": successes,
            "total_rewards": total_rewards,
            "ep_lengths": ep_lengths,
            "rewards": rewards,
            "dist_to_block": dist_to_block,
            "dist_to_goal": dist_to_goal,
            "ee_pos": ee_pos,
            "true_block_pos": true_block_pos,
            "holding": holding,
            "wrist_occluded": wrist_occluded,
            "overhead_occluded": overhead_occluded,
            "effective_sigma": effective_sigma,
            "grasp_events": grasp_events,
        }

        if self.belief_mode:
            save_dict["belief_mu"] = belief_mu
            save_dict["belief_sigma"] = belief_sigma

        # Save timestamped + latest
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"trajectory_data_{step}.npz")
        np.savez_compressed(path, **save_dict)

        latest = os.path.join(self.log_dir, "trajectory_data_latest.npz")
        np.savez_compressed(latest, **save_dict)

        if self.verbose:
            n_success = successes.sum()
            print(
                f"[TrajectoryLogger] step={step}: "
                f"{n_success}/{n_episodes} success, "
                f"mean_reward={total_rewards.mean():.1f}"
            )
