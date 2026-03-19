#!/usr/bin/env python3
"""Policy inference bridge for running trained PPO in the ROS2/MuJoCo stack.

Encapsulates model loading, observation construction (with noise), normalization,
and action inference. No ROS2 dependency — the GUI feeds raw data and gets actions back.

Usage:
    runner = PolicyRunner('models/ppo_plain', belief_mode=False)
    runner.reset_episode()
    obs = runner.build_observation(joints, block_poses, ee_pos, goal_xy, holding)
    action = runner.predict(obs)  # (dx, dy, dz, gripper_cmd)
    new_joints = runner.ik_step(ee_pos, action)
"""

import math
import os
import pickle

import numpy as np
from stable_baselines3 import PPO

from so_arm101_control.compute_workspace import (
    ARM_JOINT_NAMES,
    forward_kinematics,
    geometric_ik,
)
from so_arm101_control.occlusion import is_occluded_overhead
from so_arm101_control.particle_filter import ParticleFilter

# Constants matching lego_pick_env.py
TARGET_BLOCK = "red_lego_2x4"
DISTRACTOR_BLOCK = "blue_lego_2x2"
BLOCK_NAMES = [TARGET_BLOCK, DISTRACTOR_BLOCK]
HALF_SIZES = {
    "red_lego_2x4": (0.016, 0.008),
    "blue_lego_2x2": (0.008, 0.008),
}
TABLE_Z = 0.0055
MIN_SPACING = 0.03
SPAWN_R_MIN = 0.12
SPAWN_R_MAX = 0.22
SPAWN_ANGLE_MIN = -1.0
SPAWN_ANGLE_MAX = 1.0
SIGMA_OVERHEAD = 0.005
SIGMA_LOW = 0.003
SIGMA_HIGH = 0.020
EE_R_MIN = 0.09
EE_R_MAX = 0.31
EE_Z_MIN = 0.002
EE_Z_MAX = 0.12
GRIPPER_OPEN = -0.174533
GRIPPER_CLOSED = 1.74533


class PolicyRunner:
    """Loads a trained PPO model and runs inference with proper obs construction."""

    def __init__(self, model_dir, belief_mode=False):
        self.belief_mode = belief_mode
        self.rng = np.random.default_rng()

        # Load model
        model_path = os.path.join(model_dir, 'best_model.zip')
        self.model = PPO.load(model_path, device='cpu')

        # Load VecNormalize stats for manual normalization
        norm_path = os.path.join(model_dir, 'vec_normalize.pkl')
        with open(norm_path, 'rb') as f:
            vec_norm = pickle.load(f)
        self.obs_mean = vec_norm.obs_rms.mean.astype(np.float32)
        self.obs_var = vec_norm.obs_rms.var.astype(np.float32)
        self.clip_obs = vec_norm.clip_obs  # 10.0

        # Particle filter for belief mode
        self.pf = None
        if belief_mode:
            self.pf = ParticleFilter(n_particles=300, n_blocks=1)

        # Episode state
        self.sigma_ep = 0.01
        self.holding_block = False
        self.gripper_closed = False
        self.step_count = 0
        self._last_wrist_obs = np.zeros(3)
        self._last_overhead_obs = np.zeros(3)

    def reset_episode(self):
        """Reset episode state. Call before each pick attempt."""
        self.sigma_ep = self.rng.uniform(SIGMA_LOW, SIGMA_HIGH)
        self.holding_block = False
        self.gripper_closed = False
        self.step_count = 0
        self._last_wrist_obs = np.zeros(3)
        self._last_overhead_obs = np.zeros(3)
        if self.pf is not None:
            self.pf = ParticleFilter(n_particles=300, n_blocks=1)

    def build_observation(self, joints_dict, block_true_poses, ee_pos, goal_xy, holding):
        """Construct 18D observation matching lego_pick_env._build_observation().

        Args:
            joints_dict: {joint_name: angle} for all 6 joints
            block_true_poses: {block_name: (x, y, yaw)} for target and distractor
            ee_pos: np.array([x, y, z]) end-effector position
            goal_xy: np.array([x, y]) goal position
            holding: bool, whether holding block
        """
        # [0:6] Joint angles
        joint_obs = []
        for name in ARM_JOINT_NAMES:
            joint_obs.append(joints_dict.get(name, 0.0))
        joint_obs.append(joints_dict.get('gripper_joint', 0.0))

        # [6:9] and [9:12] Block observations
        if self.belief_mode:
            block_obs_1, block_obs_2 = self._build_belief_obs(block_true_poses, ee_pos)
        else:
            block_obs_1, block_obs_2 = self._build_plain_obs(block_true_poses, ee_pos)

        # [12:15] EE position
        ee_obs = ee_pos.tolist()

        # [15:17] Goal position
        goal_obs = goal_xy.tolist()

        # [17] Holding flag
        holding_obs = [1.0 if holding else 0.0]

        obs = np.concatenate([
            joint_obs, block_obs_1, block_obs_2, ee_obs, goal_obs, holding_obs
        ]).astype(np.float32)

        self.step_count += 1
        return obs

    def _build_plain_obs(self, block_true_poses, ee_pos):
        """Build wrist + overhead noisy observations (plain PPO mode)."""
        target = block_true_poses.get(TARGET_BLOCK)
        distractor = block_true_poses.get(DISTRACTOR_BLOCK)

        if target is None:
            return self._last_wrist_obs.copy(), self._last_overhead_obs.copy()

        # Wrist camera: noisy observation with occlusion check
        # (skip wrist occlusion in ROS mode — no camera_link body to check against)
        wrist_noise = self.rng.normal(0, self.sigma_ep, 3)
        wrist_noise[2] = self.rng.normal(0, self.sigma_ep * 10)
        wrist_obs = np.array([
            target[0] + wrist_noise[0],
            target[1] + wrist_noise[1],
            target[2] + wrist_noise[2],
        ])
        self._last_wrist_obs = wrist_obs

        # Overhead camera: noisy observation with occlusion check
        overhead_occluded = False
        if distractor is not None:
            overhead_occluded = is_occluded_overhead(
                target_pos=(target[0], target[1]),
                occluder_pos=(distractor[0], distractor[1]),
                occluder_half_size=HALF_SIZES[DISTRACTOR_BLOCK],
                occluder_yaw=distractor[2],
            )

        if overhead_occluded:
            overhead_obs = self._last_overhead_obs.copy()
        else:
            oh_noise_xy = self.rng.normal(0, SIGMA_OVERHEAD, 2)
            oh_noise_theta = self.rng.normal(0, SIGMA_OVERHEAD * 10)
            overhead_obs = np.array([
                target[0] + oh_noise_xy[0],
                target[1] + oh_noise_xy[1],
                target[2] + oh_noise_theta,
            ])
            self._last_overhead_obs = overhead_obs

        return wrist_obs, overhead_obs

    def _build_belief_obs(self, block_true_poses, ee_pos):
        """Build PF belief observations (belief PPO mode)."""
        target = block_true_poses.get(TARGET_BLOCK)
        distractor = block_true_poses.get(DISTRACTOR_BLOCK)

        if target is None:
            mu, sigma = self.pf.get_belief()
            return mu[0], sigma[0]

        # Initialize PF on first obs
        if self.step_count == 0:
            init_obs = np.array([[target[0], target[1], target[2]]])
            noise = self.rng.normal(0, self.sigma_ep * 3, (1, 3))
            self.pf.reset(init_obs + noise, sigma_init=0.05)

        # Predict step
        self.pf.predict()

        # Wrist camera update (always visible in ROS mode for simplicity)
        wrist_noise = self.rng.normal(0, self.sigma_ep, 3)
        wrist_noise[2] = self.rng.normal(0, self.sigma_ep * 10)
        wrist_obs = np.array([
            target[0] + wrist_noise[0],
            target[1] + wrist_noise[1],
            target[2] + wrist_noise[2],
        ])
        self.pf.update({0: wrist_obs}, self.sigma_ep)

        # Overhead camera update (with occlusion check)
        overhead_occluded = False
        if distractor is not None:
            overhead_occluded = is_occluded_overhead(
                target_pos=(target[0], target[1]),
                occluder_pos=(distractor[0], distractor[1]),
                occluder_half_size=HALF_SIZES[DISTRACTOR_BLOCK],
                occluder_yaw=distractor[2],
            )

        if not overhead_occluded:
            oh_noise_xy = self.rng.normal(0, SIGMA_OVERHEAD, 2)
            oh_noise_theta = self.rng.normal(0, SIGMA_OVERHEAD * 10)
            oh_obs = np.array([
                target[0] + oh_noise_xy[0],
                target[1] + oh_noise_xy[1],
                target[2] + oh_noise_theta,
            ])
            self.pf.update({0: oh_obs}, SIGMA_OVERHEAD)

        # Resample
        self.pf.resample()

        mu, sigma = self.pf.get_belief()
        return mu[0], sigma[0]

    def normalize_obs(self, obs):
        """Apply frozen VecNormalize stats."""
        obs_norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        return np.clip(obs_norm, -self.clip_obs, self.clip_obs)

    def predict(self, obs):
        """Run policy inference. Returns (dx, dy, dz, gripper_cmd)."""
        obs_norm = self.normalize_obs(obs)
        obs_tensor = obs_norm.reshape(1, -1)
        action, _ = self.model.predict(obs_tensor, deterministic=True)
        return action[0]  # (4,)

    def ik_step(self, ee_pos, action, speed_scale=1.0):
        """Compute joint targets from EE incremental action.

        Args:
            speed_scale: multiplier for step size (>1 = faster motion in ROS)

        Returns dict {joint_name: angle} or None if IK fails.
        """
        dx, dy, dz = action[0], action[1], action[2]

        # Scale actions to ±0.02m (matching env action space), then apply speed multiplier
        step = 0.02 * speed_scale
        dx = np.clip(dx, -1, 1) * step
        dy = np.clip(dy, -1, 1) * step
        dz = np.clip(dz, -1, 1) * step

        new_x = ee_pos[0] + dx
        new_y = ee_pos[1] + dy
        new_z = ee_pos[2] + dz

        # Clamp to cylindrical workspace (matching lego_pick_env)
        r = math.sqrt(new_x**2 + new_y**2)
        if r < EE_R_MIN:
            scale = EE_R_MIN / max(r, 1e-6)
            new_x *= scale
            new_y *= scale
        elif r > EE_R_MAX:
            scale = EE_R_MAX / r
            new_x *= scale
            new_y *= scale
        new_z = np.clip(new_z, EE_Z_MIN, EE_Z_MAX)

        solutions = geometric_ik(new_x, new_y, new_z, grasp_yaw=0.0)
        if not solutions:
            return None

        sol = solutions[0]
        return sol

    def check_grasp(self, ee_pos, block_true_poses):
        """Attempt grasp with probabilistic success (matching env)."""
        target = block_true_poses.get(TARGET_BLOCK)
        if target is None:
            return False

        block_xy = np.array([target[0], target[1]])
        ee_xy = ee_pos[:2]
        dist = np.linalg.norm(ee_xy - block_xy)

        # ROS mode: more forgiving than training (0.02 center vs 0.01)
        # to compensate for motion lag and observation delay
        p_grasp = 1.0 / (1.0 + math.exp(-200.0 * (0.02 - dist)))
        if self.rng.random() < p_grasp:
            self.holding_block = True
            return True
        return False

    def sample_goal(self, block_true_poses):
        """Sample a goal position on the table, away from blocks."""
        for _ in range(100):
            angle = self.rng.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
            r = self.rng.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            too_close = False
            for name in BLOCK_NAMES:
                if name in block_true_poses:
                    bx, by, _ = block_true_poses[name]
                    if math.sqrt((x - bx)**2 + (y - by)**2) < MIN_SPACING * 2:
                        too_close = True
                        break
            if not too_close:
                return np.array([x, y])

        return np.array([0.18, -0.05])

    def randomize_blocks(self):
        """Generate random block positions. Returns {name: (x, y, yaw)}."""
        positions = []
        result = {}
        for name in BLOCK_NAMES:
            for _ in range(100):
                angle = self.rng.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
                r = self.rng.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                yaw = self.rng.uniform(-math.pi, math.pi)

                too_close = False
                for px, py, _ in positions:
                    if math.sqrt((x - px)**2 + (y - py)**2) < MIN_SPACING:
                        too_close = True
                        break
                if not too_close:
                    positions.append((x, y, yaw))
                    result[name] = (x, y, yaw)
                    break
            else:
                # Fallback
                fallback_angle = SPAWN_ANGLE_MIN + (len(positions) / len(BLOCK_NAMES)) * (SPAWN_ANGLE_MAX - SPAWN_ANGLE_MIN)
                r = (SPAWN_R_MIN + SPAWN_R_MAX) / 2
                x = r * math.cos(fallback_angle)
                y = r * math.sin(fallback_angle)
                positions.append((x, y, 0.0))
                result[name] = (x, y, 0.0)
        return result
