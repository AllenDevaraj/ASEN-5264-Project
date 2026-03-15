#!/usr/bin/env python3
"""Gymnasium environment for SO-ARM101 pick-and-place under uncertainty.

Task: Pick the red lego block and place it at a random goal position.
A blue distractor block can occlude the target from the wrist and overhead cameras.

Four uncertainty sources:
  1. Episodic sigma — wrist observation = true pose + N(0, sigma_ep^2)
  2. Overhead camera noise — fixed N(0, SIGMA_OVERHEAD^2)
  3. Multi-block occlusion — occluded block gets no pose update (both cameras)
  4. Cost of looking — -1 reward per timestep

Usage:
    env = LegoPickEnv(belief_mode=False)  # Plain PPO (12D obs: joints + wrist + overhead)
    env = LegoPickEnv(belief_mode=True)   # Belief PPO (12D obs: joints + PF belief)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import math

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from so_arm101_control.compute_workspace import (
    ARM_JOINT_NAMES,
    JOINT_LIMITS,
    forward_kinematics,
    geometric_ik,
)
from so_arm101_control.model_loader import (
    build_joint_map,
    build_mocap_map,
    load_mujoco_model,
)
from so_arm101_control.occlusion import is_occluded, is_occluded_overhead
from so_arm101_control.particle_filter import ParticleFilter

# Block definitions (from randomize_legos.py)
TARGET_BLOCK = "red_lego_2x4"
DISTRACTOR_BLOCK = "blue_lego_2x2"
BLOCK_NAMES = [TARGET_BLOCK, DISTRACTOR_BLOCK]

HALF_SIZES = {
    "red_lego_2x4": (0.016, 0.008),
    "green_lego_2x3": (0.012, 0.008),
    "blue_lego_2x2": (0.008, 0.008),
}

TABLE_Z = 0.0055
MIN_SPACING = 0.03

# Workspace bounds for block spawning (within arm reach)
SPAWN_R_MIN = 0.12
SPAWN_R_MAX = 0.22
SPAWN_ANGLE_MIN = -1.0  # radians from +X axis
SPAWN_ANGLE_MAX = 1.0

# EE workspace limits
EE_R_MIN = 0.09
EE_R_MAX = 0.31
EE_Z_MIN = 0.002   # just above table surface (TABLE_Z = 0.0055)
EE_Z_MAX = 0.12

# Gripper joint limits (from URDF)
GRIPPER_OPEN = -0.174533
GRIPPER_CLOSED = 1.74533

# Overhead camera noise (fixed, independent of episodic sigma)
SIGMA_OVERHEAD = 0.005


def _yaw_to_quat(yaw):
    """Convert yaw angle to quaternion (w, x, y, z) for MuJoCo mocap."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class LegoPickEnv(gym.Env):
    """SO-ARM101 pick-and-place under observation uncertainty."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    MAX_STEPS = 200

    def __init__(
        self,
        belief_mode=False,
        use_camera_noise=False,
        sigma_low=0.003,
        sigma_high=0.020,
        approach_shaping=True,
        render_mode=None,
    ):
        """
        Args:
            belief_mode: If True, use particle filter and return 12D obs.
            use_camera_noise: If True, add distance-dependent noise on top.
            sigma_low: Minimum episodic observation noise (meters).
            sigma_high: Maximum episodic observation noise (meters).
            approach_shaping: If True, add +0.5 reward for approaching block.
            render_mode: "human" for viewer, "rgb_array" for image.
        """
        super().__init__()
        self.belief_mode = belief_mode
        self.use_camera_noise = use_camera_noise
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.approach_shaping = approach_shaping
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model, self.data = load_mujoco_model()
        self.joint_map = build_joint_map(self.model)
        self.mocap_map = build_mocap_map(self.model)

        # Look up camera_link body for occlusion checks
        self._camera_link_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "camera_link"
        )

        # Look up block body IDs for true pose reading
        self._block_body_ids = {}
        for name in BLOCK_NAMES:
            self._block_body_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )

        # Action space: [dx, dy, dz, gripper_cmd]
        self.action_space = spaces.Box(
            low=np.array([-0.02, -0.02, -0.02, -1.0], dtype=np.float32),
            high=np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float32),
        )

        # Observation space (12D for both modes)
        obs_dim = 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Particle filter
        self.pf = ParticleFilter(n_particles=300, n_blocks=1)

        # Renderer for rgb_array mode
        self._renderer = None
        self._viewer = None

        # Episode state
        self._step_count = 0
        self._sigma_ep = 0.0
        self._block_true_poses = {}  # name -> (x, y, yaw)
        self._goal_pos = None  # (x, y)
        self._gripper_closed = False
        self._holding_block = False
        self._ee_pos = np.zeros(3)
        self._prev_dist_to_block = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._holding_block = False
        self._gripper_closed = False

        # 1. Draw episodic noise level
        self._sigma_ep = self.np_random.uniform(self.sigma_low, self.sigma_high)

        # 2. Reset arm to a starting position above workspace center
        #    (all-zeros puts EE at r=0.39m which is outside workspace)
        home_target = (0.18, 0.0, 0.06)  # above workspace center
        home_solutions = geometric_ik(*home_target, grasp_yaw=0.0)
        if home_solutions:
            sol = home_solutions[0]
            for name in ARM_JOINT_NAMES:
                if name in self.joint_map and name in sol:
                    self.data.qpos[self.joint_map[name]] = sol[name]
        else:
            for name in ARM_JOINT_NAMES:
                if name in self.joint_map:
                    self.data.qpos[self.joint_map[name]] = 0.0
        if "gripper_joint" in self.joint_map:
            self.data.qpos[self.joint_map["gripper_joint"]] = GRIPPER_OPEN

        # 3. Randomize block positions
        self._randomize_blocks()

        # 4. Sample goal position
        self._goal_pos = self._sample_table_position()

        # 5. Step kinematics and get EE pos
        mujoco.mj_forward(self.model, self.data)
        self._ee_pos = self._get_ee_pos()

        # 6. Initial distance for approach shaping
        target_xy = np.array([
            self._block_true_poses[TARGET_BLOCK][0],
            self._block_true_poses[TARGET_BLOCK][1],
        ])
        self._prev_dist_to_block = np.linalg.norm(self._ee_pos[:2] - target_xy)

        # 7. Reset particle filter (feed both cameras)
        if self.belief_mode:
            noisy_obs = self._get_noisy_target_obs()
            self.pf.reset(noisy_obs.reshape(1, 3), sigma_init=self._sigma_ep * 3)
            overhead_obs = self._get_overhead_visible_observations()
            if overhead_obs:
                self.pf.update(overhead_obs, SIGMA_OVERHEAD)

        obs = self._build_observation()
        info = {
            "sigma_ep": self._sigma_ep,
            "true_target_pose": self._block_true_poses[TARGET_BLOCK],
            "goal_pos": self._goal_pos.copy(),
        }
        return obs, info

    def step(self, action):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # 1. Parse action
        dx, dy, dz = action[0], action[1], action[2]
        gripper_cmd = action[3]

        # 2. Compute new EE target
        new_ee = self._ee_pos.copy() + np.array([dx, dy, dz])
        new_ee = self._clamp_to_workspace(new_ee)

        # 3. IK and set joint positions
        solutions = geometric_ik(
            float(new_ee[0]), float(new_ee[1]), float(new_ee[2]), grasp_yaw=0.0
        )
        if solutions:
            sol = solutions[0]
            for name in ARM_JOINT_NAMES:
                if name in self.joint_map and name in sol:
                    self.data.qpos[self.joint_map[name]] = sol[name]

        # 4. Handle gripper
        want_close = gripper_cmd > 0.0
        grasp_result = None

        if want_close and not self._gripper_closed and not self._holding_block:
            # Attempting grasp
            self._gripper_closed = True
            self.data.qpos[self.joint_map["gripper_joint"]] = GRIPPER_CLOSED
            grasp_result = self._attempt_grasp()
            if grasp_result == "success":
                self._holding_block = True
        elif not want_close and self._gripper_closed:
            # Opening gripper
            self._gripper_closed = False
            self.data.qpos[self.joint_map["gripper_joint"]] = GRIPPER_OPEN

        # 5. Step kinematics
        mujoco.mj_forward(self.model, self.data)
        self._ee_pos = self._get_ee_pos()

        # 6. Move held block with EE
        if self._holding_block:
            self._update_held_block_pose()

        # 7. Update particle filter (dual camera)
        if self.belief_mode:
            self.pf.predict()
            wrist_obs = self._get_visible_observations()
            self.pf.update(wrist_obs, self._get_effective_sigma())
            overhead_obs = self._get_overhead_visible_observations()
            self.pf.update(overhead_obs, SIGMA_OVERHEAD)
            self.pf.resample()

        # 8. Compute reward
        reward = -1.0  # step cost

        if grasp_result == "fail":
            reward -= 5.0
        elif grasp_result == "success":
            pass  # no immediate reward for grasping; must place

        # Check placement
        terminated = False
        if self._holding_block and not want_close:
            # Just released the block
            self._holding_block = False
            self._release_block()
            dist_to_goal = np.linalg.norm(self._ee_pos[:2] - self._goal_pos)
            if dist_to_goal < 0.015:
                reward += 10.0
                terminated = True
            else:
                reward -= 2.0  # penalty for dropping far from goal

        # Approach shaping (only when not holding)
        if self.approach_shaping and not self._holding_block and not terminated:
            target_xy = np.array([
                self._block_true_poses[TARGET_BLOCK][0],
                self._block_true_poses[TARGET_BLOCK][1],
            ])
            dist = np.linalg.norm(self._ee_pos[:2] - target_xy)
            if self._prev_dist_to_block is not None:
                improvement = self._prev_dist_to_block - dist
                reward += 0.5 * np.clip(improvement / 0.02, -1, 1)
            self._prev_dist_to_block = dist

        truncated = self._step_count >= self.MAX_STEPS

        obs = self._build_observation()
        info = {
            "step": self._step_count,
            "sigma_ep": self._sigma_ep,
            "grasp_result": grasp_result,
            "holding": self._holding_block,
            "ee_pos": self._ee_pos.copy(),
            "overhead_occluded": self._is_target_occluded_overhead(),
        }
        if self.belief_mode:
            mu, sigma = self.pf.get_belief()
            info["belief_mu"] = mu[0]
            info["belief_sigma"] = sigma[0]
            info["sigma_at_step"] = sigma[0].copy()

        return obs, reward, terminated, truncated, info

    # ---- Internal methods ----

    def _get_ee_pos(self):
        """Get EE (TCP) position via forward kinematics from current qpos."""
        angles = []
        for name in ARM_JOINT_NAMES:
            if name in self.joint_map:
                angles.append(self.data.qpos[self.joint_map[name]])
            else:
                angles.append(0.0)
        return np.array(forward_kinematics(angles))

    def _clamp_to_workspace(self, pos):
        """Clamp EE target to reachable cylindrical workspace."""
        r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        r_clamped = np.clip(r, EE_R_MIN, EE_R_MAX)
        if r > 1e-6:
            scale = r_clamped / r
            pos[0] *= scale
            pos[1] *= scale
        pos[2] = np.clip(pos[2], EE_Z_MIN, EE_Z_MAX)
        return pos

    def _randomize_blocks(self):
        """Place blocks randomly on table within arm reach, non-overlapping."""
        positions = []
        for name in BLOCK_NAMES:
            for _ in range(100):  # rejection sampling
                angle = self.np_random.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
                r = self.np_random.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                yaw = self.np_random.uniform(-math.pi, math.pi)

                # Check spacing with already placed blocks
                too_close = False
                for px, py, _ in positions:
                    if math.sqrt((x - px) ** 2 + (y - py) ** 2) < MIN_SPACING:
                        too_close = True
                        break
                if not too_close:
                    positions.append((x, y, yaw))
                    break
            else:
                # Fallback: place at evenly spaced fixed angles to guarantee separation
                fallback_angle = SPAWN_ANGLE_MIN + (i / len(BLOCK_NAMES)) * (SPAWN_ANGLE_MAX - SPAWN_ANGLE_MIN)
                r = (SPAWN_R_MIN + SPAWN_R_MAX) / 2
                x = r * math.cos(fallback_angle)
                y = r * math.sin(fallback_angle)
                yaw = 0.0
                positions.append((x, y, yaw))

        for i, name in enumerate(BLOCK_NAMES):
            x, y, yaw = positions[i]
            self._block_true_poses[name] = (x, y, yaw)

            if name in self.mocap_map:
                mocap_id = self.mocap_map[name]
                self.data.mocap_pos[mocap_id] = [x, y, TABLE_Z]
                self.data.mocap_quat[mocap_id] = _yaw_to_quat(yaw)

    def _sample_table_position(self):
        """Sample a goal position on the table, away from blocks."""
        for _ in range(100):
            angle = self.np_random.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
            r = self.np_random.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            # Ensure goal is away from all blocks
            too_close = False
            for name in BLOCK_NAMES:
                if name in self._block_true_poses:
                    bx, by, _ = self._block_true_poses[name]
                    if math.sqrt((x - bx) ** 2 + (y - by) ** 2) < MIN_SPACING * 2:
                        too_close = True
                        break
            if not too_close:
                return np.array([x, y])

        # Fallback
        return np.array([0.18, -0.05])

    def _get_effective_sigma(self):
        """Get the effective observation noise sigma."""
        sigma = self._sigma_ep
        if self.use_camera_noise:
            target = self._block_true_poses[TARGET_BLOCK]
            target_pos = np.array([target[0], target[1], TABLE_Z])
            dist = np.linalg.norm(self._ee_pos - target_pos)
            distance_factor = dist / 0.15
            sigma_xy = 0.008 * distance_factor
            sigma = max(sigma, sigma_xy)
        return sigma

    def _get_noisy_target_obs(self):
        """Get noisy observation of target block (x, y, theta)."""
        true = self._block_true_poses[TARGET_BLOCK]
        sigma = self._get_effective_sigma()
        noise = self.np_random.normal(0, sigma, 3)
        # Scale theta noise (use sigma in radians ~ sigma * 10 for ~degrees)
        noise[2] = self.np_random.normal(0, sigma * 10)
        return np.array([true[0] + noise[0], true[1] + noise[1], true[2] + noise[2]])

    def _get_camera_state(self):
        """Get camera_link position and rotation matrix from MuJoCo."""
        cam_pos = self.data.xpos[self._camera_link_id].copy()
        cam_rot = self.data.xmat[self._camera_link_id].reshape(3, 3).copy()
        return cam_pos, cam_rot

    def _is_target_occluded(self):
        """Check if the target block is occluded by the distractor."""
        target = self._block_true_poses[TARGET_BLOCK]
        distractor = self._block_true_poses[DISTRACTOR_BLOCK]
        cam_pos, cam_rot = self._get_camera_state()

        return is_occluded(
            target_pos=(target[0], target[1]),
            target_half_size=HALF_SIZES[TARGET_BLOCK],
            target_yaw=target[2],
            occluder_pos=(distractor[0], distractor[1]),
            occluder_half_size=HALF_SIZES[DISTRACTOR_BLOCK],
            occluder_yaw=distractor[2],
            camera_pos=cam_pos,
            camera_rot=cam_rot,
        )

    def _get_visible_observations(self):
        """Get observations for visible blocks (dict: block_idx -> obs)."""
        if self._is_target_occluded():
            return {}  # target occluded, no observation

        noisy_obs = self._get_noisy_target_obs()
        return {0: noisy_obs}

    def _get_overhead_noisy_obs(self):
        """Get noisy observation of target block from overhead camera."""
        true = self._block_true_poses[TARGET_BLOCK]
        noise_xy = self.np_random.normal(0, SIGMA_OVERHEAD, 2)
        noise_theta = self.np_random.normal(0, SIGMA_OVERHEAD * 10)
        return np.array([true[0] + noise_xy[0], true[1] + noise_xy[1],
                         true[2] + noise_theta])

    def _is_target_occluded_overhead(self):
        """Check if target block is occluded from overhead by distractor."""
        target = self._block_true_poses[TARGET_BLOCK]
        distractor = self._block_true_poses[DISTRACTOR_BLOCK]
        return is_occluded_overhead(
            target_pos=(target[0], target[1]),
            occluder_pos=(distractor[0], distractor[1]),
            occluder_half_size=HALF_SIZES[DISTRACTOR_BLOCK],
            occluder_yaw=distractor[2],
        )

    def _get_overhead_visible_observations(self):
        """Get overhead camera observations for visible blocks."""
        if self._is_target_occluded_overhead():
            return {}
        noisy_obs = self._get_overhead_noisy_obs()
        return {0: noisy_obs}

    def _attempt_grasp(self):
        """Stochastic grasp based on EE-to-block distance."""
        target = self._block_true_poses[TARGET_BLOCK]
        block_xy = np.array([target[0], target[1]])
        ee_xy = self._ee_pos[:2]
        dist = np.linalg.norm(ee_xy - block_xy)

        # sigmoid(300*(0.01-d)): p(0mm)=95%, p(5mm)=82%, p(10mm)=50%, p(15mm)=18%
        p_grasp = 1.0 / (1.0 + math.exp(-300.0 * (0.01 - dist)))
        if self.np_random.random() < p_grasp:
            return "success"
        return "fail"

    def _update_held_block_pose(self):
        """Move held block to follow EE position."""
        if TARGET_BLOCK in self.mocap_map:
            mocap_id = self.mocap_map[TARGET_BLOCK]
            self.data.mocap_pos[mocap_id][0] = self._ee_pos[0]
            self.data.mocap_pos[mocap_id][1] = self._ee_pos[1]
            # Keep z at table level unless EE is above table
            self.data.mocap_pos[mocap_id][2] = max(TABLE_Z, self._ee_pos[2])
            # Update true pose tracking
            yaw = self._block_true_poses[TARGET_BLOCK][2]
            self._block_true_poses[TARGET_BLOCK] = (
                self._ee_pos[0], self._ee_pos[1], yaw
            )

    def _release_block(self):
        """Release block at current EE position (place on table)."""
        if TARGET_BLOCK in self.mocap_map:
            mocap_id = self.mocap_map[TARGET_BLOCK]
            self.data.mocap_pos[mocap_id][2] = TABLE_Z
            yaw = self._block_true_poses[TARGET_BLOCK][2]
            self._block_true_poses[TARGET_BLOCK] = (
                self._ee_pos[0], self._ee_pos[1], yaw
            )

    def _build_observation(self):
        """Construct observation vector."""
        # Joint angles
        joint_obs = []
        for name in ARM_JOINT_NAMES:
            if name in self.joint_map:
                joint_obs.append(self.data.qpos[self.joint_map[name]])
            else:
                joint_obs.append(0.0)
        gripper_val = self.data.qpos[self.joint_map["gripper_joint"]]
        joint_obs.append(gripper_val)

        if self.belief_mode:
            mu, sigma = self.pf.get_belief()
            return np.concatenate(
                [joint_obs, mu[0], sigma[0]]
            ).astype(np.float32)
        else:
            wrist_obs = self._get_noisy_target_obs()
            overhead_obs = self._get_overhead_noisy_obs()
            return np.concatenate([joint_obs, wrist_obs, overhead_obs]).astype(np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 720, 1280)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
