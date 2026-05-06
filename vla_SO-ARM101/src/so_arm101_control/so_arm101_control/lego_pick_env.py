#!/usr/bin/env python3
"""Gymnasium environment for SO-ARM101 pick-and-place under uncertainty.

Task: Pick the red lego block and place it at a random goal position.
A blue distractor block can occlude the target from the wrist and overhead cameras.

Physics-based grasping: uses mj_step for full physics (gravity, contacts).
Grasping is proximity-triggered when the gripper closes near the block, then
the block is constrained to the gripper frame. On release, the block falls
under gravity.

Four uncertainty sources:
  1. Episodic sigma — wrist observation = true pose + N(0, sigma_ep^2)
  2. Overhead camera noise — fixed N(0, SIGMA_OVERHEAD^2)
  3. Multi-block occlusion — occluded block gets no pose update (both cameras)
  4. Cost of looking — -1 reward per timestep

Usage:
    env = LegoPickEnv(belief_mode=False)  # Plain PPO (18D obs)
    env = LegoPickEnv(belief_mode=True)   # Belief PPO (18D obs)
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
    build_freejoint_map,
    build_joint_map,
    load_mujoco_model,
)
from so_arm101_control.occlusion import is_occluded, is_occluded_overhead
from so_arm101_control.particle_filter import ParticleFilter

# Block definitions
TARGET_BLOCK = "red_lego_2x4"
DISTRACTOR_BLOCKS = ["blue_lego_2x2", "blue_lego_2x2_b"]
BLOCK_NAMES = [TARGET_BLOCK] + DISTRACTOR_BLOCKS

HALF_SIZES = {
    "red_lego_2x4":     (0.016, 0.008),  # 32x16mm — target
    "blue_lego_2x2":    (0.020, 0.010),  # 40x20mm — occlusion-tuned (was 64x32, too large)
    "blue_lego_2x2_b":  (0.020, 0.010),  # 40x20mm — second blue occluder
    "blue_lego_2x2_c":  (0.020, 0.010),  # 40x20mm — third blue occluder
}

# Half-heights (Z) for each block — used for occlusion projection at top face
BLOCK_HALF_Z = {
    "red_lego_2x4":    0.0055,   # 11mm full height
    "blue_lego_2x2":   0.0165,   # 33mm full height (3x red — tall enough for wrist-camera occlusion)
    "blue_lego_2x2_b": 0.0165,
    "blue_lego_2x2_c": 0.0165,
}

TABLE_Z = 0.0055
MIN_SPACING = 0.050         # 50mm min center-to-center (covers blue-blue worst-case ~45mm + buffer)
DISTRACTOR_NEAR_TARGET_RADIUS = 0.090  # distractors spawn within 90mm of target (must be > MIN_SPACING)

# Workspace bounds for block spawning (within arm reach)
SPAWN_R_MIN = 0.12
SPAWN_R_MAX = 0.22
SPAWN_ANGLE_MIN = -1.0  # radians from +X axis
SPAWN_ANGLE_MAX = 1.0

# EE workspace limits
EE_R_MIN = 0.09
EE_R_MAX = 0.31
EE_Z_MIN = 0.002   # just above table surface
EE_Z_MAX = 0.12

# Gripper joint limits (from URDF)
GRIPPER_OPEN = -0.174533
GRIPPER_CLOSED = 1.74533

# Overhead camera noise (fixed, independent of episodic sigma)
SIGMA_OVERHEAD = 0.005

# Physics substeps per env step (at 0.0005s timestep = 5ms simulated per step)
PHYSICS_SUBSTEPS = 10

# Grasp proximity threshold (meters)
GRASP_THRESHOLD = 0.015


def _yaw_to_quat(yaw):
    """Convert yaw angle to quaternion (w, x, y, z) for MuJoCo."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class LegoPickEnv(gym.Env):
    """SO-ARM101 pick-and-place under observation uncertainty."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    MAX_STEPS = 300

    def __init__(
        self,
        belief_mode=False,
        use_camera_noise=False,
        sigma_low=0.003,
        sigma_high=0.015,
        approach_shaping=True,
        render_mode=None,
        use_overhead_camera=False,
        sigma_drift=0.0,
    ):
        super().__init__()
        self.belief_mode = belief_mode
        self.use_camera_noise = use_camera_noise
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.approach_shaping = approach_shaping
        self.render_mode = render_mode
        self.use_overhead_camera = use_overhead_camera
        self.sigma_drift = sigma_drift

        # Load MuJoCo model
        self.model, self.data = load_mujoco_model()
        self.joint_map = build_joint_map(self.model)
        self.freejoint_map = build_freejoint_map(self.model)

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

        # Build list of robot qpos/qvel indices for save/restore during mj_step
        self._robot_qpos_indices = []
        self._robot_qvel_indices = []
        robot_joint_names = list(ARM_JOINT_NAMES) + ['gripper_joint']
        for name in robot_joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self._robot_qpos_indices.append(self.model.jnt_qposadr[jnt_id])
                self._robot_qvel_indices.append(self.model.jnt_dofadr[jnt_id])

        # Action space: [dx, dy, dz, gripper_cmd]
        self.action_space = spaces.Box(
            low=np.array([-0.02, -0.02, -0.02, -1.0], dtype=np.float32),
            high=np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float32),
        )

        # Observation layout:
        #   Plain:  13D [joints(6), stale_wrist(3),        goal(2), holding(1), occluded(1)]
        #   Belief: 16D [joints(6), pf_mu(3), pf_sigma(3), goal(2), holding(1), occluded(1)]
        if self.belief_mode:
            obs_dim = 16
        elif self.use_overhead_camera:
            obs_dim = 16
        else:
            obs_dim = 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Particle filter — process noise tuned to match actual block drift
        pf_process_noise = max(0.0005, self.sigma_drift)
        self.pf = ParticleFilter(n_particles=300, n_blocks=1, process_noise_xy=pf_process_noise)

        # Renderer for rgb_array mode
        self._renderer = None
        self._viewer = None

        # Episode state
        self._step_count = 0
        self._sigma_ep = 0.0
        self._block_true_poses = {}  # name -> (x, y, yaw)
        self._goal_pos = None  # (x, y)
        self._gripper_closed = False
        self._holding_block = None  # None or block name string
        self._ee_pos = np.zeros(3)
        self._last_wrist_obs = np.zeros(3)   # stale obs for plain mode during occlusion
        self._step_noisy_obs = np.zeros(3)   # obs sampled once per step (used for reward + policy obs)
        self._prev_dist_to_block = None
        self._prev_dist_to_goal = None
        self._reached_block = False
        self._reached_goal = False
        # Offset from EE to block center when grasped (for constraint carrying)
        self._grasp_offset = np.zeros(3)
        self._prev_belief_sigma = None  # for info-gathering reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._holding_block = None
        self._gripper_closed = False
        self._reached_block = False
        self._reached_goal = False
        self._grasp_offset = np.zeros(3)
        self._prev_belief_sigma = None

        # 1. Draw episodic noise level
        self._sigma_ep = self.np_random.uniform(self.sigma_low, self.sigma_high)

        # 2. Reset arm to starting position above workspace center
        home_target = (0.18, 0.0, 0.06)
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

        # 5. Run physics to settle blocks on ground
        self._step_physics()
        self._ee_pos = self._get_ee_pos()

        # 6. Read settled block positions
        self._read_block_poses()

        # 7. Seed stale-obs cache and particle filter
        self._last_wrist_obs = self._get_noisy_target_obs()
        self._step_noisy_obs = self._last_wrist_obs.copy()
        if self.belief_mode:
            noisy_obs = self._last_wrist_obs.copy()
            self.pf.reset(noisy_obs.reshape(1, 3), sigma_init=min(self._sigma_ep * 4, 0.05))
            if self.use_overhead_camera:
                overhead_obs = self._get_overhead_visible_observations()
                if overhead_obs:
                    self.pf.update(overhead_obs, SIGMA_OVERHEAD)

        # 8b. Initial prev_dist using observed block position (not true)
        if self.belief_mode:
            _mu, _ = self.pf.get_belief()
            obs_block_xy = _mu[0, :2]
        else:
            obs_block_xy = self._last_wrist_obs[:2]
        self._prev_dist_to_block = np.linalg.norm(self._ee_pos[:2] - obs_block_xy)
        self._prev_ee_z = self._ee_pos[2]
        self._prev_dist_to_goal = None

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

        # 4. Handle gripper + auto-grasp
        want_close = gripper_cmd > 0.0
        grasp_result = None

        # Auto-grasp: if EE is within grasp range of any block, pick it up
        if not self._holding_block:
            grabbed = self._attempt_grasp()
            if grabbed is not None:
                grasp_result = grabbed  # block name (target or distractor)
                self._holding_block = grabbed
                self._gripper_closed = True
                self.data.qpos[self.joint_map["gripper_joint"]] = GRIPPER_CLOSED

        # 5. Step physics (blocks react to gravity and contacts)
        self._step_physics()
        self._ee_pos = self._get_ee_pos()

        # 6. If holding block, constrain it to gripper (position constraint)
        if self._holding_block:
            self._constrain_held_block()

        # 7. Read block true poses from freejoint qpos
        self._read_block_poses()

        # 7b. Apply target drift (if enabled and block is not held)
        if self.sigma_drift > 0 and self._holding_block != TARGET_BLOCK:
            self._apply_target_drift()

        # 8. Sample block observation once this step (used for reward + policy obs)
        # Must happen before reward so both use the same random draw.
        if not self._is_target_occluded():
            self._step_noisy_obs = self._get_noisy_target_obs()
            self._last_wrist_obs = self._step_noisy_obs.copy()
        # else: _last_wrist_obs stays stale, _step_noisy_obs stays from last visible step

        # 8b. Update particle filter
        if self.belief_mode:
            self.pf.predict()
            wrist_obs = {} if self._is_target_occluded() else {0: self._step_noisy_obs}
            self.pf.update(wrist_obs, self._get_effective_sigma())
            if self.use_overhead_camera:
                overhead_obs = self._get_overhead_visible_observations()
                self.pf.update(overhead_obs, SIGMA_OVERHEAD)
            self.pf.resample()

        # 9. Compute reward using OBSERVED block position (not true state).
        # Aligns reward signal with observation quality — belief PPO gets a more
        # accurate gradient because PF mean is closer to true than raw noisy obs.
        # Grasp and placement checks still use true physics state.
        if self.belief_mode:
            _mu, _sigma = self.pf.get_belief()
            obs_block_xy = _mu[0, :2]
        elif self._is_target_occluded():
            obs_block_xy = self._last_wrist_obs[:2]
        else:
            obs_block_xy = self._step_noisy_obs[:2]

        ee_xy = self._ee_pos[:2]
        dist_to_block = np.linalg.norm(ee_xy - obs_block_xy)
        dist_to_goal = np.linalg.norm(ee_xy - self._goal_pos)
        ee_z = self._ee_pos[2]

        terminated = False
        placement_success = False

        if not self._holding_block:
            # --- PHASE 1: Approach the block in XY ---
            reward = -1.0  # step cost

            # Sigma-reduction reward: incentivise belief convergence before grasping.
            # Only fires in belief mode when uncertainty is still high (sigma > 5mm).
            if self.belief_mode and self._prev_belief_sigma is not None:
                cur_sigma_xy = float(np.mean(_sigma[0, :2]))
                prev_sigma_xy = float(np.mean(self._prev_belief_sigma[:2]))
                sigma_reduction = prev_sigma_xy - cur_sigma_xy
                if sigma_reduction > 0 and cur_sigma_xy > 0.005:
                    reward += 1.0 * np.clip(sigma_reduction / 0.005, 0.0, 1.0)
            if self.belief_mode:
                self._prev_belief_sigma = _sigma[0].copy()

            # XY approach shaping
            if self._prev_dist_to_block is not None:
                improvement = self._prev_dist_to_block - dist_to_block
                reward += 3.0 * np.clip(improvement / 0.02, -1, 1)
            self._prev_dist_to_block = dist_to_block

            # Proximity bonus: continuous reward for being very close
            if dist_to_block < 0.025:
                reward += 2.0 * (1.0 - dist_to_block / 0.025)

            # Close-in bonus: sharp spike inside 12mm to pull policy into grasp zone
            if dist_to_block < 0.012:
                reward += 3.0 * (1.0 - dist_to_block / 0.012)

            # --- Milestone: reached block XY ---
            if dist_to_block < 0.015 and not self._reached_block:
                self._reached_block = True
                reward += 5.0

            if grasp_result == TARGET_BLOCK:
                reward += 10.0
            elif grasp_result is not None:
                # Grabbed a distractor — penalty and terminate
                reward -= 15.0
                terminated = True

        else:
            # --- PHASE 2: Carry block to goal ---
            reward = -1.0

            if self._prev_dist_to_goal is not None:
                improvement = self._prev_dist_to_goal - dist_to_goal
                reward += 5.0 * np.clip(improvement / 0.02, -1, 1)
            self._prev_dist_to_goal = dist_to_goal

            if dist_to_goal < 0.03 and not self._reached_goal:
                self._reached_goal = True
                reward += 5.0

        # --- PHASE 3: Placement check (only for target block) ---
        # Auto-release: drop block when holding target and within goal range
        if self._holding_block == TARGET_BLOCK and dist_to_goal < 0.02:
            want_close = False
        if self._holding_block == TARGET_BLOCK and not want_close:
            released = self._holding_block
            self._holding_block = None
            self._release_block(released)
            dist_to_goal = np.linalg.norm(self._ee_pos[:2] - self._goal_pos)
            if dist_to_goal < 0.01:
                reward += 50.0
                terminated = True
                placement_success = True
            elif dist_to_goal < 0.02:
                reward += 30.0
                terminated = True
                placement_success = True
            elif dist_to_goal < 0.04:
                reward += 10.0
                terminated = True
                placement_success = True
            else:
                reward -= 10.0
            self._prev_dist_to_goal = None

        truncated = self._step_count >= self.MAX_STEPS

        obs = self._build_observation()
        target_true = self._block_true_poses[TARGET_BLOCK]
        info = {
            "step": self._step_count,
            "sigma_ep": self._sigma_ep,
            "grasp_result": grasp_result,
            "holding": self._holding_block,
            "ee_pos": self._ee_pos.copy(),
            "true_block_pos": np.array([target_true[0], target_true[1], target_true[2]]),
            "wrist_occluded": self._is_target_occluded(),
            "overhead_occluded": self._is_target_occluded_overhead(),
            "effective_sigma": self._get_effective_sigma(),
            "dist_to_block": float(dist_to_block),
            "dist_to_goal": float(dist_to_goal),
            "reward": float(reward),
            "success": placement_success,
        }
        if self.belief_mode:
            mu, sigma = self.pf.get_belief()
            info["belief_mu"] = mu[0]
            info["belief_sigma"] = sigma[0]
            info["sigma_at_step"] = sigma[0].copy()

        return obs, reward, terminated, truncated, info

    # ---- Physics ----

    def _step_physics(self):
        """Run physics substeps with save/restore for kinematically-driven robot."""
        saved_qpos = [self.data.qpos[i] for i in self._robot_qpos_indices]
        saved_qvel = [self.data.qvel[i] for i in self._robot_qvel_indices]

        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            # Restore robot joints (kinematically driven)
            for i, idx in enumerate(self._robot_qpos_indices):
                self.data.qpos[idx] = saved_qpos[i]
            for i, idx in enumerate(self._robot_qvel_indices):
                self.data.qvel[idx] = saved_qvel[i]

    # ---- Block positioning (freejoint) ----

    def _set_block_pose(self, name, x, y, z, yaw):
        """Set a free body's position via its freejoint qpos."""
        if name not in self.freejoint_map:
            return
        qadr = self.freejoint_map[name]
        self.data.qpos[qadr:qadr + 3] = [x, y, z]
        self.data.qpos[qadr + 3:qadr + 7] = _yaw_to_quat(yaw)
        # Zero velocity
        body_id = self._block_body_ids.get(name, -1)
        if body_id >= 0:
            jnt_id = self.model.body_jntadr[body_id]
            if jnt_id >= 0:
                vadr = self.model.jnt_dofadr[jnt_id]
                self.data.qvel[vadr:vadr + 6] = 0.0

    def _get_block_pose(self, name):
        """Read a free body's (x, y, z, qw, qx, qy, qz) from freejoint qpos."""
        if name not in self.freejoint_map:
            return None
        qadr = self.freejoint_map[name]
        pos = self.data.qpos[qadr:qadr + 3].copy()
        quat = self.data.qpos[qadr + 3:qadr + 7].copy()
        return pos, quat

    def _read_block_poses(self):
        """Update _block_true_poses from freejoint qpos."""
        for name in BLOCK_NAMES:
            result = self._get_block_pose(name)
            if result is not None:
                pos, quat = result
                # Extract yaw from quaternion (rotation about z)
                yaw = 2.0 * math.atan2(quat[3], quat[0])
                self._block_true_poses[name] = (pos[0], pos[1], yaw)

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
        """Place blocks on table. Target spawns freely; distractors biased near target."""
        positions = []
        target_xy = None

        for idx, name in enumerate(BLOCK_NAMES):
            is_distractor = (name in DISTRACTOR_BLOCKS)
            placed = False

            # Distractors: always spawn within DISTRACTOR_NEAR_TARGET_RADIUS of target.
            # Only fall back to free spawn if near-spawn fails (rare geometry fail at workspace edge).
            attempt_ranges = []
            if is_distractor and target_xy is not None:
                attempt_ranges.append(("near", 200))
            attempt_ranges.append(("free", 100))

            for mode, n_attempts in attempt_ranges:
                for _ in range(n_attempts):
                    if mode == "near":
                        # Sample offset around target within DISTRACTOR_NEAR_TARGET_RADIUS
                        angle_off = self.np_random.uniform(0, 2 * math.pi)
                        dist_off = self.np_random.uniform(MIN_SPACING, DISTRACTOR_NEAR_TARGET_RADIUS)
                        x = target_xy[0] + dist_off * math.cos(angle_off)
                        y = target_xy[1] + dist_off * math.sin(angle_off)
                        # Skip if outside workspace
                        r = math.sqrt(x ** 2 + y ** 2)
                        if r < SPAWN_R_MIN or r > SPAWN_R_MAX:
                            continue
                    else:
                        angle = self.np_random.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
                        r = self.np_random.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
                        x = r * math.cos(angle)
                        y = r * math.sin(angle)

                    yaw = self.np_random.uniform(-math.pi, math.pi)
                    too_close = any(
                        math.sqrt((x - px) ** 2 + (y - py) ** 2) < MIN_SPACING
                        for px, py, _ in positions
                    )
                    if not too_close:
                        positions.append((x, y, yaw))
                        if idx == 0:
                            target_xy = (x, y)
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                # Last-resort fallback
                fallback_angle = SPAWN_ANGLE_MIN + (idx / len(BLOCK_NAMES)) * (
                    SPAWN_ANGLE_MAX - SPAWN_ANGLE_MIN
                )
                r = (SPAWN_R_MIN + SPAWN_R_MAX) / 2
                x = r * math.cos(fallback_angle)
                y = r * math.sin(fallback_angle)
                positions.append((x, y, 0.0))
                if idx == 0:
                    target_xy = (x, y)

        for i, name in enumerate(BLOCK_NAMES):
            x, y, yaw = positions[i]
            self._block_true_poses[name] = (x, y, yaw)
            self._set_block_pose(name, x, y, BLOCK_HALF_Z[name], yaw)

    def _sample_table_position(self):
        """Sample a goal position on the table, away from blocks."""
        for _ in range(100):
            angle = self.np_random.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
            r = self.np_random.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            too_close = False
            for name in BLOCK_NAMES:
                if name in self._block_true_poses:
                    bx, by, _ = self._block_true_poses[name]
                    if math.sqrt((x - bx) ** 2 + (y - by) ** 2) < MIN_SPACING * 2:
                        too_close = True
                        break
            if not too_close:
                return np.array([x, y])

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
        noise[2] = self.np_random.normal(0, sigma * 10)
        return np.array([true[0] + noise[0], true[1] + noise[1], true[2] + noise[2]])

    _CAM_PITCH = math.pi / 7.2  # 25° down from horizontal

    def _get_camera_state(self):
        """Virtual wrist camera: co-located with EE, pointed toward the target block
        with a fixed 25° downward pitch.

        Target-tracking ensures the camera always has the target in the frustum,
        which is essential for the occlusion check to be meaningful. Distractors
        near the target can then naturally block the view.
        """
        cam_pos = self._ee_pos.copy()

        target = self._block_true_poses[TARGET_BLOCK]
        dx = target[0] - cam_pos[0]
        dy = target[1] - cam_pos[1]
        if dx * dx + dy * dy < 1e-8:
            yaw = 0.0
        else:
            yaw = math.atan2(dy, dx)

        cp = math.cos(self._CAM_PITCH)
        sp = math.sin(self._CAM_PITCH)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        fwd = np.array([cp * cy, cp * sy, -sp])
        right = np.array([-sy, cy, 0.0])
        up = np.cross(fwd, right)

        cam_rot = np.column_stack([fwd, right, up])
        return cam_pos, cam_rot

    def _is_target_occluded(self):
        """Check if the target block is occluded by any distractor (wrist camera)."""
        target = self._block_true_poses[TARGET_BLOCK]
        cam_pos, cam_rot = self._get_camera_state()

        for d_name in DISTRACTOR_BLOCKS:
            distractor = self._block_true_poses[d_name]
            if is_occluded(
                target_pos=(target[0], target[1]),
                target_half_size=HALF_SIZES[TARGET_BLOCK],
                target_yaw=target[2],
                occluder_pos=(distractor[0], distractor[1]),
                occluder_half_size=HALF_SIZES[d_name],
                occluder_yaw=distractor[2],
                camera_pos=cam_pos,
                camera_rot=cam_rot,
                target_half_z=BLOCK_HALF_Z[TARGET_BLOCK],
                occluder_half_z=BLOCK_HALF_Z[d_name],
            ):
                return True
        return False

    def _get_visible_observations(self):
        """Get observations for visible blocks (dict: block_idx -> obs)."""
        if self._is_target_occluded():
            return {}
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
        """Check if target block is occluded from overhead by any distractor."""
        target = self._block_true_poses[TARGET_BLOCK]
        for d_name in DISTRACTOR_BLOCKS:
            distractor = self._block_true_poses[d_name]
            if is_occluded_overhead(
                target_pos=(target[0], target[1]),
                occluder_pos=(distractor[0], distractor[1]),
                occluder_half_size=HALF_SIZES[d_name],
                occluder_yaw=distractor[2],
            ):
                return True
        return False

    def _get_overhead_visible_observations(self):
        """Get overhead camera observations for visible blocks."""
        if self._is_target_occluded_overhead():
            return {}
        noisy_obs = self._get_overhead_noisy_obs()
        return {0: noisy_obs}

    def _attempt_grasp(self):
        """Check proximity-based grasp against ALL blocks (2D XY check).

        Returns the name of the closest block within GRASP_THRESHOLD,
        or None if no block is close enough. Grasping a distractor is
        valid but penalized in the reward function.
        """
        closest_name = None
        closest_dist = float('inf')

        for name in BLOCK_NAMES:
            if name not in self._block_true_poses:
                continue
            pose = self._block_true_poses[name]
            block_pos = np.array([pose[0], pose[1], BLOCK_HALF_Z[name]])
            dist_xy = np.linalg.norm(self._ee_pos[:2] - block_pos[:2])
            if dist_xy < GRASP_THRESHOLD and dist_xy < closest_dist:
                closest_dist = dist_xy
                closest_name = name

        if closest_name is not None:
            pose = self._block_true_poses[closest_name]
            block_pos = np.array([pose[0], pose[1], BLOCK_HALF_Z[closest_name]])
            self._grasp_offset = block_pos - self._ee_pos
            return closest_name
        return None

    def _constrain_held_block(self):
        """Move held block to stay attached to gripper (position constraint).

        Sets the block freejoint position relative to the EE each step.
        Zeros block velocity so it doesn't drift.
        """
        held = self._holding_block
        if not held or held not in self.freejoint_map:
            return
        qadr = self.freejoint_map[held]
        # Block follows EE with the grasp offset
        new_pos = self._ee_pos + self._grasp_offset
        # Keep z at least at table level
        new_pos[2] = max(TABLE_Z, new_pos[2])
        self.data.qpos[qadr:qadr + 3] = new_pos
        # Zero velocity
        body_id = self._block_body_ids[held]
        jnt_id = self.model.body_jntadr[body_id]
        if jnt_id >= 0:
            vadr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[vadr:vadr + 6] = 0.0

    def _apply_target_drift(self):
        """Apply small random XY perturbation to the unheld target block."""
        if TARGET_BLOCK not in self.freejoint_map:
            return
        drift = self.np_random.normal(0, self.sigma_drift, 2)
        qadr = self.freejoint_map[TARGET_BLOCK]
        self.data.qpos[qadr] += drift[0]
        self.data.qpos[qadr + 1] += drift[1]
        # Refresh true pose cache so observations and rewards use the drifted position
        self._read_block_poses()

    def _release_block(self, block_name=None):
        """Release block at current position. It will fall under gravity."""
        name = block_name or TARGET_BLOCK
        if name not in self.freejoint_map:
            return
        qadr = self.freejoint_map[name]
        # Set block at table height at EE xy position
        self.data.qpos[qadr] = self._ee_pos[0]
        self.data.qpos[qadr + 1] = self._ee_pos[1]
        self.data.qpos[qadr + 2] = TABLE_Z
        # Zero velocity
        body_id = self._block_body_ids[name]
        jnt_id = self.model.body_jntadr[body_id]
        if jnt_id >= 0:
            vadr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[vadr:vadr + 6] = 0.0
        # Update true pose
        yaw = self._block_true_poses[TARGET_BLOCK][2]
        self._block_true_poses[TARGET_BLOCK] = (
            self._ee_pos[0], self._ee_pos[1], yaw
        )

    def _build_observation(self):
        """Construct observation vector.

        Layout:
          [0:6]   joint angles + gripper
          [6:9]   block obs (wrist noisy / PF mu)
          [9:12]  block obs (PF sigma)  — belief/overhead only
          [9/12:11/14]  goal position (x, y)
          [11/14]       holding flag (0 or 1)
          [12/15]       wrist occluded flag (1.0 = block not visible, 0.0 = visible)
        Plain mode offsets shift by -3 (no sigma slot).
        """
        joint_obs = []
        for name in ARM_JOINT_NAMES:
            if name in self.joint_map:
                joint_obs.append(self.data.qpos[self.joint_map[name]])
            else:
                joint_obs.append(0.0)
        gripper_val = self.data.qpos[self.joint_map["gripper_joint"]]
        joint_obs.append(gripper_val)

        goal_obs = self._goal_pos.tolist()
        holding_obs = [1.0 if self._holding_block == TARGET_BLOCK else 0.0]
        occluded_obs = [1.0 if self._is_target_occluded() else 0.0]

        if self.belief_mode:
            mu, sigma = self.pf.get_belief()
            return np.concatenate(
                [joint_obs, mu[0], sigma[0], goal_obs, holding_obs, occluded_obs]
            ).astype(np.float32)
        else:
            wrist_obs = self._last_wrist_obs
            if self.use_overhead_camera:
                overhead_obs = self._get_overhead_noisy_obs()
                return np.concatenate(
                    [joint_obs, wrist_obs, overhead_obs, goal_obs, holding_obs, occluded_obs]
                ).astype(np.float32)
            return np.concatenate(
                [joint_obs, wrist_obs, goal_obs, holding_obs, occluded_obs]
            ).astype(np.float32)

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
