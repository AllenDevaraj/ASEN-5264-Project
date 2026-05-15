#!/usr/bin/env python3
"""Online POMCP planner for the ROS2/MuJoCo GUI using learned world model.

At each step: update particle filter belief → build 10D state →
run UCB1 MCTS using world_model.pt for rollouts → return discrete action.
Planning is near-instant (~0.04ms) since rollouts use the MLP, not real physics.
Supports full pick-and-place (approach → grasp → carry → drop at goal).

Usage:
    runner = WorldModelPOMCPRunner('models/pomcp', n_rollouts=200, max_depth=20)
    runner.reset_episode()
    runner.update_belief(block_true_poses, ee_pos)
    state  = runner.build_state(ee_pos)
    action_idx = runner.plan(state, goal_xy)   # 0-7
    joints = runner.ik_step(ee_pos, action_idx)
"""

import math
import os

import numpy as np

from so_arm101_control.compute_workspace import geometric_ik
from so_arm101_control.occlusion import is_occluded_overhead
from so_arm101_control.particle_filter import ParticleFilter
from so_arm101_control.pomcp_heuristic import heuristic_action
from so_arm101_control.world_model import WorldModel

# ── Constants matching lego_pick_env.py ────────────────────────────────────
TARGET_BLOCK       = "red_lego_2x4"
DISTRACTOR_BLOCKS  = ["blue_lego_2x2", "blue_lego_2x2_b"]
BLOCK_NAMES        = [TARGET_BLOCK] + DISTRACTOR_BLOCKS
HALF_SIZES = {
    "red_lego_2x4":    (0.016, 0.008),
    "blue_lego_2x2":   (0.020, 0.010),
    "blue_lego_2x2_b": (0.020, 0.010),
}
TABLE_Z        = 0.0055
SIGMA_OVERHEAD = 0.005
SIGMA_LOW      = 0.003
SIGMA_HIGH     = 0.015
GRASP_THRESHOLD = 0.015   # 15 mm XY, matches training
GOAL_THRESHOLD  = 0.040   # 40 mm XY to count placement as success
EE_R_MIN, EE_R_MAX = 0.09, 0.31
EE_Z_MIN, EE_Z_MAX = 0.002, 0.12

# ── 8 discrete actions matching train_pomcp.py ──────────────────────────────
DISCRETE_ACTIONS = {
    0: np.array([ 0.015,  0.0,    0.0,   -1.0]),   # +X
    1: np.array([-0.015,  0.0,    0.0,   -1.0]),   # -X
    2: np.array([ 0.0,    0.015,  0.0,   -1.0]),   # +Y
    3: np.array([ 0.0,   -0.015,  0.0,   -1.0]),   # -Y
    4: np.array([ 0.0,    0.0,   -0.015, -1.0]),   # LOWER
    5: np.array([ 0.0,    0.0,    0.015, -1.0]),   # RAISE
    6: np.array([ 0.0,    0.0,    0.0,    1.0]),   # CLOSE
    7: np.array([ 0.0,    0.0,    0.0,   -1.0]),   # OPEN
}
ACTION_NAMES = ["+X", "-X", "+Y", "-Y", "LOW", "RISE", "CLOS", "OPEN"]
N_ACTIONS = 8


class _Node:
    """Minimal UCB1 tree node."""
    __slots__ = ['visit_count', 'value_sum', 'children']

    def __init__(self):
        self.visit_count = 0
        self.value_sum   = 0.0
        self.children    = {}      # action_idx → _Node

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def ucb1(self, a, c):
        if a not in self.children or self.children[a].visit_count == 0:
            return float('inf')
        child = self.children[a]
        return child.value + c * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def best_ucb(self, c):
        return max(range(N_ACTIONS), key=lambda a: self.ucb1(a, c))

    def best_visits(self):
        if not self.children:
            return 0
        return max(self.children, key=lambda a: self.children[a].visit_count)


class WorldModelPOMCPRunner:
    """Online POMCP planner using learned world_model.pt.

    Belief maintained by a particle filter (identical setup to Belief PPO).
    MCTS uses heuristic rollouts for better value estimates than random.
    """

    def __init__(self, model_dir, n_rollouts=200, max_depth=20,
                 use_camera_noise=True, use_occlusion=True, use_drift=False):
        self.use_camera_noise = use_camera_noise
        self.use_occlusion    = use_occlusion
        self.use_drift        = use_drift
        self.n_rollouts = n_rollouts
        self.max_depth  = max_depth
        self.gamma      = 0.99
        self.ucb_c      = 1.414
        self.rng        = np.random.default_rng()

        wm_path = os.path.join(model_dir, 'world_model.pt')
        self.wm = WorldModel.load(wm_path)

        self.pf = ParticleFilter(n_particles=300, n_blocks=1)

        # Episode state
        self.holding_block  = False
        self.gripper_closed = False
        self.step_count     = 0
        self.sigma_ep       = 0.01
        self._wrist_occluded = False
        self._drifted_poses  = {}

    # ── Episode management ─────────────────────────────────────────────────

    def reset_episode(self):
        self.sigma_ep       = float(self.rng.uniform(SIGMA_LOW, SIGMA_HIGH))
        self.holding_block  = False
        self.gripper_closed = False
        self.step_count     = 0
        self._wrist_occluded = False
        self._drifted_poses  = {}

    # ── Belief update ──────────────────────────────────────────────────────

    def update_belief(self, block_true_poses, ee_pos):
        """Update PF with wrist + overhead noisy observations.
        Matches PolicyRunner._build_belief_obs() exactly."""
        target = block_true_poses.get(TARGET_BLOCK)
        if target is None:
            return

        if self.step_count == 0:
            init_obs = np.array([[target[0], target[1], target[2]]])
            noise = self.rng.normal(0, self.sigma_ep * 3, (1, 3))
            self.pf.reset(init_obs + noise, sigma_init=0.05)

        self.pf.predict()

        # Wrist observation with distance-dependent noise
        sigma = self.sigma_ep
        if self.use_camera_noise:
            dist = np.linalg.norm(ee_pos - np.array([target[0], target[1], TABLE_Z]))
            sigma = max(sigma, 0.008 * (dist / 0.15))

        wrist_noise    = self.rng.normal(0, sigma, 3)
        wrist_noise[2] = self.rng.normal(0, sigma * 10)
        wrist_obs = np.array([target[0] + wrist_noise[0],
                              target[1] + wrist_noise[1],
                              target[2] + wrist_noise[2]])
        self.pf.update({0: wrist_obs}, sigma)

        # Overhead occlusion check
        overhead_occluded = self.use_occlusion and any(
            is_occluded_overhead(
                target_pos=(target[0], target[1]),
                occluder_pos=(block_true_poses[d][0], block_true_poses[d][1]),
                occluder_half_size=HALF_SIZES[d],
                occluder_yaw=block_true_poses[d][2],
            )
            for d in DISTRACTOR_BLOCKS if d in block_true_poses
        )
        self._wrist_occluded = overhead_occluded

        if not overhead_occluded:
            oh_noise = self.rng.normal(0, SIGMA_OVERHEAD, 2)
            oh_obs = np.array([
                target[0] + oh_noise[0],
                target[1] + oh_noise[1],
                target[2] + self.rng.normal(0, SIGMA_OVERHEAD * 10),
            ])
            self.pf.update({0: oh_obs}, SIGMA_OVERHEAD)

        self.pf.resample()
        self.step_count += 1

    # ── State construction ─────────────────────────────────────────────────

    def build_state(self, ee_pos):
        """Build 10D belief state: [ee_pos(3), mu(3), sigma(3), gripper(1)]."""
        mu, sigma   = self.pf.get_belief()
        gripper_val = 1.0 if self.gripper_closed else 0.0
        return np.concatenate([ee_pos, mu[0], sigma[0], [gripper_val]]).astype(np.float32)

    # ── MCTS planning ──────────────────────────────────────────────────────

    def plan(self, state, goal_xy):
        """UCB1 MCTS with world model rollouts. Returns best action index (0-7)."""
        root = _Node()
        for _ in range(self.n_rollouts):
            self._simulate(root, state.copy(), goal_xy, depth=0)
        return root.best_visits()

    def _simulate(self, node, state, goal_pos, depth):
        if depth >= self.max_depth:
            return 0.0

        action_idx          = node.best_ucb(self.ucb_c)
        action              = DISCRETE_ACTIONS[action_idx]
        next_state, grasp_p = self.wm.predict(state, action)
        next_state[3:6]     = state[3:6]   # freeze block_mu — WM drifts it toward EE

        holding = state[9] > 0.5
        reward  = -1.0  # step cost

        terminal = False
        if action_idx == 6:  # CLOSE
            if self.rng.random() < grasp_p:
                reward += 20.0   # grasp bonus
                dist = np.linalg.norm(next_state[:2] - goal_pos)
                if dist < GOAL_THRESHOLD:
                    reward  += 80.0
                    terminal = True
            else:
                reward -= 5.0
        elif action_idx == 7 and holding:   # OPEN while carrying
            dist = np.linalg.norm(next_state[:2] - goal_pos)
            if dist < GOAL_THRESHOLD:
                reward  += 80.0
                terminal = True

        if terminal:
            self._backup(node, action_idx, reward)
            return reward

        if action_idx not in node.children:
            node.children[action_idx] = _Node()
            future = self._heuristic_rollout(next_state, goal_pos, depth + 1)
        else:
            future = self._simulate(
                node.children[action_idx], next_state, goal_pos, depth + 1)

        total = reward + self.gamma * future
        self._backup(node, action_idx, total)
        return total

    def _heuristic_rollout(self, state, goal_pos, depth):
        """Phase-based heuristic rollout for leaf value estimation."""
        total    = 0.0
        discount = 1.0
        for _ in range(depth, self.max_depth):
            ee_pos    = state[:3]
            block_mu  = state[3:6]
            holding   = state[9] > 0.5
            h_idx     = heuristic_action(
                ee_pos=ee_pos, block_mu=block_mu,
                goal_xy=goal_pos, holding=holding,
                gripper_closed=holding,
            )
            action              = DISCRETE_ACTIONS[h_idx]
            next_state, grasp_p = self.wm.predict(state, action)
            next_state[3:6]     = state[3:6]   # freeze block_mu — WM drifts it toward EE

            r = -1.0
            if h_idx == 6 and self.rng.random() < grasp_p:
                r += 20.0
                if np.linalg.norm(next_state[:2] - goal_pos) < GOAL_THRESHOLD:
                    r     += 80.0
                    total += discount * r
                    break
            elif h_idx == 7 and holding:
                if np.linalg.norm(next_state[:2] - goal_pos) < GOAL_THRESHOLD:
                    r     += 80.0
                    total += discount * r
                    break

            total    += discount * r
            discount *= self.gamma
            state     = next_state
        return total

    @staticmethod
    def _backup(node, action_idx, total):
        node.visit_count += 1
        node.value_sum   += total
        if action_idx not in node.children:
            node.children[action_idx] = _Node()
        node.children[action_idx].visit_count += 1
        node.children[action_idx].value_sum   += total

    # ── Robot interface ────────────────────────────────────────────────────

    def ik_step(self, ee_pos, action_idx):
        """Translate discrete action to joint targets via IK. Returns dict or None."""
        a = DISCRETE_ACTIONS[action_idx]
        dx, dy, dz = float(a[0]), float(a[1]), float(a[2])

        new_x = ee_pos[0] + dx
        new_y = ee_pos[1] + dy
        new_z = ee_pos[2] + dz

        r = math.sqrt(new_x**2 + new_y**2)
        if r < EE_R_MIN:
            s = EE_R_MIN / max(r, 1e-6); new_x *= s; new_y *= s
        elif r > EE_R_MAX:
            s = EE_R_MAX / r;            new_x *= s; new_y *= s
        new_z = float(np.clip(new_z, EE_Z_MIN, EE_Z_MAX))

        sols = geometric_ik(new_x, new_y, new_z, grasp_yaw=0.0)
        return sols[0] if sols else None

    def check_grasp(self, ee_pos, block_true_poses):
        """2D XY proximity check matching training env."""
        target = block_true_poses.get(TARGET_BLOCK)
        if target is None:
            return False
        if np.linalg.norm(ee_pos[:2] - np.array([target[0], target[1]])) < GRASP_THRESHOLD:
            self.holding_block  = True
            self.gripper_closed = True
            return True
        return False

    def check_placement(self, ee_pos, goal_xy):
        """True if holding and EE is within placement threshold of goal."""
        return (self.holding_block
                and np.linalg.norm(ee_pos[:2] - goal_xy) < GOAL_THRESHOLD)

    # ── Block placement helpers (shared with PolicyRunner) ─────────────────

    def randomize_blocks(self):
        SPAWN_R_MIN, SPAWN_R_MAX   = 0.12, 0.22
        SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX = -1.0, 1.0
        MIN_SPACING = 0.03
        positions, result = [], {}
        for name in BLOCK_NAMES:
            for _ in range(100):
                angle = self.rng.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
                r     = self.rng.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
                x     = r * math.cos(angle)
                y     = r * math.sin(angle)
                yaw   = self.rng.uniform(-math.pi, math.pi)
                if not any(math.hypot(x - px, y - py) < MIN_SPACING
                           for px, py, _ in positions):
                    positions.append((x, y, yaw))
                    result[name] = (x, y, yaw)
                    break
            else:
                fa = SPAWN_ANGLE_MIN + (len(positions) / len(BLOCK_NAMES)) * (
                    SPAWN_ANGLE_MAX - SPAWN_ANGLE_MIN)
                r = (SPAWN_R_MIN + SPAWN_R_MAX) / 2
                positions.append((r * math.cos(fa), r * math.sin(fa), 0.0))
                result[name] = positions[-1]
        return result

    def sample_goal(self, block_true_poses):
        SPAWN_R_MIN, SPAWN_R_MAX   = 0.12, 0.22
        SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX = -1.0, 1.0
        for _ in range(100):
            angle = self.rng.uniform(SPAWN_ANGLE_MIN, SPAWN_ANGLE_MAX)
            r     = self.rng.uniform(SPAWN_R_MIN, SPAWN_R_MAX)
            x, y  = r * math.cos(angle), r * math.sin(angle)
            if not any(
                math.hypot(x - bx, y - by) < 0.06
                for n in BLOCK_NAMES if n in block_true_poses
                for bx, by, _ in [block_true_poses[n]]
            ):
                return np.array([x, y])
        return np.array([0.18, -0.05])
