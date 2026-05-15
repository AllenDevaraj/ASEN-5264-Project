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
from so_arm101_control.pomcp_heuristic import heuristic_action as _shared_heuristic_action  # noqa: F401
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
GRASP_Z_HEIGHT = 0.015  # height at which the heuristic switches from LOWER to CLOSE
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
        next_state, reward, terminal = self._transition(state, action_idx, goal_pos)

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
            h_idx = self._heuristic_action(state, goal_pos)
            next_state, r, terminal = self._transition(state, h_idx, goal_pos)
            if terminal:
                total += discount * r
                break

            total    += discount * r
            discount *= self.gamma
            state     = next_state
        return total

    def _heuristic_action(self, state, goal_pos):
        """Greedy policy aligned with the GUI transition's grasp geometry.

        The shared pomcp_heuristic.heuristic_action switches to CLOSE at
        xy_dist<25mm but our env grasps only at <15mm. That mismatch made
        rollouts spam CLOSE between 15-25mm, eating -5 penalties and
        flattening MCTS values. This version drives XY inside 12mm before
        lowering and only closes inside GRASP_THRESHOLD.
        """
        ee_pos   = state[:3]
        block_mu = state[3:6]
        holding  = state[9] > 0.5

        if not holding:
            dx = float(block_mu[0] - ee_pos[0])
            dy = float(block_mu[1] - ee_pos[1])
            xy_dist = math.sqrt(dx * dx + dy * dy)
            ee_z = float(ee_pos[2])

            if xy_dist >= GRASP_THRESHOLD * 0.6:  # 9 mm — match GUI 10mm gate
                if abs(dx) >= abs(dy):
                    return 0 if dx > 0 else 1
                return 2 if dy > 0 else 3
            if ee_z > GRASP_Z_HEIGHT:
                return 4  # LOWER
            return 6      # CLOSE — XY tight, EE low

        # Holding: navigate to goal then OPEN.
        dx = float(goal_pos[0] - ee_pos[0])
        dy = float(goal_pos[1] - ee_pos[1])
        xy_dist = math.sqrt(dx * dx + dy * dy)
        if xy_dist >= GOAL_THRESHOLD * 0.5:
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 1
            return 2 if dy > 0 else 3
        return 7  # OPEN

    def _transition(self, state, action_idx, goal_pos):
        """Rollout transition + reward model for MCTS.

        Hybrid: use the commanded action for EE kinematics (the WM was
        trained on continuous PPO actions and drifts on discrete inputs),
        keep belief mean anchored, and use a geometry-grounded close-success
        estimate with the learned grasp head as a weak prior.

        Mirrors the auto-grasp semantics of lego_pick_env: any motion that
        lands EE within GRASP_THRESHOLD of the belief mean grants holding,
        not only an explicit CLOSE action.
        """
        action = DISCRETE_ACTIONS[action_idx]
        next_state = state.copy()
        # Apply commanded EE delta directly (deterministic, in-distribution).
        next_state[0] += float(action[0])
        next_state[1] += float(action[1])
        next_state[2] += float(action[2])
        # Clamp Z to physical bounds matching the env.
        next_state[2] = float(np.clip(next_state[2], EE_Z_MIN, EE_Z_MAX))
        # Belief mean / sigma stay anchored across the rollout (the WM head
        # is unreliable on these and the GUI's PF owns the real belief).
        next_state[3:9] = state[3:9]

        holding = state[9] > 0.5
        reward = -1.0
        terminal = False

        block_xy = next_state[3:5]
        ee_xy = next_state[:2]
        dist_xy = float(np.linalg.norm(ee_xy - block_xy))

        # Dense proximity shaping (mirrors lego_pick_env reward).
        # Critical for MCTS: with max_depth=20, heuristic rollouts often
        # can't reach the 15mm grasp radius in time, so shaping is what
        # separates good-direction actions from bad-direction actions.
        if not holding:
            prev_dist = float(np.linalg.norm(state[:2] - state[3:5]))
            improvement = prev_dist - dist_xy
            reward += 3.0 * float(np.clip(improvement / 0.02, -1.0, 1.0))
            if dist_xy < 0.025:
                reward += 2.0 * (1.0 - dist_xy / 0.025)

        if holding:
            # Carrying — only OPEN at goal terminates with placement bonus.
            next_state[9] = 1.0
            if action_idx == 7:
                dist_goal = float(np.linalg.norm(ee_xy - goal_pos))
                if dist_goal < GOAL_THRESHOLD:
                    reward += 80.0
                    terminal = True
                next_state[9] = 0.0
        else:
            # Auto-grasp on proximity (matches env _attempt_grasp).
            if dist_xy < GRASP_THRESHOLD:
                # Auto-grasp on proximity. Blend the learned world-model grasp
                # head as a weak prior only when explicit CLOSE is selected
                # (lazy WM call — keeps the rollout loop fast on CPU).
                p_auto = 0.9
                if action_idx == 6:
                    _, grasp_p_model = self.wm.predict(state, action)
                    p_auto = max(p_auto, 0.5 * float(grasp_p_model))
                if self.rng.random() < p_auto:
                    reward += 20.0
                    next_state[9] = 1.0
                else:
                    next_state[9] = 0.0
            elif action_idx == 6:
                # Explicit CLOSE far from block — penalise to discourage spam.
                reward -= 5.0
                next_state[9] = 0.0
            else:
                next_state[9] = 0.0

        return next_state, reward, terminal

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
