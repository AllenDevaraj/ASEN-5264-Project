#!/usr/bin/env python3
"""Phase 4: POMCP with Learned World Model (stretch goal).

Steps:
  1. Collect transitions from a trained Belief PPO policy
  2. Train a world model MLP on those transitions
  3. Run POMCP online planning using the world model for rollouts

Usage:
    # Step 1+2: Collect data and train world model
    python3 train_pomcp.py --collect --belief-model models/ppo_belief/best_model

    # Step 3: Evaluate POMCP planner
    python3 train_pomcp.py --evaluate --world-model models/pomcp/world_model.pt
"""

import argparse
import math
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from so_arm101_control.lego_pick_env import LegoPickEnv


# --- Discrete action mapping for tree search ---
DISCRETE_ACTIONS = {
    0: np.array([0.015, 0.0, 0.0, -1.0]),    # MOVE +X
    1: np.array([-0.015, 0.0, 0.0, -1.0]),   # MOVE -X
    2: np.array([0.0, 0.015, 0.0, -1.0]),    # MOVE +Y
    3: np.array([0.0, -0.015, 0.0, -1.0]),   # MOVE -Y
    4: np.array([0.0, 0.0, -0.015, -1.0]),   # LOWER
    5: np.array([0.0, 0.0, 0.015, -1.0]),    # RAISE
    6: np.array([0.0, 0.0, 0.0, 1.0]),       # CLOSE gripper
    7: np.array([0.0, 0.0, 0.0, -1.0]),      # OPEN gripper
}
ACTION_NAMES = ["+X", "-X", "+Y", "-Y", "LOWER", "RAISE", "CLOSE", "OPEN"]


def collect_transitions(model_path, n_transitions=50000, seed=0):
    """Collect transitions from trained Belief PPO for world model training.

    Returns:
        dict with 'states', 'actions', 'next_states', 'grasp_success' arrays.
    """
    from stable_baselines3 import PPO

    print(f"Loading Belief PPO from {model_path}...")
    policy = PPO.load(model_path)
    env = LegoPickEnv(belief_mode=True)

    states = []
    actions = []
    next_states = []
    grasp_successes = []

    obs, info = env.reset(seed=seed)
    collected = 0

    while collected < n_transitions:
        # Build state vector: [ee_pos[3], mu[3], sigma[3], gripper[1]]
        mu, sigma = env.pf.get_belief()
        gripper_val = 1.0 if env._gripper_closed else 0.0
        state = np.concatenate([env._ee_pos, mu[0], sigma[0], [gripper_val]])

        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        # Build next state
        mu_next, sigma_next = env.pf.get_belief()
        gripper_next = 1.0 if env._gripper_closed else 0.0
        next_state = np.concatenate([env._ee_pos, mu_next[0], sigma_next[0], [gripper_next]])

        grasp = 1.0 if info.get("grasp_result") == "success" else 0.0

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        grasp_successes.append(grasp)
        collected += 1

        if terminated or truncated:
            obs, info = env.reset()

        if collected % 10000 == 0:
            print(f"  Collected {collected}/{n_transitions} transitions")

    env.close()

    return {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'next_states': np.array(next_states, dtype=np.float32),
        'grasp_success': np.array(grasp_successes, dtype=np.float32),
    }


class POMCPNode:
    """Node in the POMCP search tree."""

    def __init__(self, n_actions=8):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> POMCPNode
        self.n_actions = n_actions

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb1(self, action, c=1.414):
        """UCB1 score for action selection."""
        if action not in self.children:
            return float('inf')
        child = self.children[action]
        if child.visit_count == 0:
            return float('inf')
        exploit = child.value
        explore = c * math.sqrt(math.log(self.visit_count) / child.visit_count)
        return exploit + explore

    def best_action(self, c=1.414):
        """Select action with highest UCB1."""
        best_a = 0
        best_score = -float('inf')
        for a in range(self.n_actions):
            score = self.ucb1(a, c)
            if score > best_score:
                best_score = score
                best_a = a
        return best_a


class POMCPPlanner:
    """POMCP planner using learned world model for rollouts."""

    def __init__(self, world_model, n_rollouts=500, max_depth=20,
                 gamma=0.99, ucb_c=1.414):
        """
        Args:
            world_model: WorldModel instance for simulating transitions.
            n_rollouts: Number of MCTS rollouts per planning step.
            max_depth: Maximum rollout depth.
            gamma: Discount factor.
            ucb_c: UCB1 exploration constant.
        """
        self.world_model = world_model
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.gamma = gamma
        self.ucb_c = ucb_c
        self.rng = np.random.default_rng()

    def plan(self, state, goal_pos):
        """Run POMCP and return best discrete action index.

        Args:
            state: (10,) — [ee_pos[3], mu[3], sigma[3], gripper[1]]
            goal_pos: (2,) — goal xy position

        Returns:
            best_action: int (0-7)
        """
        root = POMCPNode()

        for _ in range(self.n_rollouts):
            self._simulate(root, state.copy(), goal_pos, depth=0)

        # Select most-visited child
        best_a = max(range(8), key=lambda a: root.children.get(a, POMCPNode()).visit_count)
        return best_a

    def _simulate(self, node, state, goal_pos, depth):
        """Recursive MCTS simulation."""
        if depth >= self.max_depth:
            return 0.0

        # Select action via UCB1
        action_idx = node.best_action(self.ucb_c)
        action = DISCRETE_ACTIONS[action_idx]

        # Simulate transition using world model
        next_state, grasp_prob = self.world_model.predict(state, action)

        # Compute immediate reward
        reward = -1.0  # step cost

        # Check if grasping
        if action_idx == 6:  # CLOSE
            if self.rng.random() < grasp_prob:
                # Check if at goal
                dist_to_goal = np.linalg.norm(next_state[:2] - goal_pos)
                if dist_to_goal < 0.015:
                    reward += 10.0
                    return reward
            else:
                reward -= 5.0

        # Expand or recurse
        if action_idx not in node.children:
            node.children[action_idx] = POMCPNode()
            # Rollout with random policy
            rollout_return = self._random_rollout(next_state, goal_pos, depth + 1)
            total = reward + self.gamma * rollout_return
        else:
            total = reward + self.gamma * self._simulate(
                node.children[action_idx], next_state, goal_pos, depth + 1
            )

        # Backpropagate
        node.visit_count += 1
        node.value_sum += total
        return total

    def _random_rollout(self, state, goal_pos, depth):
        """Random rollout for value estimation."""
        total_return = 0.0
        discount = 1.0

        for d in range(depth, self.max_depth):
            action_idx = self.rng.integers(0, 8)
            action = DISCRETE_ACTIONS[action_idx]
            state, grasp_prob = self.world_model.predict(state, action)

            r = -1.0
            if action_idx == 6:
                if self.rng.random() < grasp_prob:
                    dist = np.linalg.norm(state[:2] - goal_pos)
                    if dist < 0.015:
                        r += 10.0
                        total_return += discount * r
                        break
                else:
                    r -= 5.0

            total_return += discount * r
            discount *= self.gamma

        return total_return


# --- Direct Simulator POMCP ---


def _worker_loop(task_queue, result_queue, belief_mode):
    """Persistent worker process: owns one env, runs rollouts on demand.

    Receives: (snapshot, action_idx, n_rollouts, gamma)
    Sends:    (action_idx, mean_return)
    Sentinel: None on task_queue means shutdown.
    """
    from so_arm101_control.lego_pick_env import LegoPickEnv
    from so_arm101_control.pomcp_env_bridge import restore_state
    from so_arm101_control.pomcp_heuristic import heuristic_action

    env = LegoPickEnv(belief_mode=belief_mode)
    env.reset(seed=0)

    while True:
        msg = task_queue.get()
        if msg is None:
            break

        snapshot, action_idx, n_rollouts, gamma = msg
        returns = []

        for _ in range(n_rollouts):
            restore_state(env, snapshot)

            # First step: execute the assigned action
            action = DISCRETE_ACTIONS[action_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            total_return = reward
            discount = gamma

            # Continue with heuristic policy
            while not (terminated or truncated):
                # Extract state for heuristic
                if env.belief_mode:
                    mu, _ = env.pf.get_belief()
                    block_mu = mu[0]
                else:
                    block_mu = np.array([
                        env._block_true_poses["red_lego_2x4"][0],
                        env._block_true_poses["red_lego_2x4"][1],
                        env._block_true_poses["red_lego_2x4"][2],
                    ])

                h_action_idx = heuristic_action(
                    ee_pos=env._ee_pos,
                    block_mu=block_mu,
                    goal_xy=env._goal_pos,
                    holding=env._holding_block,
                    gripper_closed=env._gripper_closed,
                )
                action = DISCRETE_ACTIONS[h_action_idx]
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += discount * reward
                discount *= gamma

            returns.append(total_return)

        result_queue.put((action_idx, float(np.mean(returns))))

    env.close()


class DirectPOMCPPlanner:
    """POMCP planner using real MuJoCo env for rollouts."""

    def __init__(self, n_rollouts=100, n_workers=8, gamma=0.99, belief_mode=True):
        self.n_rollouts = n_rollouts
        self.n_workers = n_workers
        self.gamma = gamma
        self.n_actions = len(DISCRETE_ACTIONS)

        self._task_queues = []
        self._result_queue = mp.Queue()
        self._workers = []

        for _ in range(n_workers):
            tq = mp.Queue()
            p = mp.Process(target=_worker_loop,
                           args=(tq, self._result_queue, belief_mode),
                           daemon=True)
            p.start()
            self._task_queues.append(tq)
            self._workers.append(p)

    def plan(self, snapshot):
        """Run parallel rollouts for all 8 actions and return best.

        Args:
            snapshot: dict from serialize_state(env).

        Returns:
            int: best action index (0-7).
        """
        rollouts_per_worker = max(1, self.n_rollouts // self.n_workers)
        tasks_sent = 0

        for action_idx in range(self.n_actions):
            for w in range(self.n_workers):
                self._task_queues[w].put(
                    (snapshot, action_idx, rollouts_per_worker, self.gamma)
                )
                tasks_sent += 1

        q_values = {a: [] for a in range(self.n_actions)}
        for _ in range(tasks_sent):
            action_idx, mean_return = self._result_queue.get()
            q_values[action_idx].append(mean_return)

        q_means = {a: np.mean(vs) for a, vs in q_values.items()}
        return max(q_means, key=q_means.get)

    def close(self):
        """Shutdown all worker processes."""
        for tq in self._task_queues:
            tq.put(None)
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def evaluate_pomcp(world_model_path, n_episodes=100, n_rollouts=500):
    """Evaluate POMCP planner on the environment."""
    from so_arm101_control.world_model import WorldModel

    print(f"Loading world model from {world_model_path}...")
    wm = WorldModel.load(world_model_path)
    planner = POMCPPlanner(wm, n_rollouts=n_rollouts)

    env = LegoPickEnv(belief_mode=True)
    successes = 0
    total_steps = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        goal_pos = info["goal_pos"]
        done = False
        steps = 0

        while not done:
            # Build state for planner
            mu, sigma = env.pf.get_belief()
            gripper_val = 1.0 if env._gripper_closed else 0.0
            state = np.concatenate([env._ee_pos, mu[0], sigma[0], [gripper_val]])

            # Plan
            action_idx = planner.plan(state, goal_pos)
            action = DISCRETE_ACTIONS[action_idx]

            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        if terminated:
            successes += 1
        total_steps.append(steps)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"success_rate={successes/(ep+1)*100:.1f}%, "
                  f"avg_steps={np.mean(total_steps):.1f}")

    env.close()

    print(f"\nPOMCP Results ({n_episodes} episodes):")
    print(f"  Success rate: {successes/n_episodes*100:.1f}%")
    print(f"  Mean steps: {np.mean(total_steps):.1f}")
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="POMCP with Learned World Model")
    parser.add_argument("--collect", action="store_true",
                        help="Collect transitions and train world model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate POMCP planner")
    parser.add_argument("--belief-model", type=str,
                        default="models/ppo_belief/best_model")
    parser.add_argument("--world-model", type=str,
                        default="models/pomcp/world_model.pt")
    parser.add_argument("--n-transitions", type=int, default=50000)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--n-rollouts", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./models/pomcp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.collect:
        # Step 1: Collect transitions
        print("Step 1: Collecting transitions from Belief PPO...")
        transitions = collect_transitions(
            args.belief_model, n_transitions=args.n_transitions
        )

        # Save transitions
        trans_path = os.path.join(args.output_dir, "transitions.npz")
        np.savez(trans_path, **transitions)
        print(f"Transitions saved to {trans_path}")

        # Step 2: Train world model
        print("\nStep 2: Training world model...")
        from so_arm101_control.world_model import WorldModel

        wm = WorldModel()
        wm.train_on_buffer(transitions, epochs=args.epochs)
        wm.save(os.path.join(args.output_dir, "world_model.pt"))
        print(f"World model saved to {args.output_dir}/world_model.pt")

    if args.evaluate:
        # Step 3: Evaluate POMCP
        print("Step 3: Evaluating POMCP planner...")
        evaluate_pomcp(
            args.world_model,
            n_episodes=args.n_episodes,
            n_rollouts=args.n_rollouts,
        )

    if not args.collect and not args.evaluate:
        print("Specify --collect and/or --evaluate. See --help.")


if __name__ == "__main__":
    main()
