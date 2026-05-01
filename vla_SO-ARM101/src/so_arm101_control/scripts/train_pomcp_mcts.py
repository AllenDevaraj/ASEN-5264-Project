#!/usr/bin/env python3
"""POMCP with proper MCTS tree search using direct MuJoCo simulator.

Key differences from train_pomcp.py's DirectPOMCPPlanner (flat rollout):
  - UCB1 tree: progressively focuses rollouts on promising action sequences
  - Observation hashing: branches the tree on discretized observations
  - Backpropagation: rollout returns propagate up through ancestor nodes
  - Heuristic rollout: once a leaf is reached, complete with heuristic policy

The tree is rebuilt from scratch each planning step (no persistence across
steps) because POMCP observation branching makes inter-step reuse fragile
in stochastic environments.

Usage:
    python3 train_pomcp_mcts.py --n-episodes 20 --n-simulations 200 --n-workers 3
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from so_arm101_control.lego_pick_env import LegoPickEnv
from so_arm101_control.pomcp_env_bridge import serialize_state, restore_state
from so_arm101_control.pomcp_heuristic import heuristic_action

# --- Discrete action mapping (same as train_pomcp.py) ---
DISCRETE_ACTIONS = {
    0: np.array([0.015, 0.0, 0.0, -1.0]),    # +X
    1: np.array([-0.015, 0.0, 0.0, -1.0]),   # -X
    2: np.array([0.0, 0.015, 0.0, -1.0]),    # +Y
    3: np.array([0.0, -0.015, 0.0, -1.0]),   # -Y
    4: np.array([0.0, 0.0, -0.015, -1.0]),   # LOWER
    5: np.array([0.0, 0.0, 0.015, -1.0]),    # RAISE
    6: np.array([0.0, 0.0, 0.0, 1.0]),       # CLOSE gripper
    7: np.array([0.0, 0.0, 0.0, -1.0]),      # OPEN gripper
}
N_ACTIONS = len(DISCRETE_ACTIONS)


def obs_hash(obs, resolution=0.005):
    """Discretize observation into a hashable key for tree branching.

    Bins each dimension to `resolution` meters (default 5mm).
    This gives the "O" in POMCP — the tree branches on what was observed
    after taking an action, allowing it to condition future plans on
    observations received.
    """
    discretized = tuple(int(round(x / resolution)) for x in obs)
    return discretized


class POMCPNode:
    """Node in the POMCP search tree."""

    __slots__ = ['visit_count', 'value_sum', 'children', 'obs_children']

    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        # action_idx -> ActionNode
        self.children = {}
        # Not used at this level — obs branching is in ActionNode

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb1(self, action_idx, c=1.414):
        if action_idx not in self.children:
            return float('inf')
        child = self.children[action_idx]
        if child.visit_count == 0:
            return float('inf')
        exploit = child.value
        explore = c * math.sqrt(math.log(self.visit_count) / child.visit_count)
        return exploit + explore

    def best_action_ucb(self, c=1.414):
        best_a = 0
        best_score = -float('inf')
        for a in range(N_ACTIONS):
            score = self.ucb1(a, c)
            if score > best_score:
                best_score = score
                best_a = a
        return best_a


class ActionNode:
    """Node representing an action taken from a belief node.

    Branches on observations: obs_hash -> POMCPNode (belief node).
    This is the POMCP observation branching that makes it a POMDP planner.
    """

    __slots__ = ['visit_count', 'value_sum', 'obs_children']

    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0.0
        # obs_hash -> POMCPNode (next belief node)
        self.obs_children = {}

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSPOMCPPlanner:
    """POMCP planner with proper UCB1 tree search on direct MuJoCo simulator.

    Each call to plan() builds a fresh tree with `n_simulations` rollouts.
    Each simulation:
      1. SELECT: walk down tree using UCB1 to pick actions,
         observation hashing to pick belief branches
      2. EXPAND: when a new obs branch is reached, create a leaf node
      3. ROLLOUT: from the leaf, use heuristic policy to estimate value
      4. BACKUP: propagate the return up through the tree
    """

    def __init__(self, n_simulations=200, max_depth=50, gamma=0.99,
                 ucb_c=1.414, belief_mode=True):
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.gamma = gamma
        self.ucb_c = ucb_c
        self.belief_mode = belief_mode

        # Own env for simulations
        self.sim_env = LegoPickEnv(belief_mode=belief_mode, use_camera_noise=True)
        self.sim_env.reset(seed=0)

    def plan(self, snapshot):
        """Build MCTS tree and return best action.

        Args:
            snapshot: dict from serialize_state(env)

        Returns:
            int: best action index (0-7), chosen by most-visited child
        """
        root = POMCPNode()

        for _ in range(self.n_simulations):
            # Restore to root state for each simulation
            restore_state(self.sim_env, snapshot)
            self._simulate(root, depth=0)

        # Select most-visited action (standard MCTS selection at root)
        if not root.children:
            return 0
        best_a = max(
            root.children.keys(),
            key=lambda a: root.children[a].visit_count
        )
        return best_a

    def _simulate(self, node, depth):
        """Recursive MCTS simulation: select, expand, rollout, backup."""
        if depth >= self.max_depth:
            return 0.0

        # SELECT action via UCB1
        action_idx = node.best_action_ucb(self.ucb_c)

        # Execute action in sim
        action = DISCRETE_ACTIONS[action_idx]
        obs, reward, terminated, truncated, info = self.sim_env.step(action)

        if terminated or truncated:
            # Terminal — just return immediate reward
            node.visit_count += 1
            if action_idx not in node.children:
                node.children[action_idx] = ActionNode()
            anode = node.children[action_idx]
            anode.visit_count += 1
            anode.value_sum += reward
            node.value_sum += reward
            return reward

        # Get observation hash for branching
        oh = obs_hash(obs)

        # EXPAND if action node doesn't exist
        if action_idx not in node.children:
            node.children[action_idx] = ActionNode()

        anode = node.children[action_idx]

        # Check if this observation branch exists
        if oh not in anode.obs_children:
            # New observation branch — EXPAND and ROLLOUT
            anode.obs_children[oh] = POMCPNode()
            rollout_return = self._heuristic_rollout(depth + 1)
            total = reward + self.gamma * rollout_return
        else:
            # Existing branch — RECURSE deeper into tree
            child_node = anode.obs_children[oh]
            future = self._simulate(child_node, depth + 1)
            total = reward + self.gamma * future

        # BACKUP
        node.visit_count += 1
        node.value_sum += total
        anode.visit_count += 1
        anode.value_sum += total

        return total

    def _heuristic_rollout(self, depth):
        """Complete episode from current sim state using heuristic policy."""
        total_return = 0.0
        discount = 1.0

        for _ in range(depth, self.max_depth):
            env = self.sim_env

            if env.belief_mode:
                mu, _ = env.pf.get_belief()
                block_mu = mu[0]
            else:
                t = env._block_true_poses["red_lego_2x4"]
                block_mu = np.array([t[0], t[1], t[2]])

            h_action_idx = heuristic_action(
                ee_pos=env._ee_pos,
                block_mu=block_mu,
                goal_xy=env._goal_pos,
                holding=env._holding_block,
                gripper_closed=env._gripper_closed,
            )
            action = DISCRETE_ACTIONS[h_action_idx]
            _, reward, terminated, truncated, info = env.step(action)
            total_return += discount * reward
            discount *= self.gamma

            if terminated or truncated:
                break

        return total_return

    def close(self):
        self.sim_env.close()


# --- Parallel MCTS (distribute simulations across workers) ---

def _mcts_worker_loop(task_queue, result_queue, belief_mode,
                      max_depth, gamma, ucb_c):
    """Worker that runs a batch of MCTS simulations and returns Q-values.

    Protocol (message types on task_queue):
      ("plan", snapshot, n_sims) -> result_queue.put(("plan", q_data))
      ("update", action_idx, oh) -> navigate carry_root; result_queue.put(("update", found))
      ("reset",)                 -> clear carry_root;   result_queue.put(("reset", True))
      None                       -> shutdown
    """
    env = LegoPickEnv(belief_mode=belief_mode, use_camera_noise=True)
    env.reset(seed=0)

    carry_root = None  # Subtree carried over from previous planning step

    while True:
        msg = task_queue.get()
        if msg is None:
            break

        msg_type = msg[0]

        if msg_type == "reset":
            carry_root = None
            result_queue.put(("reset", True))
            continue

        if msg_type == "update":
            _, action_idx, oh = msg
            if (carry_root is not None
                    and action_idx in carry_root.children
                    and oh in carry_root.children[action_idx].obs_children):
                carry_root = carry_root.children[action_idx].obs_children[oh]
                result_queue.put(("update", True))
            else:
                carry_root = None
                result_queue.put(("update", False))
            continue

        # msg_type == "plan"
        _, snapshot, n_sims = msg

        try:
            root = carry_root if carry_root is not None else POMCPNode()

            # Progressive bias: seed fresh roots with heuristic-informed priors
            # so UCB1 doesn't waste sims on obviously-wrong actions.
            # Computed directly from snapshot (zero env steps).
            # Only applied to fresh roots — carry-over roots already have
            # real Q-values that should not be overwritten.
            if not root.children:
                if "pf_particles" in snapshot:
                    w = snapshot["pf_weights"]
                    p = snapshot["pf_particles"]
                    block_mu = np.sum(
                        w[:, np.newaxis, np.newaxis] * p, axis=0
                    )[0]
                else:
                    bp = snapshot["block_true_poses"]["red_lego_2x4"]
                    block_mu = np.array([bp[0], bp[1], bp[2]])

                h_idx = heuristic_action(
                    ee_pos=snapshot["ee_pos"],
                    block_mu=block_mu,
                    goal_xy=snapshot["goal_pos"][:2],
                    holding=snapshot["holding_block"],
                    gripper_closed=snapshot["gripper_closed"],
                )

                # Heuristic action: neutral prior (Q=0)
                # All others: pessimistic prior (Q=V_BAD)
                # N_PRIOR=1 so real visits dominate after ~10 sims/action.
                _V_BAD, _N_PRIOR = -40.0, 1
                for a in range(N_ACTIONS):
                    anode = ActionNode()
                    anode.visit_count = _N_PRIOR
                    anode.value_sum = 0.0 if a == h_idx else _V_BAD * _N_PRIOR
                    root.children[a] = anode
                root.visit_count = N_ACTIONS * _N_PRIOR

            for _ in range(n_sims):
                restore_state(env, snapshot)
                _worker_simulate(env, root, 0, max_depth, gamma, ucb_c)

            carry_root = root  # Save for next step's tree reuse

            q_data = {}
            for a, anode in root.children.items():
                q_data[a] = (anode.visit_count, anode.value_sum)

            result_queue.put(("plan", q_data))
        except Exception:
            import traceback
            traceback.print_exc()
            carry_root = None
            result_queue.put(("plan", {}))

    env.close()


def _heuristic_rollout_to_term(env, gamma, belief_mode):
    """Run heuristic policy from current env state until episode terminates.

    Unlike the bounded rollout in the tree, this runs to completion so that
    value estimates are accurate even for long-horizon episodes (>max_depth steps).
    """
    total_return = 0.0
    discount = 1.0
    while True:
        if belief_mode:
            mu, _ = env.pf.get_belief()
            block_mu = mu[0]
        else:
            t = env._block_true_poses["red_lego_2x4"]
            block_mu = np.array([t[0], t[1], t[2]])
        h_action = heuristic_action(
            ee_pos=env._ee_pos,
            block_mu=block_mu,
            goal_xy=env._goal_pos,
            holding=env._holding_block,
            gripper_closed=env._gripper_closed,
        )
        _, r, terminated, truncated, _ = env.step(DISCRETE_ACTIONS[h_action])
        total_return += discount * r
        discount *= gamma
        if terminated or truncated:
            break
    return total_return


def _worker_simulate(env, node, depth, max_depth, gamma, ucb_c):
    """Single MCTS simulation inside a worker (non-parallel, recursive)."""
    if depth >= max_depth:
        # At tree depth limit: run a full heuristic rollout to termination
        # rather than returning 0. This prevents systematic underestimation
        # of long-horizon solutions on hard episodes.
        return _heuristic_rollout_to_term(env, gamma, env.belief_mode)

    action_idx = node.best_action_ucb(ucb_c)
    action = DISCRETE_ACTIONS[action_idx]
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        node.visit_count += 1
        if action_idx not in node.children:
            node.children[action_idx] = ActionNode()
        anode = node.children[action_idx]
        anode.visit_count += 1
        anode.value_sum += reward
        node.value_sum += reward
        return reward

    oh = obs_hash(obs)

    if action_idx not in node.children:
        node.children[action_idx] = ActionNode()
    anode = node.children[action_idx]

    # Progressive widening on observations (POMCPOW-style):
    # Only expand a new obs branch if budget allows: k * N^alpha.
    # Prevents UCB1 from being forced to explore an explosion of
    # zero-visit leaf nodes in continuous observation spaces.
    _K_OBS, _ALPHA_OBS = 2.0, 0.5
    max_obs = max(1, int(_K_OBS * (anode.visit_count + 1) ** _ALPHA_OBS))

    if oh not in anode.obs_children:
        if len(anode.obs_children) < max_obs:
            # Budget allows: create new obs branch, run heuristic rollout
            # to episode termination (not bounded at max_depth).
            anode.obs_children[oh] = POMCPNode()
            rollout_return = _heuristic_rollout_to_term(env, gamma, env.belief_mode)
            total = reward + gamma * rollout_return
        else:
            # Budget exceeded: route to the most-visited existing obs branch
            best_oh = max(anode.obs_children,
                          key=lambda k: anode.obs_children[k].visit_count)
            child_node = anode.obs_children[best_oh]
            future = _worker_simulate(env, child_node, depth + 1, max_depth, gamma, ucb_c)
            total = reward + gamma * future
    else:
        child_node = anode.obs_children[oh]
        future = _worker_simulate(env, child_node, depth + 1, max_depth, gamma, ucb_c)
        total = reward + gamma * future

    node.visit_count += 1
    node.value_sum += total
    anode.visit_count += 1
    anode.value_sum += total
    return total


class ParallelMCTSPlanner:
    """POMCP with MCTS tree, parallelized across workers.

    Each worker builds its own tree from the same root snapshot.
    Results are merged by aggregating visit counts and value sums
    across all workers' root action nodes — this is the standard
    "root parallelization" scheme for MCTS.
    """

    def __init__(self, n_simulations=200, n_workers=3, max_depth=50,
                 gamma=0.99, ucb_c=1.414, belief_mode=True):
        self.n_simulations = n_simulations
        self.n_workers = n_workers
        self.max_depth = max_depth
        self.gamma = gamma
        self.ucb_c = ucb_c

        self._task_queues = []
        self._result_queue = mp.Queue()
        self._workers = []

        for _ in range(n_workers):
            tq = mp.Queue()
            p = mp.Process(
                target=_mcts_worker_loop,
                args=(tq, self._result_queue, belief_mode,
                      max_depth, gamma, ucb_c),
                daemon=True
            )
            p.start()
            self._task_queues.append(tq)
            self._workers.append(p)

    def plan(self, snapshot):
        """Run parallel MCTS and return best action by merged visit counts."""
        sims_per_worker = max(1, self.n_simulations // self.n_workers)

        for tq in self._task_queues:
            tq.put(("plan", snapshot, sims_per_worker))

        merged = {}
        for _ in range(self.n_workers):
            try:
                msg_type, q_data = self._result_queue.get(timeout=600)
            except queue.Empty:
                raise RuntimeError("Worker timeout — MCTS simulation took >10min")
            for a, (visits, value) in q_data.items():
                if a not in merged:
                    merged[a] = (0, 0.0)
                pv, pval = merged[a]
                merged[a] = (pv + visits, pval + value)

        if not merged:
            return 0

        best_a = max(merged.keys(), key=lambda a: merged[a][0])
        return best_a

    def update(self, action_idx, obs):
        """Carry over the subtree for (action_idx, obs) into next plan call."""
        oh = obs_hash(obs)
        for tq in self._task_queues:
            tq.put(("update", action_idx, oh))
        for _ in range(self.n_workers):
            self._result_queue.get(timeout=30)

    def reset_tree(self):
        """Discard carry-over tree (call at episode start)."""
        for tq in self._task_queues:
            tq.put(("reset",))
        for _ in range(self.n_workers):
            self._result_queue.get(timeout=30)

    def close(self):
        for tq in self._task_queues:
            tq.put(None)
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


# --- Evaluation ---

def evaluate_mcts(n_episodes=20, n_simulations=200, n_workers=3,
                  max_depth=50, gamma=0.99, ucb_c=1.414, seed=0,
                  reset_every=0):
    """Evaluate MCTS POMCP planner.

    Args:
        reset_every: Reset the carry-over tree every N steps within an episode
                     (0 = never reset mid-episode). Prevents stuck episodes from
                     accumulating bad Q-values across hundreds of steps.
    """
    env = LegoPickEnv(belief_mode=True, use_camera_noise=True)
    planner = ParallelMCTSPlanner(
        n_simulations=n_simulations,
        n_workers=n_workers,
        max_depth=max_depth,
        gamma=gamma,
        ucb_c=ucb_c,
    )

    successes = 0
    perfect = 0
    precise = 0
    close = 0
    episode_lengths = []
    episode_returns = []
    planning_times = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        planner.reset_tree()  # Clear carry-over from previous episode
        done = False
        total_return = 0.0
        steps = 0

        while not done:
            step_start = time.time()
            snapshot = serialize_state(env)
            action_idx = planner.plan(snapshot)
            planning_times.append(time.time() - step_start)

            action = DISCRETE_ACTIONS[action_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            steps += 1
            done = terminated or truncated

            if not done:
                if reset_every > 0 and steps % reset_every == 0:
                    planner.reset_tree()
                else:
                    planner.update(action_idx, obs)  # Carry subtree to next step

        episode_lengths.append(steps)
        episode_returns.append(total_return)

        if info.get("success", False):
            successes += 1
            dist = info.get("dist_to_goal", 1.0)
            if dist < 0.01:
                perfect += 1
            elif dist < 0.02:
                precise += 1
            elif dist < 0.04:
                close += 1

        result_str = "✓" if info.get("success", False) else "✗"
        print(f"  [{ep+1:>3}/{n_episodes}] {result_str}  "
              f"steps={steps:>4}  return={total_return:>8.1f}  "
              f"plan={np.mean(planning_times[-steps:]):.1f}s/step  "
              f"success={successes}/{ep+1} ({successes/(ep+1)*100:.0f}%)",
              flush=True)

    planner.close()
    env.close()

    results = {
        "method": "pomcp_mcts",
        "success_rate": successes / n_episodes,
        "perfect_rate": perfect / n_episodes,
        "precise_rate": precise / n_episodes,
        "close_rate": close / n_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_planning_time_s": float(np.mean(planning_times)),
        "n_episodes": n_episodes,
        "n_simulations": n_simulations,
        "n_workers": n_workers,
        "max_depth": max_depth,
        "ucb_c": ucb_c,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "logs", "pomcp_mcts")
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print(f"\nPOMCP MCTS Results ({n_episodes} episodes, {n_simulations} sims):")
    print(f"  Success rate:     {results['success_rate']*100:.1f}%")
    print(f"  Perfect (<10mm):  {results['perfect_rate']*100:.1f}%")
    print(f"  Precise (<20mm):  {results['precise_rate']*100:.1f}%")
    print(f"  Close   (<40mm):  {results['close_rate']*100:.1f}%")
    print(f"  Mean steps:       {results['mean_episode_length']:.1f}")
    print(f"  Mean return:      {results['mean_return']:.1f} ± {results['std_return']:.1f}")
    print(f"  Mean plan time:   {results['mean_planning_time_s']:.1f}s/step")

    return results


def main():
    parser = argparse.ArgumentParser(description="POMCP with MCTS tree search")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--n-simulations", type=int, default=200,
                        help="MCTS simulations per planning step")
    parser.add_argument("--n-workers", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ucb-c", type=float, default=1.414,
                        help="UCB1 exploration constant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-every", type=int, default=0,
                        help="Reset carry-over tree every N steps (0=disabled)")
    args = parser.parse_args()

    print(f"POMCP MCTS Planner")
    print(f"  Simulations: {args.n_simulations}")
    print(f"  Workers:     {args.n_workers}")
    print(f"  Max depth:   {args.max_depth}")
    print(f"  UCB c:       {args.ucb_c}")
    print(f"  Episodes:    {args.n_episodes}")
    print(f"  Reset every: {args.reset_every if args.reset_every > 0 else 'disabled'} steps")
    print()

    evaluate_mcts(
        n_episodes=args.n_episodes,
        n_simulations=args.n_simulations,
        n_workers=args.n_workers,
        max_depth=args.max_depth,
        gamma=args.gamma,
        ucb_c=args.ucb_c,
        seed=args.seed,
        reset_every=args.reset_every,
    )


if __name__ == "__main__":
    main()
