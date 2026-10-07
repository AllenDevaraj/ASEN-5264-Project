"""PO-UCT (POMCP with root sampling) over discrete macro actions, simulated with real physics.

Each simulation draws a particle from the env's belief, writes it as the true target pose into a
restored snapshot, then descends the action tree with UCB and finishes with a heuristic rollout.
Observations are not branched on (open-loop tree over action histories), which is the standard
reduction when observations are continuous and the budget is small.
"""
import math
import time

import numpy as np

from pomdp_pick.planning import sim_bridge as sb

ALIGN_XY, ALIGN_YAW, COMMIT_Z = 0.004, 0.2, 0.035


def _heuristic(dx, dy, pz, dyaw):
    """Rollout policy: align yaw (mod π), then centre over the target, descend, commit."""
    dyaw = float(np.angle(np.exp(2j * dyaw))) / 2
    if abs(dyaw) > ALIGN_YAW:
        return 6 if dyaw > 0 else 7
    if math.hypot(dx, dy) >= ALIGN_XY:
        if abs(dx) >= abs(dy):
            return 0 if dx > 0 else 1
        return 2 if dy > 0 else 3
    return sb.COMMIT if pz <= COMMIT_Z else 4


class _Node:
    __slots__ = ("n", "q", "children")

    def __init__(self):
        self.n, self.q, self.children = 0, 0.0, {}


class Planner:
    def __init__(self, n_sims, depth=8, c=2.0, gamma=0.99, rng=None):
        self.n_sims, self.depth, self.c, self.gamma = n_sims, depth, c, gamma
        self.rng = rng if rng is not None else np.random.default_rng(0)

    @staticmethod
    def heuristic(env, txy, tyaw):
        p = env.arm.p
        return _heuristic(txy[0] - p[0], txy[1] - p[1], p[2], tyaw - env.arm.yaw)

    def _sample(self, env):
        w = env.pf.weights
        i = self.rng.choice(len(w), p=w / w.sum())
        x, y, th, _ = env.pf.particles[i]
        return np.array([x, y]), float(th)

    def plan(self, env):
        t0 = time.perf_counter()
        root, snap = _Node(), sb.snapshot(env)
        for _ in range(self.n_sims):
            sb.restore(env, snap)
            txy, tyaw = self._sample(env)
            sb.set_target(env, txy[0], txy[1], tyaw)
            self._simulate(env, root, 0, txy, tyaw)
        sb.restore(env, snap)
        q = [root.children[a].q if a in root.children else -np.inf for a in range(len(sb.MACROS))]
        best = max(root.children, key=lambda a: root.children[a].n) if root.children else sb.COMMIT
        return int(best), {"n_sims": self.n_sims, "wall_s": time.perf_counter() - t0, "q": q}

    def _simulate(self, env, node, depth, txy, tyaw):
        if depth >= self.depth:
            return 0.0
        untried = [a for a in range(len(sb.MACROS)) if a not in node.children]
        if untried:
            a = int(self.rng.choice(untried))
            node.children[a] = _Node()
            _, r, term, trunc, _ = env.step(sb.macro_action(a))
            ret = r if (term or trunc) else r + self.gamma * self._rollout(env, depth + 1, txy, tyaw)
        else:
            logn = math.log(node.n + 1)
            a = max(node.children,
                    key=lambda k: node.children[k].q + self.c * math.sqrt(logn / (node.children[k].n + 1e-9)))
            _, r, term, trunc, _ = env.step(sb.macro_action(a))
            ret = r if (term or trunc) else r + self.gamma * self._simulate(env, node.children[a], depth + 1, txy, tyaw)
        child = node.children[a]
        child.n += 1
        child.q += (ret - child.q) / child.n
        node.n += 1
        return ret

    def _rollout(self, env, depth, txy, tyaw):
        ret, disc = 0.0, 1.0
        for _ in range(depth, self.depth):
            _, r, term, trunc, _ = env.step(sb.macro_action(self.heuristic(env, txy, tyaw)))
            ret += disc * r
            disc *= self.gamma
            if term or trunc:
                break
        return ret


class WMPlanner:
    """UCT over the learned world model in the 13-D belief-summary space (no physics).

    Reward mirrors the env: -0.01 per step plus shaping on the belief-mean distance; a commit
    is terminal and pays the expected grasp-and-place value 15·p - (1 - p)."""

    def __init__(self, world_model, n_sims, depth=8, c=2.0, gamma=0.99, rng=None):
        self.wm, self.n_sims, self.depth, self.c, self.gamma = world_model, n_sims, depth, c, gamma
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.label = f"wmpomcp{n_sims}"

    @staticmethod
    def _phi(s):
        return -min(math.hypot(s[5], s[6]) / 10, 0.3) / 0.05

    @staticmethod
    def _heur(s):
        yaw, th = math.atan2(s[3], s[4]) / 2, math.atan2(s[7], s[8]) / 2
        return _heuristic(s[5] / 10, s[6] / 10, s[2] / 10, th - yaw)

    def _step(self, s, a):
        s2, p = self.wm.step_state(s, sb.MACROS[a])
        if a == sb.COMMIT:
            return s2, 15.0 * p - (1.0 - p) - 0.01 - self._phi(s), True
        return s2, -0.01 + self.gamma * self._phi(s2) - self._phi(s), False

    def plan(self, env):
        from pomdp_pick.planning.world_model import state_of
        t0 = time.perf_counter()
        root, s0 = _Node(), state_of(env._obs())
        for _ in range(self.n_sims):
            self._simulate(s0, root, 0)
        q = [root.children[a].q if a in root.children else -np.inf for a in range(len(sb.MACROS))]
        best = max(root.children, key=lambda a: root.children[a].n) if root.children else sb.COMMIT
        return int(best), {"n_sims": self.n_sims, "wall_s": time.perf_counter() - t0, "q": q}

    def _simulate(self, s, node, depth):
        if depth >= self.depth:
            return 0.0
        untried = [a for a in range(len(sb.MACROS)) if a not in node.children]
        if untried:
            a = int(self.rng.choice(untried))
            node.children[a] = _Node()
            s2, r, term = self._step(s, a)
            ret = r if term else r + self.gamma * self._rollout(s2, depth + 1)
        else:
            logn = math.log(node.n + 1)
            a = max(node.children,
                    key=lambda k: node.children[k].q + self.c * math.sqrt(logn / (node.children[k].n + 1e-9)))
            s2, r, term = self._step(s, a)
            ret = r if term else r + self.gamma * self._simulate(s2, node.children[a], depth + 1)
        child = node.children[a]
        child.n += 1
        child.q += (ret - child.q) / child.n
        node.n += 1
        return ret

    def _rollout(self, s, depth):
        ret, disc = 0.0, 1.0
        for _ in range(depth, self.depth):
            s, r, term = self._step(s, self._heur(s))
            ret += disc * r
            disc *= self.gamma
            if term:
                break
        return ret
