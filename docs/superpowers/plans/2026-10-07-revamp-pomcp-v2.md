# Revamp: POMCP v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate an online POMCP planner on exactly the same `PickPlaceEnv` episodes as the RL methods, with two simulators (exact MuJoCo copies, and a learned world model). Report success against planning budget and decision latency, plus held-out world-model accuracy.

**Architecture:**
- The real env runs in `belief` mode, so its particle filter is the planner's belief.
- Each simulation samples a particle and writes that brick pose into a scratch copy of the physics state (`mj_copyData` + arm/env bookkeeping). It then runs discrete macro actions through the real `env.step`, so grasps in simulation are physical.
- Search is PO-UCT with root sampling over action histories, plus a heuristic rollout policy.
- The learned-model variant swaps the scratch physics for an MLP over belief-summary transitions collected from belief PPO.

**Tech Stack:** `pomdp_pick` (plan 1), numpy, torch (CPU, 1 thread), scikit-learn-free AUC (rank formula).

**Spec:** `docs/superpowers/specs/2026-10-07-revamp-design.md` §7

## Global Constraints

- Same episodes as RL eval: `PickPlaceEnv(obs_mode="belief", sigma_fixed=σ)`, reset seed = 10 000 + i.
- Macro actions (9), mapped onto the 5-D env action:
  - ±x, ±y: action ±0.667 on that axis (10 mm)
  - down / up: z ∓1.0 (15 mm)
  - yaw ±0.25 rad: action ±0.833
  - commit: commit = +1
- **Assumption, stated in results:** the planner's simulator knows the distractor poses and the noise model. Only the target pose is uncertain (sampled from the belief).
- Budgets {32, 128, 512} simulations per decision, depth 8, UCB c = 2.0, γ = 0.99.
- Evaluation: σ_ep = 10 mm, 50 episodes per budget, under the watchdog, after the RL matrix finishes (never concurrently).

## Review Focus

1. **Scratch-state leakage:** after planning, the real env's physics and counters must be bit-identical to before planning (`test_plan_does_not_touch_real_env`).
2. **Empty belief:** the target was never seen, so particles come from the workspace prior. The planner must still return a legal action within budget.
3. **Terminal inside a rollout** (success, or the attempts limit): the rollout stops and doesn't step a finished env.
4. **Latency accounting:** wall time per decision is recorded per step, so budget vs time is reportable.
5. **World model off-distribution:** predicted σ must be clipped ≥ 0, and grasp probability kept in [0, 1].

---

### Task 1: Simulation bridge (snapshot / restore / particle injection)

**Files:** Create `pomdp_pick/planning/__init__.py` (empty), `pomdp_pick/planning/sim_bridge.py`, `tests/test_planning.py`

**Interfaces:**
- Produces `Snapshot = sim_bridge.snapshot(env)`, `sim_bridge.restore(env, snap)`, `sim_bridge.set_target(env, x, y, yaw)`.
- `MACROS: list[np.ndarray(5)]` (9 entries) and `macro_action(i) -> np.ndarray(5)`.

- [ ] **Step 1:** Failing tests:

```python
import numpy as np
from pomdp_pick.envs.pick_env import PickPlaceEnv
from pomdp_pick.planning import sim_bridge as sb

def test_snapshot_restore_roundtrip_is_exact():
    env = PickPlaceEnv("belief"); env.reset(seed=3)
    snap = sb.snapshot(env)
    obs_a = [env.step(sb.macro_action(i % 8))[0] for i in range(5)]
    sb.restore(env, snap)
    obs_b = [env.step(sb.macro_action(i % 8))[0] for i in range(5)]
    assert all(np.allclose(a, b) for a, b in zip(obs_a, obs_b))

def test_set_target_moves_only_the_target():
    env = PickPlaceEnv("oracle"); env.reset(seed=3)
    d_before = [env.arm.brick_pose(n) for n in ("blue_lego_2x2", "blue_lego_2x2_b")]
    sb.set_target(env, 0.2, 0.05, 0.3)
    x, y, _, yaw = env.arm.brick_pose()
    assert abs(x - 0.2) < 1e-9 and abs(y - 0.05) < 1e-9
    assert [env.arm.brick_pose(n) for n in ("blue_lego_2x2", "blue_lego_2x2_b")] == d_before
```

- [ ] **Step 2:** Run `./vpy -m pytest tests/test_planning.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement:

```python
"""Exact physics snapshots of a PickPlaceEnv for planning, and the discrete macro-action set."""
import copy

import mujoco
import numpy as np

from pomdp_pick.robot.model import TARGET

MACROS = [np.array(a, float) for a in (
    [0.667, 0, 0, 0, -1], [-0.667, 0, 0, 0, -1], [0, 0.667, 0, 0, -1], [0, -0.667, 0, 0, -1],
    [0, 0, -1.0, 0, -1], [0, 0, 1.0, 0, -1], [0, 0, 0, 0.833, -1], [0, 0, 0, -0.833, -1],
    [0, 0, 0, 0, 1.0])]
COMMIT = 8

_ENV_FIELDS = ("attempts", "success", "held", "phi", "t0", "last_reading", "age", "visible", "sigma_ep")
_ARM_FIELDS = ("p", "yaw", "grip", "sol", "sim_time", "_next_frame")


def macro_action(i):
    return MACROS[i].copy()


def snapshot(env):
    d = mujoco.MjData(env.m)
    mujoco.mj_copyData(d, env.m, env.d)
    return {"d": d,
            "env": {k: copy.deepcopy(getattr(env, k)) for k in _ENV_FIELDS},
            "arm": {k: copy.deepcopy(getattr(env.arm, k)) for k in _ARM_FIELDS},
            "rng": copy.deepcopy(env.np_random.bit_generator.state),
            "sense": copy.deepcopy(env.sense_rng.bit_generator.state),
            "pf": (env.pf.particles.copy(), env.pf.weights.copy(), copy.deepcopy(env.pf.rng.bit_generator.state)),
            "lists": (list(env.commit_err), list(env.std_at_commit))}


def restore(env, snap):
    mujoco.mj_copyData(env.d, env.m, snap["d"])
    for k, v in snap["env"].items():
        setattr(env, k, copy.deepcopy(v))
    for k, v in snap["arm"].items():
        setattr(env.arm, k, copy.deepcopy(v))
    env.np_random.bit_generator.state = copy.deepcopy(snap["rng"])
    env.sense_rng.bit_generator.state = copy.deepcopy(snap["sense"])
    env.pf.particles, env.pf.weights = snap["pf"][0].copy(), snap["pf"][1].copy()
    env.pf.rng.bit_generator.state = copy.deepcopy(snap["pf"][2])
    env.commit_err, env.std_at_commit = list(snap["lists"][0]), list(snap["lists"][1])


def set_target(env, x, y, yaw):
    _, _, z, _ = env.arm.brick_pose()
    env.arm.set_brick(x, y, yaw, name=TARGET)
```

- [ ] **Step 4:** Run the tests → PASS.
- [ ] **Step 5:** Commit: `feat: planning sim bridge (exact snapshots, macro actions)`.

---

### Task 2: PO-UCT planner with physical simulations

**Files:** Create `pomdp_pick/planning/pomcp.py`; extend `tests/test_planning.py`.

**Interfaces:**
- Consumes `sim_bridge.*`.
- Produces `Planner(n_sims, depth=8, c=2.0, gamma=0.99, rng=None)`, with `plan(env) -> (macro_index, stats)` and `heuristic(env, target_xy, target_yaw) -> macro_index`.
  - `stats` keys: `n_sims`, `wall_s`, `q` (list of 9).
- `plan` must leave `env` exactly as it found it.

- [ ] **Step 1:** Failing tests:

```python
from pomdp_pick.planning.pomcp import Planner

def test_plan_does_not_touch_real_env():
    env = PickPlaceEnv("belief"); env.reset(seed=5)
    snap = sb.snapshot(env)
    a, stats = Planner(n_sims=8, rng=np.random.default_rng(0)).plan(env)
    assert 0 <= a < 9 and stats["n_sims"] == 8
    after = sb.snapshot(env)
    assert np.array_equal(snap["d"].qpos, after["d"].qpos) and snap["env"]["attempts"] == after["env"]["attempts"]

def test_planner_handles_prior_only_belief():
    env = PickPlaceEnv("belief", sigma_fixed=0.03); env.reset(seed=6)
    env.pf.reset_from_prior((0.0, 0.0), (0.13, 0.23), (-1.0, 1.0))
    a, _ = Planner(n_sims=4, rng=np.random.default_rng(1)).plan(env)
    assert 0 <= a < 9
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement:

```python
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
        dx, dy = txy[0] - p[0], txy[1] - p[1]
        dyaw = float(np.angle(np.exp(2j * (tyaw - env.arm.yaw)))) / 2
        if math.hypot(dx, dy) < 0.004 and p[2] <= 0.035 and abs(dyaw) < 0.2:
            return sb.COMMIT
        if abs(dyaw) > 0.2:
            return 6 if dyaw > 0 else 7
        if math.hypot(dx, dy) >= 0.004:
            if abs(dx) >= abs(dy):
                return 0 if dx > 0 else 1
            return 2 if dy > 0 else 3
        return 4

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
        return best, {"n_sims": self.n_sims, "wall_s": time.perf_counter() - t0, "q": q}

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
            a = max(node.children, key=lambda k: node.children[k].q + self.c * math.sqrt(logn / (node.children[k].n + 1e-9)))
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
```

- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: PO-UCT planner with physical simulations`.

---

### Task 3: POMCP evaluation harness

**Files:** Create `pomdp_pick/eval/evaluate_pomcp.py`; extend `tests/test_planning.py`.

**Interfaces:**
- `evaluate_pomcp(n_sims, sigma_mm=10, n_episodes=50, out_csv=None) -> DataFrame`.
  - Columns: the same as `evaluate_run`, plus `n_sims` and `mean_decision_s`. `method` = `pomcp{n_sims}`, `seed` = 0.
- CLI: `./vpy -m pomdp_pick.eval.evaluate_pomcp --sims 32 128 512 --episodes 50`, writing `runs/eval/pomcp{n}_seed0_n{episodes}.csv`.

- [ ] **Step 1:** Failing test:

```python
def test_pomcp_eval_smoke(tmp_path):
    from pomdp_pick.eval.evaluate_pomcp import evaluate_pomcp
    df = evaluate_pomcp(4, sigma_mm=10, n_episodes=1, out_csv=tmp_path / "p.csv", max_decisions=6)
    assert len(df) == 1 and df.method[0] == "pomcp4" and df.mean_decision_s[0] > 0
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement:

```python
"""Evaluate POMCP on the shared eval episodes (same seeds as the RL methods)."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pomdp_pick.envs.pick_env import PickPlaceEnv
from pomdp_pick.eval.evaluate import SEED_BASE
from pomdp_pick.planning import sim_bridge as sb
from pomdp_pick.planning.pomcp import Planner


def evaluate_pomcp(n_sims, sigma_mm=10, n_episodes=50, out_csv=None, max_decisions=400):
    env = PickPlaceEnv("belief", sigma_fixed=sigma_mm / 1000.0)
    rows = []
    for ep in range(n_episodes):
        env.reset(seed=SEED_BASE + ep)
        planner = Planner(n_sims, rng=np.random.default_rng(SEED_BASE + ep))
        done, ret, times, info = False, 0.0, [], {}
        for _ in range(max_decisions):
            t = time.perf_counter()
            a, _ = planner.plan(env)
            times.append(time.perf_counter() - t)
            _, r, term, trunc, info = env.step(sb.macro_action(a))
            ret += r
            if term or trunc:
                break
        rows.append(dict(method=f"pomcp{n_sims}", seed=0, sigma_mm=sigma_mm, episode=ep, n_sims=n_sims,
                         success=info.get("success", False), attempts=info.get("attempts", 0),
                         sim_time=info.get("sim_time", np.nan),
                         commit_err_mm=np.mean(info["commit_err_mm"]) if info.get("commit_err_mm") else np.nan,
                         belief_std_at_commit=np.mean(info["belief_std_at_commit"]) if info.get("belief_std_at_commit") else np.nan,
                         mean_decision_s=float(np.mean(times)), **{"return": ret}))
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=10)
    a = ap.parse_args()
    for n in a.sims:
        out = Path("runs/eval") / f"pomcp{n}_seed0_n{a.episodes}.csv"
        if not out.exists():
            evaluate_pomcp(n, a.sigma, a.episodes, out)
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** Run → PASS. Then time one 32-sim episode under the watchdog and record decisions/s in `docs/training-log.md`. Lower the 512 budget if one episode would take more than 10 min.
- [ ] **Step 5:** Commit: `feat: POMCP evaluation harness on shared eval seeds`.

---

### Task 4: Learned world model + accuracy report + WM-POMCP

**Files:** Create `pomdp_pick/planning/world_model.py`, `pomdp_pick/planning/collect.py`; extend `tests/test_planning.py`.

**Interfaces:**
- `collect.collect(run_dir, n_steps, out_npz)`: rolls out the trained belief-PPO policy (via `load_policy`, so normalisation is correct) on fresh seeds. It logs, per transition:
  - `x = [P(3), sin2yaw, cos2yaw, μ_rel(2), sin2θ, cos2θ, σxy(2), σθ, σ_ep, action(5)]`
  - `y = [Δx(13 non-action dims), grasp_success]`
- `WorldModel(hidden=128)`:
  - `fit(X, Y, epochs, val_frac=0.15, seed=0) -> dict` with keys `val_mse_per_dim`, `grasp_auc`
  - `predict(x) -> (Δ, p_grasp)`
  - `save` / `load`
- `WMPlanner(world_model, n_sims, depth=8)`: same `plan(env)` signature as `Planner`; rollouts run in the 13-D summary space with no physics.

- [ ] **Step 1:** Failing tests:

```python
def test_world_model_learns_identity_like_map_and_reports_auc():
    from pomdp_pick.planning.world_model import WorldModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 18)).astype(np.float32)
    Y = np.concatenate([0.1 * X[:, :13], (X[:, :1] > 0).astype(np.float32)], 1)
    rep = WorldModel(hidden=64).fit(X, Y, epochs=30)
    assert np.mean(rep["val_mse_per_dim"]) < 0.01 and rep["grasp_auc"] > 0.95
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `world_model.py`:
  - A 3×128 MLP with two heads (Δ regression via MSE, grasp logit via BCE), Adam 1e-3, batch 256, inputs standardised from training stats.
  - AUC by the rank formula: `auc = (rank_sum_pos - n_pos(n_pos+1)/2) / (n_pos n_neg)`.

  `collect.py` loops `env.step(pol.act(obs))` on belief-mode envs, with the PF summary read from `env.pf.summary()`. `WMPlanner` mirrors `Planner`, but its state is the 13-D summary vector:
  - Each step adds the predicted Δ.
  - Commit returns +15 · p_grasp − 1 · (1 − p_grasp) and terminates.
  - Each non-commit step returns −0.01.

  Run the test → PASS.
- [ ] **Step 4:** Collect 200k transitions from `runs/belief/seed0`. Fit the model and save `runs/world_model/wm.pt` and `runs/world_model/report.json` (val MSE per dimension, grasp AUC). Record both in `docs/training-log.md`.
- [ ] **Step 5:** Commit: `feat: learned world model with held-out accuracy and WM-POMCP`.

---

### Task 5: Run the POMCP study and add it to the figures

- [ ] **Step 1:** Queue the jobs (`configs/queue_pomcp.json`), only after `eval_all_n200` is done:
  - `evaluate_pomcp --sims 32 128 512 --episodes 50`
  - the WM-POMCP equivalent at the same budgets
- [ ] **Step 2:** Add `figures.success_vs_budget(df, path)`: success (Wilson CI) and mean decision time against n_sims, with physics and WM as two series. Add the belief-PPO σ = 10 mm success as a horizontal reference line.
- [ ] **Step 3:** Log the results table and the figure path in `docs/training-log.md`, and commit.
