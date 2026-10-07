# Revamp: Foundation, Physics Env v2, Training/Eval Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pomdp_pick`, a standalone, tested Python package. It has a physics-grasp pick-and-place POMDP env, a joint (pose, noise-level) particle filter, five comparable agents, a guarded training queue, and shared-seed evaluation with rliable statistics. It ends by launching the first sanity training run.

**Architecture:** This is a new repo-root package that vendors three legacy modules: IK/FK, occlusion, and the URDF→MJCF merge. The env decides "where and when to commit" through policy actions. Scripted, fully physical grasp and place primitives decide success. Every method shares one reward computed from the true state. Training and eval jobs run sequentially under a thermal/memory/battery watchdog.

**Tech Stack:**
- Python 3.10, `.venv`
- mujoco 3.6.0, gymnasium 1.2.3
- stable-baselines3 2.7.1, sb3-contrib 2.7.1
- torch (CPU), numpy 2.2, rliable 1.2.0, matplotlib, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-10-07-revamp-design.md`

## Global Constraints

- **Compute limits:**
  - Training uses ≤ 4 env workers and runs one at a time, at `nice 19`, with `torch.set_num_threads(1)` per process.
  - Every long job runs under `tools/watchdog.py`. It pauses when the CPU package exceeds 85 °C (resumes below 75 °C), MemAvailable drops under 2 GB, or the laptop is on battery.
- **Episode and control timing:**
  - Decisions at 10 Hz (0.1 s per step) on a 2 ms physics timestep.
  - Episode budget: 20 s of sim time. At most 3 grasp attempts.
- **Action:** 5-D in [−1, 1]: Δx, Δy, Δz (±15 mm), Δyaw (±0.3 rad), commit (> 0.5 triggers the grasp primitive).
- **Sensing:**
  - Noise σ(d) = clip(σ_ep · d / 0.10, 0.3 σ_ep, 3 σ_ep); yaw noise 0.02 rad per mm.
  - Training σ_ep is log-uniform over [2, 20] mm. Eval levels: {2, 5, 10, 15, 20, 30} mm.
  - Drift: 0.5 mm per step.
- **Reward** (identical across observation modes):
  - −0.01 per 0.1 s.
  - Potential shaping γΦ′ − Φ, with Φ = −‖P_xy − brick_xy‖ / 0.05 and γ = 0.99.
  - −1 per failed grasp, +5 per successful grasp, +10 for placement.
- **Methods:** `oracle`, `plain`, `framestack` (4), `recurrent` (LSTM 128), `belief`. Seeds {0, 1, 2}. Eval reset seed = 10 000 + i.
- **Git:**
  - The v1 code under `vla_SO-ARM101/` stays functional (legacy). Only the `watch_policy.py` model-dir bug is fixed.
  - No history rewriting and no pushes.
- **Training log:** every fix or run gets an entry in `docs/training-log.md`.

## Review Focus

1. **IK unreachable.** The policy pushes P outside the reachable top-down workspace. P must clamp and the arm must hold its last valid setpoint; no exception, no NaN observation. *(Task 5 test `test_unreachable_target_holds_pose`.)*
2. **Target never visible.** σ_ep is huge or the brick starts out of frustum. Plain observations must stay finite (zeros, `visible=0`, age = max), and the filter must initialise from the workspace prior. *(Task 7 and Task 8 tests.)*
3. **Brick knocked off the workspace or tipped** by a miss or a distractor push. The env must not crash or report success. Placement requires the brick resting and within 20 mm of the goal. *(Task 8 test `test_tipped_brick_is_not_success`.)*
4. **Watchdog edge cases.** A missing thermal zone or AC file means "assume safe, keep running", logged once. A paused job is resumed, not left stopped, when the watchdog exits. *(Task 2 tests.)*
5. **Resume after interruption.** A training job killed mid-run continues from its last checkpoint with the same total-step target and doesn't restart from zero. *(Task 10 test `test_resume_continues_timesteps`.)*

---

## File Structure

```
pyproject.toml, requirements.txt, setup_venv.sh, .gitignore (modify)
pomdp_pick/__init__.py
pomdp_pick/robot/__init__.py
pomdp_pick/robot/kinematics.py      # vendored compute_workspace.py lines 1-308 (IK/FK)
pomdp_pick/robot/mjcf_merge.py      # vendored model_loader.py with repo-root paths
pomdp_pick/robot/model.py           # build_model(): pads, actuators, camera site, settings
pomdp_pick/robot/controller.py      # Arm: setpoints, cartesian moves, grasp/place primitives
pomdp_pick/sensing/__init__.py
pomdp_pick/sensing/occlusion.py     # vendored occlusion.py
pomdp_pick/sensing/camera.py        # WristCamera: visibility + noisy readings
pomdp_pick/belief/__init__.py
pomdp_pick/belief/particle_filter.py
pomdp_pick/envs/__init__.py
pomdp_pick/envs/pick_env.py         # PickPlaceEnv
pomdp_pick/agents/__init__.py
pomdp_pick/agents/methods.py        # method registry + policy wrappers for eval
pomdp_pick/agents/train.py          # train(method, seed, steps, ...) CLI
pomdp_pick/eval/__init__.py
pomdp_pick/eval/evaluate.py         # shared-seed rollouts -> CSV
pomdp_pick/eval/stats.py            # Wilson, IQM + bootstrap (rliable), P(improvement)
pomdp_pick/eval/figures.py          # success-vs-sigma figure
tools/watchdog.py, tools/run_queue.py, tools/bench_env.py
tests/test_*.py
```

---

### Task 1: Isolated environment, packaging, git hygiene

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `setup_venv.sh`, `pomdp_pick/__init__.py`, `tests/test_smoke.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `.venv/bin/python` with all deps, and an importable `pomdp_pick`.

- [ ] **Step 1: Write the failing smoke test** in `tests/test_smoke.py`:

```python
def test_imports():
    import mujoco, gymnasium, stable_baselines3, sb3_contrib, rliable, torch  # noqa: F401
    import pomdp_pick
    assert pomdp_pick.__version__ == "2.0.0"
    assert torch.get_num_threads() >= 1
```

- [ ] **Step 2: Create the dependency and packaging files.**

`requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
numpy==2.2.6
scipy==1.15.3
mujoco==3.6.0
gymnasium==1.2.3
torch==2.7.1+cpu
stable-baselines3==2.7.1
sb3-contrib==2.7.1
rliable==1.2.0
pandas>=2.2,<3
matplotlib>=3.9,<4
pyyaml>=6
tensorboard>=2.17
pillow>=10
pytest>=8
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pomdp_pick"
version = "2.0.0"
description = "SO-ARM101 pick-and-place under observation uncertainty: physics grasping, belief RL baselines, POMCP"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["pomdp_pick*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`setup_venv.sh`:

```bash
#!/usr/bin/env bash
# Isolated env: avoids the system cv2 (built for NumPy 1) that crashes SB3 under NumPy 2.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

`pomdp_pick/__init__.py`:

```python
"""SO-ARM101 pick-and-place under uncertainty (v2)."""
__version__ = "2.0.0"
```

- [ ] **Step 3: Run setup and the test.** If the pinned torch CPU version isn't on the index, use the newest `+cpu` wheel listed by `pip index versions torch --index-url https://download.pytorch.org/whl/cpu` and update `requirements.txt` to match.

Run: `bash setup_venv.sh && .venv/bin/pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 4: Update `.gitignore` and stop tracking binaries.** Append to `.gitignore`:

```
# v2
.venv/
runs/
*.egg-info/
__pycache__/
brag-output/work/
# model / data binaries (kept on disk and in tag v1-course-submission)
*.zip
*.pkl
*.pt
*.npz
*.nsys-rep
*.tfevents.*
vla_SO-ARM101/build/
vla_SO-ARM101/install/
vla_SO-ARM101/log/
```

Then run: `git rm -r -q --cached $(git ls-files | grep -E '\.(zip|pkl|pt|npz|nsys-rep)$|events\.out\.tfevents') && git status --short | head`

Expected: deletions staged ("D"); the files still exist on disk (`ls vla_SO-ARM101/src/so_arm101_control/scripts/models/ppo_belief/` shows `best_model.zip`).

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt pyproject.toml setup_venv.sh pomdp_pick/__init__.py tests/test_smoke.py
git commit -m "chore: isolated venv, packaging, stop tracking model binaries"
```

---

### Task 2: Watchdog and run queue

**Files:**
- Create: `tools/watchdog.py`, `tools/run_queue.py`, `tests/test_watchdog.py`

**Interfaces:**
- Produces:
  - `tools.watchdog.decide(paused: bool, temp_c: float|None, mem_avail_mb: float|None, on_ac: bool|None) -> str` returning `"pause"`, `"resume"` or `"none"`.
  - CLI: `python tools/watchdog.py --log runs/watchdog.log -- <cmd...>`.
  - `tools/run_queue.py <queue.json>` runs each `{"name": str, "cmd": [..]}` sequentially under the watchdog, records finished names in `<queue>.done`, and skips them on rerun.

- [ ] **Step 1: Write failing tests** in `tests/test_watchdog.py`:

```python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("watchdog", pathlib.Path(__file__).parents[1] / "tools/watchdog.py")
wd = importlib.util.module_from_spec(spec); spec.loader.exec_module(wd)

def test_pause_on_heat():
    assert wd.decide(False, 86.0, 6000, True) == "pause"

def test_resume_only_below_hysteresis():
    assert wd.decide(True, 80.0, 6000, True) == "none"
    assert wd.decide(True, 74.0, 6000, True) == "resume"

def test_pause_on_low_memory_and_battery():
    assert wd.decide(False, 60.0, 1500, True) == "pause"
    assert wd.decide(False, 60.0, 6000, False) == "pause"
    assert wd.decide(True, 60.0, 6000, False) == "none"

def test_unknown_sensors_are_safe():
    assert wd.decide(False, None, None, None) == "none"
    assert wd.decide(True, None, None, None) == "resume"

def test_child_runs_to_completion(tmp_path):
    rc = wd.run(["python3", "-c", "print('ok')"], log_path=tmp_path / "w.log", poll_s=0.2)
    assert rc == 0
    assert "exit 0" in (tmp_path / "w.log").read_text()
```

- [ ] **Step 2: Run them to see them fail.** `.venv/bin/pytest tests/test_watchdog.py -v` → FAIL (file not found).

- [ ] **Step 3: Implement `tools/watchdog.py`:**

```python
#!/usr/bin/env python3
"""Run a command under laptop-safety limits.

Pauses the whole process group (SIGSTOP) when the CPU package is hot, memory is low
or the laptop is on battery; resumes (SIGCONT) when safe again. The child runs at nice 19.
Usage: python tools/watchdog.py [--log runs/watchdog.log] -- cmd args...
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import time

TEMP_PAUSE_C = 85.0
TEMP_RESUME_C = 75.0
MEM_PAUSE_MB = 2000.0
MEM_KILL_MB = 700.0


def decide(paused, temp_c, mem_avail_mb, on_ac):
    """Pure policy: 'pause' | 'resume' | 'none'. Unknown readings (None) count as safe."""
    hot = temp_c is not None and temp_c > TEMP_PAUSE_C
    cool = temp_c is None or temp_c < TEMP_RESUME_C
    low_mem = mem_avail_mb is not None and mem_avail_mb < MEM_PAUSE_MB
    battery = on_ac is False
    if not paused:
        return "pause" if (hot or low_mem or battery) else "none"
    return "resume" if (cool and not low_mem and not battery) else "none"


def read_temp_c():
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(f"{zone}/type").read().strip() == "x86_pkg_temp":
                return int(open(f"{zone}/temp").read()) / 1000.0
        except OSError:
            continue
    return None


def read_mem_avail_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def read_on_ac():
    for path in glob.glob("/sys/class/power_supply/*/online"):
        if "AC" in path or "ADP" in path:
            try:
                return open(path).read().strip() == "1"
            except OSError:
                return None
    return None


def run(cmd, log_path, poll_s=5.0):
    log_path = str(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "a", buffering=1)
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    log.write(f"{stamp()} start {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, start_new_session=True, preexec_fn=lambda: os.nice(19))
    paused = False
    try:
        while proc.poll() is None:
            t, m, ac = read_temp_c(), read_mem_avail_mb(), read_on_ac()
            if m is not None and m < MEM_KILL_MB:
                log.write(f"{stamp()} memory critical ({m:.0f} MB): terminating\n")
                os.killpg(proc.pid, signal.SIGTERM)
                break
            action = decide(paused, t, m, ac)
            if action == "pause":
                os.killpg(proc.pid, signal.SIGSTOP)
                paused = True
                log.write(f"{stamp()} PAUSE temp={t} mem={m} ac={ac}\n")
            elif action == "resume":
                os.killpg(proc.pid, signal.SIGCONT)
                paused = False
                log.write(f"{stamp()} RESUME temp={t} mem={m} ac={ac}\n")
            time.sleep(poll_s)
    finally:
        if paused and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGCONT)
    rc = proc.wait()
    log.write(f"{stamp()} exit {rc}\n")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/watchdog.log")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    sys.exit(run(cmd, a.log))


if __name__ == "__main__":
    main()
```

`tools/run_queue.py`:

```python
#!/usr/bin/env python3
"""Run queued jobs one at a time under the watchdog; resumable.

queue.json: [{"name": "oracle_s0", "cmd": [".venv/bin/python", "-m", "pomdp_pick.agents.train", ...]}, ...]
Finished names are appended to queue.json.done; rerunning skips them.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watchdog import run  # noqa: E402


def main(queue_path):
    jobs = json.load(open(queue_path))
    done_path = queue_path + ".done"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    for job in jobs:
        if job["name"] in done:
            continue
        rc = run(job["cmd"], log_path=os.path.join("runs", "watchdog.log"))
        if rc != 0:
            print(f"job {job['name']} failed with {rc}; stopping queue", file=sys.stderr)
            sys.exit(rc)
        with open(done_path, "a") as fh:
            fh.write(job["name"] + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: Run the tests.** `.venv/bin/pytest tests/test_watchdog.py -v` → PASS (5 tests).

- [ ] **Step 5: Commit.** `git add tools/watchdog.py tools/run_queue.py tests/test_watchdog.py && git commit -m "feat: laptop-safe watchdog and resumable run queue"`

---

### Task 3: Vendored kinematics, occlusion, and MJCF merge

**Files:**
- Create: `pomdp_pick/robot/__init__.py` (empty), `pomdp_pick/robot/kinematics.py`, `pomdp_pick/robot/mjcf_merge.py`, `pomdp_pick/sensing/__init__.py` (empty), `pomdp_pick/sensing/occlusion.py`, `tests/test_vendored.py`

**Interfaces:**
- Produces:
  - `kinematics.geometric_ik(x, y, z, grasp_yaw=None) -> list[dict[str, float]]`
  - `kinematics.forward_kinematics(angles5) -> np.ndarray(3)`
  - `kinematics.make_transform(xyz, rpy) -> 4x4`
  - `kinematics.ARM_JOINT_NAMES`, `kinematics.JOINT_LIMITS`, `kinematics.X_PAN`, `kinematics.WRIST_ROLL_OFFSET`
  - `mjcf_merge.load_mujoco_model() -> (MjModel, MjData)`, `mjcf_merge.build_joint_map(m)`, `mjcf_merge.build_freejoint_map(m)`
  - Private helpers `_resolve_package_uris`, `_collect_mesh_assets`, `_merge_urdf_into_scene`, plus `DEFAULT_URDF`, `DEFAULT_SCENE`, `DEFAULT_MESHES`, `_DESC_PKG`, `_PKG_ROOT`.
  - `occlusion.is_occluded(...)` with the legacy signature, and `occlusion._project_to_camera`.

- [ ] **Step 1: Write the failing test** `tests/test_vendored.py`:

```python
import sys, numpy as np
LEGACY = "vla_SO-ARM101/src/so_arm101_control"

def test_ik_fk_match_legacy():
    sys.path.insert(0, LEGACY)
    from so_arm101_control import compute_workspace as old
    from pomdp_pick.robot import kinematics as new
    for x, y, z, yaw in [(0.18, 0.0, 0.06, 0.0), (0.22, 0.08, 0.02, 0.7), (0.14, -0.1, 0.1, -1.2)]:
        a, b = old.geometric_ik(x, y, z, grasp_yaw=yaw), new.geometric_ik(x, y, z, grasp_yaw=yaw)
        assert len(a) == len(b) > 0
        for sa, sb in zip(a, b):
            assert all(abs(sa[k] - sb[k]) < 1e-12 for k in sa)
        q = [b[0][k] for k in new.ARM_JOINT_NAMES]
        assert np.linalg.norm(new.forward_kinematics(q) - [x, y, z]) < 1e-3

def test_model_loads_from_repo_root():
    from pomdp_pick.robot import mjcf_merge
    m, d = mjcf_merge.load_mujoco_model()
    assert "red_lego_2x4" in mjcf_merge.build_freejoint_map(m)
    assert "gripper_joint" in mjcf_merge.build_joint_map(m)
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_vendored.py -v` → FAIL (no module `pomdp_pick.robot`).

- [ ] **Step 3: Vendor the files.**

```bash
SRC=vla_SO-ARM101/src/so_arm101_control/so_arm101_control
mkdir -p pomdp_pick/robot pomdp_pick/sensing && touch pomdp_pick/robot/__init__.py pomdp_pick/sensing/__init__.py
{ echo '"""Vendored from legacy so_arm101_control/compute_workspace.py (kinematics only, lines 1-308)."""'; sed -n '1,308p' $SRC/compute_workspace.py; } > pomdp_pick/robot/kinematics.py
cp $SRC/occlusion.py pomdp_pick/sensing/occlusion.py
cp $SRC/model_loader.py pomdp_pick/robot/mjcf_merge.py
python3 - <<'EOF'
p = 'pomdp_pick/robot/mjcf_merge.py'
s = open(p).read()
old = "_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))"
new = ("# Vendored from legacy so_arm101_control/model_loader.py; assets stay in the ROS description package.\n"
       "_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'vla_SO-ARM101', 'src'))")
assert old in s
open(p, 'w').write(s.replace(old, new))
EOF
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_vendored.py -v` → PASS.

- [ ] **Step 5: Commit.** `git add pomdp_pick/robot pomdp_pick/sensing tests/test_vendored.py && git commit -m "feat: vendor IK/FK, occlusion, MJCF merge into pomdp_pick"`

---

### Task 4: Physics model builder (pads, actuators, camera site)

**Files:**
- Create: `pomdp_pick/robot/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: Task 3 modules.
- Produces:
  - `ModelConfig` (frozen dataclass; fields below).
  - `build_model(cfg=ModelConfig()) -> (mujoco.MjModel, mujoco.MjData)`
  - Constants: `ARM`, `TARGET = "red_lego_2x4"`, `DISTRACTORS = ("blue_lego_2x2", "blue_lego_2x2_b")`.
  - Model element names:
    - actuators `a_<joint>` for the 5 arm joints, and `a_gripper`
    - geoms `gripper_pad`, `jaw_pad`
    - site `wrist_cam`, whose frame is camera_link: X forward, Y left, Z up.

- [ ] **Step 1: Write the failing test** `tests/test_model.py`:

```python
import mujoco, numpy as np
from pomdp_pick.robot.model import build_model, ARM, TARGET

def test_actuated_with_pads_and_camera():
    m, d = build_model()
    assert m.nu == 6
    assert abs(m.opt.timestep - 0.002) < 1e-12
    for g in ("gripper_pad", "jaw_pad"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert gid >= 0 and m.geom_contype[gid] == 1
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    meshes = [i for i in range(m.ngeom) if m.geom_bodyid[i] == gb and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
    assert meshes and all(m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0 for i in meshes)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_cam") >= 0

def test_arm_holds_still_without_jamming():
    m, d = build_model()
    mujoco.mj_forward(m, d)
    for _ in range(500):   # 1 s with zero setpoints
        mujoco.mj_step(m, d)
    q = np.array([d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in ARM])
    assert np.all(np.abs(q) < 0.02)
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_model.py -v` → FAIL (no module).

- [ ] **Step 3: Implement `pomdp_pick/robot/model.py`.** The pad geometry and actuator settings come from the verified prototype (24/24 lifts):

```python
"""Build the v2 MuJoCo model: SO-ARM101 with physical jaw pads, position actuators and a wrist camera site.

Settings verified by a grasp prototype (24/24 lifts): jaw meshes become visual-only and box pads
carry the contact; arm position actuators kp=50 kv=4.5 with gravity compensation; 2 ms timestep.
"""
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from pomdp_pick.robot import mjcf_merge as mm
from pomdp_pick.robot.kinematics import ARM_JOINT_NAMES as ARM, forward_kinematics, geometric_ik, make_transform

TARGET = "red_lego_2x4"
DISTRACTORS = ("blue_lego_2x2", "blue_lego_2x2_b")
ROBOT_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")
# URDF fixed chain gripper -> camera_mount -> usb_camera -> camera_link (fused away by the compiler)
_CAM_CHAIN = [((0.002359, 0.049561, -0.011838), (-2.956686, -0.384393, -2.034078)),
              ((-0.010340, 0.004827, -0.019687), (1.570625, 0.000515, 2.705234)),
              ((0.0, 0.0139, 0.0), (1.5708, 0.0, 1.5708))]
# Hull pairs that overlap at rest; once actuated they jam shoulder_pan.
_EXCLUDES = [("world", "shoulder"), ("world", "upper_arm"), ("shoulder", "lower_arm"),
             ("upper_arm", "wrist"), ("lower_arm", "gripper"), ("lower_arm", "jaw"), ("wrist", "jaw")]


@dataclass(frozen=True)
class ModelConfig:
    timestep: float = 0.002
    arm_kp: float = 50.0
    arm_kv: float = 4.5
    arm_force: float = 3.5
    grip_kp: float = 10.0
    grip_force: float = 3.5
    damping: float = 1.0
    armature: float = 0.1
    friction: tuple = (1.2, 0.02, 0.001)
    condim: int = 3
    brick_mass: float = 0.010


def _merged_xml():
    roots = {"so_arm101_description": mm._DESC_PKG,
             "so_arm101_control": os.path.join(mm._PKG_ROOT, "so_arm101_control")}
    urdf = mm._resolve_package_uris(open(mm.DEFAULT_URDF).read(), roots)
    assets = mm._collect_mesh_assets(urdf)
    for f in os.listdir(mm.DEFAULT_MESHES):
        if f.lower().endswith(".stl") and f not in assets:
            assets[f] = open(os.path.join(mm.DEFAULT_MESHES, f), "rb").read()
    return mm._merge_urdf_into_scene(urdf, mm.DEFAULT_SCENE, assets), assets


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def _pad_frames():
    """Pad boxes expressed in the gripper/jaw body frames, measured at a reference pose
    (yaw 0, jaw 0) where the jaws close along world Y: fixed-jaw inner face at TCP y≈+1 mm,
    moving-jaw inner face at y≈-16 mm, fingertip band from TCP-6 mm to TCP+15 mm."""
    m, d = mm.load_mujoco_model()
    jm = mm.build_joint_map(m)
    sol = geometric_ik(0.20, 0.0, 0.05, grasp_yaw=0.0)[0]
    for k, v in sol.items():
        d.qpos[jm[k]] = v
    d.qpos[jm["gripper_joint"]] = 0.0
    mujoco.mj_forward(m, d)
    tcp = np.array(forward_kinematics([sol[n] for n in ARM]))
    z_lo, z_hi = tcp[2] - 0.0060, tcp[2] + 0.0150
    zc, zh = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
    centers = {"gripper": np.array([tcp[0], 0.0040, zc]), "jaw": np.array([tcp[0], -0.0190, zc])}
    half = np.array([0.0055, 0.0030, zh])
    out = {}
    for body, c in centers.items():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        R = d.xmat[b].reshape(3, 3)
        out[body] = (R.T @ (c - d.xpos[b]), _quat(R.T.copy()), half)
    return out


def _camera_site():
    T = np.eye(4)
    for xyz, rpy in _CAM_CHAIN:
        T = T @ make_transform(xyz, rpy)
    return T[:3, 3], _quat(T[:3, :3])


def build_model(cfg=ModelConfig()):
    xml, assets = _merged_xml()
    root = ET.fromstring(xml)
    opt = root.find("option")
    opt.set("timestep", str(cfg.timestep))
    opt.set("integrator", "implicitfast")
    opt.set("cone", "pyramidal")
    opt.set("noslip_iterations", "0")
    pads = _pad_frames()
    cam_pos, cam_quat = _camera_site()
    fmt = lambda v: " ".join(f"{x:.6g}" for x in v)  # noqa: E731
    for body in root.iter("body"):
        name = body.get("name", "")
        if name in ROBOT_BODIES:
            body.set("gravcomp", "1")
            j = body.find("joint")
            if j is not None:
                j.set("damping", str(cfg.damping))
                j.set("armature", str(cfg.armature))
        if name in ("gripper", "jaw"):
            for g in body.findall("geom"):
                g.set("contype", "0")
                g.set("conaffinity", "0")
            pos, quat, half = pads[name]
            ET.SubElement(body, "geom", name=f"{name}_pad", type="box", pos=fmt(pos), quat=fmt(quat),
                          size=fmt(half), contype="1", conaffinity="1", condim=str(cfg.condim),
                          friction=fmt(cfg.friction), solref="0.004 1", solimp="0.95 0.99 0.001",
                          rgba="0.2 0.9 0.4 0", mass="0")
        if name == "gripper":
            ET.SubElement(body, "site", name="wrist_cam", pos=fmt(cam_pos), quat=fmt(cam_quat), size="0.003")
        if name == TARGET:
            g = body.find("geom")
            g.set("condim", str(cfg.condim))
            g.set("mass", str(cfg.brick_mass))
            g.set("friction", fmt(cfg.friction))
    con = root.find("contact")
    if con is None:
        con = ET.SubElement(root, "contact")
    for b1, b2 in _EXCLUDES:
        ET.SubElement(con, "exclude", body1=b1, body2=b2)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for jn in ARM:
        ET.SubElement(act, "position", name=f"a_{jn}", joint=jn, kp=str(cfg.arm_kp), kv=str(cfg.arm_kv),
                      forcerange=f"{-cfg.arm_force} {cfg.arm_force}")
    ET.SubElement(act, "position", name="a_gripper", joint="gripper_joint", kp=str(cfg.grip_kp), kv="0.2",
                  forcerange=f"{-cfg.grip_force} {cfg.grip_force}", ctrlrange="-0.174533 1.74533")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    return m, mujoco.MjData(m)
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_model.py -v` → PASS. If `test_arm_holds_still_without_jamming` fails, print the active contacts (`d.contact.geom1/geom2` body names) and add the overlapping pair to `_EXCLUDES`.

- [ ] **Step 5: Commit.** `git add pomdp_pick/robot/model.py tests/test_model.py && git commit -m "feat: physics model with jaw pads, actuators, wrist camera site"`

---

### Task 5: Arm controller with physical grasp and place primitives

**Files:**
- Create: `pomdp_pick/robot/controller.py`, `tests/test_controller.py`

**Interfaces:**
- Consumes: `build_model`, `ARM`, `TARGET`, `geometric_ik`, `forward_kinematics`, `X_PAN`, `WRIST_ROLL_OFFSET`, `mjcf_merge.build_joint_map/build_freejoint_map`.
- Produces `Arm(m, d)` with:
  - attributes `p` (np.ndarray(3): jaw-centre point P) and `yaw` (float)
  - `reset_pose(p, yaw)`: teleports joints and sets setpoints
  - `move(p_target, yaw_target, duration, frame_cb=None) -> bool`: True if fully reachable; otherwise clamps to the last valid pose
  - `grasp(frame_cb=None) -> bool` (held), `place(goal_xy, frame_cb=None) -> bool` (on goal)
  - `brick_pose(name=TARGET) -> (x, y, z, yaw)`
  - `sim_time` (float, seconds elapsed)
  - constants `JAW_OPEN = 0.5`, `JAW_CLOSE = -0.174533`, `Z_GRASP = 0.0085`, `Z_HOVER = 0.06`, `FIXED_CLEAR = 0.006`, `WORKSPACE`
- `frame_cb(arm)` is called every 1/30 s of sim time for footage logging.

- [ ] **Step 1: Write the failing tests** `tests/test_controller.py`:

```python
import math, numpy as np, pytest
from pomdp_pick.robot.model import build_model
from pomdp_pick.robot.controller import Arm, Z_HOVER

def _setup(bx=0.20, by=0.04, byaw=0.4):
    m, d = build_model()
    arm = Arm(m, d)
    arm.set_brick(bx, by, byaw)
    arm.park_distractors()
    arm.reset_pose(np.array([bx, by, Z_HOVER]), byaw)
    arm.settle(0.3)
    return arm

def test_grasp_at_true_pose_lifts_brick():
    arm = _setup()
    assert arm.grasp() is True
    assert arm.brick_pose()[2] > 0.03

@pytest.mark.parametrize("off", [0.015, -0.015])
def test_grasp_15mm_across_jaw_axis_fails(off):
    bx, by, byaw = 0.20, 0.04, 0.4
    arm = _setup(bx, by, byaw)
    # shift P across the jaw axis (perpendicular to brick long axis)
    p = np.array([bx - off * math.sin(byaw), by + off * math.cos(byaw), Z_HOVER])
    arm.move(p, byaw, 0.5)
    assert arm.grasp() is False

def test_place_on_goal():
    arm = _setup()
    assert arm.grasp()
    assert arm.place(np.array([0.16, -0.08]))
    x, y, z, _ = arm.brick_pose()
    assert z < 0.01 and np.hypot(x - 0.16, y + 0.08) < 0.02

def test_unreachable_target_holds_pose():
    arm = _setup()
    before = arm.p.copy()
    ok = arm.move(np.array([0.60, 0.0, 0.05]), 0.0, 0.3)
    assert ok is False
    assert np.all(np.isfinite(arm.p)) and np.linalg.norm(arm.p[:2]) <= 0.27
    assert np.linalg.norm(arm.p - before) < 0.15
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_controller.py -v` → FAIL.

- [ ] **Step 3: Implement `pomdp_pick/robot/controller.py`:**

```python
"""Cartesian control of the actuated SO-ARM101 and physical grasp / place primitives.

The controlled point P is the centre between the open jaws (where the brick should sit).
The TCP (fixed-jaw inner face) is offset from P across the jaw axis by half the brick
width plus FIXED_CLEAR, so success depends on how well P is placed over the real brick.
"""
import math

import mujoco
import numpy as np

from pomdp_pick.robot import mjcf_merge as mm
from pomdp_pick.robot.kinematics import WRIST_ROLL_OFFSET, X_PAN, forward_kinematics, geometric_ik
from pomdp_pick.robot.model import ARM, DISTRACTORS, TARGET

JAW_OPEN = 0.5
JAW_CLOSE = -0.174533
Z_GRASP = 0.0085
Z_HOVER = 0.06
Z_CARRY = 0.08
FIXED_CLEAR = 0.006
BRICK_HALF_W = 0.008
WORKSPACE = dict(r=(0.11, 0.27), ang=1.2, z=(0.0085, 0.15))
FRAME_DT = 1.0 / 30.0


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)


class Arm:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.jm = mm.build_joint_map(m)
        self.fj = mm.build_freejoint_map(m)
        self.aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{n}") for n in ARM]
        self.gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_gripper")
        self.p = np.array([0.18, 0.0, Z_HOVER])
        self.yaw = 0.0
        self.grip = JAW_OPEN
        self.sol = None
        self.sim_time = 0.0
        self._next_frame = 0.0

    # ---------- geometry ----------
    @staticmethod
    def clamp_p(p):
        p = np.asarray(p, float).copy()
        r = math.hypot(p[0], p[1])
        ang = math.atan2(p[1], p[0])
        r = min(max(r, WORKSPACE["r"][0]), WORKSPACE["r"][1])
        ang = min(max(ang, -WORKSPACE["ang"]), WORKSPACE["ang"])
        return np.array([r * math.cos(ang), r * math.sin(ang), min(max(p[2], WORKSPACE["z"][0]), WORKSPACE["z"][1])])

    @staticmethod
    def branch_yaw(p, yaw):
        """The brick is 180°-symmetric: pick yaw + kπ keeping wrist roll mid-range."""
        pan = math.atan2(-p[1], p[0] - X_PAN)
        k = min((-2, -1, 0, 1, 2), key=lambda k: abs(pan + yaw + k * math.pi - WRIST_ROLL_OFFSET))
        return yaw + k * math.pi

    def tcp_for(self, p, yaw):
        gy = self.branch_yaw(p, yaw)
        off = BRICK_HALF_W + FIXED_CLEAR
        return np.array([p[0] - off * math.sin(gy), p[1] + off * math.cos(gy), p[2]]), gy

    def _solve(self, p, yaw):
        tcp, gy = self.tcp_for(p, yaw)
        sols = geometric_ik(*tcp, grasp_yaw=gy)
        if not sols:
            return None
        if self.sol is None:
            return sols[0]
        return min(sols, key=lambda s: sum((s[k] - self.sol[k]) ** 2 for k in ARM))

    # ---------- state ----------
    def set_brick(self, x, y, yaw, name=TARGET, z=None):
        q = self.fj[name]
        half_z = self.m.geom_size[self._geom_of(name)][2]
        self.d.qpos[q:q + 3] = [x, y, half_z if z is None else z]
        self.d.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.d.qvel[self.m.jnt_dofadr[self.m.body_jntadr[self._body(name)]]:][:6] = 0
        mujoco.mj_forward(self.m, self.d)

    def park_distractors(self):
        for k, (x, y) in zip(DISTRACTORS, [(-0.3, 0.3), (-0.3, -0.3)]):
            self.set_brick(x, y, 0.0, name=k)

    def brick_pose(self, name=TARGET):
        q = self.fj[name]
        w, _, _, z = self.d.qpos[q + 3:q + 7]
        return (*self.d.qpos[q:q + 3], 2 * math.atan2(z, w))

    def brick_half_size(self, name):
        return self.m.geom_size[self._geom_of(name)].copy()

    def _body(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_of(self, name):
        b = self._body(name)
        return next(i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == b)

    def reset_pose(self, p, yaw):
        self.p, self.yaw, self.grip, self.sol = self.clamp_p(p), float(yaw), JAW_OPEN, None
        sol = self._solve(self.p, self.yaw)
        if sol is None:
            raise ValueError(f"reset pose unreachable: {p}")
        self.sol = sol
        for n in ARM:
            self.d.qpos[self.jm[n]] = sol[n]
            self.d.qvel[self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = 0
        self.d.qpos[self.jm["gripper_joint"]] = JAW_OPEN
        self._apply()
        mujoco.mj_forward(self.m, self.d)

    def _apply(self):
        for i, n in zip(self.aid, ARM):
            self.d.ctrl[i] = self.sol[n]
        self.d.ctrl[self.gid] = self.grip

    def settle(self, seconds, frame_cb=None):
        n = int(round(seconds / self.m.opt.timestep))
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.sim_time += self.m.opt.timestep
            if frame_cb is not None and self.sim_time >= self._next_frame:
                frame_cb(self)
                self._next_frame += FRAME_DT

    def tcp_actual(self):
        return np.array(forward_kinematics([self.d.qpos[self.jm[n]] for n in ARM]))

    # ---------- motion ----------
    def move(self, p_target, yaw_target, duration, frame_cb=None, substep=0.02):
        """Smooth cartesian move of P; stops at the last reachable waypoint."""
        p0, y0 = self.p.copy(), self.yaw
        p1, y1 = self.clamp_p(p_target), float(yaw_target)
        reachable = bool(np.allclose(p1, p_target, atol=1e-9))
        n = max(1, int(round(duration / substep)))
        for i in range(1, n + 1):
            s = _smooth(i / n)
            p, yaw = p0 + (p1 - p0) * s, y0 + (y1 - y0) * s
            sol = self._solve(p, yaw)
            if sol is None:
                reachable = False
            else:
                self.p, self.yaw, self.sol = p, yaw, sol
                self._apply()
            self.settle(substep, frame_cb)
        return reachable

    def set_jaw(self, target, duration, frame_cb=None, substep=0.02):
        g0 = self.grip
        n = max(1, int(round(duration / substep)))
        for i in range(1, n + 1):
            self.grip = g0 + (target - g0) * _smooth(i / n)
            self._apply()
            self.settle(substep, frame_cb)

    def holding(self):
        x, y, z, _ = self.brick_pose()
        tcp, _ = self.tcp_for(self.p, self.yaw)
        return z > 0.03 and math.hypot(x - self.p[0], y - self.p[1]) < 0.025

    def grasp(self, frame_cb=None):
        """Open, descend, close, lift 5 cm. Returns True if the brick is held (physics)."""
        hover = self.p.copy()
        self.set_jaw(JAW_OPEN, 0.2, frame_cb)
        self.move(np.array([*self.p[:2], Z_GRASP]), self.yaw, 0.8, frame_cb)
        self.settle(0.15, frame_cb)
        self.set_jaw(JAW_CLOSE, 0.5, frame_cb)
        self.settle(0.25, frame_cb)
        self.move(np.array([*self.p[:2], Z_GRASP + 0.05]), self.yaw, 0.8, frame_cb)
        self.settle(0.2, frame_cb)
        held = self.holding()
        if not held:
            self.set_jaw(JAW_OPEN, 0.25, frame_cb)
            self.move(np.array([*self.p[:2], max(hover[2], Z_HOVER)]), self.yaw, 0.5, frame_cb)
        return held

    def place(self, goal_xy, frame_cb=None):
        """Carry to goal, lower, release, retreat. True if the brick rests within 20 mm of the goal."""
        self.move(np.array([*self.p[:2], Z_CARRY]), self.yaw, 0.4, frame_cb)
        dist = float(np.linalg.norm(np.asarray(goal_xy) - self.p[:2]))
        self.move(np.array([*goal_xy, Z_CARRY]), self.yaw, max(0.6, dist / 0.2), frame_cb)
        self.move(np.array([*goal_xy, Z_GRASP + 0.004]), self.yaw, 0.7, frame_cb)
        self.set_jaw(JAW_OPEN, 0.3, frame_cb)
        self.move(np.array([*goal_xy, Z_HOVER]), self.yaw, 0.5, frame_cb)
        self.settle(0.3, frame_cb)
        x, y, z, _ = self.brick_pose()
        return bool(z < 0.01 and math.hypot(x - goal_xy[0], y - goal_xy[1]) < 0.02)
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_controller.py -v` → PASS.
  - If `test_grasp_at_true_pose_lifts_brick` fails: render `grasp` stills (see `/tmp/claude-1000/grasp_probe/grasp_proto.py:shot`) and adjust `Z_GRASP` (try 0.0075–0.0100) or the pad centres in `model._pad_frames`.
  - If the 15 mm test does *not* fail: lower `FIXED_CLEAR` to 0.004.

- [ ] **Step 5: Measure the actual tolerance** with a one-off sweep (P offset −20…+20 mm across and along the jaw axis, 3 brick poses), and record the result table in `docs/training-log.md` under "v2 grasp tolerance".

```bash
.venv/bin/python - <<'EOF'
import math, numpy as np
from pomdp_pick.robot.model import build_model
from pomdp_pick.robot.controller import Arm, Z_HOVER
m, d = build_model()
for axis in ("across", "along"):
    row = []
    for off in [-0.020, -0.015, -0.010, -0.006, -0.003, 0.0, 0.003, 0.006, 0.010, 0.015, 0.020]:
        ok = 0
        for bx, by, byaw in [(0.20, 0.04, 0.4), (0.17, -0.06, -0.9), (0.23, 0.10, 1.4)]:
            mujoco_reset = __import__("mujoco").mj_resetData(m, d)
            arm = Arm(m, d); arm.set_brick(bx, by, byaw); arm.park_distractors()
            ux, uy = (-math.sin(byaw), math.cos(byaw)) if axis == "across" else (math.cos(byaw), math.sin(byaw))
            arm.reset_pose(np.array([bx + off * ux, by + off * uy, Z_HOVER]), byaw); arm.settle(0.3)
            ok += arm.grasp()
        row.append(f"{off*1000:+.0f}mm:{ok}/3")
    print(axis, " ".join(row))
EOF
```

- [ ] **Step 6: Commit.** `git add pomdp_pick/robot/controller.py tests/test_controller.py docs/training-log.md && git commit -m "feat: actuated arm control with physical grasp/place primitives"`

---

### Task 6: Wrist camera sensing

**Files:**
- Create: `pomdp_pick/sensing/camera.py`, `tests/test_camera.py`

**Interfaces:**
- Consumes: the `wrist_cam` site; `occlusion.is_occluded` and `occlusion._project_to_camera`.
- Produces `WristCamera(m)` with:
  - `pose(d) -> (pos(3), R(3x3))`
  - `visible(d, target, occluders) -> bool`
    - `target = (x, y, yaw, half_size(3))`
    - `occluders` is a list of the same tuples.
  - `reading(rng, true_xyth, dist, sigma_ep) -> np.ndarray(3)`
- Module functions:
  - `sigma_at(dist, sigma_ep) -> float` (metres)
  - `yaw_sigma(sigma_m) -> float` (rad)
  - `wrap_pi(a) -> a in [-π/2, π/2)`, i.e. modulo π for the symmetric brick.

- [ ] **Step 1: Write the failing tests** `tests/test_camera.py`:

```python
import numpy as np
from pomdp_pick.robot.model import build_model
from pomdp_pick.robot.controller import Arm, Z_HOVER
from pomdp_pick.sensing.camera import WristCamera, sigma_at, yaw_sigma, wrap_pi

def test_noise_model_shape():
    s = 0.010
    assert abs(sigma_at(0.10, s) - 0.010) < 1e-12
    assert abs(sigma_at(0.01, s) - 0.003) < 1e-12      # floor 0.3 sigma
    assert abs(sigma_at(1.0, s) - 0.030) < 1e-12       # cap 3 sigma
    assert abs(yaw_sigma(0.010) - 0.20) < 1e-12
    assert abs(wrap_pi(np.pi * 0.75) + np.pi / 4) < 1e-12

def test_visibility_in_front_and_occlusion():
    m, d = build_model()
    arm = Arm(m, d)
    arm.set_brick(0.20, 0.0, 0.0); arm.park_distractors()
    arm.reset_pose(np.array([0.16, 0.0, 0.12]), 0.0)
    cam = WristCamera(m)
    hs = arm.brick_half_size("red_lego_2x4")
    target = (0.20, 0.0, 0.0, hs)
    assert cam.visible(d, target, [])
    pos, _ = cam.pose(d)
    # a tall block exactly between camera and target occludes it
    mid = (pos[:2] + np.array([0.20, 0.0])) / 2
    occ = (mid[0], mid[1], 0.0, np.array([0.02, 0.02, 0.05]))
    assert not cam.visible(d, target, [occ])
    # a target far behind the arm is outside the frustum
    assert not cam.visible(d, (-0.25, 0.0, 0.0, hs), [])

def test_reading_is_seeded():
    m, _ = build_model()
    cam = WristCamera(m)
    a = cam.reading(np.random.default_rng(3), np.array([0.2, 0.0, 0.3]), 0.1, 0.01)
    b = cam.reading(np.random.default_rng(3), np.array([0.2, 0.0, 0.3]), 0.1, 0.01)
    assert np.allclose(a, b) and not np.allclose(a, [0.2, 0.0, 0.3])
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_camera.py -v` → FAIL.

- [ ] **Step 3: Implement `pomdp_pick/sensing/camera.py`:**

```python
"""Wrist camera model: visibility (frustum + block occlusion) and distance-dependent noisy readings.

Unlike v1, the camera is NOT aimed at the true target: its pose is the real camera_link frame on
the gripper, so seeing the brick depends on where the policy moves the arm.
"""
import math

import mujoco
import numpy as np

from pomdp_pick.sensing import occlusion as occ

IMG_W, IMG_H = 1280, 720
MIN_DEPTH = 0.02
YAW_RAD_PER_MM = 0.02


def sigma_at(dist, sigma_ep):
    return float(np.clip(sigma_ep * dist / 0.10, 0.3 * sigma_ep, 3.0 * sigma_ep))


def yaw_sigma(sigma_m):
    return YAW_RAD_PER_MM * sigma_m * 1000.0


def wrap_pi(a):
    """Wrap an angle modulo π into [-π/2, π/2) (the brick looks the same rotated 180°)."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


class WristCamera:
    def __init__(self, m):
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_cam")
        if self.sid < 0:
            raise ValueError("model has no wrist_cam site")

    def pose(self, d):
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def visible(self, d, target, occluders):
        pos, R = self.pose(d)
        x, y, yaw, hs = target
        p = occ._project_to_camera(np.array([x, y, 2 * hs[2]]), pos, R)
        if p is None or p[2] < MIN_DEPTH or not (0 <= p[0] <= IMG_W and 0 <= p[1] <= IMG_H):
            return False
        for ox, oy, oyaw, ohs in occluders:
            if occ.is_occluded((x, y), (hs[0], hs[1]), yaw, (ox, oy), (ohs[0], ohs[1]), oyaw,
                               pos, R, target_half_z=hs[2], occluder_half_z=ohs[2]):
                return False
        return True

    def distance(self, d, xy):
        pos, _ = self.pose(d)
        return float(np.linalg.norm(pos - np.array([xy[0], xy[1], 0.0])))

    @staticmethod
    def reading(rng, true_xyth, dist, sigma_ep):
        s = sigma_at(dist, sigma_ep)
        out = np.array(true_xyth, float) + rng.normal(0.0, [s, s, yaw_sigma(s)])
        out[2] = wrap_pi(out[2])
        return out
```

Note: `occ._project_to_camera` uses the legacy image centre `_CX`/`_CY` (640/360 for 1280×720). The frustum test above therefore matches the legacy camera intrinsics.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_camera.py -v` → PASS. If the "in front" assertion fails, print `cam.pose(d)` and the projected `p`. The site axis convention (X forward) must match `occlusion.LINK_TO_OPENCV`; if the camera looks backwards, flip the last `_CAM_CHAIN` rotation in `model.py` and re-run the Task 4 and Task 6 tests.

- [ ] **Step 5: Commit.** `git add pomdp_pick/sensing/camera.py tests/test_camera.py && git commit -m "feat: wrist camera visibility and noise model"`

---

### Task 7: Particle filter with joint noise-level estimation

**Files:**
- Create: `pomdp_pick/belief/__init__.py` (empty), `pomdp_pick/belief/particle_filter.py`, `tests/test_particle_filter.py`

**Interfaces:**
- Consumes: `sigma_at`, `yaw_sigma`, `wrap_pi`.
- Produces `BeliefFilter(n=300, drift=0.0005, sigma_range=(0.0015, 0.045), rng=None)` with:
  - `reset_from_reading(z(3), dist)`
  - `reset_from_prior(center_xy, r_range, ang_range)`
  - `predict(extra_xy=0.0)`
  - `update(z(3), dist)`
  - `summary() -> dict`, with keys `mu` (x, y, θ), `std` (x, y, θ), `sigma_ep` (posterior mean, metres)
  - attributes `particles` (N×4: x, y, θ, log σ_ep) and `weights`.

- [ ] **Step 1: Write the failing tests** `tests/test_particle_filter.py`:

```python
import numpy as np
from pomdp_pick.belief.particle_filter import BeliefFilter
from pomdp_pick.sensing.camera import sigma_at, yaw_sigma, wrap_pi

def _run(seed, sigma_ep=0.010, steps=25, true=(0.2, 0.05, 1.4)):
    rng = np.random.default_rng(seed)
    pf = BeliefFilter(rng=np.random.default_rng(seed + 1))
    s = sigma_at(0.08, sigma_ep)
    z = np.array(true) + rng.normal(0, [s, s, yaw_sigma(s)])
    pf.reset_from_reading(z, 0.08)
    for _ in range(steps):
        pf.predict()
        z = np.array(true) + rng.normal(0, [s, s, yaw_sigma(s)])
        z[2] = wrap_pi(z[2])
        pf.update(z, 0.08)
    return pf

def test_deterministic_given_seed():
    a, b = _run(0).summary(), _run(0).summary()
    assert np.allclose(a["mu"], b["mu"]) and np.allclose(a["std"], b["std"])

def test_converges_and_handles_pi_wrap():
    pf = _run(1, true=(0.2, 0.05, 1.5))          # near +π/2: readings wrap to ~-π/2
    s = pf.summary()
    assert np.hypot(s["mu"][0] - 0.2, s["mu"][1] - 0.05) < 0.006
    assert abs(wrap_pi(s["mu"][2] - 1.5)) < 0.15
    assert s["std"][0] < 0.008

def test_estimates_noise_level():
    lo = _run(2, sigma_ep=0.003, steps=40).summary()["sigma_ep"]
    hi = _run(2, sigma_ep=0.025, steps=40).summary()["sigma_ep"]
    assert lo < 0.008 < hi

def test_prior_reset_spreads_over_workspace():
    pf = BeliefFilter(rng=np.random.default_rng(0))
    pf.reset_from_prior((0.0, 0.0), (0.13, 0.23), (-1.0, 1.0))
    s = pf.summary()
    assert s["std"][0] > 0.02 and np.all(np.isfinite(s["mu"]))
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_particle_filter.py -v` → FAIL.

- [ ] **Step 3: Implement `pomdp_pick/belief/particle_filter.py`:**

```python
"""Seeded SIR particle filter over the target brick pose and the unknown camera noise level.

State per particle: (x, y, θ mod π, log σ_ep). Estimating σ_ep jointly lets the likelihood
width adapt to the real noise level (v1 used a fixed 1.5× widening and collapsed at 30 mm).
"""
import numpy as np

from pomdp_pick.sensing.camera import wrap_pi, yaw_sigma


class BeliefFilter:
    def __init__(self, n=300, drift=0.0005, sigma_range=(0.0015, 0.045), rng=None, reinject=0.05):
        self.n, self.drift, self.reinject = n, drift, reinject
        self.ls_lo, self.ls_hi = np.log(sigma_range[0]), np.log(sigma_range[1])
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.particles = np.zeros((n, 4))
        self.weights = np.full(n, 1.0 / n)

    def _sigmas(self, dist):
        s_ep = np.exp(self.particles[:, 3])
        s = np.clip(s_ep * dist / 0.10, 0.3 * s_ep, 3.0 * s_ep)
        return s, yaw_sigma(s)

    def _sample_ls(self, k):
        return self.rng.uniform(self.ls_lo, self.ls_hi, k)

    def reset_from_reading(self, z, dist):
        self.particles[:, 3] = self._sample_ls(self.n)
        s, st = self._sigmas(dist)
        self.particles[:, 0] = z[0] + self.rng.normal(0, s)
        self.particles[:, 1] = z[1] + self.rng.normal(0, s)
        self.particles[:, 2] = wrap_pi(z[2] + self.rng.normal(0, st))
        self.weights[:] = 1.0 / self.n

    def reset_from_prior(self, center_xy, r_range, ang_range):
        r = self.rng.uniform(*r_range, self.n)
        a = self.rng.uniform(*ang_range, self.n)
        self.particles[:, 0] = center_xy[0] + r * np.cos(a)
        self.particles[:, 1] = center_xy[1] + r * np.sin(a)
        self.particles[:, 2] = self.rng.uniform(-np.pi / 2, np.pi / 2, self.n)
        self.particles[:, 3] = self._sample_ls(self.n)
        self.weights[:] = 1.0 / self.n

    def predict(self, extra_xy=0.0):
        sxy = np.hypot(self.drift, extra_xy)
        self.particles[:, :2] += self.rng.normal(0, sxy, (self.n, 2))
        self.particles[:, 2] = wrap_pi(self.particles[:, 2] + self.rng.normal(0, 0.01 + 5 * extra_xy, self.n))
        self.particles[:, 3] = np.clip(self.particles[:, 3] + self.rng.normal(0, 0.02, self.n), self.ls_lo, self.ls_hi)

    def update(self, z, dist):
        s, st = self._sigmas(dist)
        dx, dy = self.particles[:, 0] - z[0], self.particles[:, 1] - z[1]
        dth = wrap_pi(self.particles[:, 2] - z[2])
        logw = (-0.5 * (dx ** 2 + dy ** 2) / s ** 2 - 2 * np.log(s)
                - 0.5 * dth ** 2 / st ** 2 - np.log(st))
        logw += np.log(self.weights + 1e-300)
        logw -= logw.max()
        w = np.exp(logw)
        self.weights = w / w.sum()
        if 1.0 / np.sum(self.weights ** 2) < self.n / 2:
            self._resample(z, dist)

    def _resample(self, z, dist):
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), pos), self.n - 1)
        self.particles = self.particles[idx].copy()
        k = int(self.reinject * self.n)
        if k:
            j = self.rng.choice(self.n, k, replace=False)
            self.particles[j, 3] = self._sample_ls(k)
            s, st = self._sigmas(dist)
            self.particles[j, 0] = z[0] + self.rng.normal(0, s[j])
            self.particles[j, 1] = z[1] + self.rng.normal(0, s[j])
            self.particles[j, 2] = wrap_pi(z[2] + self.rng.normal(0, st[j]))
        self.weights[:] = 1.0 / self.n

    def summary(self):
        w = self.weights
        mu_xy = w @ self.particles[:, :2]
        c, s = w @ np.cos(2 * self.particles[:, 2]), w @ np.sin(2 * self.particles[:, 2])
        mu_th = 0.5 * np.arctan2(s, c)
        std_xy = np.sqrt(w @ (self.particles[:, :2] - mu_xy) ** 2)
        r = min(np.hypot(c, s), 1.0)
        std_th = 0.5 * np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0))
        return {"mu": np.array([*mu_xy, mu_th]), "std": np.array([*std_xy, std_th]),
                "sigma_ep": float(w @ np.exp(self.particles[:, 3]))}
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_particle_filter.py -v` → PASS.

- [ ] **Step 5: Commit.** `git add pomdp_pick/belief tests/test_particle_filter.py && git commit -m "feat: seeded particle filter with joint noise-level estimation"`

---

### Task 8: `PickPlaceEnv`

**Files:**
- Create: `pomdp_pick/envs/__init__.py`, `pomdp_pick/envs/pick_env.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: `build_model`, `Arm` (+ constants), `WristCamera`, `BeliefFilter`, `wrap_pi`.
- Produces `PickPlaceEnv(obs_mode: str in {"oracle", "plain", "belief"}, sigma_range=(0.002, 0.020), sigma_fixed=None, drift=0.0005, max_time=20.0, max_attempts=3, frame_cb=None)`.
  - Observation dims: oracle 12, plain 14, belief 17.
  - `info` keys: `success`, `attempts`, `sigma_ep`, `sim_time`, `commit_err_mm` (list), `belief_std_at_commit` (list), `visible`.
  - Gymnasium registration id `"PickPlace-v2"` (kwargs pass through).

- [ ] **Step 1: Write the failing tests** `tests/test_env.py`:

```python
import numpy as np, pytest
from pomdp_pick.envs.pick_env import PickPlaceEnv

DIMS = {"oracle": 12, "plain": 14, "belief": 17}

@pytest.mark.parametrize("mode", list(DIMS))
def test_shapes_and_finite(mode):
    env = PickPlaceEnv(obs_mode=mode)
    obs, info = env.reset(seed=5)
    assert obs.shape == (DIMS[mode],) and np.all(np.isfinite(obs))
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample() * np.array([1, 1, 1, 1, 0]))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)

def test_reset_deterministic():
    a, b = PickPlaceEnv("belief"), PickPlaceEnv("belief")
    oa, _ = a.reset(seed=11); ob, _ = b.reset(seed=11)
    assert np.allclose(oa, ob)
    act = np.array([0.3, -0.2, -0.5, 0.1, -1.0])
    for _ in range(4):
        oa, ra, *_ = a.step(act); ob, rb, *_ = b.step(act)
    assert np.allclose(oa, ob) and ra == rb

def test_reward_identical_across_modes():
    act = np.array([0.4, 0.1, -0.6, 0.0, -1.0])
    rewards = []
    for mode in DIMS:
        env = PickPlaceEnv(mode); env.reset(seed=3)
        rewards.append([env.step(act)[1] for _ in range(6)])
    assert np.allclose(rewards[0], rewards[1]) and np.allclose(rewards[0], rewards[2])

def test_oracle_scripted_pick_succeeds():
    env = PickPlaceEnv("oracle"); env.reset(seed=7)
    done, info = False, {}
    while not done:
        bx, by, _, byaw = env.arm.brick_pose()
        p = env.arm.p
        dx, dy = np.clip((np.array([bx, by]) - p[:2]) / 0.015, -1, 1)
        dz = np.clip((0.03 - p[2]) / 0.015, -1, 1)
        dyaw = np.clip(float(np.angle(np.exp(2j * (byaw - env.arm.yaw)))) / 2 / 0.3, -1, 1)
        commit = 1.0 if np.hypot(bx - p[0], by - p[1]) < 0.002 and abs(dyaw) < 0.2 else -1.0
        _, _, term, trunc, info = env.step(np.array([dx, dy, dz, dyaw, commit]))
        done = term or trunc
    assert info["success"] is True

def test_tipped_brick_is_not_success():
    env = PickPlaceEnv("oracle"); env.reset(seed=7)
    x, y, _, yaw = env.arm.brick_pose()
    env.arm.set_brick(env.goal[0], env.goal[1], yaw, z=0.03)   # dropped from height onto goal, not placed
    assert env._placement_ok() is False or env.arm.brick_pose()[2] < 0.01
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_env.py -v` → FAIL.

- [ ] **Step 3: Implement `pomdp_pick/envs/pick_env.py`:**

```python
"""PickPlaceEnv: SO-ARM101 pick-and-place POMDP with physical grasping.

The policy moves the jaw-centre point P and the gripper yaw, and decides when to commit.
Committing runs a physical grasp; a held brick is carried to the known goal by a scripted
physical place. Every observation mode shares the same dynamics, reward and seeds.
"""
import math

import gymnasium as gym
import numpy as np

from pomdp_pick.belief.particle_filter import BeliefFilter
from pomdp_pick.robot.controller import Arm, Z_HOVER
from pomdp_pick.robot.model import DISTRACTORS, TARGET, build_model
from pomdp_pick.sensing.camera import WristCamera, wrap_pi

STEP_S = 0.1
GAMMA = 0.99
HOME = np.array([0.17, 0.0, 0.12])
DIMS = {"oracle": 12, "plain": 14, "belief": 17}
MAX_AGE = 50


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_mode="belief", sigma_range=(0.002, 0.020), sigma_fixed=None, drift=0.0005,
                 max_time=20.0, max_attempts=3, frame_cb=None):
        if obs_mode not in DIMS:
            raise ValueError(f"obs_mode must be one of {list(DIMS)}")
        self.obs_mode, self.sigma_range, self.sigma_fixed = obs_mode, sigma_range, sigma_fixed
        self.drift, self.max_time, self.max_attempts, self.frame_cb = drift, max_time, max_attempts, frame_cb
        self.m, self.d = build_model()
        self.arm = Arm(self.m, self.d)
        self.cam = WristCamera(self.m)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (DIMS[obs_mode],), np.float32)
        self.pf = BeliefFilter(drift=drift)

    # ---------------- episode setup ----------------
    def _spawn(self, rng):
        pts = []
        while len(pts) < 4:
            r, a = rng.uniform(0.13, 0.23), rng.uniform(-1.0, 1.0)
            q = np.array([r * math.cos(a), r * math.sin(a)])
            need = 0.08 if len(pts) == 3 else 0.05
            if all(np.linalg.norm(q - p) >= need for p in pts):
                pts.append(q)
        return pts  # target, distractor, distractor, goal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        import mujoco
        mujoco.mj_resetData(self.m, self.d)
        self.arm = Arm(self.m, self.d)
        self.arm.frame_cb = self.frame_cb
        tgt, d1, d2, self.goal = self._spawn(rng)
        self.arm.set_brick(*tgt, rng.uniform(-math.pi / 2, math.pi / 2))
        for name, p in zip(DISTRACTORS, (d1, d2)):
            self.arm.set_brick(*p, rng.uniform(-math.pi / 2, math.pi / 2), name=name)
        self.arm.reset_pose(HOME, 0.0)
        self.arm.settle(0.3, self.frame_cb)
        lo, hi = np.log(self.sigma_range[0]), np.log(self.sigma_range[1])
        self.sigma_ep = self.sigma_fixed if self.sigma_fixed is not None else float(np.exp(rng.uniform(lo, hi)))
        self.pf = BeliefFilter(drift=self.drift, rng=np.random.default_rng(int(rng.integers(2**31))))
        self.sense_rng = np.random.default_rng(int(rng.integers(2**31)))
        self.t0 = self.arm.sim_time
        self.attempts, self.success, self.held = 0, False, False
        self.last_reading, self.age = None, MAX_AGE
        self.commit_err, self.std_at_commit = [], []
        self._sense(first=True)
        self.phi = self._potential()
        return self._obs(), self._info()

    # ---------------- sensing ----------------
    def _target(self):
        x, y, _, yaw = self.arm.brick_pose()
        return (x, y, yaw, self.arm.brick_half_size(TARGET))

    def _occluders(self):
        out = []
        for name in DISTRACTORS:
            x, y, _, yaw = self.arm.brick_pose(name)
            out.append((x, y, yaw, self.arm.brick_half_size(name)))
        return out

    def _sense(self, first=False, extra_xy=0.0):
        tx, ty, tyaw, _ = self._target()
        self.visible = bool(self.cam.visible(self.d, self._target(), self._occluders()))
        z = None
        if self.visible:
            dist = self.cam.distance(self.d, (tx, ty))
            z = self.cam.reading(self.sense_rng, np.array([tx, ty, wrap_pi(tyaw)]), dist, self.sigma_ep)
            self.last_reading, self.age = z, 0
        else:
            self.age = min(self.age + 1, MAX_AGE)
        if first:
            if z is not None:
                self.pf.reset_from_reading(z, self.cam.distance(self.d, (tx, ty)))
            else:
                self.pf.reset_from_prior((0.0, 0.0), (0.13, 0.23), (-1.0, 1.0))
        else:
            self.pf.predict(extra_xy)
            if z is not None:
                self.pf.update(z, self.cam.distance(self.d, (tx, ty)))

    # ---------------- reward pieces ----------------
    def _potential(self):
        x, y, _, _ = self.arm.brick_pose()
        return -min(math.hypot(self.arm.p[0] - x, self.arm.p[1] - y), 0.3) / 0.05

    def _placement_ok(self):
        x, y, z, _ = self.arm.brick_pose()
        return bool(z < 0.01 and math.hypot(x - self.goal[0], y - self.goal[1]) < 0.02)

    def _drift_target(self):
        x, y, z, yaw = self.arm.brick_pose()
        if z < 0.01 and self.drift > 0:
            dx, dy = self.np_random.normal(0.0, self.drift, 2)
            self.arm.set_brick(x + dx, y + dy, yaw)

    # ---------------- step ----------------
    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        t_before = self.arm.sim_time
        reward, extra_xy = 0.0, 0.0
        if a[4] > 0.5:
            self.attempts += 1
            bx, by, _, byaw = self.arm.brick_pose()
            self.commit_err.append(1000 * math.hypot(self.arm.p[0] - bx, self.arm.p[1] - by))
            self.std_at_commit.append(float(np.hypot(*self.pf.summary()["std"][:2]) * 1000))
            before = np.array([bx, by])
            if self.arm.grasp(self.frame_cb):
                reward += 5.0
                self.held = True
                self.success = self.arm.place(self.goal, self.frame_cb) and self._placement_ok()
                reward += 10.0 if self.success else 0.0
            else:
                reward -= 1.0
                bx2, by2, _, _ = self.arm.brick_pose()
                extra_xy = float(np.linalg.norm(np.array([bx2, by2]) - before)) + 0.01
        else:
            p = self.arm.p + a[:3] * 0.015
            self.arm.move(p, self.arm.yaw + a[3] * 0.3, STEP_S, self.frame_cb, substep=0.02)
            self._drift_target()
        dt = self.arm.sim_time - t_before
        reward -= 0.01 * dt / STEP_S
        phi = self._potential()
        reward += GAMMA * phi - self.phi
        self.phi = phi
        if not self.held:
            self._sense(extra_xy=extra_xy)
        elapsed = self.arm.sim_time - self.t0
        terminated = self.held or self.attempts >= self.max_attempts
        truncated = (not terminated) and elapsed >= self.max_time
        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info()

    # ---------------- observation ----------------
    def _common(self):
        p = self.arm.p
        return [p[0] * 10, p[1] * 10, p[2] * 10, math.sin(2 * self.arm.yaw), math.cos(2 * self.arm.yaw),
                self.arm.grip, self.attempts / self.max_attempts,
                1.0 - min((self.arm.sim_time - self.t0) / self.max_time, 1.0)]

    def _rel(self, x, y, th):
        return [(x - self.arm.p[0]) * 10, (y - self.arm.p[1]) * 10, math.sin(2 * th), math.cos(2 * th)]

    def _obs(self):
        o = self._common()
        if self.obs_mode == "oracle":
            x, y, _, yaw = self.arm.brick_pose()
            o += self._rel(x, y, yaw)
        elif self.obs_mode == "plain":
            if self.last_reading is None:
                o += [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                o += self._rel(*self.last_reading) + [float(self.visible), self.age / MAX_AGE]
        else:
            s = self.pf.summary()
            o += self._rel(*s["mu"]) + [s["std"][0] * 100, s["std"][1] * 100, s["std"][2],
                                        s["sigma_ep"] * 100, float(self.visible)]
        return np.asarray(o, np.float32)

    def _info(self):
        return {"success": bool(self.success), "attempts": self.attempts, "sigma_ep": self.sigma_ep,
                "sim_time": self.arm.sim_time - self.t0, "commit_err_mm": list(self.commit_err),
                "belief_std_at_commit": list(self.std_at_commit), "visible": bool(self.visible)}
```

`pomdp_pick/envs/__init__.py`:

```python
from gymnasium.envs.registration import register

register(id="PickPlace-v2", entry_point="pomdp_pick.envs.pick_env:PickPlaceEnv")
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_env.py -v` → PASS.
  - If `test_oracle_scripted_pick_succeeds` fails, re-run the Task 5 tolerance sweep. The scripted policy commits at < 2 mm error, so a failure means a primitive problem, not an RL problem.
  - In `test_tipped_brick_is_not_success`, a brick dropped from 3 cm should settle; the assertion accepts either outcome but must not raise.

- [ ] **Step 5: Commit.** `git add pomdp_pick/envs tests/test_env.py && git commit -m "feat: PickPlaceEnv v2 with physical grasp, shared reward, three obs modes"`

---

### Task 9: Throughput benchmark and legacy `watch_policy` fix

**Files:**
- Create: `tools/bench_env.py`
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/watch_policy.py:61-66`, `docs/training-log.md`

- [ ] **Step 1: Write `tools/bench_env.py`:**

```python
#!/usr/bin/env python3
"""Measure env steps/s: single process, then N SubprocVecEnv workers (default 4)."""
import sys
import time

import numpy as np
import torch


def _make(mode):
    def f():
        torch.set_num_threads(1)
        from pomdp_pick.envs.pick_env import PickPlaceEnv
        return PickPlaceEnv(mode)
    return f


def single(mode, steps=600):
    env = _make(mode)()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(steps):
        a = rng.uniform(-1, 1, 5); a[4] = -1 if rng.random() > 0.03 else 1
        if any(env.step(a)[2:4]):
            env.reset()
    return steps / (time.perf_counter() - t)


def vec(mode, n, steps=1200):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([_make(mode) for _ in range(n)], start_method="forkserver")
    venv.reset()
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(steps // n):
        a = rng.uniform(-1, 1, (n, 5)); a[:, 4] = np.where(rng.random(n) > 0.03, -1, 1)
        venv.step(a)
    r = steps / (time.perf_counter() - t)
    venv.close()
    return r


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for mode in ("plain", "belief"):
        print(f"{mode}: single {single(mode):.0f} steps/s, {n} workers {vec(mode, n):.0f} steps/s")
```

- [ ] **Step 2: Run it under the watchdog** and record the numbers in `docs/training-log.md`:

`.venv/bin/python tools/watchdog.py --log runs/watchdog.log -- .venv/bin/python tools/bench_env.py 4`

Then compute the run length: `steps_per_run = floor(0.9 * 60h * 3600 * sps_4workers / 15)`, capped at 3M and floored to 100k. Write the value into `configs/train.yaml` (Task 10).

- [ ] **Step 3: Fix the legacy `watch_policy.py` default model dir.** It must prefer `scripts/models/`, whose 13-D/16-D models match the current v1 env. Replace the candidate list:

```python
        for candidate in [
            os.path.join(scripts_dir, 'models', name),
            os.path.join(pkg_dir, 'models', name),
        ]:
```

- [ ] **Step 4: Verify** the change: `grep -n "scripts_dir, 'models', name" vla_SO-ARM101/src/so_arm101_control/scripts/watch_policy.py`. Expected: the scripts/models candidate appears first.

- [ ] **Step 5: Commit.** `git add tools/bench_env.py vla_SO-ARM101/src/so_arm101_control/scripts/watch_policy.py docs/training-log.md && git commit -m "chore: env throughput benchmark; fix legacy watch_policy default model dir"`

---

### Task 10: Method registry and training entry point

**Files:**
- Create: `pomdp_pick/agents/__init__.py` (empty), `pomdp_pick/agents/methods.py`, `pomdp_pick/agents/train.py`, `configs/train.yaml`, `tests/test_train.py`

**Interfaces:**
- Consumes: `PickPlaceEnv`.
- Produces:
  - `methods.METHODS: dict[str, MethodSpec]`, where `MethodSpec` has `obs_mode`, `algo` ("ppo" | "recurrent"), `n_stack` (int).
  - `train.train(method, seed, total_steps, out_root="runs", n_envs=4, ckpt_every=100_000, hparams=None) -> Path`
    - Writes `runs/<method>/seed<seed>/`: `model.zip`, `vecnorm.pkl`, `progress.json` (`{"timesteps": int, "done": bool}`), and a `monitor/` folder.
  - CLI: `python -m pomdp_pick.agents.train --method M --seed S [--steps N] [--config configs/train.yaml]`.
  - `methods.load_policy(run_dir) -> Policy`, an object with `reset()` and `act(obs) -> np.ndarray(5)` that applies saved normalisation, frame stacking and LSTM state internally. Used by evaluation.

- [ ] **Step 1: Write the failing tests** `tests/test_train.py`:

```python
import json, numpy as np, pytest
from pomdp_pick.agents.methods import METHODS, load_policy
from pomdp_pick.agents.train import train
from pomdp_pick.envs.pick_env import PickPlaceEnv

@pytest.mark.parametrize("method", list(METHODS))
def test_tiny_training_and_policy_roundtrip(tmp_path, method):
    run = train(method, seed=0, total_steps=512, out_root=tmp_path, n_envs=2, ckpt_every=256,
                hparams={"n_steps": 128, "batch_size": 64, "n_epochs": 1})
    prog = json.loads((run / "progress.json").read_text())
    assert prog["done"] and prog["timesteps"] >= 512
    pol = load_policy(run)
    env = PickPlaceEnv(METHODS[method].obs_mode)
    obs, _ = env.reset(seed=1); pol.reset()
    a = pol.act(obs)
    assert a.shape == (5,) and np.all(np.abs(a) <= 1.0)

def test_resume_continues_timesteps(tmp_path):
    run = train("plain", 0, 256, tmp_path, n_envs=2, ckpt_every=128, hparams={"n_steps": 64, "batch_size": 64, "n_epochs": 1})
    run = train("plain", 0, 512, tmp_path, n_envs=2, ckpt_every=128, hparams={"n_steps": 64, "batch_size": 64, "n_epochs": 1})
    prog = json.loads((run / "progress.json").read_text())
    assert 512 <= prog["timesteps"] < 512 + 256
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_train.py -v -x` → FAIL.

- [ ] **Step 3: Implement.**

`pomdp_pick/agents/methods.py`:

```python
"""Method registry: which observation each method sees and how its policy is evaluated."""
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    obs_mode: str
    algo: str = "ppo"      # "ppo" | "recurrent"
    n_stack: int = 1


METHODS = {
    "oracle": MethodSpec("oracle"),
    "plain": MethodSpec("plain"),
    "framestack": MethodSpec("plain", n_stack=4),
    "recurrent": MethodSpec("plain", algo="recurrent"),
    "belief": MethodSpec("belief"),
}


class Policy:
    """Deterministic evaluation wrapper: normalise -> (stack) -> predict (with LSTM state)."""

    def __init__(self, run_dir):
        import pickle
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "progress.json").read_text())
        self.spec = METHODS[meta["method"]]
        algo = RecurrentPPO if self.spec.algo == "recurrent" else PPO
        self.model = algo.load(run_dir / "model.zip", device="cpu")
        with open(run_dir / "vecnorm.pkl", "rb") as fh:
            vn = pickle.load(fh)
        self.mean, self.var = vn.obs_rms.mean, vn.obs_rms.var
        self.clip, self.eps = vn.clip_obs, vn.epsilon
        self.reset()

    def reset(self):
        self.stack = collections.deque(maxlen=self.spec.n_stack)
        self.state, self.start = None, np.ones((1,), bool)

    def _norm(self, obs):
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)

    def act(self, obs):
        o = np.asarray(obs, np.float32)
        if self.spec.n_stack > 1:
            if not self.stack:
                self.stack.extend([o] * self.spec.n_stack)
            else:
                self.stack.append(o)
            o = np.concatenate(self.stack)
        o = self._norm(o)
        if self.spec.algo == "recurrent":
            a, self.state = self.model.predict(o[None], state=self.state, episode_start=self.start,
                                               deterministic=True)
            self.start = np.zeros((1,), bool)
            return np.clip(a[0], -1, 1)
        a, _ = self.model.predict(o[None], deterministic=True)
        return np.clip(a[0], -1, 1)


def load_policy(run_dir):
    return Policy(run_dir)
```

Frame-stacking note: the stack is built over raw observations, and VecNormalize statistics are computed on the stacked vector. Training must therefore apply `VecFrameStack` *before* `VecNormalize`, as `train.py` does below, so `Policy` mirrors that order.

`pomdp_pick/agents/train.py`:

```python
"""Train one (method, seed) run with resumable checkpoints. Laptop-safe: ≤4 workers, 1 torch thread."""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from pomdp_pick.agents.methods import METHODS

DEFAULT_HP = dict(n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4, ent_coef=0.01,
                  gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5)


def _env_fn(obs_mode, rank, seed, monitor_dir):
    def f():
        torch.set_num_threads(1)
        from stable_baselines3.common.monitor import Monitor
        from pomdp_pick.envs.pick_env import PickPlaceEnv
        env = PickPlaceEnv(obs_mode)
        env.reset(seed=seed * 1000 + rank)
        return Monitor(env, os.path.join(monitor_dir, f"w{rank}"), info_keywords=("success", "attempts"))
    return f


def _linear(lr):
    return lambda frac: lr * frac


def train(method, seed, total_steps, out_root="runs", n_envs=4, ckpt_every=100_000, hparams=None):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

    torch.set_num_threads(1)
    n_envs = min(n_envs, 4)
    spec = METHODS[method]
    run = Path(out_root) / method / f"seed{seed}"
    (run / "monitor").mkdir(parents=True, exist_ok=True)
    prog_path = run / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"timesteps": 0, "done": False}
    hp = {**DEFAULT_HP, **(hparams or {})}

    venv = SubprocVecEnv([_env_fn(spec.obs_mode, i, seed, str(run / "monitor")) for i in range(n_envs)],
                         start_method="forkserver")
    if spec.n_stack > 1:
        venv = VecFrameStack(venv, spec.n_stack)
    resuming = (run / "model.zip").exists() and prog["timesteps"] > 0
    if resuming:
        venv = VecNormalize.load(str(run / "vecnorm.pkl"), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=hp["gamma"])
    algo = RecurrentPPO if spec.algo == "recurrent" else PPO
    policy = "MlpLstmPolicy" if spec.algo == "recurrent" else "MlpPolicy"
    kw = dict(hp, learning_rate=_linear(hp["learning_rate"]))
    pk = dict(net_arch=[256, 256])
    if spec.algo == "recurrent":
        pk.update(lstm_hidden_size=128, n_lstm_layers=1)
    if resuming:
        model = algo.load(run / "model.zip", env=venv, device="cpu")
    else:
        model = algo(policy, venv, seed=seed, device="cpu", verbose=0, policy_kwargs=pk,
                     tensorboard_log=str(run / "tb"), **kw)

    def save(ts, done):
        model.save(run / "model.zip")
        venv.save(str(run / "vecnorm.pkl"))
        prog_path.write_text(json.dumps({"method": method, "seed": seed, "timesteps": int(ts), "done": done,
                                         "target": int(total_steps), "hparams": hp}))

    class Ckpt(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = model.num_timesteps + ckpt_every

        def _on_step(self):
            if self.num_timesteps >= self.next:
                save(self.num_timesteps, False)
                self.next += ckpt_every
            return True

    remaining = max(0, total_steps - prog["timesteps"])
    if remaining > 0:
        model.learn(remaining, callback=Ckpt(), reset_num_timesteps=not resuming, tb_log_name="ppo")
    save(model.num_timesteps, True)
    venv.close()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)) if os.path.exists(a.config) else {}
    steps = a.steps or cfg.get("total_steps", 1_000_000)
    train(a.method, a.seed, steps, a.out, n_envs=cfg.get("n_envs", 4), ckpt_every=cfg.get("ckpt_every", 100_000),
          hparams=cfg.get("hparams"))


if __name__ == "__main__":
    main()
```

`configs/train.yaml` (`total_steps` is set from the Task 9 benchmark):

```yaml
total_steps: 1000000
n_envs: 4
ckpt_every: 100000
hparams:
  n_steps: 1024
  batch_size: 256
  n_epochs: 10
  learning_rate: 0.0003
  ent_coef: 0.01
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_train.py -v` → PASS (6 tests; uses 2 workers and finishes in about a minute). Run it under the watchdog if the laptop is warm.

- [ ] **Step 5: Commit.** `git add pomdp_pick/agents configs tests/test_train.py && git commit -m "feat: method registry, resumable training entry point, eval policy wrapper"`

---

### Task 11: Shared-seed evaluation, statistics, figures

**Files:**
- Create: `pomdp_pick/eval/__init__.py` (empty), `pomdp_pick/eval/evaluate.py`, `pomdp_pick/eval/stats.py`, `pomdp_pick/eval/figures.py`, `tests/test_eval.py`

**Interfaces:**
- Consumes: `load_policy`, `METHODS`, `PickPlaceEnv`.
- Produces:
  - `evaluate.evaluate_run(run_dir, sigmas_mm, n_episodes, out_csv) -> pandas.DataFrame`, with columns `method, seed, sigma_mm, episode, success, attempts, sim_time, commit_err_mm, belief_std_at_commit, return`.
  - `stats.wilson(k, n) -> (lo, hi)`
  - `stats.aggregate(df) -> DataFrame`, per method: IQM success with 95% CI over seeds × levels.
  - `stats.prob_improvement(df, a, b) -> (p, lo, hi)`
  - `figures.success_vs_sigma(df, path)`
  - CLIs: `python -m pomdp_pick.eval.evaluate --runs runs --episodes 200` and `python -m pomdp_pick.eval.stats runs/eval/*.csv`.

- [ ] **Step 1: Write the failing tests** `tests/test_eval.py`:

```python
import numpy as np, pandas as pd
from pomdp_pick.eval.stats import wilson, aggregate, prob_improvement

def _fake():
    rows = []
    for method, p in [("a", 0.8), ("b", 0.5)]:
        for seed in range(3):
            for sigma in (2, 10, 30):
                rng = np.random.default_rng(seed * 100 + sigma)
                for ep in range(50):
                    rows.append(dict(method=method, seed=seed, sigma_mm=sigma, episode=ep, success=rng.random() < p))
    return pd.DataFrame(rows)

def test_wilson_bounds():
    lo, hi = wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88
    assert wilson(0, 10)[0] == 0.0

def test_aggregate_orders_methods():
    agg = aggregate(_fake())
    a, b = agg.set_index("method").loc["a"], agg.set_index("method").loc["b"]
    assert a["iqm"] > b["iqm"] and a["lo"] <= a["iqm"] <= a["hi"]

def test_probability_of_improvement():
    p, lo, hi = prob_improvement(_fake(), "a", "b")
    assert p > 0.9 and lo <= p <= hi
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_eval.py -v` → FAIL.

- [ ] **Step 3: Implement.**

`pomdp_pick/eval/stats.py`:

```python
"""Statistics: Wilson intervals per cell; rliable IQM + stratified bootstrap across seeds x levels."""
import math
import sys

import numpy as np
import pandas as pd


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _score_matrix(df, method):
    """(n_seeds, n_levels) success rates for one method."""
    sub = df[df.method == method]
    piv = sub.groupby(["seed", "sigma_mm"]).success.mean().unstack("sigma_mm").sort_index(axis=1)
    return piv.to_numpy()


def aggregate(df, reps=5000):
    from rliable import library as rly
    from rliable import metrics
    scores = {m: _score_matrix(df, m) for m in sorted(df.method.unique())}
    point, ci = rly.get_interval_estimates(scores, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=reps)
    return pd.DataFrame([{"method": m, "iqm": point[m][0], "lo": ci[m][0, 0], "hi": ci[m][1, 0]} for m in scores])


def prob_improvement(df, a, b, reps=2000):
    from rliable import library as rly
    from rliable import metrics
    pair = {f"{a},{b}": (_score_matrix(df, a), _score_matrix(df, b))}
    point, ci = rly.get_interval_estimates(pair, metrics.probability_of_improvement, reps=reps)
    k = f"{a},{b}"
    return float(point[k]), float(ci[k][0]), float(ci[k][1])


def per_cell(df):
    g = df.groupby(["method", "sigma_mm"]).success.agg(["sum", "count"]).reset_index()
    g[["lo", "hi"]] = g.apply(lambda r: pd.Series(wilson(r["sum"], r["count"])), axis=1)
    g["rate"] = g["sum"] / g["count"]
    return g


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]])
    print(per_cell(df).to_string(index=False))
    print(aggregate(df).to_string(index=False))
    for m in sorted(df.method.unique()):
        if m != "belief" and "belief" in set(df.method):
            print(f"P(belief > {m}) =", prob_improvement(df, "belief", m))
```

`pomdp_pick/eval/evaluate.py`:

```python
"""Shared-seed deterministic evaluation of trained runs -> one CSV per (method, seed)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pomdp_pick.agents.methods import METHODS, load_policy
from pomdp_pick.envs.pick_env import PickPlaceEnv

EVAL_SIGMAS_MM = (2, 5, 10, 15, 20, 30)
SEED_BASE = 10_000


def evaluate_run(run_dir, sigmas_mm=EVAL_SIGMAS_MM, n_episodes=200, out_csv=None):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "progress.json").read_text())
    pol = load_policy(run_dir)
    rows = []
    for s_mm in sigmas_mm:
        env = PickPlaceEnv(METHODS[meta["method"]].obs_mode, sigma_fixed=s_mm / 1000.0)
        for ep in range(n_episodes):
            obs, info = env.reset(seed=SEED_BASE + ep)
            pol.reset()
            done, ret = False, 0.0
            while not done:
                obs, r, term, trunc, info = env.step(pol.act(obs))
                ret += r
                done = term or trunc
            rows.append(dict(method=meta["method"], seed=meta["seed"], sigma_mm=s_mm, episode=ep,
                             success=info["success"], attempts=info["attempts"], sim_time=info["sim_time"],
                             commit_err_mm=np.mean(info["commit_err_mm"]) if info["commit_err_mm"] else np.nan,
                             belief_std_at_commit=np.mean(info["belief_std_at_commit"]) if info["belief_std_at_commit"] else np.nan,
                             **{"return": ret}))
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for prog in sorted(Path(a.runs).glob("*/seed*/progress.json")):
        meta = json.loads(prog.read_text())
        if not meta.get("done") or (a.only and meta["method"] not in a.only):
            continue
        out = Path(a.runs) / "eval" / f"{meta['method']}_seed{meta['seed']}.csv"
        if out.exists():
            continue
        evaluate_run(prog.parent, n_episodes=a.episodes, out_csv=out)
        print("wrote", out)


if __name__ == "__main__":
    main()
```

`pomdp_pick/eval/figures.py`:

```python
"""Figures from evaluation CSVs."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pomdp_pick.eval.stats import per_cell  # noqa: E402

COLORS = {"oracle": "#2E7D32", "plain": "#8A8F98", "framestack": "#B26B00", "recurrent": "#7B3FA0", "belief": "#0E8FA0"}


def success_vs_sigma(df, path):
    cell = per_cell(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for m, g in cell.groupby("method"):
        ax.plot(g.sigma_mm, g.rate * 100, "-o", color=COLORS.get(m), label=m, lw=2, ms=4)
        ax.fill_between(g.sigma_mm, g.lo * 100, g.hi * 100, color=COLORS.get(m), alpha=0.15, lw=0)
    ax.axvline(20, ls=":", color="#999", lw=1)
    ax.text(20.5, 3, "training max", color="#777", fontsize=8)
    ax.set_xlabel("Camera noise σ_ep (mm)")
    ax.set_ylabel("Pick-and-place success (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(p) for p in sys.argv[2:]])
    success_vs_sigma(df, sys.argv[1])
```

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_eval.py -v` → PASS. Then run a smoke eval on one tiny run: `.venv/bin/python -c "from pomdp_pick.eval.evaluate import evaluate_run; print(evaluate_run('<tmp run from Task 10>', (10,), 2).shape)"`.

- [ ] **Step 5: Commit.** `git add pomdp_pick/eval tests/test_eval.py && git commit -m "feat: shared-seed evaluation, rliable stats, success-vs-noise figure"`

---

### Task 12: Oracle sanity gate, then queue the run matrix

**Files:**
- Create: `configs/queue_phase1.json`
- Modify: `docs/training-log.md`

- [ ] **Step 1: Run the whole test suite** `.venv/bin/pytest -q` → all pass.
- [ ] **Step 2: Write `configs/queue_phase1.json`.** Oracle seed 0 runs first as the sanity gate, then plain seed 0 to check the memory gap:

```json
[
  {"name": "oracle_s0", "cmd": [".venv/bin/python", "-m", "pomdp_pick.agents.train", "--method", "oracle", "--seed", "0"]},
  {"name": "plain_s0",  "cmd": [".venv/bin/python", "-m", "pomdp_pick.agents.train", "--method", "plain",  "--seed", "0"]},
  {"name": "eval_phase1", "cmd": [".venv/bin/python", "-m", "pomdp_pick.eval.evaluate", "--episodes", "100", "--only", "oracle", "plain"]}
]
```

- [ ] **Step 3: Launch it detached under the watchdog:** `nohup .venv/bin/python tools/run_queue.py configs/queue_phase1.json > runs/queue_phase1.out 2>&1 &`
- [ ] **Step 4: Log it.** Record in `docs/training-log.md`: the v2 env summary, the grasp tolerance table, the benchmark, the queue contents and the start time.
- [ ] **Step 5: Commit.** `git add configs/queue_phase1.json docs/training-log.md && git commit -m "run: phase-1 queue (oracle sanity gate, plain memory-gap check)"`

**Gate after phase 1** (next session): oracle success must be ≥ 70% and oracle − plain must be significant (non-overlapping Wilson intervals at ≥ 2 noise levels).
- If oracle fails, tune the reward and primitive using oracle only.
- If the gap is small, raise the drift or distractor height per spec §10.
- Only then queue belief, framestack and recurrent, plus seeds 1–2.
