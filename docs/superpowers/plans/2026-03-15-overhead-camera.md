# Overhead Camera Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed overhead camera with top-down occlusion to LegoPickEnv, providing dual-camera observations for training stability.

**Architecture:** Fixed overhead camera at (0.15, 0, 0.5) looking straight down. Uses XY point-in-rotated-rectangle occlusion. Two independent PF updates per step. Plain PPO obs grows 9D→12D; belief PPO obs stays 12D with better-informed PF.

**Tech Stack:** Python, NumPy, Gymnasium, MuJoCo (no new deps)

---

## Chunk 1: Implementation

### Task 1: Add `is_occluded_overhead()` to occlusion.py

**Files:**
- Modify: `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/occlusion.py:143` (append)

- [ ] **Step 1: Add point-in-rotated-rectangle function and overhead occlusion check**

```python
def _point_in_rotated_rect(point_xy, rect_center, rect_half_size, rect_yaw):
    """Check if a 2D point falls inside a rotated rectangle.

    Args:
        point_xy: (x, y) point to test.
        rect_center: (x, y) center of rectangle.
        rect_half_size: (half_length, half_width) of rectangle.
        rect_yaw: Yaw angle of rectangle (radians).

    Returns:
        True if point is inside the rotated rectangle.
    """
    dx = point_xy[0] - rect_center[0]
    dy = point_xy[1] - rect_center[1]
    cy, sy = np.cos(rect_yaw), np.sin(rect_yaw)
    # Rotate point into rectangle's local frame
    local_x = dx * cy + dy * sy
    local_y = -dx * sy + dy * cy
    hl, hw = rect_half_size
    return abs(local_x) <= hl and abs(local_y) <= hw


def is_occluded_overhead(target_pos, occluder_pos, occluder_half_size, occluder_yaw):
    """Check if target block center is occluded by occluder from directly overhead.

    Uses top-down shadow model: occluder's 2D footprint shadows the target center.

    Args:
        target_pos: (x, y) of target block center.
        occluder_pos: (x, y) of occluder block center.
        occluder_half_size: (half_length, half_width) of occluder.
        occluder_yaw: Yaw angle of occluder block (radians).

    Returns:
        True if target center falls inside occluder's XY footprint.
    """
    return _point_in_rotated_rect(target_pos, occluder_pos,
                                   occluder_half_size, occluder_yaw)
```

- [ ] **Step 2: Commit occlusion changes**

```bash
git add vla_SO-ARM101/src/so_arm101_control/so_arm101_control/occlusion.py
git commit -m "feat: add is_occluded_overhead() for top-down shadow occlusion"
```

---

### Task 2: Update lego_pick_env.py with overhead camera

**Files:**
- Modify: `vla_SO-ARM101/src/so_arm101_control/so_arm101_control/lego_pick_env.py`

- [ ] **Step 1: Add import and constant**

Add `is_occluded_overhead` to the import from occlusion, and add `SIGMA_OVERHEAD = 0.005` constant after `GRIPPER_CLOSED`.

- [ ] **Step 2: Update obs_dim in `__init__`**

Change `obs_dim = 12 if belief_mode else 9` to `obs_dim = 12` (both modes are now 12D).

- [ ] **Step 3: Add overhead camera methods**

Add three new methods after `_get_visible_observations()`:

```python
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
```

- [ ] **Step 4: Update `reset()` PF initialization for dual camera**

In `reset()`, after the existing PF reset block, add overhead camera PF update:

```python
# 7. Reset particle filter
if self.belief_mode:
    noisy_obs = self._get_noisy_target_obs()
    self.pf.reset(noisy_obs.reshape(1, 3), sigma_init=self._sigma_ep * 3)
    # Also feed overhead observation at reset
    overhead_obs = self._get_overhead_visible_observations()
    if overhead_obs:
        self.pf.update(overhead_obs, SIGMA_OVERHEAD)
```

- [ ] **Step 5: Update `step()` PF section for dual camera**

Replace the PF update block in `step()`:

```python
# 7. Update particle filter
if self.belief_mode:
    self.pf.predict()
    wrist_obs = self._get_visible_observations()
    self.pf.update(wrist_obs, self._get_effective_sigma())
    overhead_obs = self._get_overhead_visible_observations()
    self.pf.update(overhead_obs, SIGMA_OVERHEAD)
    self.pf.resample()
```

- [ ] **Step 6: Update `_build_observation()` for 12D plain mode**

Replace the plain mode branch:

```python
else:
    wrist_obs = self._get_noisy_target_obs()
    overhead_obs = self._get_overhead_noisy_obs()
    return np.concatenate([joint_obs, wrist_obs, overhead_obs]).astype(np.float32)
```

- [ ] **Step 7: Add overhead info to step() info dict**

Add `overhead_occluded` to the info dict in `step()`:

```python
info["overhead_occluded"] = self._is_target_occluded_overhead()
```

- [ ] **Step 8: Update module docstring**

Update the docstring at the top to reflect 12D for both modes.

- [ ] **Step 9: Commit env changes**

```bash
git add vla_SO-ARM101/src/so_arm101_control/so_arm101_control/lego_pick_env.py
git commit -m "feat: add overhead camera with dual PF updates to LegoPickEnv"
```

---

### Task 3: Update tests

**Files:**
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/test_env.py`

- [ ] **Step 1: Update obs shape test**

Change plain mode assertion from `(9,)` to `(12,)`.

- [ ] **Step 2: Add overhead occlusion test**

Add `test_overhead_occlusion()` function:

```python
def test_overhead_occlusion():
    """Test 9: Overhead occlusion detects blue-over-red correctly."""
    from so_arm101_control.occlusion import is_occluded_overhead

    # Blue block directly on top of red block center -> occluded
    result = is_occluded_overhead(
        target_pos=(0.18, 0.03),
        occluder_pos=(0.18, 0.03),
        occluder_half_size=(0.008, 0.008),
        occluder_yaw=0.0,
    )
    assert result == True, "Same position should be occluded"
    print(f"  Blue on red center: {result}")

    # Blocks side by side -> not occluded
    result2 = is_occluded_overhead(
        target_pos=(0.18, 0.03),
        occluder_pos=(0.18, 0.08),
        occluder_half_size=(0.008, 0.008),
        occluder_yaw=0.0,
    )
    assert result2 == False, "Side-by-side should not be occluded"
    print(f"  Blocks side by side: {result2}")

    # Blue slightly offset but still covering red center -> occluded
    result3 = is_occluded_overhead(
        target_pos=(0.18, 0.03),
        occluder_pos=(0.185, 0.035),
        occluder_half_size=(0.008, 0.008),
        occluder_yaw=0.0,
    )
    assert result3 == True, "Small offset should still occlude"
    print(f"  Small offset: {result3}")

    return True
```

- [ ] **Step 3: Add test to test list and update numbering**

Add `("9. Overhead occlusion", test_overhead_occlusion)` to the tests list.

- [ ] **Step 4: Run tests and verify all pass**

```bash
cd vla_SO-ARM101/src/so_arm101_control/scripts && python3 test_env.py
```

Expected: 9/9 passed, 0 failed

- [ ] **Step 5: Commit test changes**

```bash
git add vla_SO-ARM101/src/so_arm101_control/scripts/test_env.py
git commit -m "test: update obs shapes and add overhead occlusion test"
```

---

### Task 4: Update watch scripts

**Files:**
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/watch_env.py`
- Modify: `vla_SO-ARM101/src/so_arm101_control/scripts/watch_belief_env.py`

- [ ] **Step 1: Update watch_belief_env.py to show overhead status**

In the print line inside the loop, add overhead occlusion status from `info.get("overhead_occluded")`.

- [ ] **Step 2: Update watch_env.py print to show 12D obs**

Update the step print to show overhead obs values (obs indices 9-11).

- [ ] **Step 3: Commit watch script changes**

```bash
git add vla_SO-ARM101/src/so_arm101_control/scripts/watch_env.py \
       vla_SO-ARM101/src/so_arm101_control/scripts/watch_belief_env.py
git commit -m "chore: update watch scripts to display overhead camera info"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cd vla_SO-ARM101/src/so_arm101_control/scripts && python3 test_env.py
```

Expected: 9/9 passed

- [ ] **Step 2: Quick smoke test with watch_belief_env.py**

```bash
python3 watch_belief_env.py
```

Verify overhead info is printed and sigma behavior is correct.
