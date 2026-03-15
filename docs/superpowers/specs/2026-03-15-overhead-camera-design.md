# Overhead Camera Addition — Design Spec

**Date:** 2026-03-15
**Status:** Approved
**Context:** The SO-ARM101 LegoPickEnv has only a wrist-mounted camera. When the arm wanders away from the workspace, the particle filter never loses confidence because the occlusion model only checks blue-occludes-red, not field-of-view. This causes training instability — the agent can get "forever sidetracked" with no signal to return. Adding a fixed overhead camera provides a second observation source that keeps the agent globally aware of the block position, matching real-world manipulation setups.

---

## 1. Overhead Camera Geometry

- **Position:** Fixed at `(0.15, 0.0, 0.50)` in world frame, centered over workspace
- **Orientation:** Looking straight down (−Z axis)
- **Not rendered in MuJoCo** — purely a simulated observation source (no image, just noisy XY+theta)

## 2. Overhead Occlusion Model (Top-Down Shadow)

Uses XY footprint overlap — no projection math needed since the camera is directly overhead.

**Algorithm:**
1. Compute the blue block's 2D oriented bounding box (4 corners from center + half_size + yaw)
2. Check if the red block's center point falls inside the blue block's rotated rectangle
3. If inside → red block is occluded from overhead; observation withheld

**Why center-point-in-rectangle instead of full rectangle overlap:**
The overhead camera has a clear line of sight. The blue block (2x2) is smaller than the red block (2x4). For the blue block to meaningfully occlude the red, it needs to cover the red block's center. Partial edge overlap shouldn't count as full occlusion from directly above.

## 3. Noise Model

- `SIGMA_OVERHEAD = 0.005` (5mm fixed, constant across episodes)
- Independent of `sigma_ep` (episodic wrist camera noise)
- Lower than wrist camera because overhead is fixed (no arm vibration, constant distance)
- Applied as: `obs = true_pose + N(0, sigma_overhead²)` for x, y; `theta_noise = N(0, sigma_overhead * 10)` matching the wrist camera's theta scaling

## 4. Observation Space Changes

### Plain PPO: 9D → 12D

```
Index  | Content
-------|--------
0-4    | joint angles (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
5      | gripper_joint
6-8    | wrist camera noisy obs (x, y, theta) — always provided, no occlusion in plain mode
9-11   | overhead camera noisy obs (x, y, theta) — always provided, no occlusion in plain mode
```

Plain PPO always receives both observations. No occlusion handling in plain mode — this maintains the "no belief state" baseline design.

### Belief PPO: stays 12D (unchanged shape)

```
Index  | Content
-------|--------
0-4    | joint angles
5      | gripper_joint
6-8    | particle filter mean (mu_x, mu_y, mu_theta)
9-11   | particle filter sigma (sigma_x, sigma_y, sigma_theta)
```

The particle filter internally fuses both camera sources. Its output shape is unchanged. The policy sees the same 12D belief — just better-informed because the PF receives observations from two independent viewpoints.

## 5. Particle Filter Integration

No changes to `ParticleFilter` class. The change is in `LegoPickEnv.step()`:

```python
# Current (wrist only):
visible_obs = self._get_visible_observations()          # wrist
self.pf.update(visible_obs, self._get_effective_sigma()) # one update

# Proposed (wrist + overhead):
wrist_obs = self._get_visible_observations()             # wrist camera
self.pf.update(wrist_obs, self._get_effective_sigma())

overhead_obs = self._get_overhead_visible_observations() # overhead camera
self.pf.update(overhead_obs, SIGMA_OVERHEAD)
```

Two sequential `pf.update()` calls with different sigmas. Each call independently tightens the belief. Scenarios:
- **Both visible:** Two updates per step → fast convergence, low sigma
- **Wrist occluded, overhead visible:** One update from overhead → prevents drift
- **Both occluded:** No updates → sigma grows (correct uncertainty behavior)
- **Wrist visible, overhead occluded:** One update from wrist → still functional

## 6. File Changes

| File | Change | Size |
|------|--------|------|
| `occlusion.py` | Add `is_occluded_overhead()` — point-in-rotated-rectangle check | ~30 lines |
| `lego_pick_env.py` | Add `SIGMA_OVERHEAD` constant, `_get_overhead_noisy_obs()`, `_is_target_occluded_overhead()`, `_get_overhead_visible_observations()`; update `_build_observation()` for 12D plain; update `step()` for dual PF update; update `__init__` obs_dim for plain mode | ~40 lines |
| `test_env.py` | Update obs shape test (9→12 plain), add overhead occlusion test | ~30 lines |
| `watch_env.py` | Print overhead obs info | ~5 lines |
| `watch_belief_env.py` | Print overhead visibility status | ~5 lines |

**No changes to:**
- `particle_filter.py` — interface already supports this
- `train_ppo.py` — obs space auto-detected from env
- `train_belief_ppo.py` — obs space auto-detected from env
- `evaluate.py` — uses env directly
- `train_pomcp.py` — uses env directly

## 7. Testing Criteria

1. `test_env.py` passes all 8 tests (with updated obs shapes)
2. Overhead occlusion test: blue block directly over red → occluded; blocks side by side → not occluded
3. `watch_belief_env.py` shows overhead visibility status and sigma behavior:
   - sigma stays low when at least one camera sees the block
   - sigma grows only when both cameras are occluded
4. Plain PPO obs shape is (12,), Belief PPO obs shape is (12,)
