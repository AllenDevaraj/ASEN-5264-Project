# Overhead Camera Addition — Design Spec

**Date:** 2026-03-15
**Status:** Approved
**Context:** The SO-ARM101 LegoPickEnv has only a wrist-mounted camera. When the arm wanders away from the workspace, the particle filter never loses confidence because the occlusion model only checks blue-occludes-red, not field-of-view. This causes training instability — the agent can get "forever sidetracked" with no signal to return. Adding a fixed overhead camera provides a second observation source that keeps the agent globally aware of the block position, matching real-world manipulation setups.

---

## 1. Overhead Camera Geometry

- **Position:** Fixed at `(0.15, 0.0, 0.50)` in world frame, centered over workspace
- **Orientation:** Looking straight down (−Z axis)
- **FOV coverage:** At 0.50m height, the furthest block (SPAWN_R_MAX=0.22m) is ~32° from nadir — well within typical camera FOV. The entire workspace is always visible from the overhead camera.
- **Not rendered in MuJoCo** — purely a simulated observation source (no image, just noisy XY+theta). A future enhancement could render this as an actual MuJoCo camera for visualization.

## 2. Overhead Occlusion Model (Top-Down Shadow)

Uses XY footprint overlap — no projection math needed since the camera is directly overhead.

**Algorithm:**
1. Compute the blue block's 2D oriented bounding box (4 corners from center + half_size + yaw)
2. Check if the red block's center point falls inside the blue block's rotated rectangle
3. If inside → red block is occluded from overhead; observation withheld

**Why center-point-in-rectangle instead of full rectangle overlap:**
The overhead camera has a clear line of sight. The blue block (2x2, footprint ~16mm x 16mm) is smaller than the red block (2x4, footprint ~32mm x 16mm). For the blue block to meaningfully occlude the red from directly above, it needs to cover the red block's center.

**Note:** Because the blue block is small relative to the red block, overhead occlusion will be rare in practice — it requires the blue block to be placed nearly on top of the red block's center. This is intentional: the overhead camera should be a reliable second source that only loses visibility in unusual configurations.

## 3. Noise Model

- `SIGMA_OVERHEAD = 0.005` (5mm fixed, constant across episodes)
- Independent of `sigma_ep` (episodic wrist camera noise)
- Lower than wrist camera because overhead is fixed (no arm vibration, constant distance)
- Applied as: `obs = true_pose + N(0, sigma_overhead²)` for x, y; `theta_noise = N(0, sigma_overhead * 10)` matching the wrist camera's theta scaling

**Known limitation (pre-existing):** The particle filter's `update()` takes a single scalar `sigma_obs` and applies it uniformly to x, y, and theta. In reality, theta noise is 10x larger than xy noise. This mismatch exists in the wrist camera path already and is preserved here. A future improvement would change `update()` to accept a per-dimension sigma vector, but this is out of scope for this change.

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

**Note on 12D overlap:** Both plain and belief modes now produce 12D observations, but with completely different semantics (raw sensor readings vs. belief statistics). This is acceptable because separate network weights are trained for each mode — the policies never share parameters.

## 5. Particle Filter Integration

No changes to `ParticleFilter` class. The change is in `LegoPickEnv.step()`:

```python
# Current (wrist only):
self.pf.predict()
visible_obs = self._get_visible_observations()
self.pf.update(visible_obs, self._get_effective_sigma())
self.pf.resample()

# Proposed (wrist + overhead):
self.pf.predict()

wrist_obs = self._get_visible_observations()
self.pf.update(wrist_obs, self._get_effective_sigma())

overhead_obs = self._get_overhead_visible_observations()
self.pf.update(overhead_obs, SIGMA_OVERHEAD)

self.pf.resample()  # single resample AFTER both updates
```

Two sequential `pf.update()` calls with different sigmas, followed by a single `resample()`. This is mathematically correct: each `update()` multiplies weights by the observation likelihood, and `resample()` normalizes and resamples once after all evidence is incorporated. No intermediate resample between the two updates.

Scenarios:
- **Both visible:** Two updates per step → fast convergence, low sigma
- **Wrist occluded, overhead visible:** One update from overhead → prevents drift
- **Both occluded:** No updates → sigma grows (correct uncertainty behavior)
- **Wrist visible, overhead occluded:** One update from wrist → still functional

The same dual-update pattern applies in `reset()` for the initial PF observation.

## 6. File Changes

| File | Change | Size |
|------|--------|------|
| `occlusion.py` | Add `is_occluded_overhead()` — point-in-rotated-rectangle check | ~30 lines |
| `lego_pick_env.py` | Add `SIGMA_OVERHEAD` constant, `_get_overhead_noisy_obs()`, `_is_target_occluded_overhead()`, `_get_overhead_visible_observations()`; update `_build_observation()` for 12D plain; update `step()` and `reset()` for dual PF update; update `__init__` obs_dim for plain mode; add `overhead_occluded` to info dict | ~45 lines |
| `test_env.py` | Update obs shape test (9→12 plain), add overhead occlusion test, add dual-camera PF convergence test | ~40 lines |
| `watch_env.py` | Print overhead obs info | ~5 lines |
| `watch_belief_env.py` | Print overhead visibility status | ~5 lines |

**No changes to:**
- `particle_filter.py` — interface already supports this
- `train_ppo.py` — obs space auto-detected from env
- `train_belief_ppo.py` — obs space auto-detected from env
- `evaluate.py` — uses env directly
- `train_pomcp.py` — uses env directly

## 7. Testing Criteria

1. `test_env.py` passes all tests (with updated obs shapes)
2. Overhead occlusion test: blue block center over red center → occluded; blocks side by side → not occluded
3. Plain PPO obs shape is (12,), Belief PPO obs shape is (12,)
4. Dual-camera PF convergence test: PF with both cameras converges faster (lower sigma after N steps) than PF with wrist only
5. Overhead observation is independent of `sigma_ep` (episodic wrist noise)
6. `watch_belief_env.py` shows overhead visibility status and sigma behavior:
   - sigma stays low when at least one camera sees the block
   - sigma grows only when both cameras are occluded
7. Both `watch_env.py` and `watch_belief_env.py` display overhead camera info correctly
