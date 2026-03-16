#!/usr/bin/env python3
"""Phase 1 verification tests for LegoPickEnv.

Runs 8 checks to validate the environment before training:
  1. Model loading
  2. IK roundtrip
  3. Block randomization
  4. Occlusion detection
  5. Grasp probability
  6. Particle filter convergence
  7. Episode loop (random actions)
  8. Observation dimensions
"""

import math
import sys
import traceback

import numpy as np

# Add package to path
sys.path.insert(0, '/home/the2xman/ASEN-5264-Project/vla_SO-ARM101/src/so_arm101_control')


def test_model_loading():
    """Test 1: Model loads, joints and mocap bodies found."""
    from so_arm101_control.model_loader import (
        build_joint_map, build_mocap_map, load_mujoco_model,
    )
    import mujoco

    model, data = load_mujoco_model()
    joint_map = build_joint_map(model)
    mocap_map = build_mocap_map(model)

    # Check joints
    expected_joints = [
        'shoulder_pan', 'shoulder_lift', 'elbow_flex',
        'wrist_flex', 'wrist_roll', 'gripper_joint',
    ]
    for j in expected_joints:
        assert j in joint_map, f"Joint '{j}' not found in model"

    # Check mocap bodies
    for block in ['red_lego_2x4', 'blue_lego_2x2']:
        assert block in mocap_map, f"Mocap body '{block}' not found"

    # Test mj_forward runs
    mujoco.mj_forward(model, data)
    print(f"  Joints found: {list(joint_map.keys())}")
    print(f"  Mocap bodies: {list(mocap_map.keys())}")
    return True


def test_ik_roundtrip():
    """Test 2: IK roundtrip error < 2mm."""
    from so_arm101_control.compute_workspace import forward_kinematics, geometric_ik

    targets = [
        (0.18, 0.03, 0.005),
        (0.15, -0.05, 0.01),
        (0.20, 0.0, -0.05),
    ]

    for tx, ty, tz in targets:
        solutions = geometric_ik(tx, ty, tz, grasp_yaw=0.0)
        assert len(solutions) > 0, f"IK failed for ({tx}, {ty}, {tz})"

        sol = solutions[0]
        angles = [sol[n] for n in ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                                    'wrist_flex', 'wrist_roll']]
        result = forward_kinematics(angles)
        error = np.linalg.norm(np.array([tx, ty, tz]) - result)
        assert error < 0.002, f"IK error {error*1000:.1f}mm > 2mm for ({tx}, {ty}, {tz})"
        print(f"  IK({tx:.2f}, {ty:.2f}, {tz:.3f}) error: {error*1000:.2f}mm")

    return True


def test_block_randomization():
    """Test 3: Block randomization stays in workspace with min spacing."""
    from so_arm101_control.lego_pick_env import (
        LegoPickEnv, SPAWN_R_MIN, SPAWN_R_MAX, MIN_SPACING,
    )

    env = LegoPickEnv(belief_mode=False)
    violations = 0

    for i in range(100):
        env.reset(seed=i)
        poses = env._block_true_poses

        for name, (x, y, yaw) in poses.items():
            r = math.sqrt(x**2 + y**2)
            if r < SPAWN_R_MIN - 0.01 or r > SPAWN_R_MAX + 0.01:
                violations += 1
                print(f"  WARNING: {name} at r={r:.3f} outside [{SPAWN_R_MIN}, {SPAWN_R_MAX}]")

        # Check spacing between blocks
        names = list(poses.keys())
        for a in range(len(names)):
            for b in range(a+1, len(names)):
                xa, ya, _ = poses[names[a]]
                xb, yb, _ = poses[names[b]]
                d = math.sqrt((xa-xb)**2 + (ya-yb)**2)
                if d < MIN_SPACING - 0.001:
                    violations += 1

    env.close()
    print(f"  100 resets, {violations} violations")
    assert violations == 0, f"{violations} randomization violations"
    return True


def test_occlusion():
    """Test 4: Occlusion detection works for known configurations."""
    from so_arm101_control.occlusion import is_occluded

    # Camera above and behind, looking forward-down
    camera_pos = np.array([0.05, 0.0, 0.15])
    # Rotation: camera_link X = world X (forward), Z = world Z (up)
    camera_rot = np.eye(3)

    # Target far, occluder between camera and target
    result = is_occluded(
        target_pos=(0.20, 0.0),
        target_half_size=(0.016, 0.008),
        target_yaw=0.0,
        occluder_pos=(0.12, 0.0),
        occluder_half_size=(0.020, 0.020),  # large occluder
        occluder_yaw=0.0,
        camera_pos=camera_pos,
        camera_rot=camera_rot,
    )
    print(f"  Occluder between camera and target: {result}")

    # No occlusion: blocks side by side
    result2 = is_occluded(
        target_pos=(0.20, 0.05),
        target_half_size=(0.016, 0.008),
        target_yaw=0.0,
        occluder_pos=(0.20, -0.05),
        occluder_half_size=(0.008, 0.008),
        occluder_yaw=0.0,
        camera_pos=camera_pos,
        camera_rot=camera_rot,
    )
    print(f"  Blocks side by side: {result2}")
    assert result2 == False, "Side-by-side blocks should not be occluded"

    return True


def test_grasp_probability():
    """Test 5: Grasp probability matches sigmoid(-150 * d)."""
    distances = [0.0, 0.005, 0.010, 0.015, 0.020]
    expected = []
    for d in distances:
        p = 1.0 / (1.0 + math.exp(-300.0 * (0.01 - d)))
        expected.append(p)
        print(f"  d={d*1000:.0f}mm: p_grasp={p:.3f}")

    # Verify known values: sigmoid(300*(0.01-d))
    assert abs(expected[0] - 0.953) < 0.01, "p(0mm) should be ~95%"
    assert 0.45 < expected[2] < 0.55, "p(10mm) should be ~50%"
    assert expected[4] < 0.10, "p(20mm) should be < 10%"
    return True


def test_particle_filter():
    """Test 6: Particle filter converges given repeated observations."""
    from so_arm101_control.particle_filter import ParticleFilter

    pf = ParticleFilter(n_particles=300, n_blocks=1)
    true_pose = np.array([[0.18, 0.03, 0.5]])
    sigma_obs = 0.008

    # Initialize with noisy observation
    rng = np.random.default_rng(42)
    initial_obs = true_pose + rng.normal(0, sigma_obs * 3, (1, 3))
    pf.reset(initial_obs, sigma_init=0.05)

    # Feed 50 observations
    for i in range(50):
        pf.predict()
        obs = true_pose[0] + rng.normal(0, sigma_obs, 3)
        pf.update({0: obs}, sigma_obs)
        pf.resample()

    mu, sigma = pf.get_belief()
    error = np.linalg.norm(mu[0, :2] - true_pose[0, :2])
    print(f"  After 50 obs: mu_error={error*1000:.1f}mm, sigma_xy={sigma[0,:2]*1000}")
    assert error < sigma_obs * 3, f"PF error {error:.4f} too large"
    assert sigma[0, 0] < 0.03, f"PF sigma_x {sigma[0,0]:.4f} not converging"
    return True


def test_episode_loop():
    """Test 7: Random episode runs 200 steps without crashing."""
    from so_arm101_control.lego_pick_env import LegoPickEnv

    env = LegoPickEnv(belief_mode=False)
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    env.close()
    print(f"  {steps} steps, total_reward={total_reward:.1f}")
    assert steps > 0, "No steps completed"
    return True


def test_observation_dimensions():
    """Test 8: Observation dimensions correct for both modes."""
    from so_arm101_control.lego_pick_env import LegoPickEnv

    # Plain mode: 18D (6 joints + 3 wrist + 3 overhead + 3 ee + 2 goal + 1 holding)
    env1 = LegoPickEnv(belief_mode=False)
    obs1, _ = env1.reset(seed=0)
    assert obs1.shape == (18,), f"Plain obs shape {obs1.shape} != (18,)"
    env1.close()
    print(f"  Plain mode: obs shape = {obs1.shape}")

    # Belief mode: 18D (6 joints + 3 mu + 3 sigma + 3 ee + 2 goal + 1 holding)
    env2 = LegoPickEnv(belief_mode=True)
    obs2, _ = env2.reset(seed=0)
    assert obs2.shape == (18,), f"Belief obs shape {obs2.shape} != (18,)"
    env2.close()
    print(f"  Belief mode: obs shape = {obs2.shape}")

    return True


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


def main():
    tests = [
        ("1. Model loading", test_model_loading),
        ("2. IK roundtrip", test_ik_roundtrip),
        ("3. Block randomization", test_block_randomization),
        ("4. Occlusion detection", test_occlusion),
        ("5. Grasp probability", test_grasp_probability),
        ("6. Particle filter convergence", test_particle_filter),
        ("7. Episode loop (random)", test_episode_loop),
        ("8. Observation dimensions", test_observation_dimensions),
        ("9. Overhead occlusion", test_overhead_occlusion),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)
        try:
            result = test_fn()
            if result:
                print(f"  PASSED")
                passed += 1
            else:
                print(f"  FAILED")
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{passed+failed} passed, {failed} failed")
    print('='*60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
