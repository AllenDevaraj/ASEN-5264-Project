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
