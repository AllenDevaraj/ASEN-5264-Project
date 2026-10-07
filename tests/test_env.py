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
