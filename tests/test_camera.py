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
    arm.reset_pose(np.array([0.16, 0.0, 0.08]), 0.0)
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
