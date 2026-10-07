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


def test_primitives_fit_a_tight_sim_time_budget():
    """Each commit costs sim time = training wall time; keep the grasp+place chain short."""
    arm = _setup()
    t0 = arm.sim_time
    assert arm.grasp()
    t1 = arm.sim_time
    assert arm.place(np.array([0.16, -0.08]))
    t2 = arm.sim_time
    assert t1 - t0 <= 1.8
    assert t2 - t1 <= 2.6
