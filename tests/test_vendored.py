import sys, numpy as np
LEGACY = "vla_SO-ARM101/src/so_arm101_control"

def test_ik_fk_match_legacy():
    sys.path.insert(0, LEGACY)
    from so_arm101_control import compute_workspace as old
    from pomdp_pick.robot import kinematics as new
    for x, y, z, yaw in [(0.18, 0.0, 0.06, 0.0), (0.22, 0.08, 0.02, 0.7), (0.15, -0.08, 0.08, -0.6)]:
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
