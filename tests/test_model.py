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
