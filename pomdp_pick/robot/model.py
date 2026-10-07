"""Build the v2 MuJoCo model: SO-ARM101 with physical jaw pads, position actuators and a wrist camera site.

Settings verified by a grasp prototype (24/24 lifts): jaw meshes become visual-only and box pads
carry the contact; arm position actuators kp=50 kv=4.5 with gravity compensation; 4 ms timestep
(grasps verified at 2 and 4 ms; 4 ms halves the physics cost per decision).
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from pomdp_pick.robot import mjcf_merge as mm
from pomdp_pick.robot.kinematics import ARM_JOINT_NAMES as ARM, forward_kinematics, geometric_ik, make_transform

TARGET = "red_lego_2x4"
DISTRACTORS = ("blue_lego_2x2", "blue_lego_2x2_b")
ROBOT_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")
# URDF fixed chain gripper -> camera_mount -> usb_camera -> camera_link (fused away by the compiler)
_CAM_CHAIN = [((0.002359, 0.049561, -0.011838), (-2.956686, -0.384393, -2.034078)),
              ((-0.010340, 0.004827, -0.019687), (1.570625, 0.000515, 2.705234)),
              ((0.0, 0.0139, 0.0), (1.5708, 0.0, 1.5708))]
# Hull pairs that overlap at rest; once actuated they jam shoulder_pan.
_EXCLUDES = [("world", "shoulder"), ("world", "upper_arm"), ("shoulder", "lower_arm"),
             ("upper_arm", "wrist"), ("lower_arm", "gripper"), ("lower_arm", "jaw"), ("wrist", "jaw")]


@dataclass(frozen=True)
class ModelConfig:
    timestep: float = 0.004
    arm_kp: float = 50.0
    arm_kv: float = 4.5
    arm_force: float = 3.5
    grip_kp: float = 10.0
    grip_force: float = 3.5
    damping: float = 1.0
    armature: float = 0.1
    friction: tuple = (1.2, 0.02, 0.001)
    condim: int = 3
    brick_mass: float = 0.010


def _merged_xml():
    roots = {"so_arm101_description": mm._DESC_PKG,
             "so_arm101_control": os.path.join(mm._PKG_ROOT, "so_arm101_control")}
    urdf = mm._resolve_package_uris(open(mm.DEFAULT_URDF).read(), roots)
    assets = mm._collect_mesh_assets(urdf)
    for f in os.listdir(mm.DEFAULT_MESHES):
        if f.lower().endswith(".stl") and f not in assets:
            assets[f] = open(os.path.join(mm.DEFAULT_MESHES, f), "rb").read()
    return mm._merge_urdf_into_scene(urdf, mm.DEFAULT_SCENE, assets), assets


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def _pad_frames():
    """Pad boxes expressed in the gripper/jaw body frames, measured at a reference pose
    (yaw 0, jaw 0) where the jaws close along world Y: fixed-jaw inner face at TCP y≈+1 mm,
    moving-jaw inner face at y≈-16 mm, fingertip band from TCP-6 mm to TCP+15 mm."""
    m, d = mm.load_mujoco_model()
    jm = mm.build_joint_map(m)
    sol = geometric_ik(0.20, 0.0, 0.05, grasp_yaw=0.0)[0]
    for k, v in sol.items():
        d.qpos[jm[k]] = v
    d.qpos[jm["gripper_joint"]] = 0.0
    mujoco.mj_forward(m, d)
    tcp = np.array(forward_kinematics([sol[n] for n in ARM]))
    z_lo, z_hi = tcp[2] - 0.0060, tcp[2] + 0.0150
    zc, zh = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
    centers = {"gripper": np.array([tcp[0], 0.0040, zc]), "jaw": np.array([tcp[0], -0.0190, zc])}
    half = np.array([0.0055, 0.0030, zh])
    out = {}
    for body, c in centers.items():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        R = d.xmat[b].reshape(3, 3)
        out[body] = (R.T @ (c - d.xpos[b]), _quat(R.T.copy()), half)
    return out


def _camera_site():
    T = np.eye(4)
    for xyz, rpy in _CAM_CHAIN:
        T = T @ make_transform(xyz, rpy)
    return T[:3, 3], _quat(T[:3, :3])


def build_model(cfg=ModelConfig()):
    xml, assets = _merged_xml()
    root = ET.fromstring(xml)
    opt = root.find("option")
    opt.set("timestep", str(cfg.timestep))
    opt.set("integrator", "implicitfast")
    opt.set("cone", "pyramidal")
    opt.set("noslip_iterations", "0")
    pads = _pad_frames()
    cam_pos, cam_quat = _camera_site()
    fmt = lambda v: " ".join(f"{x:.6g}" for x in v)  # noqa: E731
    for body in root.iter("body"):
        name = body.get("name", "")
        if name in ROBOT_BODIES:
            body.set("gravcomp", "1")
            j = body.find("joint")
            if j is not None:
                j.set("damping", str(cfg.damping))
                j.set("armature", str(cfg.armature))
        if name in ("gripper", "jaw"):
            for g in body.findall("geom"):
                g.set("contype", "0")
                g.set("conaffinity", "0")
            pos, quat, half = pads[name]
            ET.SubElement(body, "geom", name=f"{name}_pad", type="box", pos=fmt(pos), quat=fmt(quat),
                          size=fmt(half), contype="1", conaffinity="1", condim=str(cfg.condim),
                          friction=fmt(cfg.friction), solref="0.008 1", solimp="0.95 0.99 0.001",
                          rgba="0.2 0.9 0.4 0", mass="0")
        if name == "gripper":
            ET.SubElement(body, "site", name="wrist_cam", pos=fmt(cam_pos), quat=fmt(cam_quat), size="0.003")
        if name == TARGET:
            g = body.find("geom")
            g.set("condim", str(cfg.condim))
            g.set("mass", str(cfg.brick_mass))
            g.set("friction", fmt(cfg.friction))
    con = root.find("contact")
    if con is None:
        con = ET.SubElement(root, "contact")
    for b1, b2 in _EXCLUDES:
        ET.SubElement(con, "exclude", body1=b1, body2=b2)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for jn in ARM:
        ET.SubElement(act, "position", name=f"a_{jn}", joint=jn, kp=str(cfg.arm_kp), kv=str(cfg.arm_kv),
                      forcerange=f"{-cfg.arm_force} {cfg.arm_force}")
    ET.SubElement(act, "position", name="a_gripper", joint="gripper_joint", kp=str(cfg.grip_kp), kv="0.2",
                  forcerange=f"{-cfg.grip_force} {cfg.grip_force}", ctrlrange="-0.174533 1.74533")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    return m, mujoco.MjData(m)
