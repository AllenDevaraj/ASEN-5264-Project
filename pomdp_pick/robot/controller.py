"""Cartesian control of the actuated SO-ARM101 and physical grasp / place primitives.

The controlled point P is the centre between the open jaws (where the brick should sit).
The TCP (fixed-jaw inner face) is offset from P across the jaw axis by half the brick
width plus FIXED_CLEAR, so success depends on how well P is placed over the real brick.
"""
import math

import mujoco
import numpy as np

from pomdp_pick.robot import mjcf_merge as mm
from pomdp_pick.robot.kinematics import JOINT_LIMITS, WRIST_ROLL_OFFSET, X_PAN, forward_kinematics, geometric_ik
from pomdp_pick.robot.model import ARM, DISTRACTORS, TARGET

JAW_OPEN = 0.12          # pre-grasp gap ~27 mm: symmetric ~±7 mm tolerance across the jaw axis
JAW_CLOSE = -0.174533
Z_GRASP = 0.0085
Z_HOVER = 0.06
Z_CARRY = 0.06            # top-down reach tops out near 0.08-0.09 m
FIXED_CLEAR = 0.004
BRICK_HALF_W = 0.008
WORKSPACE = dict(r=(0.11, 0.27), ang=1.2, z=(0.0085, 0.09))
FRAME_DT = 1.0 / 30.0
FLIP_MIN_Z = 0.05         # a π wrist flip sweeps the open jaws: do it with the tips above the 33 mm distractors


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)


class Arm:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.jm = mm.build_joint_map(m)
        self.fj = mm.build_freejoint_map(m)
        self.aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{n}") for n in ARM]
        self.gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_gripper")
        self.p = np.array([0.18, 0.0, Z_HOVER])
        self.yaw = 0.0
        self.grip = JAW_OPEN
        self.sol = None
        self.sim_time = 0.0
        self._next_frame = 0.0

    # ---------- geometry ----------
    @staticmethod
    def clamp_p(p):
        p = np.asarray(p, float).copy()
        r = math.hypot(p[0], p[1])
        ang = math.atan2(p[1], p[0])
        r = min(max(r, WORKSPACE["r"][0]), WORKSPACE["r"][1])
        ang = min(max(ang, -WORKSPACE["ang"]), WORKSPACE["ang"])
        return np.array([r * math.cos(ang), r * math.sin(ang), min(max(p[2], WORKSPACE["z"][0]), WORKSPACE["z"][1])])

    def branch_yaw(self, p, yaw):
        """The brick is 180°-symmetric, so yaw and yaw + kπ grasp the same way. Pick the branch whose
        wrist roll stays inside its limits and closest to the current roll (hysteresis), so the open
        jaws never sweep half a turn in the middle of a move; with no current pose, prefer mid-range."""
        pan = math.atan2(-p[1], p[0] - X_PAN)
        lo, hi = JOINT_LIMITS["wrist_roll"]
        ref = self.sol["wrist_roll"] if self.sol is not None else 0.0
        cands = [k for k in range(-3, 4) if lo + 0.05 <= pan + yaw + k * math.pi - WRIST_ROLL_OFFSET <= hi - 0.05]
        if not cands:
            cands = list(range(-3, 4))
        k = min(cands, key=lambda k: abs(pan + yaw + k * math.pi - WRIST_ROLL_OFFSET - ref))
        return yaw + k * math.pi

    def _flips(self, p, gy):
        if self.sol is None:
            return False
        pan = math.atan2(-p[1], p[0] - X_PAN)
        return abs(pan + gy - WRIST_ROLL_OFFSET - self.sol["wrist_roll"]) > math.pi / 2

    def tcp_for(self, p, yaw):
        gy = self.branch_yaw(p, yaw)
        off = BRICK_HALF_W + FIXED_CLEAR
        return np.array([p[0] - off * math.sin(gy), p[1] + off * math.cos(gy), p[2]]), gy

    def _solve(self, p, yaw):
        """IK for P; returns (solution or None, P)."""
        p = np.asarray(p, float)
        tcp, gy = self.tcp_for(p, yaw)
        sols = geometric_ik(*tcp, grasp_yaw=gy)
        if not sols:
            return None, p
        if self.sol is None:
            return sols[0], p
        return min(sols, key=lambda s: sum((s[k] - self.sol[k]) ** 2 for k in ARM)), p

    # ---------- state ----------
    def _body(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_of(self, name):
        b = self._body(name)
        return next(i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == b)

    def set_brick(self, x, y, yaw, name=TARGET, z=None):
        q = self.fj[name]
        half_z = self.m.geom_size[self._geom_of(name)][2]
        self.d.qpos[q:q + 3] = [x, y, half_z if z is None else z]
        self.d.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        dof = self.m.jnt_dofadr[self.m.body_jntadr[self._body(name)]]
        self.d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def park_distractors(self):
        for k, (x, y) in zip(DISTRACTORS, [(-0.3, 0.3), (-0.3, -0.3)]):
            self.set_brick(x, y, 0.0, name=k)

    def brick_pose(self, name=TARGET):
        q = self.fj[name]
        w, _, _, z = self.d.qpos[q + 3:q + 7]
        return (*self.d.qpos[q:q + 3], 2 * math.atan2(z, w))

    def brick_half_size(self, name):
        return self.m.geom_size[self._geom_of(name)].copy()

    def reset_pose(self, p, yaw):
        self.p, self.yaw, self.grip, self.sol = self.clamp_p(p), float(yaw), JAW_OPEN, None
        sol, _ = self._solve(self.p, self.yaw)
        if sol is None:
            raise ValueError(f"reset pose unreachable: {p}")
        self.sol = sol
        for n in ARM:
            self.d.qpos[self.jm[n]] = sol[n]
            self.d.qvel[self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = 0
        self.d.qpos[self.jm["gripper_joint"]] = JAW_OPEN
        self._apply()
        mujoco.mj_forward(self.m, self.d)

    def _apply(self):
        for i, n in zip(self.aid, ARM):
            self.d.ctrl[i] = self.sol[n]
        self.d.ctrl[self.gid] = self.grip

    def settle(self, seconds, frame_cb=None):
        n = int(round(seconds / self.m.opt.timestep))
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.sim_time += self.m.opt.timestep
            if frame_cb is not None and self.sim_time >= self._next_frame:
                frame_cb(self)
                self._next_frame += FRAME_DT

    def tcp_actual(self):
        return np.array(forward_kinematics([self.d.qpos[self.jm[n]] for n in ARM]))

    # ---------- motion ----------
    def move(self, p_target, yaw_target, duration, frame_cb=None, substep=0.02):
        """Smooth cartesian move of P; stops at the last reachable waypoint."""
        p1, y1 = self.clamp_p(p_target), float(yaw_target)
        reachable = bool(np.allclose(p1, p_target, atol=1e-9))
        n = max(1, int(round(duration / substep)))
        # A wrist flip near the bricks would sweep the open jaws through them: raise first, then move.
        path = [self.p + (p1 - self.p) * _smooth(i / n) for i in range(1, n + 1)]
        yaws = [self.yaw + (y1 - self.yaw) * _smooth(i / n) for i in range(1, n + 1)]
        if any(q[2] < FLIP_MIN_Z and self._flips(q, self.branch_yaw(q, y)) for q, y in zip(path, yaws)):
            self._raise_to(FLIP_MIN_Z, frame_cb, substep)
            p1 = np.array([p1[0], p1[1], max(p1[2], FLIP_MIN_Z)])
        p0, y0 = self.p.copy(), self.yaw
        for i in range(1, n + 1):
            s = _smooth(i / n)
            p, yaw = p0 + (p1 - p0) * s, y0 + (y1 - y0) * s
            sol, p_used = self._solve(p, yaw)
            if sol is None:
                reachable = False
            else:
                self.p, self.yaw, self.sol = p_used, yaw, sol
                self._apply()
            self.settle(substep, frame_cb)
        # Yaw is only meaningful mod π; keep it bounded (the branch choice follows the current roll).
        self.yaw = (self.yaw + math.pi / 2) % math.pi - math.pi / 2
        return reachable

    def _raise_to(self, z, frame_cb, substep):
        """Vertical lift at the current yaw (no branch change), used before a wrist flip."""
        p0 = self.p.copy()
        n = max(1, int(round(0.15 / substep)))
        for i in range(1, n + 1):
            p = np.array([p0[0], p0[1], p0[2] + (z - p0[2]) * _smooth(i / n)])
            sol, _ = self._solve(p, self.yaw)
            if sol is not None:
                self.p, self.sol = p, sol
                self._apply()
            self.settle(substep, frame_cb)

    def set_jaw(self, target, duration, frame_cb=None, substep=0.02):
        g0 = self.grip
        n = max(1, int(round(duration / substep)))
        for i in range(1, n + 1):
            self.grip = g0 + (target - g0) * _smooth(i / n)
            self._apply()
            self.settle(substep, frame_cb)

    def holding(self):
        x, y, z, _ = self.brick_pose()
        return bool(z > 0.03 and math.hypot(x - self.p[0], y - self.p[1]) < 0.025)

    def grasp(self, frame_cb=None):
        """Open, descend, close, lift 5 cm (~1.75 s). Returns True if the brick is held (physics)."""
        hover = self.p.copy()
        self.set_jaw(JAW_OPEN, 0.1, frame_cb)
        self.move(np.array([*self.p[:2], Z_GRASP]), self.yaw, 0.5, frame_cb)
        self.settle(0.1, frame_cb)
        self.set_jaw(JAW_CLOSE, 0.3, frame_cb)
        self.settle(0.15, frame_cb)
        self.move(np.array([*self.p[:2], Z_GRASP + 0.05]), self.yaw, 0.5, frame_cb)
        self.settle(0.1, frame_cb)
        held = self.holding()
        if not held:
            self.set_jaw(JAW_OPEN, 0.2, frame_cb)
            self.move(np.array([*self.p[:2], max(hover[2], Z_HOVER)]), self.yaw, 0.4, frame_cb)
        return held

    def place(self, goal_xy, frame_cb=None):
        """Carry to goal, lower, release, retreat (~2.2 s). True if the brick rests within 20 mm of the goal."""
        self.move(np.array([*self.p[:2], Z_CARRY]), self.yaw, 0.3, frame_cb)
        dist = float(np.linalg.norm(np.asarray(goal_xy) - self.p[:2]))
        self.move(np.array([*goal_xy, Z_CARRY]), self.yaw, max(0.4, dist / 0.3), frame_cb)
        self.move(np.array([*goal_xy, Z_GRASP + 0.004]), self.yaw, 0.45, frame_cb)
        self.set_jaw(JAW_OPEN, 0.2, frame_cb)
        self.move(np.array([*goal_xy, Z_HOVER]), self.yaw, 0.35, frame_cb)
        self.settle(0.2, frame_cb)
        x, y, z, _ = self.brick_pose()
        return bool(z < 0.01 and math.hypot(x - goal_xy[0], y - goal_xy[1]) < 0.02)
