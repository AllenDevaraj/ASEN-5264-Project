"""PickPlaceEnv: SO-ARM101 pick-and-place POMDP with physical grasping.

The policy moves the jaw-centre point P and the gripper yaw, and decides when to commit.
Committing runs a physical grasp; a held brick is carried to the known goal by a scripted
physical place. Every observation mode shares the same dynamics, reward and seeds.
"""
import math

import gymnasium as gym
import mujoco
import numpy as np

from pomdp_pick.belief.particle_filter import BeliefFilter
from pomdp_pick.robot.controller import Arm
from pomdp_pick.robot.model import DISTRACTORS, TARGET, build_model
from pomdp_pick.sensing.camera import WristCamera, wrap_pi

STEP_S = 0.1
GAMMA = 0.99
HOME = np.array([0.18, 0.0, 0.08])
DIMS = {"oracle": 12, "plain": 14, "belief": 17}
MAX_AGE = 50


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_mode="belief", sigma_range=(0.002, 0.020), sigma_fixed=None, drift=0.0005,
                 max_time=20.0, max_attempts=3, frame_cb=None, gamma=GAMMA):
        if obs_mode not in DIMS:
            raise ValueError(f"obs_mode must be one of {list(DIMS)}")
        self.obs_mode, self.sigma_range, self.sigma_fixed = obs_mode, sigma_range, sigma_fixed
        self.drift, self.max_time, self.max_attempts, self.frame_cb = drift, max_time, max_attempts, frame_cb
        self.gamma = gamma
        self.m, self.d = build_model()
        self.arm = Arm(self.m, self.d)
        self.cam = WristCamera(self.m)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (DIMS[obs_mode],), np.float32)
        self.pf = BeliefFilter(drift=drift)

    # ---------------- episode setup ----------------
    def _spawn(self, rng):
        """Target, two distractors, goal. Bricks are separated by the sum of their footprint
        half-diagonals (+3 mm) so no spawn interpenetrates, whatever the yaws."""
        hd = [float(np.hypot(*self.arm.brick_half_size(n)[:2])) for n in (TARGET, *DISTRACTORS)]
        pts, radii = [], []
        while len(pts) < 4:
            r, a = rng.uniform(0.13, 0.23), rng.uniform(-1.0, 1.0)
            q = np.array([r * math.cos(a), r * math.sin(a)])
            if len(pts) < 3:
                ok = all(np.linalg.norm(q - p) >= hd[len(pts)] + rp + 0.003 for p, rp in zip(pts, radii))
            else:
                ok = all(np.linalg.norm(q - p) >= max(0.08, hd[0] + rp + 0.01) for p, rp in zip(pts, radii))
            if ok:
                pts.append(q)
                radii.append(hd[len(pts) - 1] if len(pts) <= 3 else 0.0)
        return pts

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        mujoco.mj_resetData(self.m, self.d)
        self.arm = Arm(self.m, self.d)
        tgt, d1, d2, self.goal = self._spawn(rng)
        self.arm.set_brick(*tgt, rng.uniform(-math.pi / 2, math.pi / 2))
        for name, p in zip(DISTRACTORS, (d1, d2)):
            self.arm.set_brick(*p, rng.uniform(-math.pi / 2, math.pi / 2), name=name)
        self.arm.reset_pose(HOME, 0.0)
        self.arm.settle(0.3, self.frame_cb)
        lo, hi = np.log(self.sigma_range[0]), np.log(self.sigma_range[1])
        self.sigma_ep = self.sigma_fixed if self.sigma_fixed is not None else float(np.exp(rng.uniform(lo, hi)))
        self.pf = BeliefFilter(drift=self.drift, rng=np.random.default_rng(int(rng.integers(2**31))))
        self.sense_rng = np.random.default_rng(int(rng.integers(2**31)))
        self.t0 = self.arm.sim_time
        self.attempts, self.success, self.held = 0, False, False
        self.last_reading, self.age = None, MAX_AGE
        self.commit_err, self.std_at_commit = [], []
        self._sense(first=True)
        self.phi = self._potential()
        return self._obs(), self._info()

    # ---------------- sensing ----------------
    def _target(self):
        x, y, _, yaw = self.arm.brick_pose()
        return (x, y, yaw, self.arm.brick_half_size(TARGET))

    def _occluders(self):
        out = []
        for name in DISTRACTORS:
            x, y, _, yaw = self.arm.brick_pose(name)
            out.append((x, y, yaw, self.arm.brick_half_size(name)))
        return out

    def _sense(self, first=False, extra_xy=0.0):
        tx, ty, tyaw, _ = self._target()
        self.visible = bool(self.cam.visible(self.d, self._target(), self._occluders()))
        z = None
        if self.visible:
            dist = self.cam.distance(self.d, (tx, ty))   # true distance drives the sensor noise only
            z = self.cam.reading(self.sense_rng, np.array([tx, ty, wrap_pi(tyaw)]), dist, self.sigma_ep)
            self.last_reading, self.age = z, 0
        else:
            self.age = min(self.age + 1, MAX_AGE)
        if first:
            if z is not None:
                self.pf.reset_from_reading(z, self._reading_distance(z))
            else:
                self.pf.reset_from_prior((0.0, 0.0), (0.13, 0.23), (-1.0, 1.0))
        else:
            self.pf.predict(extra_xy)
            if z is not None:
                self.pf.update(z, self._reading_distance(z))

    def _reading_distance(self, z):
        """The filter only knows where it *saw* the brick, so its likelihood width uses that."""
        return self.cam.distance(self.d, z[:2])

    # ---------------- reward pieces ----------------
    def _potential(self):
        x, y, _, _ = self.arm.brick_pose()
        return -min(math.hypot(self.arm.p[0] - x, self.arm.p[1] - y), 0.3) / 0.05

    def _upright(self, name=TARGET, max_tilt_deg=20.0):
        q = self.arm.fj[name]
        _, qx, qy, _ = self.d.qpos[q + 3:q + 7]
        return bool(1.0 - 2.0 * (qx * qx + qy * qy) > math.cos(math.radians(max_tilt_deg)))

    def _placement_ok(self):
        x, y, z, _ = self.arm.brick_pose()
        resting = z < self.arm.brick_half_size(TARGET)[2] + 0.002
        return bool(resting and self._upright(TARGET) and math.hypot(x - self.goal[0], y - self.goal[1]) < 0.02)

    def _drift_target(self):
        """Random-walk the resting brick's position; orientation and velocity are left to physics."""
        _, _, z, _ = self.arm.brick_pose()
        if z < 0.01 and self.drift > 0:
            q = self.arm.fj[TARGET]
            self.d.qpos[q:q + 2] += self.np_random.normal(0.0, self.drift, 2)
            mujoco.mj_forward(self.m, self.d)

    # ---------------- step ----------------
    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        t_before = self.arm.sim_time
        reward, extra_xy = 0.0, 0.0
        if a[4] > 0.5:
            self.attempts += 1
            bx, by, _, _ = self.arm.brick_pose()
            self.commit_err.append(1000 * math.hypot(self.arm.p[0] - bx, self.arm.p[1] - by))
            self.std_at_commit.append(float(np.hypot(*self.pf.summary()["std"][:2]) * 1000))
            if self.arm.grasp(self.frame_cb):
                reward += 5.0
                self.held = True
                self.success = self.arm.place(self.goal, self.frame_cb) and self._placement_ok()
                reward += 10.0 if self.success else 0.0
            else:
                reward -= 1.0
                extra_xy = 0.01   # a miss may have shoved the brick; the filter is not told by how much
        else:
            p = self.arm.p + a[:3] * 0.015
            self.arm.move(p, self.arm.yaw + a[3] * 0.3, STEP_S, self.frame_cb, substep=0.02)
            self._drift_target()
        dt = self.arm.sim_time - t_before
        reward -= 0.01 * dt / STEP_S
        elapsed = self.arm.sim_time - self.t0
        terminated = self.held or self.attempts >= self.max_attempts
        truncated = (not terminated) and elapsed >= self.max_time
        # Potential-based shaping with Φ(terminal) = 0 keeps the optimal policy unchanged (Ng et al. 1999).
        phi = 0.0 if terminated else self._potential()
        reward += self.gamma * phi - self.phi
        self.phi = phi
        if not self.held:
            self._sense(extra_xy=extra_xy)
        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info()

    # ---------------- observation ----------------
    def _common(self):
        p = self.arm.p
        return [p[0] * 10, p[1] * 10, p[2] * 10, math.sin(2 * self.arm.yaw), math.cos(2 * self.arm.yaw),
                self.arm.grip, self.attempts / self.max_attempts,
                1.0 - min((self.arm.sim_time - self.t0) / self.max_time, 1.0)]

    def _rel(self, x, y, th):
        return [(x - self.arm.p[0]) * 10, (y - self.arm.p[1]) * 10, math.sin(2 * th), math.cos(2 * th)]

    def _obs(self):
        o = self._common()
        if self.obs_mode == "oracle":
            x, y, _, yaw = self.arm.brick_pose()
            o += self._rel(x, y, yaw)
        elif self.obs_mode == "plain":
            if self.last_reading is None:
                o += [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                o += self._rel(*self.last_reading) + [float(self.visible), self.age / MAX_AGE]
        else:
            s = self.pf.summary()
            o += self._rel(*s["mu"]) + [s["std"][0] * 100, s["std"][1] * 100, s["std"][2],
                                        s["sigma_ep"] * 100, float(self.visible)]
        return np.asarray(o, np.float32)

    def _info(self):
        return {"success": bool(self.success), "attempts": self.attempts, "sigma_ep": self.sigma_ep,
                "sim_time": self.arm.sim_time - self.t0, "commit_err_mm": list(self.commit_err),
                "belief_std_at_commit": list(self.std_at_commit), "visible": bool(self.visible)}
