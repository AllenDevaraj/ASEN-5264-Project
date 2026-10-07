"""Wrist camera model: visibility (frustum + block occlusion) and distance-dependent noisy readings.

Unlike v1, the camera is NOT aimed at the true target: its pose is the real camera_link frame on
the gripper, so seeing the brick depends on where the policy moves the arm.
"""
import mujoco
import numpy as np

from pomdp_pick.sensing import occlusion as occ

IMG_W, IMG_H = 1280, 720
MIN_DEPTH = 0.02
YAW_RAD_PER_MM = 0.02


def sigma_at(dist, sigma_ep):
    return float(np.clip(sigma_ep * dist / 0.10, 0.3 * sigma_ep, 3.0 * sigma_ep))


def yaw_sigma(sigma_m):
    return YAW_RAD_PER_MM * sigma_m * 1000.0


def wrap_pi(a):
    """Wrap an angle modulo π into [-π/2, π/2) (the brick looks the same rotated 180°)."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


class WristCamera:
    def __init__(self, m):
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_cam")
        if self.sid < 0:
            raise ValueError("model has no wrist_cam site")

    def pose(self, d):
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def visible(self, d, target, occluders):
        pos, R = self.pose(d)
        x, y, yaw, hs = target
        p = occ._project_to_camera(np.array([x, y, 2 * hs[2]]), pos, R)
        if p is None or p[2] < MIN_DEPTH or not (0 <= p[0] <= IMG_W and 0 <= p[1] <= IMG_H):
            return False
        for ox, oy, oyaw, ohs in occluders:
            if occ.is_occluded((x, y), (hs[0], hs[1]), yaw, (ox, oy), (ohs[0], ohs[1]), oyaw,
                               pos, R, target_half_z=hs[2], occluder_half_z=ohs[2]):
                return False
        return True

    def distance(self, d, xy):
        pos, _ = self.pose(d)
        return float(np.linalg.norm(pos - np.array([xy[0], xy[1], 0.0])))

    @staticmethod
    def reading(rng, true_xyth, dist, sigma_ep):
        s = sigma_at(dist, sigma_ep)
        out = np.array(true_xyth, float) + rng.normal(0.0, [s, s, yaw_sigma(s)])
        out[2] = wrap_pi(out[2])
        return out
