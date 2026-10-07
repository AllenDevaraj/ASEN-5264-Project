"""Render recorded episodes (pomdp_pick.footage.record) to MP4 with MuJoCo + ffmpeg.

Every video frame is a logged physics state. Overlays come from the same log:
- particle cloud (cyan) and belief mean (white ring) for belief agents;
- the latest raw camera reading (ghost brick) for memoryless agents;
- the goal (green).
A wrist-camera inset shows what the robot's camera sees, with a visibility light.
"""
import argparse
import math
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pomdp_pick.footage.record import FPS  # noqa: E402
from pomdp_pick.robot.model import TARGET, build_model  # noqa: E402

CYAN, GHOST, GOAL = (0.36, 0.88, 0.90), (0.96, 0.95, 0.92), (0.30, 0.95, 0.35)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WRIST_VFOV = math.degrees(2 * math.atan(360 / 537.0))   # sensing intrinsics: 1280x720, fx = 537


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


class Stage:
    def __init__(self, w, h):
        self.m, self.d = build_model()
        m = self.m
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 1920), max(h, 1080)
        m.vis.quality.offsamples = 8
        m.vis.quality.shadowsize = 4096
        mat = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n)  # noqa: E731
        m.mat_rgba[mat("ground_mat")] = (0.15, 0.16, 0.19, 1)
        m.mat_reflectance[mat("ground_mat")] = 0.0
        m.mat_rgba[mat("red_lego")] = (0.93, 0.16, 0.14, 1)
        m.mat_rgba[mat("blue_lego")] = (0.16, 0.30, 0.92, 1)
        m.vis.headlight.ambient[:] = (0.28, 0.28, 0.30)
        m.vis.headlight.diffuse[:] = (0.45, 0.45, 0.45)
        m.vis.rgba.haze[:] = (0.043, 0.051, 0.071, 1)
        m.cam_mode[0] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        m.cam_fovy[0] = WRIST_VFOV
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_cam")
        self.tgt_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TARGET)
        self.w, self.h = w, h
        self.r = mujoco.Renderer(m, h, w, max_geom=4000)
        iw, ih = (w // 4) & ~1, (h // 4) & ~1
        self.r_in = mujoco.Renderer(m, ih, iw, max_geom=4000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.wcam = mujoco.MjvCamera()
        self.wcam.type, self.wcam.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, 0
        self.opt = mujoco.MjvOption()
        self.font, self.font_b = _font(FONT, max(12, h // 36)), _font(FONT_B, max(14, h // 26))

    def set_qpos(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        # wrist camera = camera_link: optical axis +x, image right -y, image up +z (MuJoCo looks down -z)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        Rc = np.column_stack([-R[:, 1], R[:, 2], -R[:, 0]])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rc.reshape(-1))
        self.m.cam_pos[0], self.m.cam_quat[0] = self.d.site_xpos[self.site], q
        mujoco.mj_camlight(self.m, self.d)

    def _add(self, scn, gtype, size, pos, rgba, emission=0.0, mat=None):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                            (np.eye(3) if mat is None else mat).flatten(), np.asarray(rgba, np.float32))
        g.emission = emission
        scn.ngeom += 1

    def _overlays(self, scn, f):
        gx, gy = f["goal"]
        self._add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.024, 0.0004, 0), (gx, gy, 0.0006), (*GOAL, 0.30), 0.6)
        self._add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.0035, 0.0006, 0), (gx, gy, 0.0008), (*GOAL, 0.9), 0.8)
        if f["mode"] == "belief" and not f["lifted"]:
            for x, y in f["particles"]:
                self._add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.0017, 0, 0), (x, y, 0.0128), (*CYAN, 0.75), 0.9)
            mx, my, _ = f["mu"]
            self._add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, (0.0055, 0.0004, 0), (mx, my, 0.0135),
                      (1, 1, 1, 0.95), 1.0)
        if f["mode"] == "plain" and not f["lifted"] and np.all(np.isfinite(f["reading"])):
            x, y, th = f["reading"]
            c, s = math.cos(th), math.sin(th)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self._add(scn, mujoco.mjtGeom.mjGEOM_BOX, (0.016, 0.008, 0.0055), (x, y, 0.0056), (*GHOST, 0.42), 0.7, rot)

    def frame(self, f):
        self.set_qpos(f["qpos"])
        self.r.update_scene(self.d, self.cam, self.opt)
        self._overlays(self.r.scene, f)
        img = Image.fromarray(self.r.render())
        if f.get("inset", True):
            self.r_in.update_scene(self.d, self.wcam, self.opt)
            ins = Image.fromarray(self.r_in.render())
            pad = self.h // 40
            x0 = self.w - ins.width - pad
            dr = ImageDraw.Draw(img)
            dr.rectangle([x0 - 2, pad - 2, x0 + ins.width + 1, pad + ins.height + 1], outline=(230, 230, 230))
            img.paste(ins, (x0, pad))
            col = (90, 230, 120) if f["visible"] else (235, 80, 70)
            r = max(4, self.h // 120)
            cy = pad + ins.height + 3 * r
            dr.ellipse([x0, cy - r, x0 + 2 * r, cy + r], fill=col)
            dr.text((x0 + 3 * r, cy - self.font.size // 2 - 1), "wrist cam: " + ("target visible" if f["visible"]
                    else "target hidden"), font=self.font, fill=(225, 225, 225))
        self._hud(img, f)
        return np.asarray(img)

    def _hud(self, img, f):
        dr = ImageDraw.Draw(img)
        pad = self.h // 40
        dr.text((pad, pad), f["title"], font=self.font_b, fill=(245, 245, 245))
        dr.text((pad, pad + int(self.font_b.size * 1.4)), f["subtitle"], font=self.font, fill=(190, 195, 205))
        dr.text((pad, self.h - pad - self.font_b.size), f["status"], font=self.font_b, fill=f["status_col"])


def _status(rec, i, n_att):
    if rec["lifted"][i]:
        return "GRASPED · carrying to goal", (120, 220, 255)
    if rec["success"] and i > rec["last_lift"]:
        return "PLACED ON GOAL", (90, 230, 120)
    if rec["commit"][i]:
        return f"GRASP ATTEMPT {rec['attempts'][i]}/{n_att}", (255, 196, 64)
    return f"searching · attempts used {rec['attempts'][i]}/{n_att}", (225, 225, 225)


class FollowCam:
    """Three-quarter view whose look-at point eases toward the gripper, so grasps fill the frame."""

    def __init__(self, cam, centre=(0.17, 0.0, 0.02), dist=0.36, az=135.0, el=-24.0, ease=0.06):
        self.cam, self.look, self.ease = cam, np.array(centre, float), ease
        cam.distance, cam.azimuth, cam.elevation = dist, az, el

    def update(self, p):
        tgt = np.array([p[0], p[1], 0.02])
        self.look += self.ease * (tgt - self.look)
        self.cam.lookat[:] = self.look


def render_episode(npz, out_mp4, title, subtitle="", w=1280, h=720, inset=True, hold_end_s=1.0, n_attempts=3):
    rec = dict(np.load(npz, allow_pickle=False))
    mode = str(rec["obs_mode"])
    st = Stage(w, h)
    adr = st.m.jnt_qposadr[st.m.body_jntadr[st.tgt_body]]
    lifted = rec["qpos"][:, adr + 2] > 0.02
    rec["lifted"] = lifted
    rec["last_lift"] = int(np.nonzero(lifted)[0][-1]) if lifted.any() else len(lifted)
    cam = FollowCam(st.cam)
    cam.look[:] = [rec["p"][0][0], rec["p"][0][1], 0.02]
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2", str(out_mp4)],
                          stdin=subprocess.PIPE)
    n = len(rec["t"])
    for i in range(n):
        cam.update(rec["p"][i])
        status, col = _status(rec, i, n_attempts)
        f = dict(qpos=rec["qpos"][i], particles=rec["particles"][i], mu=rec["mu"][i], reading=rec["reading"][i],
                 visible=bool(rec["visible"][i]), goal=rec["goal"], mode=mode, lifted=bool(lifted[i]),
                 title=title, subtitle=subtitle or f"camera noise σ = {1000 * float(rec['sigma_ep']):.0f} mm · "
                 f"t = {rec['t'][i]:4.1f} s", status=status, status_col=col, inset=inset)
        ff.stdin.write(st.frame(f).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    if hold_end_s > 0:
        _hold_last(out_mp4, hold_end_s)
    return out_mp4


def _hold_last(mp4, seconds):
    tmp = mp4.with_suffix(".hold.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", f"tpad=stop_mode=clone:stop_duration={seconds}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
                    str(tmp)], check=True)
    os.replace(tmp, mp4)


def side_by_side(left, right, out):
    """Two clips side by side; the shorter one freezes on its last frame."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right), "-filter_complex",
                    "[0:v]tpad=stop_mode=clone:stop_duration=60[a];[1:v]tpad=stop_mode=clone:stop_duration=60[b];"
                    "[a][b]hstack=inputs=2:shortest=1[v]", "-map", "[v]", "-t", str(_dur(left, right)),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
                    str(out)], check=True)
    return out


def _dur(*mp4s):
    ds = []
    for p in mp4s:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True)
        ds.append(float(r.stdout.strip()))
    return max(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--no-inset", action="store_true")
    a = ap.parse_args()
    render_episode(a.npz, a.out, a.title, a.subtitle, a.w, a.h, inset=not a.no_inset)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
