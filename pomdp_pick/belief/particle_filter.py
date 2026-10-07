"""Seeded SIR particle filter over the target brick pose and the unknown camera noise level.

State per particle: (x, y, θ mod π, log σ_ep). Estimating σ_ep jointly lets the likelihood
width adapt to the real noise level (v1 used a fixed 1.5× widening and collapsed at 30 mm).
"""
import numpy as np

from pomdp_pick.sensing.camera import wrap_pi, yaw_sigma


class BeliefFilter:
    def __init__(self, n=300, drift=0.0005, sigma_range=(0.0015, 0.045), rng=None, reinject=0.05):
        self.n, self.drift, self.reinject = n, drift, reinject
        self.ls_lo, self.ls_hi = np.log(sigma_range[0]), np.log(sigma_range[1])
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.particles = np.zeros((n, 4))
        self.weights = np.full(n, 1.0 / n)

    def _sigmas(self, dist):
        s_ep = np.exp(self.particles[:, 3])
        s = np.clip(s_ep * dist / 0.10, 0.3 * s_ep, 3.0 * s_ep)
        return s, yaw_sigma(s)

    def _sample_ls(self, k):
        return self.rng.uniform(self.ls_lo, self.ls_hi, k)

    def reset_from_reading(self, z, dist):
        self.particles[:, 3] = self._sample_ls(self.n)
        s, st = self._sigmas(dist)
        self.particles[:, 0] = z[0] + self.rng.normal(0, s)
        self.particles[:, 1] = z[1] + self.rng.normal(0, s)
        self.particles[:, 2] = wrap_pi(z[2] + self.rng.normal(0, st))
        self.weights[:] = 1.0 / self.n

    def reset_from_prior(self, center_xy, r_range, ang_range):
        r = self.rng.uniform(*r_range, self.n)
        a = self.rng.uniform(*ang_range, self.n)
        self.particles[:, 0] = center_xy[0] + r * np.cos(a)
        self.particles[:, 1] = center_xy[1] + r * np.sin(a)
        self.particles[:, 2] = self.rng.uniform(-np.pi / 2, np.pi / 2, self.n)
        self.particles[:, 3] = self._sample_ls(self.n)
        self.weights[:] = 1.0 / self.n

    def predict(self, extra_xy=0.0):
        sxy = np.hypot(self.drift, extra_xy)
        self.particles[:, :2] += self.rng.normal(0, sxy, (self.n, 2))
        self.particles[:, 2] = wrap_pi(self.particles[:, 2] + self.rng.normal(0, 0.01 + 5 * extra_xy, self.n))
        self.particles[:, 3] = np.clip(self.particles[:, 3] + self.rng.normal(0, 0.02, self.n), self.ls_lo, self.ls_hi)

    def update(self, z, dist):
        s, st = self._sigmas(dist)
        dx, dy = self.particles[:, 0] - z[0], self.particles[:, 1] - z[1]
        dth = wrap_pi(self.particles[:, 2] - z[2])
        logw = (-0.5 * (dx ** 2 + dy ** 2) / s ** 2 - 2 * np.log(s)
                - 0.5 * dth ** 2 / st ** 2 - np.log(st))
        logw += np.log(self.weights + 1e-300)
        logw -= logw.max()
        w = np.exp(logw)
        self.weights = w / w.sum()
        if 1.0 / np.sum(self.weights ** 2) < self.n / 2:
            self._resample(z, dist)

    def _resample(self, z, dist):
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), pos), self.n - 1)
        self.particles = self.particles[idx].copy()
        k = int(self.reinject * self.n)
        if k:
            j = self.rng.choice(self.n, k, replace=False)
            self.particles[j, 3] = self._sample_ls(k)
            s, st = self._sigmas(dist)
            self.particles[j, 0] = z[0] + self.rng.normal(0, s[j])
            self.particles[j, 1] = z[1] + self.rng.normal(0, s[j])
            self.particles[j, 2] = wrap_pi(z[2] + self.rng.normal(0, st[j]))
        self.weights[:] = 1.0 / self.n

    def summary(self):
        w = self.weights
        mu_xy = w @ self.particles[:, :2]
        c, s = w @ np.cos(2 * self.particles[:, 2]), w @ np.sin(2 * self.particles[:, 2])
        mu_th = 0.5 * np.arctan2(s, c)
        std_xy = np.sqrt(w @ (self.particles[:, :2] - mu_xy) ** 2)
        r = min(np.hypot(c, s), 1.0)
        std_th = 0.5 * np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0))
        return {"mu": np.array([*mu_xy, mu_th]), "std": np.array([*std_xy, std_th]),
                "sigma_ep": float(w @ np.exp(self.particles[:, 3]))}
