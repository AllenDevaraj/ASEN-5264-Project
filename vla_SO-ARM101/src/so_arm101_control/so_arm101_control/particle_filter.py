#!/usr/bin/env python3
"""Particle filter for tracking block pose belief under observation uncertainty.

Used by the Belief-Augmented PPO agent to maintain a belief distribution
over block poses (x, y, theta) given noisy, intermittent observations.

Usage:
    pf = ParticleFilter(n_particles=300)
    pf.reset(initial_obs)
    pf.predict()
    pf.update({0: obs_array}, sigma_obs=0.01)
    mu, sigma = pf.get_belief()
"""

import numpy as np


class ParticleFilter:
    """Bootstrap particle filter for block pose tracking.

    State per particle per block: (x, y, theta).
    """

    def __init__(
        self,
        n_particles=300,
        n_blocks=1,
        process_noise_xy=0.001,
        process_noise_theta=0.01,
        injection_ratio=0.05,
    ):
        """
        Args:
            n_particles: Number of particles.
            n_blocks: Number of blocks to track.
            process_noise_xy: Std of random walk in xy (meters/step).
            process_noise_theta: Std of random walk in theta (rad/step).
            injection_ratio: Fraction of particles redrawn from latest obs.
        """
        self.n_particles = n_particles
        self.n_blocks = n_blocks
        self.process_noise_xy = process_noise_xy
        self.process_noise_theta = process_noise_theta
        self.injection_ratio = injection_ratio

        # particles: (n_particles, n_blocks, 3) = [x, y, theta]
        self.particles = np.zeros((n_particles, n_blocks, 3))
        self.weights = np.ones(n_particles) / n_particles
        self._rng = np.random.default_rng()
        self._last_obs = None

    def reset(self, initial_obs, sigma_init=0.03):
        """Initialize particles around a noisy initial observation.

        Args:
            initial_obs: (n_blocks, 3) array of [x, y, theta] (noisy).
            sigma_init: Initial spread (meters for xy, radians for theta).
        """
        initial_obs = np.asarray(initial_obs).reshape(self.n_blocks, 3)
        self._last_obs = initial_obs.copy()

        noise = self._rng.normal(0, sigma_init, (self.n_particles, self.n_blocks, 3))
        self.particles = initial_obs[np.newaxis, :, :] + noise
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self):
        """Propagate particles with process noise (stationary block assumption)."""
        noise_xy = self._rng.normal(
            0, self.process_noise_xy, (self.n_particles, self.n_blocks, 2)
        )
        noise_theta = self._rng.normal(
            0, self.process_noise_theta, (self.n_particles, self.n_blocks, 1)
        )
        self.particles[:, :, :2] += noise_xy
        self.particles[:, :, 2:] += noise_theta

    def update(self, observations, sigma_obs):
        """Update weights using observation likelihood.

        Args:
            observations: dict mapping block_idx -> (x, y, theta) observation.
                Only visible (non-occluded) blocks are included.
            sigma_obs: Observation noise std (episodic sigma).
        """
        if not observations:
            return

        for block_idx, obs in observations.items():
            obs = np.asarray(obs)
            self._last_obs = np.copy(self._last_obs) if self._last_obs is not None else np.zeros((self.n_blocks, 3))
            self._last_obs[block_idx] = obs

            diff = self.particles[:, block_idx, :] - obs
            # Wrap theta difference to [-pi, pi]
            diff[:, 2] = (diff[:, 2] + np.pi) % (2 * np.pi) - np.pi

            # Log-likelihood: sum of independent Gaussians
            log_lik = -0.5 * np.sum((diff / sigma_obs) ** 2, axis=1)
            # Multiply weights (in log space for stability)
            log_weights = np.log(self.weights + 1e-300) + log_lik
            log_weights -= np.max(log_weights)  # shift for numerical stability
            self.weights = np.exp(log_weights)

        # Normalize
        w_sum = self.weights.sum()
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

    def resample(self):
        """Systematic resampling when effective sample size drops below N/2."""
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff >= self.n_particles / 2:
            return

        # Systematic resampling
        n = self.n_particles
        positions = (self._rng.random() + np.arange(n)) / n
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # ensure exact

        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n

        # Injection: replace a fraction of particles with samples near latest obs
        if self._last_obs is not None:
            n_inject = max(1, int(self.injection_ratio * n))
            inject_idx = self._rng.choice(n, n_inject, replace=False)
            noise = self._rng.normal(0, 0.005, (n_inject, self.n_blocks, 3))
            self.particles[inject_idx] = self._last_obs[np.newaxis, :, :] + noise

    def get_belief(self):
        """Compute weighted mean and std for each block.

        Returns:
            mu: (n_blocks, 3) weighted mean [x, y, theta].
            sigma: (n_blocks, 3) weighted std [x, y, theta].
        """
        w = self.weights[:, np.newaxis, np.newaxis]  # (N, 1, 1)
        mu = np.sum(w * self.particles, axis=0)  # (n_blocks, 3)

        diff = self.particles - mu[np.newaxis, :, :]
        # Wrap theta for circular std
        diff[:, :, 2] = (diff[:, :, 2] + np.pi) % (2 * np.pi) - np.pi
        sigma = np.sqrt(np.sum(w * diff ** 2, axis=0) + 1e-8)  # (n_blocks, 3)

        return mu, sigma
