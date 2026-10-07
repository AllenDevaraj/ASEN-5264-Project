"""Method registry: which observation each method sees and how its policy is evaluated."""
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    obs_mode: str
    algo: str = "ppo"      # "ppo" | "recurrent"
    n_stack: int = 1


METHODS = {
    "oracle": MethodSpec("oracle"),
    "plain": MethodSpec("plain"),
    "framestack": MethodSpec("plain", n_stack=4),
    "recurrent": MethodSpec("plain", algo="recurrent"),
    "belief": MethodSpec("belief"),
}


class Policy:
    """Deterministic evaluation wrapper: (stack) -> normalise -> predict (with LSTM state)."""

    def __init__(self, run_dir):
        import pickle
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "progress.json").read_text())
        self.spec = METHODS[meta["method"]]
        algo = RecurrentPPO if self.spec.algo == "recurrent" else PPO
        self.model = algo.load(run_dir / "model.zip", device="cpu")
        with open(run_dir / "vecnorm.pkl", "rb") as fh:
            vn = pickle.load(fh)
        self.mean, self.var = vn.obs_rms.mean, vn.obs_rms.var
        self.clip, self.eps = vn.clip_obs, vn.epsilon
        self.reset()

    def reset(self):
        self.stack = collections.deque(maxlen=self.spec.n_stack)
        self.state, self.start = None, np.ones((1,), bool)

    def _norm(self, obs):
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)

    def act(self, obs):
        o = np.asarray(obs, np.float32)
        if self.spec.n_stack > 1:
            if not self.stack:
                self.stack.extend([np.zeros_like(o)] * (self.spec.n_stack - 1))
            self.stack.append(o)
            o = np.concatenate(self.stack)
        o = self._norm(o)
        if self.spec.algo == "recurrent":
            a, self.state = self.model.predict(o[None], state=self.state, episode_start=self.start,
                                               deterministic=True)
            self.start = np.zeros((1,), bool)
            return np.clip(a[0], -1, 1)
        a, _ = self.model.predict(o[None], deterministic=True)
        return np.clip(a[0], -1, 1)


def load_policy(run_dir):
    return Policy(run_dir)
