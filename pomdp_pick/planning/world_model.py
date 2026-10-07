"""Learned world model over the belief summary, for planning without physics.

State (13-D, the env's belief-mode obs units): P(3), sin/cos 2·yaw, μ_rel(2), sin/cos 2θ,
σx, σy, σθ, σ_ep. Input = state + 5-D action. Outputs: Δstate (MSE head) and the grasp logit
(BCE head, trained on commit transitions only, so it cannot shortcut on the commit flag).
"""
import json
from pathlib import Path

import numpy as np

# belief obs: [P*10 (0-2), sin2yaw, cos2yaw (3-4), grip, attempts, time (5-7), μ_rel (8-9),
#              sin2θ, cos2θ (10-11), σx, σy, σθ (12-14), σ_ep (15), visible (16)]
STATE_IDX = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15])
STATE_DIM = len(STATE_IDX)
SIGMA_DIMS = slice(9, 13)


def state_of(obs):
    return np.asarray(obs, np.float32)[STATE_IDX]


def auc(scores, labels):
    """Rank-sum (Mann-Whitney) AUC; ties get average ranks."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class WorldModel:
    def __init__(self, hidden=128):
        self.hidden = hidden
        self.params = None   # numpy weights for fast single-sample prediction

    @staticmethod
    def _commit_mask(X):
        return X[:, -1] > 0.5

    def fit(self, X, Y, epochs=30, val_frac=0.15, seed=0, batch=256, lr=1e-3):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
        idx = np.random.default_rng(seed).permutation(len(X))
        n_val = int(len(X) * val_frac)
        va, tr = idx[:n_val], idx[n_val:]
        self.x_mu, self.x_sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        self.y_mu, self.y_sd = Y[tr, :-1].mean(0), Y[tr, :-1].std(0) + 1e-6
        h = self.hidden
        trunk = nn.Sequential(nn.Linear(X.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                              nn.Linear(h, h), nn.ReLU())
        head_d, head_g = nn.Linear(h, Y.shape[1] - 1), nn.Linear(h, 1)
        opt = torch.optim.Adam([*trunk.parameters(), *head_d.parameters(), *head_g.parameters()], lr=lr)
        xt = torch.tensor((X - self.x_mu) / self.x_sd)
        dt = torch.tensor((Y[:, :-1] - self.y_mu) / self.y_sd)
        gt = torch.tensor(Y[:, -1])
        mt = torch.tensor(self._commit_mask(X))
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // batch)):
                z = trunk(xt[b])
                loss = nn.functional.mse_loss(head_d(z), dt[b])
                m = mt[b]
                if m.any():
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(head_g(z)[m, 0], gt[b][m])
                opt.zero_grad()
                loss.backward()
                opt.step()
        lin = [l for l in trunk if isinstance(l, nn.Linear)] + [head_d, head_g]
        self.params = [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in lin]
        d_pred, p_pred = self.predict(X[va])
        mse = ((d_pred - Y[va, :-1]) ** 2).mean(0)
        var = Y[va, :-1].var(0) + 1e-12
        cm = self._commit_mask(X[va])
        return {"val_mse_per_dim": mse.tolist(), "val_r2_per_dim": (1 - mse / var).tolist(),
                "grasp_auc": auc(p_pred[cm], Y[va, -1][cm] > 0.5), "n_train": int(len(tr)),
                "n_val": int(n_val), "n_val_commits": int(cm.sum())}

    def predict(self, x):
        """x: (18,) or (N, 18) -> (Δstate in obs units, p_grasp in [0, 1])."""
        x = np.atleast_2d(np.asarray(x, np.float32))
        z = (x - self.x_mu) / self.x_sd
        for W, b in self.params[:3]:
            z = np.maximum(z @ W + b, 0.0)
        d = (z @ self.params[3][0] + self.params[3][1]) * self.y_sd + self.y_mu
        logit = np.clip(z @ self.params[4][0] + self.params[4][1], -30, 30)[:, 0]
        p = 1.0 / (1.0 + np.exp(-logit))
        return (d[0], float(p[0])) if len(x) == 1 else (d, p)

    def step_state(self, s, action):
        """One imagined transition; off-distribution predictions cannot make σ negative."""
        d, p = self.predict(np.concatenate([s, action]).astype(np.float32))
        s2 = np.asarray(s, np.float32) + d
        s2[SIGMA_DIMS] = np.maximum(s2[SIGMA_DIMS], 0.0)
        return s2, float(np.clip(p, 0.0, 1.0))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz") if path.suffix != ".npz" else path,
                 hidden=self.hidden, x_mu=self.x_mu, x_sd=self.x_sd, y_mu=self.y_mu, y_sd=self.y_sd,
                 **{f"W{i}": W for i, (W, _) in enumerate(self.params)},
                 **{f"b{i}": b for i, (_, b) in enumerate(self.params)})

    @classmethod
    def load(cls, path):
        path = Path(path)
        f = np.load(path.with_suffix(".npz") if path.suffix != ".npz" else path)
        wm = cls(int(f["hidden"]))
        wm.x_mu, wm.x_sd, wm.y_mu, wm.y_sd = f["x_mu"], f["x_sd"], f["y_mu"], f["y_sd"]
        wm.params = [(f[f"W{i}"], f[f"b{i}"]) for i in range(5)]
        return wm


def write_report(rep, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rep, indent=2))
