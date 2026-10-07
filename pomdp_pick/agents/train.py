"""Train one (method, seed) run with resumable checkpoints. Laptop-safe: ≤4 workers, 1 torch thread."""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from pomdp_pick.agents.methods import METHODS

DEFAULT_HP = dict(n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4, ent_coef=0.01,
                  gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5)


def env_seed_base(seed, timesteps):
    """Episode-seed stream for a run: disjoint across training seeds (10M apart) and advanced on
    resume so a restarted run does not replay the episodes it has already trained on."""
    return int(seed) * 10_000_000 + int(timesteps)


def _env_fn(obs_mode, rank, monitor_dir, gamma):
    def f():
        torch.set_num_threads(1)
        from stable_baselines3.common.monitor import Monitor
        from pomdp_pick.envs.pick_env import PickPlaceEnv
        env = PickPlaceEnv(obs_mode, gamma=gamma)
        return Monitor(env, os.path.join(monitor_dir, f"w{rank}"), info_keywords=("success", "attempts"),
                       override_existing=False)
    return f


def _atomic(save_fn, path):
    tmp = str(path) + ".tmp"
    save_fn(tmp)
    os.replace(tmp if os.path.exists(tmp) else tmp + ".zip", path)


def _linear(lr):
    return lambda frac: lr * frac


def train(method, seed, total_steps, out_root="runs", n_envs=4, ckpt_every=100_000, hparams=None):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

    torch.set_num_threads(1)
    n_envs = min(n_envs, 4)
    spec = METHODS[method]
    run = Path(out_root) / method / f"seed{seed}"
    (run / "monitor").mkdir(parents=True, exist_ok=True)
    prog_path = run / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"timesteps": 0, "done": False}
    hp = {**DEFAULT_HP, **(hparams or {})}

    venv = SubprocVecEnv([_env_fn(spec.obs_mode, i, str(run / "monitor"), hp["gamma"]) for i in range(n_envs)],
                         start_method="forkserver")
    if spec.n_stack > 1:
        venv = VecFrameStack(venv, spec.n_stack)
    resuming = (run / "model.zip").exists() and prog["timesteps"] > 0
    if resuming:
        venv = VecNormalize.load(str(run / "vecnorm.pkl"), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=hp["gamma"])
    algo = RecurrentPPO if spec.algo == "recurrent" else PPO
    policy = "MlpLstmPolicy" if spec.algo == "recurrent" else "MlpPolicy"
    kw = dict(hp, learning_rate=_linear(hp["learning_rate"]))
    pk = dict(net_arch=[256, 256])
    if spec.algo == "recurrent":
        pk.update(lstm_hidden_size=128, n_lstm_layers=1)
    if resuming:
        model = algo.load(run / "model.zip", env=venv, device="cpu")
    else:
        model = algo(policy, venv, seed=seed, device="cpu", verbose=0, policy_kwargs=pk,
                     tensorboard_log=str(run / "tb"), **kw)

    # SB3 seeds envs with seed+rank, which overlaps across runs and replays on resume; override.
    venv.seed(env_seed_base(seed, model.num_timesteps))

    def save(ts, done):
        # model and stats first, progress.json last: a kill mid-save never claims unsaved progress
        _atomic(lambda p: model.save(p), run / "model.zip")
        _atomic(lambda p: venv.save(p), run / "vecnorm.pkl")
        _atomic(lambda p: open(p, "w").write(json.dumps({"method": method, "seed": seed, "timesteps": int(ts),
                                                          "done": done, "target": int(total_steps),
                                                          "hparams": hp})), prog_path)

    class Ckpt(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = model.num_timesteps + ckpt_every

        def _on_step(self):
            if self.num_timesteps >= self.next:
                save(self.num_timesteps, False)
                self.next += ckpt_every
            return True

    remaining = max(0, total_steps - model.num_timesteps)
    if remaining > 0:
        model.learn(remaining, callback=Ckpt(), reset_num_timesteps=not resuming, tb_log_name="ppo")
    save(model.num_timesteps, True)
    venv.close()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)) if os.path.exists(a.config) else {}
    steps = a.steps or cfg.get("total_steps", 1_000_000)
    train(a.method, a.seed, steps, a.out, n_envs=cfg.get("n_envs", 4), ckpt_every=cfg.get("ckpt_every", 100_000),
          hparams=cfg.get("hparams"))


if __name__ == "__main__":
    main()
