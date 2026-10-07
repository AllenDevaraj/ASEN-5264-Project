import json, numpy as np, pytest
from pomdp_pick.agents.methods import METHODS, load_policy
from pomdp_pick.agents.train import train
from pomdp_pick.envs.pick_env import PickPlaceEnv

@pytest.mark.parametrize("method", list(METHODS))
def test_tiny_training_and_policy_roundtrip(tmp_path, method):
    run = train(method, seed=0, total_steps=512, out_root=tmp_path, n_envs=2, ckpt_every=256,
                hparams={"n_steps": 128, "batch_size": 64, "n_epochs": 1})
    prog = json.loads((run / "progress.json").read_text())
    assert prog["done"] and prog["timesteps"] >= 512
    pol = load_policy(run)
    env = PickPlaceEnv(METHODS[method].obs_mode)
    obs, _ = env.reset(seed=1); pol.reset()
    a = pol.act(obs)
    assert a.shape == (5,) and np.all(np.abs(a) <= 1.0)

def test_resume_continues_timesteps(tmp_path):
    run = train("plain", 0, 256, tmp_path, n_envs=2, ckpt_every=128, hparams={"n_steps": 64, "batch_size": 64, "n_epochs": 1})
    run = train("plain", 0, 512, tmp_path, n_envs=2, ckpt_every=128, hparams={"n_steps": 64, "batch_size": 64, "n_epochs": 1})
    prog = json.loads((run / "progress.json").read_text())
    assert 512 <= prog["timesteps"] < 512 + 256
