def test_imports():
    import mujoco, gymnasium, stable_baselines3, sb3_contrib, rliable, torch  # noqa: F401
    import pomdp_pick
    assert pomdp_pick.__version__ == "2.0.0"
    assert torch.get_num_threads() >= 1
