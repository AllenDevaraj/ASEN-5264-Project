from gymnasium.envs.registration import register

register(id="PickPlace-v2", entry_point="pomdp_pick.envs.pick_env:PickPlaceEnv")
