#!/usr/bin/env python3
"""Watch the LegoPickEnv live with a random policy."""

import os
import sys
import time

sys.path.insert(0, '..')

from so_arm101_control.lego_pick_env import LegoPickEnv

env = LegoPickEnv(render_mode="human")
obs, _ = env.reset(seed=42)
print("Viewer open — watching random policy for 1000 steps...")
print("Obs is 12D: [6 joints | 3 wrist noisy | 3 overhead noisy]")
print("Close the MuJoCo window to exit early.")

try:
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Stop if viewer window was closed
        if env._viewer is not None and not env._viewer.is_running():
            print("Viewer closed.")
            break

        time.sleep(0.03)
        if terminated or truncated:
            print(f"  Episode done at step {i}, resetting...")
            obs, _ = env.reset()
finally:
    print("Done.")
    os._exit(0)  # hard exit — avoids MuJoCo viewer thread segfault on cleanup
