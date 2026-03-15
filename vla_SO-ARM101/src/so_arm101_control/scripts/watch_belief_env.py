#!/usr/bin/env python3
"""Watch the LegoPickEnv live with belief mode (particle filter) and a random policy.

Prints belief uncertainty (sigma) each step so you can see the particle filter
working — sigma grows when the red block is occluded, shrinks when visible.
"""

import os
import sys
import time

sys.path.insert(0, '..')

from so_arm101_control.lego_pick_env import LegoPickEnv

env = LegoPickEnv(belief_mode=True, render_mode="human")
obs, info = env.reset(seed=42)

print("Belief PPO env — random policy, 1000 steps")
print(f"  Episode sigma_ep (noise level): {info['sigma_ep']*1000:.1f}mm")
print(f"  Obs is 12D: [6 joints | mu_x mu_y mu_theta | sigma_x sigma_y sigma_theta]")
print()

try:
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if env._viewer is not None and not env._viewer.is_running():
            print("Viewer closed.")
            break

        # Print belief state every 20 steps
        if i % 20 == 0:
            mu = info.get("belief_mu")
            sigma = info.get("belief_sigma")
            overhead_occ = info.get("overhead_occluded", False)
            cam_status = "OCCLUDED" if (sigma is not None and sigma[0] > 0.015) else "visible"
            oh_status = "OH:occ" if overhead_occ else "OH:vis"
            if mu is not None and sigma is not None:
                print(f"  step {i:4d} | mu=({mu[0]:.3f}, {mu[1]:.3f}) "
                      f"| sigma_xy=({sigma[0]*1000:.1f}, {sigma[1]*1000:.1f})mm "
                      f"| reward={reward:+.1f} | {cam_status} | {oh_status}")

        time.sleep(0.03)

        if terminated or truncated:
            obs, info = env.reset()
            print(f"\n--- Episode reset | new sigma_ep={info['sigma_ep']*1000:.1f}mm ---\n")

finally:
    print("Done.")
    os._exit(0)  # hard exit — avoids MuJoCo viewer thread segfault on cleanup
