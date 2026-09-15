import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mujco_env import G1Env

env = G1Env()
env.reset()
print("initial base_height:", round(env.base_height(), 3))
target = env.default_angles.copy()
for step in range(1200):
    env.data.ctrl[:] = env.pd_tau(target)
    env.step_physics()
    if step % 100 == 0:
        print(step, "t=", round(env.data.time, 2), "h=", round(env.base_height(), 3),
              "vx=", round(env.forward_velocity(), 3))
print("defaults(xml):", env.default_angles.round(3))
print("joints qpos now:", env.data.qpos[7:].round(3))