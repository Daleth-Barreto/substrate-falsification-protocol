import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mujco_env import G1Env

env = G1Env()
env.reset()
target = env.default_angles.copy()
env.data.ctrl[:] = env.pd_tau(target)
durations = []
for _ in range(3000):
    env.data.ctrl[:] = env.pd_tau(target)
    env.step_physics()
    if env.data.time > 10:
        break
print("sim_time_s:", round(env.data.time, 2))
print("base_height:", round(env.base_height(), 3))
print("fwd_vel:", round(env.forward_velocity(), 3), "fallen:", env.has_fallen())
print("max_qpos_joint_before_idx7:", env.data.qpos[:7])
print("nq:", env.model.nq, "nv:", env.model.nv, "njnt:", env.model.njnt)
print("policy_to_xml:", env.policy_to_xml)
print("xml_to_policy:", env.xml_to_policy)