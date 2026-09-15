import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mujco_env import G1Env, REPO_ROOT
from ann_checkpoint import ANNPolicy

results_dir = REPO_ROOT / "results"
results_dir.mkdir(exist_ok=True)

env = G1Env()
policy = ANNPolicy(REPO_ROOT / "exported" / "policy.onnx")
env.reset()

cmd = np.array([0.5, 0.0, 0.0])
action_xml = np.zeros(29)
target_dof_pos = env.default_angles.copy()
tau = env.pd_tau(target_dof_pos)

t0 = time.time()
steps = 0
ctrl = 0
heights = []
vels = []
duration = 15.0
while env.data.time < duration and not env.has_fallen():
    tau = env.pd_tau(target_dof_pos)
    env.data.ctrl[:] = tau
    env.step_physics()
    steps += 1
    ctrl += 1
    if ctrl % env.decimation == 0:
        obs = env.stacked_obs(cmd, action_xml)
        action_policy = policy.act(obs)
        action_xml = action_policy[env.policy_to_xml]
        target_dof_pos = action_xml * env.action_scale + env.default_angles
    if steps % 100 == 0:
        heights.append(env.base_height())
        vels.append(env.forward_velocity())

wall = time.time() - t0
print("backend:", policy.backend)
print("sim_time_s:", round(env.data.time, 2), "wall_s:", round(wall, 2), "warp:", round(env.data.time / wall, 2))
print("base_height_end:", round(env.base_height(), 3))
print("fwd_vel_end:", round(env.forward_velocity(), 3), "max_fwd_vel:", round(max(vels), 3))
print("avg_base_height:", round(float(np.mean(heights)), 3), "min:", round(float(np.min(heights)), 3))
print("settled_walking:", env.data.time > 5 and not env.has_fallen())
np.savez(results_dir / "smoke_ann.npz", heights=heights, vels=vels)

env.release()