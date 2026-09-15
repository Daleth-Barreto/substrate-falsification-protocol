import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mujoco
from collections import deque
from mujco_env import REPO_ROOT, THIRD_PARTY, DEFAULT_CONFIG

with open(DEFAULT_CONFIG, "r") as f:
    import yaml
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = THIRD_PARTY / "g1_deploy_mujoco" / config["xml_path"]
kps = np.array(config["kps"], dtype=np.float32)
kds = np.array(config["kds"], dtype=np.float32)
policy_joints = config["policy_joints"]
default_angles = np.array(config["default_angles"], dtype=np.float32)
ang_vel_scale = config["ang_vel_scale"]
dof_pos_scale = config["dof_pos_scale"]
dof_vel_scale = config["dof_vel_scale"]
action_scale = config["action_scale"]
cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
num_actions, num_obs = config["num_actions"], config["num_obs"]
cmd = np.array(config["cmd_init"], dtype=np.float32)
simulation_dt = config["simulation_dt"]
control_decimation = config["control_decimation"]

m = mujoco.MjModel.from_xml_path(str(xml_path))
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

floor_id = m.geom("floor").id
m.geom_friction[floor_id, 0] = 1.5
print("floor friction:", m.geom_friction[floor_id])

policy_to_xml = []
for i in range(1, m.njnt):
    jname = mujoco.mj_id2name(m, 3, i)
    policy_to_xml.append(policy_joints.index(jname))
xml_to_policy = [policy_to_xml.index(i) for i in range(len(policy_to_xml))]
default_angles = default_angles[policy_to_xml]
target_dof_pos = default_angles.copy()

import onnxruntime as ort
sess = ort.InferenceSession(str(REPO_ROOT / "exported" / "policy.onnx"), providers=["CPUExecutionProvider"])

def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz*qx + qw*qy), -2*(qz*qy + qw*qx), 1 - 2*(qw*qw + qz*qz)])

def pd_control(tq, q, kp, tdq, dq, kd):
    return (tq - q) * kp + (tdq - dq) * kd

action = np.zeros(num_actions, dtype=np.float32)
obs = np.zeros(num_obs, dtype=np.float32)
frame_stack = deque(maxlen=5)
for _ in range(5):
    frame_stack.append(obs.copy())
    mujoco.mj_step(m, d)

counter = 0
log = []
while d.time < 12.0 and d.qpos[2] > 0.35:
    d.ctrl[:] = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
    mujoco.mj_step(m, d)
    counter += 1
    if counter % control_decimation == 0:
        obs[:3] = d.qvel[3:6] * ang_vel_scale
        obs[3:6] = get_gravity_orientation(d.qpos[3:7])
        obs[6:9] = cmd * cmd_scale
        obs[9:38] = (d.qpos[7:] - default_angles) * dof_pos_scale
        obs[38:67] = d.qvel[6:] * dof_vel_scale
        obs[67:96] = action[xml_to_policy]
        frame_stack.append(obs.copy())
        stacked = np.concatenate(frame_stack, axis=0).reshape(1, -1)
        action = sess.run(None, {"obs": stacked.astype(np.float32)})[0].squeeze().astype(np.float32)
        action = action[policy_to_xml]
        target_dof_pos = action * action_scale + default_angles
    if counter % 500 == 0:
        log.append((d.time, d.qpos[2], d.qvel[0], float(np.mean(np.abs(action)))))

print("end: h:", round(d.qpos[2], 3), "vx:", round(d.qvel[0], 3))
for row in log[-8:]:
    print("t=%.2f h=%.3f vx=%.3f |a|=%.3f" % row)