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

policy_to_xml = []
for i in range(1, m.njnt):
    jname = mujoco.mj_id2name(m, 3, i)
    policy_to_xml.append(policy_joints.index(jname))
xml_to_policy = [policy_to_xml.index(i) for i in range(len(policy_to_xml))]
default_angles = default_angles[policy_to_xml]
target_dof_pos = default_angles.copy()

action = np.zeros(num_actions, dtype=np.float32)
obs = np.zeros(num_obs, dtype=np.float32)
frame_stack = deque(maxlen=5)
for _ in range(5):
    frame_stack.append(obs.copy())
    mujoco.mj_step(m, d)

def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz)])


def pd_control(tq, q, kp, tdq, dq, kd):
    return (tq - q) * kp + (tdq - dq) * kd

policy = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "exported" / "policy.onnx")
if policy.endswith(".onnx"):
    import onnxruntime as ort
    sess = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])
    inf = lambda x: sess.run(None, {"obs": x})[0]
else:
    import torch
    mod = torch.jit.load(policy, map_location="cpu").eval()
    inf = lambda x: mod(torch.from_numpy(x)).detach().numpy()

counter = 0
log = []
while d.time < 10.0 and d.qpos[2] > 0.35:
    tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    counter += 1
    if counter % control_decimation == 0:
        qj = d.qpos[7:]
        dqj = d.qvel[6:]
        quat = d.qpos[3:7]
        omega = d.qvel[3:6]
        qj = (qj - default_angles) * dof_pos_scale
        dqj = dqj * dof_vel_scale
        obs[:3] = omega * ang_vel_scale
        obs[3:6] = get_gravity_orientation(quat)
        obs[6:9] = cmd * cmd_scale
        obs[9:38] = qj[xml_to_policy]
        obs[38:67] = dqj[xml_to_policy]
        obs[67:96] = action[xml_to_policy]
        frame_stack.append(obs.copy())
        stacked = np.concatenate(frame_stack, axis=0).reshape(1, -1)
        action = inf(stacked.astype(np.float32)).squeeze().astype(np.float32)
        action = action[policy_to_xml]
        target_dof_pos = action * action_scale + default_angles
    if counter % 500 == 0:
        log.append((d.time, d.qpos[2], d.qvel[0], float(np.mean(np.abs(action)))))

print("policy:", policy)
print("end_t:", round(d.time, 2), "h:", round(d.qpos[2], 3), "vx:", round(d.qvel[0], 3))
for row in log:
    print("t=%.2f h=%.3f vx=%.3f |action|=%.3f" % row)