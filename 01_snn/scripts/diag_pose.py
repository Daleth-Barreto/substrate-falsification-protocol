import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mujoco
from mujco_env import REPO_ROOT, THIRD_PARTY, DEFAULT_CONFIG

with open(DEFAULT_CONFIG, "r") as f:
    import yaml
    config = yaml.load(f, Loader=yaml.FullLoader)

xml_path = THIRD_PARTY / "g1_deploy_mujoco" / config["xml_path"]
kps = np.array(config["kps"], dtype=np.float32)
kds = np.array(config["kds"], dtype=np.float32)
policy_joints = config["policy_joints"]
action_scale = config["action_scale"]
simulation_dt = config["simulation_dt"]

m = mujoco.MjModel.from_xml_path(str(xml_path))
d = mujoco.MjData(m)
m.opt.timestep = simulation_dt

policy_to_xml = []
for i in range(1, m.njnt):
    jname = mujoco.mj_id2name(m, 3, i)
    policy_to_xml.append(policy_joints.index(jname))

default_angles_policy = np.array(config["default_angles"], dtype=np.float32)
default_angles_xml = default_angles_policy[policy_to_xml]

d.qpos[2] = 0.80
d.qpos[7:] = default_angles_xml
mujoco.mj_forward(m, d)

def pd_control(tq, q, kp, tdq, dq, kd):
    return (tq - q) * kp + (tdq - dq) * kd

targets_variants = {
    "scrambled(policy[xmlmap])": default_angles_xml,
    "raw policy order": default_angles_policy[np.arange(29)],
}
for label, tgt in targets_variants.items():
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.80
    d.qpos[7:] = tgt
    mujoco.mj_forward(m, d)
    fallen = False
    for _ in range(2500):
        d.ctrl[:] = pd_control(tgt, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.35:
            fallen = True
            break
    print(label, "-> survived %.2fs" % d.time, "h=%.3f" % d.qpos[2], "fallen:", fallen)