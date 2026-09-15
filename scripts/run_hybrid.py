import sys
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deploy12 import Deploy12
from convert import ESNReader, SpikeHeadManifold

data = np.load(Path(__file__).resolve().parents[1] / "exported" / "lstm_walk_data.npz")
obs_all, h_all, _, act_all = data["obs"], data["h"], data["c"], data["act"]

split = int(len(obs_all) * 0.7)
esn = ESNReader(neurons=1500, spectral_radius=0.7)
esn.fit(obs_all[:split], h_all[:split])

from convert import load_policy_params, elu
w1, b1, _, _ = load_policy_params()
z1 = elu(w1 @ h_all[:split].T + b1[:, None]).T
head2 = SpikeHeadManifold(neurons=2000, settle=8)
head2.fit(h_all[:split], z1, act_all[:split])

dep = Deploy12()
dep.reset(cmd=[0.8, 0.0, 0.0])
dep.obs_noise_std = 0.0
esn.state = np.zeros(esn.n)

vx, hgt, pitch, tlog = [], [], [], []
t0 = time.time()
steps = 0
act_prev = np.zeros(12)
while dep.data.time < 8.0 and dep.data.qpos[2] > 0.35:
    dep.refresh_cmd()
    dep.step_ctrl()
    steps += 1
    if steps % dep.dec == 0:
        obs = dep.obs()
        esn._step_obs_only(obs)
        hhat = esn.D.T @ esn.state
        head2._h["x"] = hhat
        head2.sim.run_steps(head2.settle)
        action = head2.sim.data[head2._q][-1]
        dep.action = action
        dep.target = action * dep.action_scale + dep.default_angles
    if steps % dep.dec == 0:
        vx.append(dep.data.qvel[0]); hgt.append(dep.data.qpos[2])
        pitch.append(dep.data.xquat[[0]].copy())
        tlog.append(dep.data.time)

print("fall:", dep.data.qpos[2] <= 0.35, "h_end=%.3f" % dep.data.qpos[2])
if tlog:
    print("mean_vx=%.3f max_vx=%.3f mean_h=%.3f pitch=%.3f wall=%.2fs"
          % (np.mean(vx), np.max(vx), np.mean(hgt), float(np.mean([abs(p) for p in pitch])), time.time() - t0))