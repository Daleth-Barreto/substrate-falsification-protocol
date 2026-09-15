import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deploy12 import Deploy12

dep = Deploy12()
dep.reset(cmd=[0.8, 0.0, 0.0])
dep.obs_noise_std = 0.0

data = {"obs": [], "h": [], "c": [], "act": [], "t": []}
steps = 0
policy = dep.policy
with torch.inference_mode():
    while dep.data.time < 12.0 and dep.data.qpos[2] > 0.35:
        dep.refresh_cmd()
        dep.step_ctrl()
        steps += 1
        if steps % dep.dec == 0:
            obs = dep.obs()
            out = policy(torch.from_numpy(obs).unsqueeze(0))
            action = out.detach().numpy().squeeze().astype(np.float32)
            dep.action = action
            dep.target = action * dep.action_scale + dep.default_angles
            h = policy.hidden_state.detach().numpy().copy()
            c = policy.cell_state.detach().numpy().copy()
            data["obs"].append(obs.astype(np.float32))
            data["h"].append(h.reshape(-1).astype(np.float32))
            data["c"].append(c.reshape(-1).astype(np.float32))
            data["act"].append(action.astype(np.float32))
            data["t"].append(dep.data.time)

out = Path(__file__).resolve().parents[1] / "exported" / "lstm_walk_data.npz"
np.savez(out,
         obs=np.asarray(data["obs"]), h=np.asarray(data["h"]),
         c=np.asarray(data["c"]), act=np.asarray(data["act"]), t=np.asarray(data["t"]))
print("saved", out, "n=", len(data["obs"]))

import numpy as np
print("walk survived:", dep.data.qpos[2] > 0.35, "h_end=", round(dep.data.qpos[2], 3),
      "vx_end=", round(dep.data.qvel[0], 3))