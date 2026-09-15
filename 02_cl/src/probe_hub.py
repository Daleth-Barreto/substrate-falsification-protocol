import sys
from pathlib import Path

import numpy as np
import nengo

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import demo_walk as D

holder = {"last_x": np.array([0.2, 0.1, 0.0, 1.0, 1.0])}
inp, out, p, sim = D.build_hub(holder)
sim.run_steps(120)
vals = sim.data[p][:, 0]
print("input", holder["last_x"])
print("out min=%.3f max=%.3f last=%.3f mean(last40)=%.3f"
      % (vals.min(), vals.max(), vals[-1], vals[-40:].mean()))

# vary input: stop condition (nspk 0)
holder["last_x"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
sim.run_steps(120)
vals = sim.data[p][:, 0]
print("nspk=0 out last=%.3f mean(last40)=%.3f" % (vals[-1], vals[-40:].mean()))

# distress: roll high
holder["last_x"] = np.array([3.0, 0.1, 0.0, 1.0, 1.0])
sim.run_steps(120)
vals = sim.data[p][:, 0]
print("roll=3 out last=%.3f mean(last40)=%.3f" % (vals[-1], vals[-40:].mean()))

# very high nspk
holder["last_x"] = np.array([0.2, 0.1, 0.0, 5.0, 1.0])
sim.run_steps(600)
vals = sim.data[p][:, 0]
print("nspk=5 out last=%.3f mean(last40)=%.3f" % (vals[-1], vals[-40:].mean()))

# nspk=2
holder["last_x"] = np.array([0.2, 0.1, 0.0, 2.0, 1.0])
sim.run_steps(300)
vals = sim.data[p][:, 0]
print("nspk=2 out last=%.3f mean(last40)=%.3f" % (vals[-1], vals[-40:].mean()))
sim.close()