import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG = RESULTS / "figures"

with open(RESULTS / "summary.json") as f:
    S = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0, 0]
for ctrl in ("pd", "spd"):
    s = S["e1"][ctrl]
    t = s["series"]["t"]
    ax.plot(t, s["series"]["v"], label=f"{ctrl} vx (fall={s['fallen']})")
ax.set_title("E1 sprint profile: commanded velocity tracking")
ax.set_ylabel("vx (m/s)")
ax.legend()

ax = axes[0, 1]
for ctrl in ("pd", "spd"):
    s = S["e1"][ctrl]
    ax.plot(s["series"]["t"], s["series"]["h"], label=f"{ctrl}")
ax.set_title("E1 CoM height")
ax.set_ylabel("h (m)")
ax.legend()

ax = axes[1, 0]
noises = sorted({k.split("_")[0] for k in S["e3"]})
for ctrl in ("pd", "spd"):
    means, errs = [], []
    for n in noises:
        vals = [S["e3"][f"{n}_{ctrl}_s{s}"]["mean_vx"] for s in range(3)]
        vals = [v for v in vals if v == v]
        means.append(np.nanmean(vals))
        errs.append(np.nanstd(vals) if len(vals) else 0.0)
    lt = noises if False else [n.replace("n", "") for n in noises]
    ax.errorbar(range(len(noises)), means, yerr=errs, marker="o", label=ctrl)
ax.set_xticks(range(len(noises)))
ax.set_xticklabels([n[1:] for n in noises])
ax.set_title("E3 obs-noise robustness (mean vx)")
ax.set_ylabel("mean vx")

ax = axes[1, 1]
ps = S["e1"]["pd"]
ax.plot(ps["series"]["t"], ps["series"]["tau"], label="pd |tau|")
ax.set_title("E1 torque norm (pd)")
ax.set_ylabel("|tau| (Nm)")
ax.legend()

fig.tight_layout()
fig.savefig(FIG / "e1_e3_summary.png", dpi=150)
print("saved", FIG / "e1_e3_summary.png")
print("S keys:", list(S.keys()))
print("E3 n keys:", len(S["e3"]))

E = S["e4"]
print("E4 spd:", E["stats"])