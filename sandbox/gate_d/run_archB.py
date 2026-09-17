"""Gate D on a SECOND hub architecture (additive; artifact in results_archB/).

Reviewer concern: is the LIF-vs-IZH ranking an artifact of the one hub we
happened to build (N=2000, p_re=0.01)? This runner redoes the full Gate D
battery -- identical substrates, profiles, seeds, decoders and decision map --
on a different architecture B:

    N = 4000, p_re = 0.02   (bigger hub, denser recurrent wiring)

The wiring stays variance-normalized (w ~ N(0, 1/sqrt(N p_re))), so the
per-neuron recurrent drive has the same variance as in architecture A and the
operating point is comparable; what changes is hub size and connectivity
density. If LIF still carries task information above the matched dead null and
IZH still does not, the substrate ranking is not an artifact of the specific
hub. The verdict is an AND over the same criteria run.py already uses.

Writes results/archB/ (raw_*.json, gate_d_summary.json, gate_d.png) and a
side-by-side results/archB/archB_summary.json. gate_d_summary.json (v1,
N=2000) is never touched.

Usage:
    ..\\..\\02_cl\\.venv312\\Scripts\\python.exe run_archB.py
"""

import json
import os

import numpy as np

import hub
import run as gd

ARCH_B = {"N": 4000, "P_RE": 0.02}
OUTB = os.path.join(os.path.dirname(__file__), "results_archB")
CANON = os.path.join(os.path.dirname(__file__), "results", "gate_d_summary.json")


def main():
    # patch the hub geometry BEFORE any wiring/substrate is built; run.py
    # reads hub.N / hub.P_RE at call time, so this is enough.
    hub.N = ARCH_B["N"]
    hub.P_RE = ARCH_B["P_RE"]
    gd.OUTDIR = OUTB                      # keep v1 artifacts untouched
    os.makedirs(OUTB, exist_ok=True)

    gd.main()

    with open(os.path.join(OUTB, "gate_d_summary.json"), encoding="utf-8") as f:
        b = json.load(f)
    with open(CANON, encoding="utf-8") as f:
        a = json.load(f)

    def row(s):
        p = s["pooled"]
        return {
            "lif_carried_info": p["lif"]["carried_info"],
            "izh_carried_info": p["izh"]["carried_info"],
            "null_carried_info": p["poisson"]["carried_info"],
            "lif_ridge_mi_sig_frac": p["lif"]["ridge_mi_sig_frac"],
            "izh_ridge_mi_sig_frac": p["izh"]["ridge_mi_sig_frac"],
            "null_ridge_mi_sig_frac": p["poisson"]["ridge_mi_sig_frac"],
            "lif_ridge_rmse": p["lif"]["ridge_rmse_mean"],
            "izh_ridge_rmse": p["izh"]["ridge_rmse_mean"],
            "canon_rmse_lif": p["lif"]["canon_rmse_mean"],
            "canon_rmse_izh": p["izh"]["canon_rmse_mean"],
        }

    ra, rb = row(a), row(b)
    floor_a = ra["null_carried_info"]
    floor_b = rb["null_carried_info"]
    # same decision rule as run.py's criteria, evaluated per architecture
    verdict_b = {
        "lif_real": bool(rb["lif_carried_info"] - floor_b > 0.05),
        "izh_weak": bool((rb["izh_carried_info"] - floor_b > 0.03)
                         or (rb["izh_ridge_mi_sig_frac"] > 0.5)),
        "null_dead": bool(floor_b > -0.02 and rb["null_ridge_mi_sig_frac"] < 0.4),
    }
    verdict_b["ranking_reproduces"] = bool(
        verdict_b["lif_real"] and verdict_b["null_dead"]
        and (rb["lif_carried_info"] - floor_b) > (rb["izh_carried_info"] - floor_b))

    out = {
        "architecture_A": {"N": 2000, "p_re": 0.01, **ra},
        "architecture_B": {"N": ARCH_B["N"], "P_RE": ARCH_B["P_RE"], **rb},
        "criteria_B": verdict_b,
        "seeds": b["seeds"],
        "profiles": b["profiles"],
        "verdict": (
            f"architecture B (N={ARCH_B['N']}, p_re={ARCH_B['P_RE']}): "
            f"LIF info-over-null {rb['lif_carried_info'] - floor_b:+.3f}, "
            f"IZH {rb['izh_carried_info'] - floor_b:+.3f}, "
            f"null floor {floor_b:+.3f}; ranking reproduces="
            f"{verdict_b['ranking_reproduces']}."
        ),
    }
    with open(os.path.join(OUTB, "archB_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
