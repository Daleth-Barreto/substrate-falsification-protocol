"""verify_null.py — dead-substrate null on unseen seeds.

Re-runs the Poisson dead-null (rates matched to LIF, input-independent) on two
seeds OUTSIDE the calibrated set {1,7,13,29,55,91} and checks that under the
canonical fixed decode it still carries ~zero task information (MI, TE, rho).
Proving "zero info" is not an artifact of the chosen seed list.

Usage: ..\..\02_cl\.venv312\Scripts\python.exe verify_null.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gate_d"))
import hub as hubd
import metrics as metricd
import run as rund

NEW_SEEDS = [103, 201]


def main():
    cal = rund.calibrate()
    g, w, wr = hubd.make_wiring(7)
    worst = 0.0
    max_te = 0.0
    for pname in ("multi_step", "pulse"):
        u = rund.make_profile(pname)
        for seed in NEW_SEEDS:
            gw, ww, wrr = hubd.make_wiring(seed)
            sub = hubd.PoissonNull(cal["poisson_rate"])
            sub.reset(seed)
            thsig, counts = hubd.run_profile(
                sub, (gw, ww, wrr), u, 0.0, 0.0, 0.0, cal["poisson_k_ro"])
            xhat = hubd.decide(thsig)
            vref = 0.75 * u
            rmse = float(np.sqrt(np.mean((xhat - vref) ** 2)))
            mi, mi_p = metricd.mi_bias_corrected(xhat, u)
            s = (thsig[:-1] > rund.C62_TH).astype(int)
            te, te_p = metricd.te_bias_corrected(s, xhat)
            rho = 0.0
            if np.std(xhat) > 1e-9 and np.std(u) > 1e-9:
                rho = float(np.corrcoef(xhat, u)[0, 1])
            print(f"{pname:10s} seed {seed}: rmse={rmse:.4f} mi={mi:.4f} "
                  f"p2={mi_p:.3f} te={te:.1e} rho={rho:+.4f}")
            worst = max(worst, abs(mi), abs(rho))
            max_te = max(max_te, te)
    ok = worst < 0.05 and max_te < 1e-6
    print(f"VERIFY_NULL {'PASS' if ok else 'FAIL'} (worst |mi/rho| = {worst:.4f}, max_te = {max_te:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())