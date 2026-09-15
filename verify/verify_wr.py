"""verify_wr.py — is the IZH 'tracking' carried by the aligned readout channel?

Hypothesis from Gate D: IZH's apparent Gate-B tracking came from the readout
channel `wr` being aligned with the input projection `g`, not from task
information in the population spikes. `wr` = g is the calibrated choice.

Test: rebuild the readout with `wr_random` = a zero-mean random vector
(decorrelated from g, same norm scaling then k_ro re-calibrated). If
`wr`-alignment is what inflated IZH, then under `wr_random` the IZH canon RMSE
collapses toward the dead-null floor, while LIF, whose rate code carries the
task (Gate D carried +0.094), should degrade less.

Run on sine+pulse, seeds 1/29/55. Compares canon RMSE under aligned wr vs
random wr for LIF, IZH, POISSON.

Usage: ..\..\02_cl\.venv312\Scripts\python.exe verify_wr.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gate_d"))
import hub as hubd
import run as rund


def canon_rmse(sub, wiring, u_win, bi, dg, rc, k_ro):
    thsig, _ = hubd.run_profile(sub, wiring, u_win, bi, dg, rc, k_ro)
    xhat = hubd.decide(thsig)
    vref = 0.75 * u_win
    return float(np.sqrt(np.mean((xhat - vref) ** 2)))


def main():
    cal = rund.calibrate()
    g, w, wr = hubd.make_wiring(7)
    rng = np.random.default_rng(4242)
    wr_rand = rng.normal(0.0, 1.0, size=g.shape[0])
    # rescale random readout to same |wr| so k_ro calibration transfers
    wr_rand = wr_rand * (np.linalg.norm(wr) / np.linalg.norm(wr_rand))

    for pname in ("sine", "pulse"):
        u = rund.make_profile(pname)
        for seed in (1, 29, 55):
            gw, ww, wrr = hubd.make_wiring(seed)
            row = [f"{pname:6s} s{seed:2d}"]
            for sn in ("lif", "izh", "poisson"):
                sub = (hubd.LIF() if sn == "lif" else
                       hubd.IZH() if sn == "izh" else
                       hubd.PoissonNull(cal["poisson_rate"]))
                sub.reset(seed)
                k_ro = cal[sn + "_k_ro"] if sn != "poisson" else cal["poisson_k_ro"]
                bi = rund.BASE_I.get(sn, 0.0)
                dg = rund.DRIVE_GAIN.get(sn, 0.0)
                rc = rund.REC_SCALE.get(sn, 0.0)
                r_aligned = canon_rmse(sub, (gw, ww, wrr), u, bi, dg, rc, k_ro)
                r_random = canon_rmse(sub, (gw, ww, wr_rand), u, bi, dg, rc, k_ro)
                row.append(f"{sn}: aligned={r_aligned:.3f} random={r_random:.3f} "
                           f"d={r_random - r_aligned:+.3f}")
            print("  ".join(row))

    print("EXPECT: IZH degrades strongly under wr_random (d >> 0); LIF degrades "
          "little or stays well below Poisson; POISSON stays ~0.39 regardless.")


if __name__ == "__main__":
    main()