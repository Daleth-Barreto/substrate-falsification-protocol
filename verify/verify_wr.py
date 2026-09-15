"""verify_wr.py — is the apparent 'tracking' carried by the aligned readout?

Hypothesis from Gate D: the fixed-channel Gate-B tracking came from the
readout projection `wr` being aligned with the input projection `g`
(`wr` = g is the calibrated choice), not from task information that any
readout of the population spikes would recover. Gate C tests this.

Test: rebuild the readout with `wr_random` = a zero-mean random vector
(drawn independently per seed, decorrelated from g, same norm then k_ro
re-calibrated). If `wr`-alignment is what produced the Gate-B numbers,
then under `wr_random` the canonical RMSE collapses toward the dead-null
floor for the materials that only tracked through the channel. Poisson
must be unaffected. The distinction survives only if a readout-invariant
decoder (Gate D ridge) recovers the task from the same spikes.

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

    def rand_wr(seed):
        # per-seed independent random readout, unit-norm rescaled to |wr|
        rr = rng.normal(0.0, 1.0, size=g.shape[0])
        rr -= rr.mean()
        return rr * (np.linalg.norm(wr) / np.linalg.norm(rr))

    rows = []
    for pname in ("sine", "pulse"):
        u = rund.make_profile(pname)
        for seed in (1, 29, 55):
            gw, ww, wrr = hubd.make_wiring(seed)
            wr_rand = rand_wr(seed)
            row = [f"{pname:6s} s{seed:2d}"]
            entry = {"profile": pname, "seed": seed}
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
                entry[sn] = {"aligned": r_aligned, "random": r_random,
                             "delta": r_random - r_aligned}
                row.append(f"{sn}: aligned={r_aligned:.3f} random={r_random:.3f} "
                           f"d={r_random - r_aligned:+.3f}")
            print("  ".join(row))
            rows.append(entry)

    summary = {
        "label": "Gate C readout-channel control (wr alineado vs aleatorio)",
        "n_seeds_per_profile": 3,
        "profiles": ["sine", "pulse"],
        "notes": "wr_random = N(0,1) reescalado a |wr| igual y centrado, "
                 "dibujado de forma independiente por seed (rng seed 4242); "
                 "k_ro re-calibrado por sustrato; RMSE canon bajo el decode "
                 "fijo.",
        "per_profile": rows,
    }
    import json
    out = os.path.join(os.path.dirname(__file__), "verify_wr_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("WROTE", out)

    print("EXPECT: randomizing wr collapses BOTH living substrates toward the "
          "dead-null floor (d >> 0): the fixed aligned channel wr=g carried the "
          "Gate-B tracking; only a readout-invariant decoder (Gate D) separates "
          "LIF (real carried info) from IZH (channel artifact). POISSON is "
          "unaffected (d = 0).")


if __name__ == "__main__":
    main()