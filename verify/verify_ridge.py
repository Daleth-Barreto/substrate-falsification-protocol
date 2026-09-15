"""verify_ridge.py — does the Woodbury ridge match a direct solve?

Gate D/E use the kernel (Woodbury) identity for ridge. This verifies the
implementation against a direct O(N^3) solve of the primal normal equations
on a few (profile, seed, substrate) runs. If they match to ~1e-9 the kernel
implementation is correct and the "optimal linear readout" numbers are real.

Does NOT rerun the hub (reads gate_d raw results are insufficient: ridge
features come from counts, which are not stored in raw JSONs). So this script
RE-RUNS two (profile, seed) pairs per substrate with both solve methods.

Usage (run from sandbox/verify):
    ..\..\02_cl\.venv312\Scripts\python.exe verify_ridge.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gate_d"))
import hub as hubd
import metrics as metricd
import run as rund


def woodbury(Xtr, ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xs = (Xtr - mu) / sd
    Xe = (Xte - mu) / sd
    yb = ytr.mean()
    Kt = Xs @ Xs.T
    alpha = np.linalg.solve(Kt + lam * np.eye(Kt.shape[0]), ytr - yb)
    return Xe @ (Xs.T @ alpha) + yb


def direct(Xtr, ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xs = (Xtr - mu) / sd
    Xe = (Xte - mu) / sd
    yb = ytr.mean()
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    beta = np.linalg.solve(A, Xs.T @ (ytr - yb))
    return Xe @ beta + yb


def main():
    cal = rund.calibrate()
    g, w, wr = hubd.make_wiring(7)
    max_err = 0.0
    for pname in ("sine", "pulse"):
        u = rund.make_profile(pname)
        vref = 0.75 * u
        for seed in (1, 29):
            for sn in ("lif", "izh", "poisson"):
                sub = {"lif": hubd.LIF, "izh": hubd.IZH}.get(sn, hubd.PoissonNull)
                s = sub() if sn != "poisson" else sub(cal["poisson_rate"])
                s.reset(seed)
                k_ro = cal[sn + "_k_ro"] if sn != "poisson" else cal["poisson_k_ro"]
                bi = rund.BASE_I.get(sn, 0.0)
                dg = rund.DRIVE_GAIN.get(sn, 0.0)
                rc = rund.REC_SCALE.get(sn, 0.0)
                _, counts = hubd.run_profile(s, (g, w, wr), u, bi, dg, rc, k_ro)
                X = counts.T.astype(np.float64)
                idx = np.arange(hubd.N_WIN)
                tr, te = idx[::2][:int(0.7 * len(idx[::2]))], idx[1::2]
                y = vref.astype(np.float64)
                for lam in (1e-3, 1.0):
                    pw = woodbury(X[tr], y[tr], X[te], lam)
                    pd = direct(X[tr], y[tr], X[te], lam)
                    err = float(np.max(np.abs(pw - pd)))
                    max_err = max(max_err, err)
                    rm1 = float(np.sqrt(np.mean((pw - y[te]) ** 2)))
                    rm2 = float(np.sqrt(np.mean((pd - y[te]) ** 2)))
                    print(f"{pname:8s} s{seed:2d} {sn:7s} lam={lam:.0e} "
                          f"max|diff|={err:.2e} rmse_woodbury={rm1:.4f} rmse_direct={rm2:.4f}")
    ok = max_err < 1e-9
    print(f"VERIFY_RIDGE {'PASS' if ok else 'FAIL'} (max|diff|={max_err:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())