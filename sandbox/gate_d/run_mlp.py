"""Gate D bis: nonlinear (MLP) decoder on the SAME Gate D spike runs.

Additive extension, does NOT touch gate_d_summary.json. Re-simulates the Gate D
battery (identical seeds/profiles/calibration) and adds a nonlinear readout to
the decoder roster: the SAME cross-validated ridge readout plus a small
one-hidden-layer MLP trained to correct its residual (a linear skip). This is
the honest way to ask the reviewer's question "does a nonlinear decoder recover
task information the best linear readout misses?":

  xhat = ridge(X)  +  MLP_residual(X)

so the model starts exactly at the optimal linear readout and a nonlinear gain
is possible only if the MLP generalizes on held-out windows. Architecture is
deliberately tiny (N=2000 -> 32 -> 1, tanh, L2, gradient clipping) because the
train split has only ~66 windows; a bigger net overfits catastrophically, which
we observed and which is itself a data-budget finding for this hub.

If LIF gains under the MLP, linearity was hiding readout information; if IZH
does not, the Gate D "IRH is a readout-channel artifact" conclusion is robust
to nonlinear decoding.  Either outcome is informative and is reported as-is.
"""

import json
import os

import numpy as np
from numpy.random import default_rng

import hub
import metrics
import run as gd

MLP_HIDDEN = 32
MLP_LAM_GRID = [1e-3, 1e-2, 1e-1, 1.0]
MLP_EPOCHS = 300
MLP_LR = 0.05
MLP_CLIP = 1.0
MLP_SEED = 4242
OUTDIR = os.path.join(os.path.dirname(__file__), "results", "mlp")
MLP_JSON = os.path.join(OUTDIR, "mlp_summary.json")


def _standardize(Xtr, Xa):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xa - mu) / sd


def _ridge_preds(X, y, tr, va, te, lam_grid):
    """Cross-validated ridge with intercept; returns train/val/test preds."""
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    Xts, Xqs = _standardize(Xtr, Xva)
    _, Xes = _standardize(Xtr, Xte)
    xb = ytr.mean()
    ytc = ytr - xb
    Kt = Xts @ Xts.T
    Xts_T = Xts.T

    def pred(Xa, alpha):
        return Xa @ (Xts_T @ alpha) + xb

    best = (np.inf, None)
    for lam in lam_grid:
        alpha = np.linalg.solve(Kt + lam * np.eye(Kt.shape[0]), ytc)
        r = float(np.sqrt(np.mean((pred(Xqs, alpha) - yva) ** 2)))
        if r < best[0]:
            best = (r, lam)
    lam_star = best[1]
    alpha = np.linalg.solve(Kt + lam_star * np.eye(Kt.shape[0]), ytc)
    return pred(Xts, alpha), pred(Xqs, alpha), pred(Xes, alpha), lam_star


def _forward(X, W1, b1, W2, b2):
    Z = np.tanh(X @ W1 + b1)
    return Z @ W2 + b2, Z


def _fit_mlp(Xtr, ytr, lam, rng):
    """One hidden layer, full-batch GD with L2 on both layers + grad clipping."""
    n, d = Xtr.shape
    h = MLP_HIDDEN
    W1 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, h))
    b1 = np.zeros(h)
    W2 = np.zeros(h)
    b2 = 0.0
    lr = MLP_LR
    for _ in range(MLP_EPOCHS):
        y, Z = _forward(Xtr, W1, b1, W2, b2)
        err = (y - ytr) / n
        gW2 = Z.T @ err + 2.0 * lam * W2 / n
        gb2 = float(err.sum())
        gZ = np.outer(err, W2) * (1.0 - Z ** 2)
        gW1 = Xtr.T @ gZ + 2.0 * lam * W1 / n
        gb1 = gZ.sum(0)
        gnorm = np.sqrt(np.sum(np.square(gW1)) + np.sum(np.square(gW2))
                        + np.sum(np.square(gb1)) + gb2 * gb2)
        if gnorm > MLP_CLIP:
            scale = MLP_CLIP / gnorm
            gW1 *= scale; gb1 *= scale; gW2 *= scale; gb2 *= scale
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


def _mlp_metrics(counts, u_win, vref):
    """Ridge skip + L2 MLP residual correction, same split policy as ridge."""
    W = counts.shape[1]
    X = counts.T.astype(np.float64)
    y = vref.astype(np.float64)
    idx = np.arange(W)
    tr, va, te = idx[::3], idx[1::3], idx[2::3]
    p_tr, p_va, p_te, lam_ridge = _ridge_preds(X, y, tr, va, te, gd.LAM_GRID)
    Xts, Xqs = _standardize(X[tr], X[va])
    _, Xes = _standardize(X[tr], X[te])
    res_tr = y[tr] - p_tr
    res_va = y[va] - p_va
    res_te = y[te] - p_te
    # center residual target on train, add back
    rb = res_tr.mean()
    ytr = res_tr - rb
    best = (np.inf, None, None)
    for lam in MLP_LAM_GRID:
        rng = default_rng(MLP_SEED)
        params = _fit_mlp(Xts, ytr, lam, rng)
        pv, _ = _forward(Xqs, *params)
        rv = float(np.sqrt(np.mean((p_va + pv + rb - y[va]) ** 2)))
        if rv < best[0]:
            best = (rv, lam, params)
    rv_best, lam_mlp, params = best
    p_te_corr, _ = _forward(Xes, *params)
    pred_te = p_te + p_te_corr + rb
    rmse = float(np.sqrt(np.mean((pred_te - y[te]) ** 2)))
    mi, mi_p = metrics.mi_bias_corrected(pred_te, u_win[te])
    rho = 0.0
    if np.std(pred_te) > 1e-9 and np.std(u_win[te]) > 1e-9:
        rho = float(np.corrcoef(pred_te, u_win[te])[0, 1])
    return {"rmse": rmse, "mi": mi, "mi_p": mi_p, "rho": rho,
            "lam": lam_mlp, "hidden": MLP_HIDDEN,
            "rmse_va_ridge_skip": float(np.sqrt(np.mean((p_va - y[va]) ** 2))),
            "rmse_va_mlp": rv_best}


def run_one(seed, pname, u_win, cal):
    """Same sim as Gate D run_one, returns counts + full decoder roster."""
    g, w, wr = hub.make_wiring(seed)
    vref = 0.75 * u_win
    subs = {"lif": hub.LIF(), "izh": hub.IZH(),
            "poisson": hub.PoissonNull(cal["poisson_rate"])}
    out = {}
    for sn, sub in subs.items():
        sub.reset(seed)
        k_ro = cal[sn + "_k_ro"] if sn != "poisson" else cal["poisson_k_ro"]
        bi = gd.BASE_I[sn] if sn != "poisson" else 0.0
        dg = gd.DRIVE_GAIN[sn] if sn != "poisson" else 0.0
        rc = gd.REC_SCALE[sn] if sn != "poisson" else 0.0
        thsig, counts = hub.run_profile(sub, (g, w, wr), u_win, bi, dg, rc, k_ro)
        d = gd._decoders(thsig, counts, u_win, cal, sn)
        d["mlp"] = _mlp_metrics(counts, u_win, vref)
        d["rate_hz"] = float(counts.sum() / hub.N_WIN / hub.CW / (hub.DT_MS / 1000.0))
        out[sn] = d
    return out


def _load():
    for pname in gd.PROFILES:
        for seed in gd.SEEDS:
            yield pname, seed, json.load(
                open(os.path.join(OUTDIR, f"raw_{pname}_s{seed}.json"), encoding="utf-8"))


def _aggregate():
    rows = list(_load())
    subs = ["lif", "izh", "poisson"]
    pooled = {}
    for sn in subs:
        base = [r[2][sn] for r in rows]
        d = {}
        for name in ("canon", "ridge", "meanpd", "mlp"):
            for kk in ("rmse", "mi", "te", "rho"):
                if kk in base[0][name]:
                    v = np.array([b[name][kk] for b in base])
                    d[f"{name}_{kk}_mean"] = float(v.mean())
                    d[f"{name}_{kk}_se"] = float(v.std() / np.sqrt(len(v)))
            if "mi_p" in base[0][name]:
                d[f"{name}_mi_sig_frac"] = float(
                    np.mean([b[name]["mi_p"] < 0.05 for b in base]))
        d["carried_info"] = float(d["meanpd_rmse_mean"] - d["ridge_rmse_mean"])
        d["carried_info_mlp"] = float(d["meanpd_rmse_mean"] - d["mlp_rmse_mean"])
        pooled[sn] = d

    lif_mlp = pooled["lif"]["carried_info_mlp"] - pooled["poisson"]["carried_info_mlp"]
    izh_mlp = pooled["izh"]["carried_info_mlp"] - pooled["poisson"]["carried_info_mlp"]
    lif_lin = pooled["lif"]["carried_info"] - pooled["poisson"]["carried_info"]
    izh_lin = pooled["izh"]["carried_info"] - pooled["poisson"]["carried_info"]
    mlp_gain_lif = lif_mlp - lif_lin
    mlp_gain_izh = izh_mlp - izh_lin

    criteria = {
        "carried_info_lif_linear": float(lif_lin),
        "carried_info_lif_mlp": float(lif_mlp),
        "carried_info_izh_linear": float(izh_lin),
        "carried_info_izh_mlp": float(izh_mlp),
        "mlp_minus_linear_lif": float(mlp_gain_lif),
        "mlp_minus_linear_izh": float(mlp_gain_izh),
        "ridge_mi_sig_frac_lif": float(pooled["lif"]["ridge_mi_sig_frac"]),
        "ridge_mi_sig_frac_izh": float(pooled["izh"]["ridge_mi_sig_frac"]),
        "mlp_mi_sig_frac_lif": float(pooled["lif"]["mlp_mi_sig_frac"]),
        "mlp_mi_sig_frac_izh": float(pooled["izh"]["mlp_mi_sig_frac"]),
        "mlp_mi_sig_frac_poisson": float(pooled["poisson"]["mlp_mi_sig_frac"]),
    }
    verdict = (
        "Gate D bis (MLP): nonlinear readout "
        f"carried-info lif {lif_mlp:+.3f} vs linear {lif_lin:+.3f} "
        f"(delta {mlp_gain_lif:+.3f}); "
        f"izh {izh_mlp:+.3f} vs linear {izh_lin:+.3f} "
        f"(delta {mlp_gain_izh:+.3f}); "
        f"null mlp floor {pooled['poisson']['carried_info_mlp']:+.3f}."
    )
    return {"profiles": gd.PROFILES, "seeds": gd.SEEDS,
            "decoders": ["canon", "ridge", "meanpd", "mlp"],
            "pooled": pooled, "verdict": verdict, "criteria": criteria}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cal = gd.calibrate()
    for pname in gd.PROFILES:
        u = gd.make_profile(pname)
        for seed in gd.SEEDS:
            r = run_one(seed, pname, u, cal)
            with open(os.path.join(OUTDIR, f"raw_{pname}_s{seed}.json"), "w") as f:
                json.dump(r, f, indent=1)
    summary = _aggregate()
    with open(MLP_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({k: summary[k] for k in ("verdict", "criteria")}, indent=2))


if __name__ == "__main__":
    main()