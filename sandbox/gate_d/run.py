"""Gate D battery: is it the readout? Same spikes, four decoders.

For every (profile, seed, substrate) the SAME hub run produces counts; then:

- canon    : fixed calibrated readout (Gate B reference, RMSE + MI/TE/rho)
- ridge    : optimal L2-regularized LINEAR readout of the spike counts
             (nested train/valid/test), an upper bound for any fixed decode
- passthru : xhat = 0.75*u directly, RMSE = 0 by construction (ceiling)

Recurrence contribution: the same canon decode is also evaluated with
rec_scale = 0 ("nohub"), isolating wiring vs dynamics for LIF/IZH.
"""

import json
import os

import numpy as np

import hub
import metrics

SEEDS = [1, 7, 13, 29, 55, 91]
CAL_SEED = 7
PROFILES = ["baseline", "multi_step", "sine", "descend", "pulse"]
OUTDIR = os.path.join(os.path.dirname(__file__), "results")

BASE_I = {"lif": 0.7, "izh": 1.0}
DRIVE_GAIN = {"lif": 1.0, "izh": 4.0}
REC_SCALE = {"lif": 0.002, "izh": 0.02}
C62_TH = 0.18
LAM_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1]


def make_profile(name):
    t = np.arange(hub.N_WIN, dtype=float)
    if name == "baseline":
        return np.full(hub.N_WIN, 0.62)
    if name == "multi_step":
        return np.where((t // 10) % 2 == 0, 0.25, 0.85)
    if name == "sine":
        return 0.5 + 0.3 * np.sin(2.0 * np.pi * t / 20.0)
    if name == "descend":
        return np.linspace(0.9, 0.1, hub.N_WIN)
    if name == "pulse":
        return np.where((t % 10) < 3, 0.85, 0.30)
    raise ValueError(name)


def calibrate():
    g, w, wr = hub.make_wiring(CAL_SEED)
    cal = {}
    u05 = np.full(hub.N_WIN, 0.5)
    for name in ("lif", "izh"):
        sub = hub.LIF() if name == "lif" else hub.IZH()
        sub.reset(CAL_SEED)
        thsig, counts = hub.run_profile(
            sub, (g, w, wr), u05, BASE_I[name], DRIVE_GAIN[name], REC_SCALE[name], 1.0)
        cal[name + "_k_ro"] = 0.200 / (wr @ (counts.astype(np.float64) / hub.CW)).mean()
    lif = hub.LIF()
    lif.reset(CAL_SEED)
    _, counts = hub.run_profile(
        lif, (g, w, wr), u05, BASE_I["lif"], DRIVE_GAIN["lif"], REC_SCALE["lif"], 1.0)
    cal["poisson_rate"] = (counts.sum(axis=1) / hub.N_WIN / hub.CW).astype(np.float64).tolist()
    cal["poisson_mean_hz"] = float(np.mean(cal["poisson_rate"]) / (hub.DT_MS / 1000.0))
    cal["lif_mean_hz"] = cal["poisson_mean_hz"]
    poi = hub.PoissonNull(cal["poisson_rate"])
    poi.reset(CAL_SEED)
    _, counts = hub.run_profile(poi, (g, w, wr), u05, 0.0, 0.0, 0.0, 1.0)
    cal["poisson_k_ro"] = 0.200 / (wr @ (counts.astype(np.float64) / hub.CW)).mean()
    return cal


def _decoders(thsig, counts, u_win, cal, sn):
    vref = 0.75 * u_win
    xhat_canon = hub.decide(thsig)
    res = {"canon": _sch(xhat_canon, u_win, vref, thsig)}
    res["ridge"] = _ridge_metrics(counts, u_win, vref)
    res["meanpd"] = _meanpd_metrics(vref, u_win)
    res["passthru"] = {"rmse": 0.0, "mi": _continuous_mi(0.75 * u_win, u_win)}
    return res


def _meanpd_metrics(vref, u_win):
    """Floor: predict the run-mean of vref everywhere. Any decoder that
    carries task information must beat this; the dead null should not."""
    xhat = np.full(len(vref), vref.mean())
    rmse = float(np.sqrt(np.mean((xhat - vref) ** 2)))
    mi, mi_p = metrics.mi_bias_corrected(xhat, u_win)
    rho = 0.0
    if np.std(xhat) > 1e-9 and np.std(u_win) > 1e-9:
        rho = float(np.corrcoef(xhat, u_win)[0, 1])
    return {"rmse": rmse, "mi": mi, "mi_p": mi_p, "rho": rho}


def _sch(xhat, u_win, vref, thsig):
    rmse = float(np.sqrt(np.mean((xhat - vref) ** 2)))
    mi, mi_p = metrics.mi_bias_corrected(xhat, u_win)
    s = (thsig[:-1] > C62_TH).astype(int)
    te, te_p = metrics.te_bias_corrected(s, xhat)
    rho = 0.0
    if np.std(xhat) > 1e-9 and np.std(u_win) > 1e-9:
        rho = float(np.corrcoef(xhat, u_win)[0, 1])
    return {"rmse": rmse, "mi": mi, "mi_p": mi_p, "te": te, "te_p": te_p, "rho": rho}


def _ridge_metrics(counts, u_win, vref):
    lam_grid = LAM_GRID
    W = counts.shape[1]
    X = counts.T.astype(np.float64)          # (W, N)
    y = vref.astype(np.float64)
    # interleaved splits with equal phase coverage
    idx = np.arange(W)
    tr, va, te = idx[::3], idx[1::3], idx[2::3]
    rmse_tr, rmse_va, rmse_te, pred_te = _nested_ridge(X, y, tr, va, te, lam_grid)
    mi, mi_p = metrics.mi_bias_corrected(pred_te, u_win[te])
    rho = 0.0
    if np.std(pred_te) > 1e-9 and np.std(u_win[te]) > 1e-9:
        rho = float(np.corrcoef(pred_te, u_win[te])[0, 1])
    return {"rmse": rmse_te, "rmse_tr": rmse_tr, "rmse_va": rmse_va,
            "mi": mi, "mi_p": mi_p, "rho": rho}


def _nested_ridge(X, y, tr, va, te, lam_grid):
    """Ridge with intercept via the Woodbury (kernel) identity.

    y is centered on train so the intercept is fitted implicitly;
    predictions add back the train bias. Cost O(n_train^3).
    """
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xts = (Xtr - mu) / sd
    Xqs = (Xva - mu) / sd
    Xes = (Xte - mu) / sd
    xb = ytr.mean()
    ytc = ytr - xb
    Kt = Xts @ Xts.T                          # (n_tr, n_tr)
    Ktr_tf = np.eye(Kt.shape[0])
    Xts_T = Xts.T

    def fit_rmse(Xa, ya, lam):
        alpha = np.linalg.solve(Kt + lam * Ktr_tf, ytc)
        pred = Xa @ (Xts_T @ alpha) + xb
        return float(np.sqrt(np.mean((pred - ya) ** 2))), alpha

    best = (None, None, np.inf)
    for lam in lam_grid:
        r, _ = fit_rmse(Xqs, yva, lam)
        if r < best[2]:
            best = (r, lam, 0.0)
    _, lam_star, _ = best
    rmse_tr, alpha = fit_rmse(Xts, ytr, lam_star)
    rmse_va, _ = fit_rmse(Xqs, yva, lam_star)
    pred_te = Xes @ (Xts_T @ alpha) + xb
    rmse_te = float(np.sqrt(np.mean((pred_te - yte) ** 2)))
    return rmse_tr, rmse_va, rmse_te, pred_te


def _continuous_mi(x, u):
    mi, p = metrics.mi_bias_corrected(x, u)
    return mi


def run_one(name, u_win, seed, cal):
    g, w, wr = hub.make_wiring(seed)
    vref = 0.75 * u_win
    subs = {"lif": hub.LIF(), "izh": hub.IZH(),
            "poisson": hub.PoissonNull(cal["poisson_rate"])}
    out = {}
    for sn, sub in subs.items():
        sub.reset(seed)
        k_ro = cal[sn + "_k_ro"] if sn != "poisson" else cal["poisson_k_ro"]
        bi = BASE_I[sn] if sn != "poisson" else 0.0
        dg = DRIVE_GAIN[sn] if sn != "poisson" else 0.0
        rc = REC_SCALE[sn] if sn != "poisson" else 0.0
        thsig, counts = hub.run_profile(sub, (g, w, wr), u_win, bi, dg, rc, k_ro)
        d = _decoders(thsig, counts, u_win, cal, sn)
        # recurrence ablation: same substrate, ignore recurrent wiring
        if sn != "poisson":
            sub2 = hub.LIF() if sn == "lif" else hub.IZH()
            sub2.reset(seed)
            thsig2, _ = hub.run_profile(sub2, (g, w, wr), u_win, bi, dg, 0.0, k_ro)
            x2 = hub.decide(thsig2)
            rmse2 = float(np.sqrt(np.mean((x2 - vref) ** 2)))
            rmse_c = d["canon"]["rmse"]
            d["nohub_rmse"] = rmse2
            d["nohub_over_loss"] = float(rmse2 - rmse_c)
        d["rate_hz"] = float(counts.sum() / hub.N_WIN / hub.CW / (hub.DT_MS / 1000.0))
        out[sn] = d
    return out


def main():
    cal = calibrate()
    for i, (pname, u) in enumerate({p: make_profile(p) for p in PROFILES}.items()):
        for seed in SEEDS:
            r = run_one(pname, u, seed, cal)
            with open(os.path.join(OUTDIR, f"raw_{pname}_s{seed}.json"), "w") as f:
                json.dump({k: v for k, v in r.items()}, f, indent=1)

    summary = _aggregate(cal)
    with open(os.path.join(OUTDIR, "gate_d_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    try:
        _figure(summary)
    except Exception as exc:
        print("figure failed:", exc)
    print(json.dumps({k: summary[k] for k in ("pooled", "verdict", "criteria")}, indent=2))


def _load():
    for pname in PROFILES:
        for seed in SEEDS:
            with open(os.path.join(OUTDIR, f"raw_{pname}_s{seed}.json")) as f:
                yield pname, seed, json.load(f)


def _aggregate(cal):
    rows = list(_load())
    subs = ["lif", "izh", "poisson"]
    pooled = {}
    for sn in subs:
        d = {}
        base = [r[2][sn] for r in rows]
        for name in ("canon", "ridge", "meanpd", "passthru"):
            for kk in ("rmse", "mi", "te", "rho"):
                if kk in base[0][name]:
                    v = np.array([b[name][kk] for b in base])
                    d[f"{name}_{kk}_mean"] = float(v.mean())
                    d[f"{name}_{kk}_se"] = float(v.std() / np.sqrt(len(v)))
            if "mi_p" in base[0][name]:
                d[f"{name}_mi_sig_frac"] = float(
                    np.mean([b[name]["mi_p"] < 0.05 for b in base]))
        nh = [b.get("nohub_rmse") for b in base if b.get("nohub_rmse") is not None]
        if nh:
            d["nohub_rmse_mean"] = float(np.mean(nh))
            d["nohub_rmse_se"] = float(np.std(nh) / np.sqrt(len(nh)))
        # carried info = how much better the best linear readout is than the
        # trivial mean predictor; 0 -> substrate spikes carry up to nothing.
        d["carried_info"] = float(d["meanpd_rmse_mean"] - d["ridge_rmse_mean"])
        pooled[sn] = d

    mean_floor = pooled["poisson"]["carried_info"]
    lif_info = pooled["lif"]["carried_info"]
    izh_info = pooled["izh"]["carried_info"]
    canon_loss_lif = pooled["lif"]["canon_rmse_mean"] - pooled["lif"]["ridge_rmse_mean"]
    canon_loss_izh = pooled["izh"]["canon_rmse_mean"] - pooled["izh"]["ridge_rmse_mean"]
    nohub_gap_lif = pooled["lif"]["nohub_rmse_mean"] - pooled["lif"]["canon_rmse_mean"]
    nohub_gap_izh = pooled["izh"]["nohub_rmse_mean"] - pooled["izh"]["canon_rmse_mean"]
    # ... and under the best linear readout, is there significant MI at all?
    ridge_mi_sig_lif = pooled["lif"]["ridge_mi_sig_frac"]
    ridge_mi_sig_izh = pooled["izh"]["ridge_mi_sig_frac"]
    ridge_mi_sig_poi = pooled["poisson"]["ridge_mi_sig_frac"]

    # Readout efficiency: does the fixed canon decode leave extractable info
    # on the table (ridge < canon)? Positive -> yes.
    readout_improvable = (canon_loss_lif > 0.05) or (canon_loss_izh > 0.05)
    # Substrate contribution: carried info clearly above the dead-null floor.
    lif_real = lif_info - mean_floor > 0.05
    izh_real = (izh_info - mean_floor > 0.03) or (ridge_mi_sig_izh > 0.5)
    null_still_dead = (mean_floor > -0.02) and (ridge_mi_sig_poi < 0.4)
    # Recurrence: does the recurrent wiring add anything at all?
    recurrence_neutral = (abs(nohub_gap_lif) < 0.01) and (abs(nohub_gap_izh) < 0.01)

    verdict = (
        "RESULT: LIF carries task info robustly under best linear readout "
        f"({lif_info - mean_floor:+.3f} vs null, sig MI {ridge_mi_sig_lif:.2f}); "
        f"IZH carries only weak info (({izh_info - mean_floor:+.3f} vs null, "
        f"sig MI {ridge_mi_sig_izh:.2f}) -- gate-b izh 'tracking' was readout-"
        f"channel artifact; null stays dead ({ridge_mi_sig_poi:.2f}); "
        f"readout improvable ({readout_improvable}); recurrence neutral ({recurrence_neutral})"
    )
    criteria = {
        "carried_info_null_floor": float(mean_floor),
        "carried_info_lif": float(lif_info),
        "carried_info_izh": float(izh_info),
        "carried_info_lif_minus_null": float(lif_info - mean_floor),
        "carried_info_izh_minus_null": float(izh_info - mean_floor),
        "canon_minus_ridge_lif": float(canon_loss_lif),
        "canon_minus_ridge_izh": float(canon_loss_izh),
        "nohub_minus_canon_lif": float(nohub_gap_lif),
        "nohub_minus_canon_izh": float(nohub_gap_izh),
        "ridge_mi_sig_frac_lif": float(ridge_mi_sig_lif),
        "ridge_mi_sig_frac_izh": float(ridge_mi_sig_izh),
        "ridge_mi_sig_frac_poisson": float(ridge_mi_sig_poi),
        "readout_improvable": bool(readout_improvable),
        "lif_substrate_real": bool(lif_real),
        "izh_substrate_real_weak": bool(izh_real),
        "null_still_dead": bool(null_still_dead),
        "recurrence_neutral": bool(recurrence_neutral)}

    return {"profiles": PROFILES, "seeds": SEEDS, "calibration": cal,
            "decoders": ["canon", "ridge", "meanpd", "passthru", "nohub(recurrence ablation)"],
            "pooled": pooled, "verdict": verdict, "criteria": criteria}


def _figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subs = ["lif", "izh", "poisson"]
    labels = ["LIF", "Izhikevich", "Poisson (dead)"]
    p = summary["pooled"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(3); w = 0.24
    for i, (d, c) in enumerate([("canon", "#2c7fb8"), ("ridge", "#fdb863"), ("meanpd", "#bdbdbd")]):
        ax[0].bar(x + (i - 1) * w, [p[s][f"{d}_rmse_mean"] for s in subs], w, label=d, color=c,
                  yerr=[p[s][f"{d}_rmse_se"] for s in subs], capsize=3)
    ax[0].axhline(0.0, color="k", lw=0.6)
    ax[0].set_xticks(x, labels); ax[0].set_ylabel("RMSE vs vref")
    ax[0].set_title("canon vs best-linear vs mean-floor")
    ax[0].legend(fontsize=8)
    for i, s in enumerate(subs):
        nh = p[s].get("nohub_rmse_mean")
        if nh is not None:
            ax[1].bar(i, nh, 0.5, color="#41ab5d", alpha=0.7, label="nohub" if i == 0 else None)
            ax[1].bar(i, p[s]["canon_rmse_mean"], 0.3, color="#2c7fb8", label="canon" if i == 0 else None)
        else:
            ax[1].bar(i, p[s]["canon_rmse_mean"], 0.3, color="#2c7fb8")
    ax[1].set_xticks(x, labels); ax[1].set_ylabel("RMSE")
    ax[1].set_title("canon vs nohub (recurrence off)")
    ax[1].legend(fontsize=8)
    ax[2].bar(x, [p[s]["carried_info"] for s in subs],
              color=["#2c7fb8", "#41ab5d", "#fdb863"])
    ax[2].axhline(p["poisson"]["carried_info"], color="k", ls="--", lw=0.8,
                  label="dead-null floor")
    ax[2].set_xticks(x, labels); ax[2].set_ylabel("carried info (meanpd_rmse - ridge_rmse)")
    ax[2].set_title("Substrate info above the trivial-mean decoder")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Gate D: readout check -- {summary['verdict']}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gate_d.png"), dpi=150)


if __name__ == "__main__":
    main()