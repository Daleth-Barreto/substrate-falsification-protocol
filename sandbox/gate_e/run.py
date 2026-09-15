"""Gate E — memory-requiring tasks: does substrate state carry the past?

Gate B/C ran feed-forward command tracking where vref = 0.75*u is a
deterministic function of the CURRENT window. Under any memoryless probe the
dead null can only ride the mean. Here vref depends on the HISTORY of u (hold,
lagged response, leaky integral, pulse-increment staircase), so a substrate
with real state must transpose the past INTO its present-window spikes to be
decodable. A dead Poisson substrate (input-independent counts) cannot.

Readouts (all, where relevant, from PRESENT-window features only, so memory
must be in the substrate, not in a lagged readout):

- canon   : k_ro * (wr @ rates) -> decide()         (fixed, memoryless)
- meanpd  : predict run-mean of vref                 (floor)
- ridge   : best L2-regularized linear readout of counts(w) -> vref(w)
- passthru: best ridge on INPUT taps [u(w)..u(w-8)] -> vref(w)  (trivial FIR
            filter on the input signal alone: the "no substrate needed" bound)

Verdict: null must stay at the floor under the optimal linear readout;
LIF/IZH must beat the floor AND not be dominated by the trivial input FIR.
"""

import json
import os

import numpy as np

import hub
import metrics

SEEDS = [1, 7, 13, 29, 55, 91]
CAL_SEED = 7
PROFILES = ["lag", "integ", "stair"]
N_MEM = 8                 # taps for the passthru input FIR ridge
BASE_I = {"lif": 0.7, "izh": 1.0}
DRIVE_GAIN = {"lif": 1.0, "izh": 4.0}
REC_SCALE = {"lif": 0.002, "izh": 0.02}
C62_TH = 0.18
LAM_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1]
RECIPE = np.random.default_rng(2026)
OUTDIR = os.path.join(os.path.dirname(__file__), "results")


def make_profile(name):
    """Return (u (N_WIN,), vref (N_WIN,)). u drives the hub; vref is the
    memory-dependent target that correct tracking must reach."""
    W = hub.N_WIN
    t = np.arange(W, dtype=float)
    if name == "lag":
        u = np.clip(0.5 + 0.25 * np.sin(2 * np.pi * t / 24.0) + 0.15 * np.sin(2 * np.pi * t / 7.0),
                    0.1, 0.9)
        vref = np.empty(W)
        vref[:4] = 0.75 * u[0]
        vref[4:] = 0.75 * u[:-4]            # answer of window w = input at w-4
        return u, vref
    if name == "integ":
        u = np.clip(0.5 + 0.20 * np.sin(2 * np.pi * t / 30.0)
                    + 0.10 * np.sin(2 * np.pi * t / 9.0) + 0.06 * RECIPE.normal(size=W),
                    0.2, 0.8)
        a = 1.0 / 6.0                        # 6-window leaky integration
        s = np.zeros(W)
        for w in range(1, W):
            s[w] = (1 - a) * s[w - 1] + a * (u[w] - 0.5)
        vref = np.clip(0.75 * (0.5 + 2.5 * s), 0.05, 0.7)
        return u, vref
    if name == "stair":
        u = np.full(W, 0.30)
        u[10:14] = 0.85; u[34:37] = 0.85; u[60:62] = 0.85; u[92:95] = 0.85
        u[120:122] = 0.85; u[150:153] = 0.85; u[180:182] = 0.85
        count = np.zeros(W, dtype=int)
        c = 0
        for w in range(1, W):
            if u[w] > 0.55 and u[w - 1] <= 0.55:
                c = min(c + 1, 5)
            count[w] = c
        vref = 0.15 + 0.11 * count             # step up after each pulse
        return u, vref
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


def _canon(thsig, vref, u_win):
    xhat = hub.decide(thsig)
    rmse = float(np.sqrt(np.mean((xhat - vref) ** 2)))
    mi, mi_p = metrics.mi_bias_corrected(xhat, vref)
    s = (thsig[:-1] > C62_TH).astype(int)
    te, te_p = metrics.te_bias_corrected(s, xhat)
    rho = 0.0
    if np.std(xhat) > 1e-9 and np.std(vref) > 1e-9:
        rho = float(np.corrcoef(xhat, vref)[0, 1])
    return {"rmse": rmse, "mi": mi, "mi_p": mi_p, "te": te, "te_p": te_p, "rho": rho}


def _nested_ridge(X, y, tr, va, te, lam_grid):
    Xi, yi = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-9
    Xs = (Xi - mu) / sd
    Xqs = (Xva - mu) / sd
    Xes = (Xte - mu) / sd
    yb = yi.mean()
    yc = yi - yb
    Kt = Xs @ Xs.T
    KtI = np.eye(Kt.shape[0])
    XsT = Xs.T

    def rmse_of(Xa, ya, lam):
        alpha = np.linalg.solve(Kt + lam * KtI, yc)
        pred = Xa @ (XsT @ alpha) + yb
        return float(np.sqrt(np.mean((pred - ya) ** 2))), alpha

    best = (None, None, np.inf)
    for lam in lam_grid:
        r, _ = rmse_of(Xqs, yva, lam)
        if r < best[2]:
            best = (r, lam, 0.0)
    _, lam_star, _ = best
    r_tr, alpha = rmse_of(Xs, yi, lam_star)
    r_va, _ = rmse_of(Xqs, yva, lam_star)
    pred_te = Xes @ (XsT @ alpha) + yb
    r_te = float(np.sqrt(np.mean((pred_te - yte) ** 2)))
    return r_tr, r_va, r_te, pred_te


def _ridge_metrics(counts, vref, taps=N_MEM):
    """Linear readout of PRESENT-window spike counts -> vref (no lag used, so
    memory must be in the substrate: counts(w) already carries the past)."""
    W = counts.shape[1]
    X = counts.T[:W - taps].astype(np.float64)     # windows with input taps available
    y = vref[taps:]
    idx = np.arange(X.shape[0])
    tr, va, te = idx[::3], idx[1::3], idx[2::3]
    r_tr, r_va, r_te, pred = _nested_ridge(X, y, tr, va, te, LAM_GRID)
    mi, mi_p = metrics.mi_bias_corrected(pred, y[te])
    rho = 0.0
    if np.std(pred) > 1e-9 and np.std(y[te]) > 1e-9:
        rho = float(np.corrcoef(pred, y[te])[0, 1])
    return {"rmse": r_te, "rmse_tr": r_tr, "rmse_va": r_va,
            "mi": mi, "mi_p": mi_p, "rho": rho}


def _passthru(u, vref, taps=N_MEM):
    """Trivial input FIR bound: ridge on u(w)..u(w-8) -> vref. This is what a
    memoryless 'no substrate' filter achieves; the substrate must beat it."""
    W = len(u)
    X = np.empty((W - taps, taps))
    for w in range(W - taps):
        X[w] = u[w:w + taps]
    y = vref[taps:]
    idx = np.arange(X.shape[0])
    tr, va, te = idx[::3], idx[1::3], idx[2::3]
    r_tr, r_va, r_te, pred = _nested_ridge(X, y, tr, va, te, LAM_GRID)
    mi, mi_p = metrics.mi_bias_corrected(pred, y[te])
    rho = 0.0
    if np.std(pred) > 1e-9 and np.std(y[te]) > 1e-9:
        rho = float(np.corrcoef(pred, y[te])[0, 1])
    return {"rmse": r_te, "rmse_tr": r_tr, "rmse_va": r_va,
            "mi": mi, "mi_p": mi_p, "rho": rho}


def _meanpd(vref):
    x = np.full(len(vref), vref.mean())
    rmse = float(np.sqrt(np.mean((x - vref) ** 2)))
    return {"rmse": rmse, "mi": 0.0, "mi_p": 1.0, "rho": 0.0}


def run_one(name, u_win, vref, seed, cal):
    g, w, wr = hub.make_wiring(seed)
    subs = {"lif": hub.LIF(), "izh": hub.IZH(),
            "poisson": hub.PoissonNull(cal["poisson_rate"])}
    out = {"u": u_win.tolist(), "vref": vref.tolist()}
    for sn, sub in subs.items():
        sub.reset(seed)
        k_ro = cal[sn + "_k_ro"] if sn != "poisson" else cal["poisson_k_ro"]
        bi = BASE_I[sn] if sn != "poisson" else 0.0
        dg = DRIVE_GAIN[sn] if sn != "poisson" else 0.0
        rc = REC_SCALE[sn] if sn != "poisson" else 0.0
        thsig, counts = hub.run_profile(sub, (g, w, wr), u_win, bi, dg, rc, k_ro)
        out[sn] = {"canon": _canon(thsig, vref, u_win),
                   "ridge": _ridge_metrics(counts, vref),
                   "meanpd": _meanpd(vref),
                   "rate_hz": float(counts.sum() / hub.N_WIN / hub.CW / (hub.DT_MS / 1000.0))}
    out["passthru"] = _passthru(u_win, vref)
    return out


def main():
    cal = calibrate()
    for pname in PROFILES:
        u, vref = make_profile(pname)
        for seed in SEEDS:
            r = run_one(pname, u, vref, seed, cal)
            with open(os.path.join(OUTDIR, f"raw_{pname}_s{seed}.json"), "w") as f:
                json.dump(r, f, indent=1)

    summary = _aggregate(cal)
    with open(os.path.join(OUTDIR, "gate_e_summary.json"), "w", encoding="utf-8") as f:
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


def _agg(sub):
    base = [r[2][sn] for sn in (sub,) for r in rows() if r[2].get(sn)]
    return base


def _aggregate(cal):
    rows = list(_load())
    subs = ["lif", "izh", "poisson"]
    pooled = {}
    per_profile = {}
    for pname in PROFILES:
        per_profile[pname] = {sn: {} for sn in subs + ["passthru"]}
    for sn in subs:
        d = {}
        base = [r[2][sn] for r in rows]
        for name in ("canon", "ridge", "meanpd"):
            for kk in ("rmse", "mi", "te", "rho"):
                if kk in base[0][name]:
                    v = np.array([b[name][kk] for b in base])
                    d[f"{name}_{kk}_mean"] = float(v.mean())
                    d[f"{name}_{kk}_se"] = float(v.std() / np.sqrt(len(v)))
            if "mi_p" in base[0][name]:
                d[f"{name}_mi_sig_frac"] = float(np.mean([b[name]["mi_p"] < 0.05 for b in base]))
        d["carried_info"] = float(d["meanpd_rmse_mean"] - d["ridge_rmse_mean"])
        pooled[sn] = d
        for pname in PROFILES:
            pb = [r[2][sn] for r in rows if r[0] == pname]
            mp = np.mean([b["meanpd"]["rmse"] for b in pb])
            rg = np.mean([b["ridge"]["rmse"] for b in pb])
            per_profile[pname][sn] = {"carried_info": float(mp - rg),
                                      "ridge_rmse": float(rg), "canon_rmse": float(np.mean([b["canon"]["rmse"] for b in pb]))}
    # passthru FIR bound per profile
    for pname in PROFILES:
        pb = [r[2] for r in rows if r[0] == pname]
        mp_mean = np.mean([b["lif"]["meanpd"]["rmse"] for b in pb])
        per_profile[pname]["passthru"] = {"carried_info": float(
            mp_mean - np.mean([b["passthru"]["rmse"] for b in pb])),
            "ridge_rmse": float(np.mean([b["passthru"]["rmse"] for b in pb]))}

    floor = pooled["poisson"]["carried_info"]
    lif_info = pooled["lif"]["carried_info"]
    izh_info = pooled["izh"]["carried_info"]
    # per-profile: substrate vs input-FIR and vs null floor
    best_prof_lif = max((per_profile[p]["lif"]["carried_info"] for p in PROFILES))
    best_prof_izh = max((per_profile[p]["izh"]["carried_info"] for p in PROFILES))
    best_prof_pt = max((per_profile[p]["passthru"]["carried_info"] for p in PROFILES))
    prof_floor = min((per_profile[p]["poisson"]["carried_info"] for p in PROFILES))

    null_memoryless = pooled["poisson"]["ridge_mi_sig_frac"] < 0.30 and floor <= 0.02
    neural_implements = any(ci - prof_floor > 0.05 for ci in
                            (per_profile[p][s]["carried_info"]
                             for p in PROFILES for s in ("lif", "izh")))
    beats_input_fir = any(per_profile[p]["lif"]["carried_info"] - per_profile[p]["passthru"]["carried_info"] > -0.02
                          for p in PROFILES)

    verdict = f"null memoryless ({null_memoryless}), neural implements ({neural_implements}), lif-not-dominated-by-input-FIR ({beats_input_fir})"
    criteria = {"carried_info_null_floor": float(floor),
                "carried_info_lif": float(lif_info), "carried_info_izh": float(izh_info),
                "best_carried_lif": float(best_prof_lif), "best_carried_izh": float(best_prof_izh),
                "best_carried_inputFIR": float(best_prof_pt), "profile_floor": float(prof_floor),
                "ridge_mi_sig_frac_poisson": float(pooled["poisson"]["ridge_mi_sig_frac"]),
                "null_memoryless": bool(null_memoryless),
                "neural_implements_memory": bool(neural_implements),
                "lif_not_dominated_by_input_fir": bool(beats_input_fir)}

    return {"profiles": PROFILES, "seeds": SEEDS, "taps": N_MEM, "calibration": cal,
            "pooled": pooled, "per_profile": per_profile, "verdict": verdict,
            "criteria": criteria}


def rows():
    return _load()


def _figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subs = ["lif", "izh", "poisson"]
    labels = ["LIF", "Izhikevich", "Poisson (dead)"]
    p = summary["pooled"]
    pp = summary["per_profile"]
    fig, ax = plt.subplots(1, len(summary["profiles"]) + 1, figsize=(4 * (len(summary["profiles"]) + 1), 3.6))
    x = np.arange(len(subs)); w = 0.32
    for i, pn in enumerate(summary["profiles"]):
        vals = [pp[pn][s]["carried_info"] for s in subs]
        vals.append(pp[pn]["passthru"]["carried_info"])
        labs = labels + ["input FIR"]
        ax[i].bar(np.arange(len(labs)), vals, color=["#2c7fb8", "#41ab5d", "#fdb863", "#636363"])
        ax[i].set_xticks(np.arange(len(labs)), labs, rotation=20, fontsize=8)
        ax[i].set_ylabel("carried info"); ax[i].set_title(pn)
    ax[-1].bar(x, [p[s]["carried_info"] for s in subs], color=["#2c7fb8", "#41ab5d", "#fdb863"])
    ax[-1].set_xticks(x, labels); ax[-1].set_ylabel("carried info (pooled)")
    ax[-1].set_title("pooled (3 memory tasks)")
    fig.suptitle(f"Gate E: memory-requiring tasks -- {summary['verdict']}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gate_e.png"), dpi=150)


if __name__ == "__main__":
    main()