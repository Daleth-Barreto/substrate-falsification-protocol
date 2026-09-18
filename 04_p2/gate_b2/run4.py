"""Gate B2v3: robustness battery for the substrate claim (second paper).

B2v3 is the reviewer-responsive extension of B2v2 (run3.py, 6 seeds). The B2v2
artifacts are untouched; this battery supersedes them with three strengthens:

1. EXTENDED SEEDS: 24 fixed seeds (the B2v2 set {1,7,13,29,55,91} is a subset),
   so pooled statements cover 10 profiles x 24 seeds x 2 substrates = 480 runs.

2. FOURTH NULL: "taskshuf" -- a population-matched surrogate that preserves the
   REAL count matrix verbatim (hence all per-neuron marginals, all cross-neuron
   correlations and the full population temporal structure) and permutes only
   the pairing of task signal to windows.  It is the strictest population-
   matched null: the only thing removed is window-to-task alignment.

3. MLP READOUT (numpy, no new deps): readout-invariance is recomputed under a
   fixed single-hidden-layer MLP decoder in addition to the fixed ridge.  The
   MLP and ridge are both fitted once on the real counts and applied to the
   null matrices (same readout, different spike statistics).

Everything else (hub, wiring, decide map, calibration seed, K, metric
definitions, p_perm convention) is byte-identical to B2v2.  All numbers in the
output JSON are computed, never hand-typed.
"""

import argparse
import json
import os

import numpy as np
from numpy.random import default_rng

import hub
import metrics
import nulls
from run2 import BASE_I, calibrate, metric_on
from run3 import (K_NULL, NULL_TYPES, PROFILES, RIDGE_NULLS,
                  _ridge_pair, make_profile_adv, metric_on_null, null_counts,
                  run_real)

OUTDIR = os.path.join(os.path.dirname(__file__), "results")

SEEDS_EXT = [1, 3, 7, 13, 17, 29, 31, 37, 41, 47, 53, 55,
             59, 61, 67, 71, 73, 79, 83, 89, 91, 97, 101, 103]
NULL_TYPES_EXT = NULL_TYPES + ["taskshuf"]

# MLP hyperparameters, fixed a priori (no tuning on the battery results).
MLP_HIDDEN = 32
MLP_EPOCHS = 120
MLP_LR = 1e-2
MLP_L2 = 1e-3
MLP_RNG_SEED = 20260917


def taskshuf_ups(u, k, seed):
    """K permutations of u preserving the population exactly (counts untouched)."""
    rng = default_rng(nulls.NULL_BASE + 303 + seed)
    return np.stack([rng.permutation(u) for _ in range(k)])


def null_pairs(counts, u, ntype, k, seed):
    """List of (counts_i, u_i) for null replicate i."""
    if ntype == "taskshuf":
        perms = taskshuf_ups(u, k, seed)
        return [(counts, perms[i]) for i in range(k)]
    nc = null_counts(counts, ntype, k, seed)
    return [(nc[i], u) for i in range(k)]


def mlp_fit(X_tr, y_tr, X_va, y_va, rng):
    """Single-hidden-layer MLP (tanh), MSE + l2, early-stop by best va epoch.

    Standardization moments are computed on tr only and stored with the fitted
    weights so the fixed readout can be applied to any later matrix.
    Returns callable proj(CM) -> (W,) decoded series.
    """
    mu = X_tr.mean(0)
    sd = X_tr.std(0) + 1e-9
    Xs = (X_tr - mu) / sd
    Xv = (X_va - mu) / sd
    n, d = Xs.shape
    h = MLP_HIDDEN
    W1 = rng.normal(0.0, 1.0 / np.sqrt(d), (d, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0.0, 1.0 / np.sqrt(h), (h, 1))
    b2 = np.zeros(1)
    yb = y_tr.mean()
    yt = (y_tr - yb)[:, None]
    m = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)]
    v = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)]
    lr, l2 = MLP_LR, MLP_L2
    best = (np.inf, None)
    for ep in range(MLP_EPOCHS):
        Z = np.tanh(Xs @ W1 + b1)
        Yh = Z @ W2 + b2 + yb
        dY = (Yh - yt) / n
        gW2 = Z.T @ dY + l2 * W2
        gb2 = dY.sum(0, keepdims=True)
        dZ = dY @ W2.T
        dH = dZ * (1.0 - Z * Z)
        gW1 = Xs.T @ dH + l2 * W1
        gb1 = dH.sum(0)
        grads = [gW1, gb1, gW2, gb2]
        params = [W1, b1, W2, b2]
        t = ep + 1
        for p, g, mm, vv in zip(params, grads, m, v):
            mm[:] = 0.9 * mm + 0.1 * g
            vv[:] = 0.999 * vv + 0.001 * g * g
            mh = mm / (1 - 0.9 ** t)
            vh = vv / (1 - 0.999 ** t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        Zv = np.tanh(Xv @ W1 + b1)
        Yv = Zv @ W2 + b2 + yb
        val_r = float(np.sqrt(np.mean((Yv[:, 0] - y_va) ** 2)))
        if val_r < best[0]:
            best = (val_r, (W1.copy(), b1.copy(), W2.copy(), b2.copy()))
    _, (W1f, b1f, W2f, b2f) = best

    def proj(CM):
        Xq = (CM.T.astype(np.float64) - mu) / sd
        return (np.tanh(Xq @ W1f + b1f) @ W2f + b2f + yb)[:, 0]

    return proj


def _mlp_pair(counts, null_counts_i, vref, u, k, rng):
    """Observed and null MI under a FIXED trained MLP readout (cf. ridge pair)."""
    X = counts.T.astype(np.float64)
    W = X.shape[0]
    idx = np.arange(W)
    tr, va, te = idx[::3], idx[1::3], idx[2::3]
    proj = mlp_fit(X[tr], vref[tr], X[va], vref[va], rng)
    mi_obs, _ = metrics.mi_bias_corrected(proj(counts), u)
    mi_null = []
    for i in range(k):
        mi, _ = metrics.mi_bias_corrected(proj(null_counts_i[i]), u)
        mi_null.append(mi)
    return mi_obs, np.asarray(mi_null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        profiles = ["sine", "noise"]
        seeds = SEEDS_EXT[:2]
        k_null = 10
        run_decoders = True
        outname = "gate_b2v3_smoke.json"
    else:
        profiles = PROFILES
        seeds = SEEDS_EXT
        k_null = K_NULL
        run_decoders = True
        outname = "gate_b2v3_summary.json"

    cal = calibrate()
    print("calibration k_ro: lif=%.4f izh=%.4f" % (cal["lif_k_ro"], cal["izh_k_ro"]))
    rows = {pname: [] for pname in profiles}
    for pname in profiles:
        for seed in seeds:
            cell = {"seed": seed, "subs": {}}
            for sn in ("lif", "izh"):
                print("  %s seed %2d %s ..." % (pname, seed, sn), flush=True)
                u, wiring, counts, obs = run_real(seed, pname, cal, sn, None)
                per_type = {}
                for nt in NULL_TYPES_EXT:
                    pairs = null_pairs(counts, u, nt, k_null, seed)
                    dims = ["mi", "rmse", "rho"]
                    vals = {d: np.empty(k_null) for d in dims}
                    for i, (ci, ui) in enumerate(pairs):
                        m = metric_on_null(ci, ui, wiring, cal[sn + "_k_ro"])
                        for d in dims:
                            vals[d][i] = m[d]
                    per_type[nt] = {
                        "mi": cell_stats(obs["mi"], vals["mi"], k_null, "higher"),
                        "rmse": cell_stats(obs["rmse"], vals["rmse"], k_null, "lower"),
                        "rho": cell_stats(obs["rho"], vals["rho"], k_null, "higher"),
                    }
                cell["subs"][sn] = {"observed": obs, "nulls": per_type}
            rows[pname].append(cell)

    # False-positive control (drive_gain = 0), circshift + taskshuf, extended seeds.
    fp = {}
    for sn in ("lif", "izh"):
        for pname in ("sine", "noise", "ramp"):
            sig = {"circshift": 0, "taskshuf": 0}
            for seed in seeds:
                u = make_profile_adv(pname)
                g, w, wr = hub.make_wiring(seed)
                sub = hub.LIF() if sn == "lif" else hub.IZH()
                sub.reset(seed)
                _, counts = hub.run_profile(
                    sub, (g, w, wr), u, BASE_I[sn], 0.0, 0.0,
                    cal[sn + "_k_ro"])
                obs = metric_on_null(counts, u, (g, w, wr), cal[sn + "_k_ro"])
                for nt in ("circshift", "taskshuf"):
                    pairs = null_pairs(counts, u, nt, k_null, seed)
                    mi_n = np.array([metric_on_null(ci, ui, (g, w, wr),
                                                   cal[sn + "_k_ro"])["mi"]
                                     for ci, ui in pairs])
                    p, z, _, _ = cell_stats(obs["mi"], mi_n, k_null, "higher")
                    sig[nt] += int(p <= 0.05)
            fp.setdefault(pname, {}).setdefault(sn, sig)

    # Readout invariance: canonical vs ridge vs MLP on the primary null (sine).
    decoders = {}
    if run_decoders:
        for sn in ("lif", "izh"):
            pc = []
            for seed in seeds:
                u, wiring, counts, obs = run_real(seed, "sine", cal, sn, None)
                vref = 0.75 * u
                nc = null_counts(counts, "circshift", RIDGE_NULLS, seed)
                mi_canon_null = np.array([
                    metric_on(nc[i], u, wiring, cal[sn + "_k_ro"])["mi"]
                    for i in range(RIDGE_NULLS)])
                p_canon, _, _, _ = cell_stats(obs["mi"], mi_canon_null,
                                              RIDGE_NULLS, "higher")
                mi_ridge_obs, mi_ridge_null = _ridge_pair(
                    counts, nc, vref, u, RIDGE_NULLS)
                p_ridge, _, _, _ = cell_stats(mi_ridge_obs, mi_ridge_null,
                                              RIDGE_NULLS, "higher")
                rng = default_rng(MLP_RNG_SEED + seed)
                mi_mlp_obs, mi_mlp_null = _mlp_pair(
                    counts, nc, vref, u, RIDGE_NULLS, rng)
                p_mlp, _, _, _ = cell_stats(mi_mlp_obs, mi_mlp_null,
                                            RIDGE_NULLS, "higher")
                pc.append({"seed": seed, "p_canon": p_canon, "p_ridge": p_ridge,
                           "p_mlp": p_mlp, "mi_obs_ridge": mi_ridge_obs,
                           "mi_obs_mlp": mi_mlp_obs,
                           "mi_null_ridge_mean": float(mi_ridge_null.mean()),
                           "mi_null_mlp_mean": float(mi_mlp_null.mean())})
            decoders[sn] = pc

    summary = {
        "experiment": "gate_b2v3",
        "supersedes": "gate_b2v2 (run3.py); B2v2 artifacts left untouched",
        "profiles": profiles, "seeds": seeds, "k_null": k_null,
        "null_types": NULL_TYPES_EXT,
        "seed_set_note": "24 fixed seeds; B2v2 set {1,7,13,29,55,91} is a subset",
        "calibration": cal,
        "null_suite": {
            "circshift": "per-neuron circular shift (rotated sequence)",
            "blockshuffle": "per-neuron B-window block shuffle",
            "identityswap": "per-window permutation of neuron identity",
            "taskshuf": "real counts verbatim; window-to-task alignment permuted "
                        "(population correlations preserved exactly)",
        },
        "false_positive_control": {
            "note": "drive_gain=0 (no task signal); circshift and taskshuf nulls",
            "significant_cells": fp,
        },
        "readout_invariance": {
            "note": "canonical vs fixed ridge vs fixed MLP on circshift null, sine",
            "mlp_hyperparameters": {"hidden": MLP_HIDDEN, "epochs": MLP_EPOCHS,
                                    "lr": MLP_LR, "l2": MLP_L2,
                                    "seed": MLP_RNG_SEED},
            "per_substrate": decoders,
        },
        "per_profile": {}, "pooled": {}, "verdict": "", "criteria": {},
    }

    task = [p for p in profiles if p != "baseline"]
    for pname in profiles:
        pp = {}
        for sn in ("lif", "izh"):
            cell_sn = {nt: [] for nt in NULL_TYPES_EXT}
            for cell in rows[pname]:
                for nt in NULL_TYPES_EXT:
                    cell_sn[nt].append(cell["subs"][sn]["nulls"][nt]["mi"][0])
            arr = {nt: np.array(cell_sn[nt]) for nt in NULL_TYPES_EXT}
            pp[sn] = {nt: {
                "signif_l05": int(np.sum(arr[nt] <= 0.05)),
                "signif_frac": float(np.mean(arr[nt] <= 0.05)),
                "p_min": float(np.min(arr[nt])),
            } for nt in NULL_TYPES_EXT}
        summary["per_profile"][pname] = pp

    pooled = {}
    for sn in ("lif", "izh"):
        for nt in NULL_TYPES_EXT:
            all_p = np.array([cell["subs"][sn]["nulls"][nt]["mi"][0]
                              for pn in task for cell in rows[pn]])
            pooled.setdefault(sn, {})[nt] = {
                "signif_l05_frac": float(np.mean(all_p <= 0.05)),
                "signif_l05_n": int(np.sum(all_p <= 0.05)),
                "n_cells": int(all_p.size),
            }
    summary["pooled"] = pooled

    lif_ok = all(pooled["lif"][nt]["signif_l05_frac"] >= 0.95
                 for nt in NULL_TYPES_EXT)
    izh_ok = all(pooled["izh"][nt]["signif_l05_frac"] >= 0.95
                 for nt in NULL_TYPES_EXT)
    fp_bad = any(vv >= 2 for pv in fp.values() for sn in pv.values()
                 for vv in sn.values())
    verdict = "PASS" if (lif_ok and izh_ok and not fp_bad) else "REVIEW"
    summary["verdict"] = verdict
    summary["criteria"] = {
        "lif_all_null_types_l95": bool(lif_ok),
        "izh_all_null_types_l95": bool(izh_ok),
        "fp_control_clean": bool(not fp_bad),
    }

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, outname), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    if not args.smoke:
        try:
            _figure(summary)
        except Exception as exc:
            print("figure failed:", exc)
    print(json.dumps({k: summary[k] for k in
                      ("verdict", "criteria", "pooled", "false_positive_control")},
                     indent=2))


def _figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    task = [p for p in summary["profiles"] if p != "baseline"]
    subs = ["lif", "izh"]
    nts = summary["null_types"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(task))
    wdt = 0.8 / (len(nts) * len(subs) + 1)
    for (i, sn) in enumerate(subs):
        for (j, nt) in enumerate(nts):
            fracs = [summary["per_profile"][p][sn][nt]["signif_frac"] for p in task]
            ax[0].bar(x + (j * len(subs) + i - len(nts) * len(subs) / 2) * wdt,
                      fracs, width=wdt, color=["#2c7fb8", "#41ab5d"][i],
                      alpha=0.45 + 0.15 * j, label=(sn + " " + nt) if j == 0 else None)
    ax[0].axhline(0.95, ls="--", c="k", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(task, rotation=40, ha="right")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("frac seeds MI p_perm <= 0.05")
    ax[0].set_title("B2v3: null suite vs profiles")
    zi = np.arange(len(nts))
    for (i, sn) in enumerate(subs):
        fracs = [summary["pooled"][sn][nt]["signif_l05_frac"] for nt in nts]
        ax[1].bar(zi + (i - 0.5) * 0.38, fracs, width=0.38,
                  color=["#2c7fb8", "#41ab5d"][i], label=sn.upper())
    ax[1].set_xticks(zi); ax[1].set_xticklabels(nts, rotation=20)
    ax[1].set_ylim(0, 1.05); ax[1].axhline(0.95, ls="--", c="k", lw=0.8)
    ax[1].set_ylabel("pooled frac sig"); ax[1].legend()
    ax[1].set_title("Pooled across task profiles")
    fpk = summary["false_positive_control"]["significant_cells"]
    worst = 0
    for pv in fpk.values():
        for snd in pv.values():
            worst = max(worst, max(snd.values()))
    ax[2].bar([0], [worst], color="#999")
    ax[2].set_xticks([0]); ax[2].set_xticklabels(["max sig (drive=0)"])
    ax[2].set_ylabel("sig cells"); ax[2].set_title("False-positive control")
    fig.suptitle("Gate B2v3 (second paper) -- VERDICT: %s" % summary["verdict"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gate_b2v3.png"), dpi=150)


def cell_stats(obs, null_metrics, k, direction="higher"):
    arr = np.asarray(null_metrics, dtype=float)
    mu, sd = float(arr.mean()), float(arr.std(ddof=0))
    n_ge = int((arr >= obs).sum()) if direction == "higher" else int((arr <= obs).sum())
    p_perm = (1.0 + n_ge) / (k + 1.0)
    z = (obs - mu) / (sd + 1e-12)
    return p_perm, z, mu, sd


if __name__ == "__main__":
    main()