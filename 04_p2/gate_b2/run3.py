"""Gate B2v2: robustness battery for the substrate claim (second paper).

Extends run2.py with the controls a methods / TNNLS reader will demand:

1. NULL SUITE (3 paired constructions, K=50 each): circshift (B2v1), block-
   shuffle, identity-swap -- each preserving a *different* marginal structure and
   destroying a different task-locked property.  If the substrate separates from
   all three, the information is not an artifact of any single matched feature.

2. FALSE-POSITIVE CONTROL: the identical battery applied to a LIF/IZH substrate
   whose DRIVE_GAIN = 0 (no task-coupled input, same base drive).  A valid test
   must NOT reject a system that received no task signal.

3. READOUT INVARIANCE: for the primary null (circshift) the same separation is
   recomputed under the cross-validated RIDGE readout (opt. linear), not just
   the fixed canonical decode.

4. ADVERSARIAL PROFILES: chirp (swept frequency) and noise (band-limited random
   walk) added to the 5 original task profiles + ramp/perturb/step_pulse.

Pipeline is the same for every cell: real hub run -> counts -> observed metrics
via the fixed canonical decode -> K=50 paired nulls -> null metric distribution
-> p_perm (one-sided) and z.  Nothing is ever hand-typed; all numbers come from
the JSON artifacts produced here.
"""

import argparse
import json
import os

import numpy as np

import hub
import metrics
import nulls
import run2
from run2 import (SEEDS, calibrate, make_profile, metric_on,
                  null_counts_by_shift)

K_NULL = 50
NULL_TYPES = ["circshift", "blockshuffle", "identityswap"]
RIDGE_NULLS = 20        # ridge is O(n_tr^3); keep the null count smaller
RIDGE_SEEDS = [1, 7, 13, 29, 55, 91]
RIDGE_PROFILE = "sine"
LAM_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1]

OUTDIR = os.path.join(os.path.dirname(__file__), "results")

PROFILES = ["baseline", "multi_step", "sine", "descend", "pulse",
            "ramp", "perturb", "step_pulse", "chirp", "noise"]


def make_profile_adv(name):
    """Adversarial profiles; falls back to run2.make_profile for the rest."""
    t = np.arange(hub.N_WIN, dtype=float)
    if name == "chirp":
        f = np.linspace(1.0 / 20.0, 1.0 / 5.0, hub.N_WIN)   # 0.05..0.2 cyc/win
        return 0.5 + 0.3 * np.sin(2.0 * np.pi * np.cumsum(f))
    if name == "noise":
        rng = np.random.default_rng(7)
        w = rng.normal(0.0, 1.0, hub.N_WIN)
        sm = np.convolve(w, np.ones(5) / 5.0, mode="same")
        sm = sm / sm.std()
        return np.clip(0.5 + 0.32 * sm, 0.10, 0.95)
    return make_profile(name)


def null_counts(counts, ntype, k, seed):
    if ntype == "circshift":
        return null_counts_by_shift(counts, k, np.random.default_rng(
            nulls.NULL_BASE + seed * 131))
    if ntype == "blockshuffle":
        return nulls.blockshuffle(counts, k, seed)
    if ntype == "identityswap":
        return nulls.identityswap(counts, k, seed)
    raise ValueError(ntype)


def metric_on_null(counts, u, wiring, k_ro):
    """Lighter metric for null replicates: RMSE + bias-corrected MI + rho.

    TE is dropped: it is reported only for the observed run (real dynamics) and
    is not part of the null decision, saving ~half the null-loop time.
    """
    g, w, wr = wiring
    c0 = wr @ (counts.astype(np.float64) / hub.CW)
    thsig = k_ro * c0
    xhat = hub.decide(thsig)
    vref = 0.75 * u
    rmse = float(np.sqrt(np.mean((xhat - vref) ** 2)))
    mi, _ = metrics.mi_bias_corrected(xhat, u)
    rho = 0.0
    if np.std(xhat) > 1e-9 and np.std(u) > 1e-9:
        rho = float(np.corrcoef(xhat, u)[0, 1])
    return {"rmse": rmse, "mi": mi, "rho": rho}


def cell_stats(obs, null_metrics, k, direction="higher"):
    """p_perm (one-sided) and z vs the K-null distribution.

    direction='higher': task-carrying is higher MI/rho (nulls >= obs count).
    direction='lower' : lower RMSE (nulls <= obs count).
    """
    arr = np.asarray(null_metrics, dtype=float)
    mu, sd = float(arr.mean()), float(arr.std(ddof=0))
    if direction == "higher":
        n_ge = int((arr >= obs).sum())
    else:
        n_ge = int((arr <= obs).sum())
    p_perm = (1.0 + n_ge) / (k + 1.0)
    z = (obs - mu) / (sd + 1e-12)
    return p_perm, z, mu, sd


def run_real(seed, pname, cal, sn, rng_app):
    """Real simulated run and observed metrics (canonical decode)."""
    g, w, wr = hub.make_wiring(seed)
    u = make_profile_adv(pname)
    wiring = (g, w, wr)
    sub = hub.LIF() if sn == "lif" else hub.IZH()
    sub.reset(seed)
    _, counts = hub.run_profile(sub, wiring, u,
                                run2.BASE_I[sn], run2.DRIVE_GAIN[sn],
                                run2.REC_SCALE[sn], cal[sn + "_k_ro"])
    obs = metric_on(counts, u, wiring, cal[sn + "_k_ro"])
    return u, wiring, counts, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    global K_NULL
    if args.smoke:
        profiles = ["sine", "perturb", "noise"]
        seeds = SEEDS[:2]
        K_NULL = 10
        use_ridge = False
    else:
        profiles = PROFILES
        seeds = SEEDS
        use_ridge = True

    cal = calibrate()
    print("calibration k_ro: lif=%.4f izh=%.4f" % (cal["lif_k_ro"], cal["izh_k_ro"]))
    rows = {pname: [] for pname in profiles}
    # Main battery: real runs + K per null type.
    for pname in profiles:
        for seed in seeds:
            cell = {"seed": seed, "subs": {}}
            for sn in ("lif", "izh"):
                print("  %s seed %2d %s ..." % (pname, seed, sn), flush=True)
                u, wiring, counts, obs = run_real(seed, pname, cal, sn, None)
                per_type = {}
                for nt in NULL_TYPES:
                    nc = null_counts(counts, nt, K_NULL, seed)
                    dims = ["mi", "rmse", "rho"]
                    vals = {d: np.empty(K_NULL) for d in dims}
                    for i in range(K_NULL):
                        m = metric_on_null(nc[i], u, wiring, cal[sn + "_k_ro"])
                        for d in dims:
                            vals[d][i] = m[d]
                    per_type[nt] = {
                        "mi": cell_stats(obs["mi"], vals["mi"], K_NULL, "higher"),
                        "rmse": cell_stats(obs["rmse"], vals["rmse"], K_NULL, "lower"),
                        "rho": cell_stats(obs["rho"], vals["rho"], K_NULL, "higher"),
                    }
                cell["subs"][sn] = {"observed": obs, "nulls": per_type}
            rows[pname].append(cell)

    # False-positive control: same pipeline, drive_gain = 0 (no task input).
    fp = {}
    for sn in ("lif", "izh"):
        for pname in ("sine", "noise", "ramp"):
            n_sig = 0
            for seed in seeds:
                u = make_profile_adv(pname)
                g, w, wr = hub.make_wiring(seed)
                sub = hub.LIF() if sn == "lif" else hub.IZH()
                sub.reset(seed)
                _, counts = hub.run_profile(
                    sub, (g, w, wr), u, run2.BASE_I[sn], 0.0, 0.0,
                    cal[sn + "_k_ro"])
                obs = metric_on_null(counts, u, (g, w, wr), cal[sn + "_k_ro"])
                nc = null_counts(counts, "circshift", K_NULL, seed)
                mi_n = np.array([metric_on_null(nc[i], u, (g, w, wr),
                                           cal[sn + "_k_ro"])["mi"]
                                 for i in range(K_NULL)])
                p, z, _, _ = cell_stats(obs["mi"], mi_n, K_NULL, "higher")
                n_sig += int(p <= 0.05)
            fp.setdefault(pname, {}).setdefault(sn, n_sig)

    # Readout invariance: canonical vs ridge on the primary null.
    ridge = {}
    if use_ridge:
        for sn in ("lif", "izh"):
            pc = []
            for seed in RIDGE_SEEDS:
                u, wiring, counts, obs = run_real(seed, "sine", cal, sn, None)
                vref = 0.75 * u
                # canonical null MI (circshift) - reuse from battery when present
                nc = null_counts(counts, "circshift", RIDGE_NULLS, seed)
                mi_canon_null = np.array([
                    metric_on(nc[i], u, wiring, cal[sn + "_k_ro"])["mi"]
                    for i in range(RIDGE_NULLS)])
                p_canon, _, _, _ = cell_stats(obs["mi"], mi_canon_null,
                                              RIDGE_NULLS, "higher")
                # ridge observed and null
                mi_ridge_obs, mi_ridge_null = _ridge_pair(
                    counts, nc, vref, u, RIDGE_NULLS)
                p_ridge, _, _, _ = cell_stats(mi_ridge_obs, mi_ridge_null,
                                              RIDGE_NULLS, "higher")
                pc.append({"seed": seed, "p_canon": p_canon, "p_ridge": p_ridge,
                           "mi_obs_ridge": mi_ridge_obs})
            ridge[sn] = pc

    summary = {
        "experiment": "gate_b2v2",
        "profiles": profiles, "seeds": seeds, "k_null": K_NULL,
        "null_types": NULL_TYPES,
        "calibration": cal,
        "null_suite": {
            "circshift": "per-neuron circular shift (rotated sequence)",
            "blockshuffle": "per-neuron B-window block shuffle",
            "identityswap": "per-window permutation of neuron identity",
        },
        "false_positive_control": {
            "note": "drive_gain=0 (no task signal) under circshift null",
            "significant_cells": fp,
        },
        "ridge_invariance": ridge,
        "per_profile": {}, "pooled": {},
    }

    task = [p for p in profiles if p != "baseline"]
    for pname in profiles:
        pp = {}
        for sn in ("lif", "izh"):
            cell_sn = {nt: [] for nt in NULL_TYPES}
            for cell in rows[pname]:
                for nt in NULL_TYPES:
                    cell_sn[nt].append(cell["subs"][sn]["nulls"][nt]["mi"][0])
            arr = {nt: np.array(cell_sn[nt]) for nt in NULL_TYPES}
            pp[sn] = {nt: {
                "signif_l05": int(np.sum(arr[nt] <= 0.05)),
                "signif_frac": float(np.mean(arr[nt] <= 0.05)),
                "p_min": float(np.min(arr[nt])),
            } for nt in NULL_TYPES}
        summary["per_profile"][pname] = pp

    pooled = {}
    for sn in ("lif", "izh"):
        for nt in NULL_TYPES:
            all_p = np.array([cell["subs"][sn]["nulls"][nt]["mi"][0]
                              for pn in task for cell in rows[pn]])
            pooled.setdefault(sn, {})[nt] = {
                "signif_l05_frac": float(np.mean(all_p <= 0.05)),
                "signif_l05_n": int(np.sum(all_p <= 0.05)),
                "n_cells": int(all_p.size),
            }
    summary["pooled"] = pooled

    # VERDICT: for BOTH substrates, every null type has >= 95% significant
    # cells over the pooled task-varying battery AND the FP control shows <= 1
    # false positive.
    lif_ok = all(pooled["lif"][nt]["signif_l05_frac"] >= 0.95 for nt in NULL_TYPES)
    izh_ok = all(pooled["izh"][nt]["signif_l05_frac"] >= 0.95 for nt in NULL_TYPES)
    fp_bad = any(v >= 2 for pv in fp.values() for v in pv.values())
    verdict = "PASS" if (lif_ok and izh_ok and not fp_bad) else "REVIEW"
    summary["verdict"] = verdict
    summary["criteria"] = {
        "lif_all_null_types_l95": bool(lif_ok),
        "izh_all_null_types_l95": bool(izh_ok),
        "fp_control_clean": bool(not fp_bad),
    }

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "gate_b2v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    try:
        _figure(summary)
    except Exception as exc:
        print("figure failed:", exc)
    print(json.dumps({k: summary[k] for k in
                      ("verdict", "criteria", "pooled", "false_positive_control")},
                     indent=2))


def _ridge_pair(counts, null_counts, vref, u, k):
    """Observed and null MI under a FIXED cross-validated ridge readout.

    Readout invariance, same logic as Gate D but paired: the ridge weights are
    fitted ONCE on the real counts (interleaved tr/va/te, lam over LAM_GRID) and
    that fixed readout is applied to the null count matrices.  This mirrors the
    canonical decode exactly: same readout, different spike statistics.  It does
    NOT re-fit the readout per null (re-fitting lets the readout adapt to the
    null's retained structure and measures model adaptivity, not stimulus
    dependence).
    """
    X = counts.T.astype(np.float64)
    W = X.shape[0]
    idx = np.arange(W)
    tr, va, te = idx[::3], idx[1::3], idx[2::3]
    # standardize on train only; alpha (fixed weights) from nested lam selection
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xts = (X[tr] - mu) / sd
    Xqs = (X[va] - mu) / sd
    xb = vref[tr].mean()
    ytc = vref[tr] - xb
    Kt = Xts @ Xts.T
    best = (None, None, np.inf)
    for lam in LAM_GRID:
        alpha = np.linalg.solve(Kt + lam * np.eye(Kt.shape[0]), ytc)
        r = float(np.sqrt(np.mean(((Xqs @ (Xts.T @ alpha) + xb) - vref[va]) ** 2)))
        if r < best[2]:
            best = (r, lam, 0.0)
    alpha = np.linalg.solve(Kt + best[1] * np.eye(Kt.shape[0]), ytc)
    P = (Xts.T @ alpha)                 # fixed readout over standardized inputs
    proj = lambda CM: ((CM.T - mu) / sd) @ P + xb

    mi_obs, _ = metrics.mi_bias_corrected(proj(counts), u)
    mi_null = []
    for i in range(k):
        mi, _ = metrics.mi_bias_corrected(proj(null_counts[i]), u)
        mi_null.append(mi)
    return mi_obs, np.asarray(mi_null)


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
                      fracs, width=wdt,
                      color=["#2c7fb8", "#41ab5d"][i],
                      alpha=0.5 + 0.25 * j, label=(sn + " " + nt) if j == 0 else None)
    ax[0].axhline(0.95, ls="--", c="k", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(task, rotation=40, ha="right")
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("frac seeds MI p_perm <= 0.05")
    ax[0].set_title("B2v2: null suite vs profiles")
    # pooled bar
    zi = np.arange(len(nts))
    for (i, sn) in enumerate(subs):
        fracs = [summary["pooled"][sn][nt]["signif_l05_frac"] for nt in nts]
        ax[1].bar(zi + (i - 0.5) * 0.38, fracs, width=0.38,
                  color=["#2c7fb8", "#41ab5d"][i], label=sn.upper())
    ax[1].set_xticks(zi); ax[1].set_xticklabels(nts)
    ax[1].set_ylim(0, 1.05); ax[1].axhline(0.95, ls="--", c="k", lw=0.8)
    ax[1].set_ylabel("pooled frac sig"); ax[1].legend()
    ax[1].set_title("Pooled across task profiles")
    # FP control
    fpk = sorted(summary["false_positive_control"]["significant_cells"].keys())
    ax[2].bar(np.arange(len(fpk)), [
        max(summary["false_positive_control"]["significant_cells"][p].values())
        for p in fpk], color="#999")
    ax[2].set_xticks(np.arange(len(fpk))); ax[2].set_xticklabels(fpk)
    ax[2].set_ylabel("max sig cells (drive=0)"); ax[2].set_title("False-positive control")
    fig.suptitle("Gate B2v2 (second paper) -- VERDICT: %s" % summary["verdict"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gate_b2v2.png"), dpi=150)


if __name__ == "__main__":
    main()