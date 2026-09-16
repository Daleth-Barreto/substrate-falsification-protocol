"""Gate B2: population paired null (surrogate circular-shift) for the substrate claim.

Gate B showed LIF/IZH carry the task under a fixed wiring+decode, against a single
Poisson null matched only in per-neuron mean rates. The reader of a follow-up
will ask: is the separation robust to a *better matched* null -- one that keeps
per-neuron marginals AND per-neuron temporal statistics, destroying only the
population's task-locked alignment?

B2 answers with a paired-null DISTRIBUTION instead of a single null:

  For every real LIF (and IZH) run we keep its counts matrix C (N x N_WIN).
  Each null replicate k applies an independent random CIRCULAR SHIFT with a
  per-neuron lag  l_n ~ U{0..N_WIN-1}  to each neuron's count sequence:

      C'_k[n, w] = C[n, (w - l_n[k]) mod N_WIN]

  This preserves, per neuron, the exact count total, the count histogram and the
  autocorrelation structure (same sequence, rotated); it destroys the
  cross-neuron synchronized response to the task that the readout wr.C picks
  up.  The readout, decide map and calibration k_ro are byte-identical to the
  real run (same seed, same wiring).  K=50 nulls per run form the null
  distribution; we report p_perm (fraction of nulls >= observed, one-sided) and
  z = (obs - mean_null)/std_null for MI, RMSE, rho, TE.

Battery: 8 profiles (5 original + ramp, perturb, step_pulse) x 6 seeds x 2
substrates = 96 real runs, each with K=50 paired nulls = 4,800 null replicates.
"""

import argparse
import json
import os

import numpy as np
from numpy.random import default_rng

import hub
import metrics

SEEDS = [1, 7, 13, 29, 55, 91]
CAL_SEED = 7
K_NULL = 50

BASE_I = {"lif": 0.7, "izh": 1.0}
DRIVE_GAIN = {"lif": 1.0, "izh": 4.0}
REC_SCALE = {"lif": 0.002, "izh": 0.02}
C62_TH = 0.18

OUTDIR = os.path.join(os.path.dirname(__file__), "results")
NULL_RNG_BASE = 4242

PROFILES = ["baseline", "multi_step", "sine", "descend", "pulse",
            "ramp", "perturb", "step_pulse"]


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
    if name == "ramp":
        return np.linspace(0.15, 0.85, hub.N_WIN)
    if name == "perturb":
        u = 0.5 + 0.3 * np.sin(2.0 * np.pi * t / 20.0)
        u[60:64] += 0.35
        u[130:132] -= 0.40
        return np.clip(u, 0.10, 0.95)
    if name == "step_pulse":
        u = np.where((t // 5) % 2 == 0, 0.35, 0.80)
        u[120:140] = 0.70
        return u
    raise ValueError(name)


def calibrate():
    """Same calibration as Gate B (single source): k_ro per substrate, u=0.5."""
    g, w, wr = hub.make_wiring(CAL_SEED)
    cal = {}
    u05 = np.full(hub.N_WIN, 0.5)
    for name in ("lif", "izh"):
        sub = hub.LIF() if name == "lif" else hub.IZH()
        sub.reset(CAL_SEED)
        _, counts = hub.run_profile(
            sub, (g, w, wr), u05, BASE_I[name], DRIVE_GAIN[name],
            REC_SCALE[name], 1.0)
        cal[name + "_k_ro"] = 0.200 / (wr @ (counts.astype(np.float64) / hub.CW)).mean()
    return cal


def null_counts_by_shift(counts, k, rng):
    """K paired surrogates: per-neuron circular shift with random lags.

    counts: (N, N_WIN) real count matrix. Returns (K, N, N_WIN).
    """
    n, w = counts.shape
    out = np.empty((k, n, w), dtype=counts.dtype)
    for i in range(k):
        lags = rng.integers(0, w, size=(n, 1))
        idx = (np.arange(w)[None, :] - lags) % w
        out[i] = counts[np.arange(n)[:, None], idx]
    return out


def metric_on(counts, u, wiring, k_ro):
    """Metrics for a single count matrix through the fixed decode."""
    g, w, wr = wiring
    c0 = wr @ (counts.astype(np.float64) / hub.CW)
    thsig = k_ro * c0
    xhat = hub.decide(thsig)
    vref = 0.75 * u
    rmse = float(np.sqrt(np.mean((xhat - vref) ** 2)))
    mi, mi_p = metrics.mi_bias_corrected(xhat, u)
    s = (thsig[:-1] > C62_TH).astype(int)
    te, te_p = metrics.te_bias_corrected(s, xhat)
    rho = 0.0
    if np.std(xhat) > 1e-9 and np.std(u) > 1e-9:
        rho = float(np.corrcoef(xhat, u)[0, 1])
    return {"rmse": rmse, "mi": mi, "mi_p": mi_p, "te": te,
            "te_p": te_p, "rho": rho}


def run_cell(pname, u, seed, cal, k_null):
    """One (profile, seed): real LIF+IZH runs and their paired-null distributions."""
    g, w, wr = hub.make_wiring(seed)
    wiring = (g, w, wr)
    rng_seed = NULL_RNG_BASE + seed * 131 + (abs(hash(pname)) % (1 << 20))
    rng = default_rng(rng_seed)
    out = {}
    for sn, cls in (("lif", hub.LIF), ("izh", hub.IZH)):
        sub = cls()
        sub.reset(seed)
        thsig, counts = hub.run_profile(
            sub, wiring, u, BASE_I[sn], DRIVE_GAIN[sn], REC_SCALE[sn],
            cal[sn + "_k_ro"])
        obs = metric_on(counts, u, wiring, cal[sn + "_k_ro"])
        nulls = null_counts_by_shift(counts, k_null, rng)
        dims = ["rmse", "mi", "te", "rho"]
        null_rows = {d: np.empty(k_null) for d in dims}
        for i in range(k_null):
            m = metric_on(nulls[i], u, wiring, cal[sn + "_k_ro"])
            for d in dims:
                null_rows[d][i] = m[d]
        cell = {"observed": obs, "k_null": k_null}
        for d in dims:
            mu, sd = float(null_rows[d].mean()), float(null_rows[d].std(ddof=0))
            # one-sided: task-carrying is HIGHER MI/rho, LOWER RMSE
            if d in ("mi", "rho", "te"):
                n_ge = int((null_rows[d] >= obs[d]).sum())
            else:  # rmse, te? TE informational upward:
                n_ge = int((null_rows[d] <= obs[d]).sum())
            p_perm = (1 + n_ge) / (k_null + 1)
            z = (obs[d] - mu) / (sd + 1e-12)
            cell[d + "_p"] = float(p_perm)
            cell[d + "_z"] = float(z)
            cell[d + "_null_mean"] = mu
            cell[d + "_null_sd"] = sd
        out[sn] = cell
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 profile x 2 seeds, K=10")
    args = ap.parse_args()

    profiles = PROFILES if not args.smoke else ["sine", "perturb"]
    seeds = SEEDS if not args.smoke else SEEDS[:2]
    k_null = K_NULL if not args.smoke else 10

    cal = calibrate()
    print("calibration k_ro: lif=%.4f izh=%.4f" % (cal["lif_k_ro"], cal["izh_k_ro"]))
    rows = {}
    for pname in profiles:
        u = make_profile(pname)
        rows[pname] = []
        for seed in seeds:
            cell = run_cell(pname, u, seed, cal, k_null)
            rows[pname].append([seed, cell])

    summary = {"profiles": profiles, "seeds": seeds, "k_null": k_null,
               "calibration": cal, "null": "paired per-neuron circular shift",
               "decide_map": {"thsig_base": hub.TH0, "thsig_gain": hub.K,
                              "cap": 0.75, "gate": hub.GATE, "slow": hub.SLOW},
               "subs": ["lif", "izh"], "per_profile": {}, "pooled": {}}

    for pname in profiles:
        pp = {}
        for sn in ("lif", "izh"):
            obs_mi = np.array([r[1][sn]["observed"]["mi"] for r in rows[pname]])
            p_mi = np.array([r[1][sn]["mi_p"] for r in rows[pname]])
            z_mi = np.array([r[1][sn]["mi_z"] for r in rows[pname]])
            p_rmse = np.array([r[1][sn]["rmse_p"] for r in rows[pname]])
            z_rmse = np.array([r[1][sn]["rmse_z"] for r in rows[pname]])
            rho = np.array([r[1][sn]["observed"]["rho"] for r in rows[pname]])
            rho_p = np.array([r[1][sn]["rho_p"] for r in rows[pname]])
            n_seeds = len(rows[pname])
            pp[sn] = {
                "mi_obs_mean": float(obs_mi.mean()),
                "mi_signif_l05": int((p_mi <= 0.05).sum()),
                "mi_signif_frac": float((p_mi <= 0.05).mean()),
                "mi_z_mean": float(z_mi.mean()),
                "rmse_signif_l05": int((p_rmse <= 0.05).sum()),
                "rmse_z_mean": float(z_rmse.mean()),
                "rho_mean": float(rho.mean()),
                "rho_signif_l05": int((rho_p <= 0.05).sum()),
                "n_seeds": n_seeds,
            }
        summary["per_profile"][pname] = pp

    # pooled across task-varying profiles only; baseline is the negative control.
    task_profiles = [p for p in profiles if p != "baseline"]
    pooled = {}
    for sn in ("lif", "izh"):
        mi_p = np.array([r[1][sn]["mi_p"] for pn in task_profiles for r in rows[pn]])
        rmse_p = np.array([r[1][sn]["rmse_p"] for pn in task_profiles for r in rows[pn]])
        z_mi = np.array([r[1][sn]["mi_z"] for pn in task_profiles for r in rows[pn]])
        n_task = len(task_profiles) * len(seeds)
        pooled[sn] = {
            "mi_signif_l05_frac": float((mi_p <= 0.05).mean()),
            "mi_signif_l05_n": int((mi_p <= 0.05).sum()),
            "rmse_signif_l05_frac": float((rmse_p <= 0.05).mean()),
            "rmse_signif_l05_n": int((rmse_p <= 0.05).sum()),
            "mi_z_mean": float(z_mi.mean()),
            "n_cells": n_task,
        }
    summary["pooled"] = pooled

    # Verdict: LIF beats its own paired null in MI on >= 95% of task-varying
    # cells (strictly: <= 0.05 fraction non-significant), and every profile has
    # at least floor(0.8) of its seeds significant.
    lif_ok = pooled["lif"]["mi_signif_l05_frac"] >= 0.95 and all(
        summary["per_profile"][p]["lif"]["mi_signif_frac"] >= 0.8
        for p in task_profiles) if task_profiles else False
    izh_info = pooled["izh"]["mi_signif_l05_frac"]
    baseline_ok = all(
        summary["per_profile"]["baseline"][sn]["mi_signif_l05"] == 0
        for sn in ("lif", "izh") if "baseline" in summary["per_profile"])

    verdict = "PASS" if (lif_ok and baseline_ok) else "REVIEW"
    summary["verdict"] = verdict
    summary["criteria"] = {
        "lif_beat_own_null_frac": float(pooled["lif"]["mi_signif_l05_frac"]),
        "izh_beat_own_null_frac": float(izh_info),
        "baseline_control": bool(baseline_ok),
    }

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "gate_b2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    try:
        _figure(summary)
    except Exception as exc:
        print("figure failed:", exc)
    print(json.dumps({k: summary[k] for k in ("verdict", "criteria", "pooled")}, indent=2))


def _figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    task = [p for p in summary["profiles"] if p != "baseline"]
    subs = ["lif", "izh"]
    n = len(task)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(n)
    wdt = 0.36
    for i, sn in enumerate(subs):
        fracs = [summary["per_profile"][p][sn]["mi_signif_frac"] for p in task]
        c = "#2c7fb8" if sn == "lif" else "#41ab5d"
        ax[0].bar(x + (i - 0.5) * wdt, fracs, width=wdt, label=sn.upper(), color=c)
    ax[0].axhline(0.95, ls="--", c="k", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(task, rotation=30, ha="right")
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("frac seeds MI p_perm <= 0.05")
    ax[0].set_title("Gate B2: neural beats its own paired null (MI)")
    for i, sn in enumerate(subs):
        zs = [summary["per_profile"][p][sn]["mi_z_mean"] for p in task]
        c = "#2c7fb8" if sn == "lif" else "#41ab5d"
        ax[1].bar(x + (i - 0.5) * wdt, zs, width=wdt, label=sn.upper(), color=c)
    ax[1].axhline(0, ls="--", c="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(task, rotation=30, ha="right")
    ax[1].set_ylabel("z = (obs - mean_null) / sd_null  (MI)")
    ax[1].set_title("Separation from the paired-null distribution")
    ax[1].legend()
    fig.suptitle("Gate B2: population paired null  -- VERDICT: %s" % summary["verdict"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gate_b2.png"), dpi=150)


if __name__ == "__main__":
    main()