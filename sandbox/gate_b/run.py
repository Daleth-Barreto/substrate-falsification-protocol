"""Gate B battery: 5 task profiles x 6 seeds x 3 substrates, fixed decode."""

import json
import os

import numpy as np

import hub
import metrics

SEEDS = [1, 7, 13, 29, 55, 91]
CAL_SEED = 7
PROFILES = ["baseline", "multi_step", "sine", "descend", "pulse"]
N_WIN = hub.N_WIN
OUTDIR = os.path.join(os.path.dirname(__file__), "results")

# Substrate drive parameters (tuned with smoke.py so both substrates fire in
# the 5-30 Hz band and the channel tracks the command monotonically)
BASE_I = {"lif": 0.7, "izh": 1.0}
DRIVE_GAIN = {"lif": 1.0, "izh": 4.0}
REC_SCALE = {"lif": 0.002, "izh": 0.02}
C62_TH = 0.18          # channel crossing threshold for the event series


def make_profile(name):
    t = np.arange(N_WIN, dtype=float)
    if name == "baseline":
        return np.full(N_WIN, 0.62)
    if name == "multi_step":
        return np.where((t // 10) % 2 == 0, 0.25, 0.85)
    if name == "sine":
        return 0.5 + 0.3 * np.sin(2.0 * np.pi * t / 20.0)
    if name == "descend":
        return np.linspace(0.9, 0.1, N_WIN)
    if name == "pulse":
        return np.where((t % 10) < 3, 0.85, 0.30)
    raise ValueError(name)


def calibrate():
    """Fix k_ro per substrate and the Poisson-null rates, once (CAL_SEED)."""
    g, w, wr = hub.make_wiring(CAL_SEED)
    cal = {}
    u05 = np.full(N_WIN, 0.5)
    for name in ("lif", "izh"):
        sub = hub.LIF() if name == "lif" else hub.IZH()
        sub.reset(CAL_SEED)
        thsig, counts = hub.run_profile(
            sub, (g, w, wr), u05, BASE_I[name], DRIVE_GAIN[name], REC_SCALE[name], 1.0)
        c0 = (wr @ (counts.astype(np.float64) / hub.CW)).mean()
        cal[name + "_k_ro"] = 0.200 / c0
    # Poisson-null rates matched to the LIF mean firing (per-neuron per-tick)
    lif = hub.LIF()
    lif.reset(CAL_SEED)
    thsig, counts = hub.run_profile(
        lif, (g, w, wr), u05, BASE_I["lif"], DRIVE_GAIN["lif"], REC_SCALE["lif"], 1.0)
    cal["poisson_rate"] = (counts.sum(axis=1) / N_WIN / hub.CW).astype(np.float64).tolist()
    cal["poisson_mean_hz"] = float(np.mean(cal["poisson_rate"]) / (hub.DT_MS / 1000.0))
    cal["lif_mean_hz"] = cal["poisson_mean_hz"]
    # same k_ro for the dead substrate (its output is input-independent)
    poi = hub.PoissonNull(cal["poisson_rate"])
    poi.reset(CAL_SEED)
    thsig, counts = hub.run_profile(poi, (g, w, wr), u05, 0.0, 0.0, 0.0, 1.0)
    cal["poisson_k_ro"] = 0.200 / (wr @ (counts.astype(np.float64) / hub.CW)).mean()
    return cal


def run_one(name, u_win, seed, cal):
    g, w, wr = hub.make_wiring(seed)
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
        xhat = hub.decide(thsig)
        vref = 0.75 * u_win
        rmse = float(np.sqrt(np.mean((xhat - vref) ** 2)))
        mi, mi_p = metrics.mi_bias_corrected(xhat, u_win)
        s = (thsig[:-1] > C62_TH).astype(int)
        te, te_p = metrics.te_bias_corrected(s, xhat)
        rho = 0.0
        if np.std(xhat) > 1e-9 and np.std(u_win) > 1e-9:
            rho = float(np.corrcoef(xhat, u_win)[0, 1])
        rate_hz = float(counts.sum() / N_WIN / hub.CW / (hub.DT_MS / 1000.0))
        out[sn] = {"rmse": rmse, "mi": mi, "mi_p": mi_p, "te": te,
                   "te_p": te_p, "rho": rho, "rate_hz": rate_hz}
    return out


def main():
    cal = calibrate()
    profiles = {p: make_profile(p) for p in PROFILES}
    rows = {}
    for pname, u in profiles.items():
        for seed in SEEDS:
            rows.setdefault(pname, []).append([seed, run_one(pname, u, seed, cal)])

    summary = {"profiles": PROFILES, "seeds": SEEDS, "calibration": cal,
               "decide_map": {"thsig_base": hub.TH0, "thsig_gain": hub.K,
                              "cap": 0.75, "gate": hub.GATE, "slow": hub.SLOW},
               "per_profile": {}, "pooled": {}}
    subs_order = ["lif", "izh", "poisson"]
    for pname in PROFILES:
        pp = {}
        for sn in subs_order:
            vals = np.array([[r[1][sn][kk] for kk in
                              ("rmse", "mi", "te", "rho", "rate_hz")] for r in rows[pname]])
            te_p = np.array([r[1][sn]["te_p"] for r in rows[pname]])
            pp[sn] = {
                "rmse_mean": float(vals[:, 0].mean()), "rmse_se": float(vals[:, 0].std()/np.sqrt(len(vals))),
                "mi_msd": float(vals[:, 1].mean()), "mi_se": float(vals[:, 1].std()/np.sqrt(len(vals))),
                "te_mean": float(vals[:, 2].mean()), "te_se": float(vals[:, 2].std()/np.sqrt(len(vals))),
                "te_p_two": float(te_p.mean()), "te_p_two_l05": int((te_p < 0.05).sum()),
                "rho_mean": float(vals[:, 3].mean()), "rate_hz": float(vals[:, 4].mean())}
        summary["per_profile"][pname] = pp

    pooled = {}
    for sn in subs_order:
        te_p = np.array([r[1][sn]["te_p"] for r in [rr for pn in PROFILES for rr in rows[pn]]])
        rmse = np.array([r[1][sn]["rmse"] for pn in PROFILES for r in rows[pn]])
        mi = np.array([r[1][sn]["mi"] for pn in PROFILES for r in rows[pn]])
        mi_p = np.array([r[1][sn]["mi_p"] for pn in PROFILES for r in rows[pn]])
        te = np.array([r[1][sn]["te"] for pn in PROFILES for r in rows[pn]])
        rho = np.array([r[1][sn]["rho"] for pn in PROFILES for r in rows[pn]])
        pooled[sn] = {"rmse_mean": float(rmse.mean()), "rmse_se": float(rmse.std()/np.sqrt(len(rmse))),
                      "mi_msd": float(mi.mean()), "mi_se": float(mi.std()/np.sqrt(len(mi))),
                      "mi_signif_p2_l05": int((mi_p < 0.05).sum()),
                      "te_mean": float(te.mean()),
                      "te_signif_p2_l05": int((te_p < 0.05).sum()), "rho_mean": float(rho.mean())}
    summary["pooled"] = pooled

    # Decisive verdict. The discriminator is task-carried information (MI) and
    # tracking correlation; TE is reported informationally (the mini-model is
    # feed-forward, so the canonical closed-loop TE is not expected here).
    neural_ok = all(
        pooled[sn]["mi_msd"] > 0.05 and pooled[sn]["rho_mean"] > 0.3
        for sn in ("lif", "izh"))
    poisson_dead = (pooled["poisson"]["mi_msd"] <= 0.01
                    and abs(pooled["poisson"]["rho_mean"]) <= 0.1
                    and pooled["poisson"]["te_mean"] <= 1e-6)
    if neural_ok and poisson_dead:
        verdict = "PASS"
    elif poisson_dead and not neural_ok:
        verdict = "FAIL (neural substrates do not carry the task)"
    elif neural_ok and not poisson_dead:
        verdict = "FAIL (dead Poisson substrate carries the task)"
    else:
        verdict = "INCONCLUSIVE"
    summary["verdict"] = verdict
    summary["criteria"] = {"neural_ok": bool(neural_ok), "poisson_dead": bool(poisson_dead)}

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "gate_b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    try:
        _figure(summary)
    except Exception as exc:  # figures are auxiliary
        print("figure failed:", exc)
    print(json.dumps({k: summary[k] for k in ("pooled", "verdict", "criteria")}, indent=2))


def _figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    subs = ["lif", "izh", "poisson"]
    labels = ["LIF", "Izhikevich", "Poisson (dead)"]
    pooled = summary["pooled"]
    ax[0].bar(labels, [pooled[s]["rmse_mean"] for s in subs], yerr=[pooled[s]["rmse_se"] for s in subs],
              capsize=4, color=["#2c7fb8", "#41ab5d", "#fdb863"])
    ax[0].set_ylabel("RMSE vs v_ref"); ax[0].set_title("Tracking")
    ax[1].bar(labels, [pooled[s]["mi_msd"] for s in subs], color=["#2c7fb8", "#41ab5d", "#fdb863"])
    ax[1].set_ylabel("MI (nats)"); ax[1].set_title("MI(command; task)")
    for i, s in enumerate(subs):
        n = pooled[s]["te_signif_p2_l05"]
        ax[2].bar(labels[i], pooled[s]["te_mean"], color=["#2c7fb8", "#41ab5d", "#fdb863"][i])
        ax[2].text(labels[i], pooled[s]["te_mean"], f"{n}/30", ha="center", va="bottom", fontsize=8)
    ax[2].set_ylabel("TE (nats, bias-corrected)"); ax[2].set_title("TE(c62 -> cmd), p2<0.05 count")
    fig.suptitle(f"Gate B: substrate interchangeability probe -- VERDICT: {summary['verdict']}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gate_b.png"), dpi=150)


if __name__ == "__main__":
    main()