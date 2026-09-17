"""verify_paper_numbers.py -- audit the numbers quoted in paper/main_v2.tex.

Every value that appears in the v2 manuscript (prose and generated tables)
is re-derived here from the JSON artifacts produced by the runners. If any
claim disagrees with its artifact the script prints FAIL and exits non-zero,
so the manuscript cannot silently drift from the data.

This script does NOT re-run any experiment and does NOT write to the
artifact directories; it is read-only with respect to results/.

Usage:
    ..\\..\\02_cl\\.venv312\\Scripts\\python.exe verify_paper_numbers.py
"""

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, "..")
REPO = os.path.join(SB, "..")

PATHS = {
    "gate_b": os.path.join(SB, "gate_b", "results", "gate_b_summary.json"),
    "gate_d": os.path.join(SB, "gate_d", "results", "gate_d_summary.json"),
    "mlp": os.path.join(SB, "gate_d", "results", "mlp", "mlp_summary.json"),
    "gate_d20": os.path.join(SB, "gate_d", "results20", "gate_d_summary.json"),
    "seeds20": os.path.join(SB, "gate_d", "results20", "gate_d20_seed_stats.json"),
    "gate_e": os.path.join(SB, "gate_e", "results", "gate_e_summary.json"),
    "wr": os.path.join(SB, "verify", "verify_wr_summary.json"),
    "gate_b2": os.path.join(REPO, "04_p2", "gate_b2", "results",
                            "gate_b2v2_summary.json"),
    "archb": os.path.join(SB, "gate_d", "results_archB", "archB_summary.json"),
}


def load():
    out = {}
    for k, p in PATHS.items():
        with open(p, encoding="utf-8") as f:
            out[k] = json.load(f)
    return out


CHECKS = []


def check(label, claimed, actual, tol):
    ok = abs(float(claimed) - float(actual)) <= tol
    CHECKS.append((ok, label, claimed, actual))
    return ok


def check_eq(label, claimed, actual):
    ok = claimed == actual
    CHECKS.append((ok, label, claimed, actual))
    return ok


def run(A):
    b, d, m, d20, s20, e, wr, b2 = (
        A["gate_b"], A["gate_d"], A["mlp"], A["gate_d20"],
        A["seeds20"], A["gate_e"], A["wr"], A["gate_b2"],
    )
    ab = A["archb"]

    # ---- Table tab_hist: matched-dead marginal -------------------------
    r = np.asarray(b["calibration"]["poisson_rate"], dtype=float)
    hz = r * 2000.0
    check("hist mean rate (Hz)", 8.091, hz.mean(), 5e-4)
    check("hist per-neuron SD (Hz)", 15.296, hz.std(), 5e-4)
    check("hist Fano", 28.916, (hz.std() ** 2) / hz.mean(), 5e-4)
    check("hist silent fraction", 0.740, float((r == 0).mean()), 5e-4)

    # ---- Gate B2: paired null suite -----------------------------------
    for nt in ("circshift", "blockshuffle", "identityswap"):
        for sub in ("lif", "izh"):
            n = b2["pooled"][sub][nt]["signif_l05_n"]
            tot = b2["pooled"][sub][nt]["n_cells"]
            check_eq(f"B2 {nt}/{sub} sig cells", 54, n)
            check_eq(f"B2 {nt}/{sub} total cells", 54, tot)
    fp = b2["false_positive_control"]["significant_cells"]
    fp_total = sum(v for prof in fp.values() for v in prof.values())
    check_eq("B2 FP control significant cells", 0, fp_total)
    check_eq("B2 FP control clean flag", True, b2["criteria"]["fp_control_clean"])
    inv = b2["ridge_invariance"]
    n_inv = len(inv["lif"]) + len(inv["izh"])
    check_eq("B2 fixed-weight invariance cells", 12, n_inv)
    for sub in ("lif", "izh"):
        for row in inv[sub]:
            check(f"B2 invariance p ({sub} s{row['seed']})",
                  0.0476, row["p_ridge"], 5e-5)

    # ---- Gate D: linear readout on the same spikes --------------------
    check("GateD LIF canon RMSE", 0.168, d["pooled"]["lif"]["canon_rmse_mean"], 5e-4)
    check("GateD LIF ridge RMSE", 0.055, d["pooled"]["lif"]["ridge_rmse_mean"], 5e-4)
    check("GateD LIF carried-info", 0.094, d["pooled"]["lif"]["carried_info"], 5e-4)
    check("GateD LIF ridge MI-sig", 0.800, d["pooled"]["lif"]["ridge_mi_sig_frac"], 5e-4)
    check("GateD LIF canon MI-sig", 0.800, d["pooled"]["lif"]["canon_mi_sig_frac"], 5e-4)
    check("GateD LIF canon-to-ridge gap", 0.113,
          d["pooled"]["lif"]["canon_rmse_mean"] - d["pooled"]["lif"]["ridge_rmse_mean"],
          5e-4)
    check("GateD IZH canon RMSE", 0.239, d["pooled"]["izh"]["canon_rmse_mean"], 5e-4)
    check("GateD IZH ridge RMSE", 0.157, d["pooled"]["izh"]["ridge_rmse_mean"], 5e-4)
    check("GateD IZH carried-info", -0.008, d["pooled"]["izh"]["carried_info"], 5e-4)
    check("GateD IZH ridge MI-sig", 0.767, d["pooled"]["izh"]["ridge_mi_sig_frac"], 5e-4)
    check("GateD dead-null ridge RMSE", 0.158, d["pooled"]["poisson"]["ridge_rmse_mean"], 5e-4)
    check("GateD dead-null carried-info", -0.009, d["pooled"]["poisson"]["carried_info"], 5e-4)
    check("GateD dead-null ridge MI-sig", 0.033,
          d["pooled"]["poisson"]["ridge_mi_sig_frac"], 5e-4)

    # ---- Gate D bis: nonlinear readout --------------------------------
    check("GateD-bis LIF linear carried", 0.094,
          m["pooled"]["lif"]["carried_info"], 5e-4)
    check("GateD-bis LIF MLP carried", 0.105,
          m["pooled"]["lif"]["carried_info_mlp"], 5e-4)
    check("GateD-bis IZH MLP carried", 0.012,
          m["pooled"]["izh"]["carried_info_mlp"], 5e-4)
    check("GateD-bis dead-null MLP carried", -0.009,
          m["pooled"]["poisson"]["carried_info_mlp"], 5e-4)
    check("GateD-bis LIF MLP MI-sig", 0.800,
          m["pooled"]["lif"]["mlp_mi_sig_frac"], 5e-4)
    check("GateD-bis IZH MLP MI-sig", 0.800,
          m["pooled"]["izh"]["mlp_mi_sig_frac"], 5e-4)
    check("GateD-bis dead-null MLP MI-sig", 0.033,
          m["pooled"]["poisson"]["mlp_mi_sig_frac"], 5e-4)

    # ---- Gate E: memory (negative) ------------------------------------
    check("GateE LIF carried-info", -0.131, e["pooled"]["lif"]["carried_info"], 5e-4)
    check("GateE IZH carried-info", -0.061, e["pooled"]["izh"]["carried_info"], 5e-4)
    check("GateE dead-null carried-info", -0.006,
          e["pooled"]["poisson"]["carried_info"], 5e-4)
    check("GateE input-FIR best", 0.152, e["criteria"]["best_carried_inputFIR"], 5e-4)

    # ---- Gate F: recurrence ablation ----------------------------------
    check("GateF n=6 LIF nohub", 0.1678, d["pooled"]["lif"]["nohub_rmse_mean"], 5e-5)
    check("GateF n=6 LIF canon", 0.1678, d["pooled"]["lif"]["canon_rmse_mean"], 5e-5)
    check("GateF n=6 IZH nohub", 0.2388, d["pooled"]["izh"]["nohub_rmse_mean"], 5e-5)
    check("GateF n=6 IZH canon", 0.2390, d["pooled"]["izh"]["canon_rmse_mean"], 5e-5)
    gap_lif = (d20["pooled"]["lif"]["nohub_rmse_mean"]
               - d20["pooled"]["lif"]["canon_rmse_mean"])
    gap_izh = (d20["pooled"]["izh"]["nohub_rmse_mean"]
               - d20["pooled"]["izh"]["canon_rmse_mean"])
    check("GateF n=20 LIF nohub-canon gap", -3.3e-5, gap_lif, 5e-6)
    check("GateF n=20 IZH nohub-canon gap", 3.5e-4, gap_izh, 5e-5)

    # ---- Seeds n=20: exact paired seed-level test ---------------------
    check("Seeds20 LIF effect vs null", 0.103,
          s20["lif"]["mean_effect_vs_poisson"], 5e-4)
    check("Seeds20 LIF perm p", 1.0e-5,
          s20["lif"]["paired_perm_p_one_sided"], 5e-7)
    check_eq("Seeds20 LIF seeds same direction", 20,
             s20["lif"]["seeds_same_direction"])
    check("Seeds20 IZH effect vs null", 0.008,
          s20["izh"]["mean_effect_vs_poisson"], 5e-4)
    check("Seeds20 IZH perm p", 0.00356,
          s20["izh"]["paired_perm_p_one_sided"], 5e-6)
    check_eq("Seeds20 IZH seeds same direction", 16,
             s20["izh"]["seeds_same_direction"])
    check("Seeds20 pooled LIF canon", 0.167,
          d20["pooled"]["lif"]["canon_rmse_mean"], 5e-4)
    check("Seeds20 pooled LIF ridge", 0.058,
          d20["pooled"]["lif"]["ridge_rmse_mean"], 5e-4)
    check("Seeds20 pooled IZH canon", 0.239,
          d20["pooled"]["izh"]["canon_rmse_mean"], 5e-4)
    check("Seeds20 pooled IZH ridge", 0.153,
          d20["pooled"]["izh"]["ridge_rmse_mean"], 5e-4)
    check("Seeds20 dead-null ridge MI-sig", 0.05,
          d20["pooled"]["poisson"]["ridge_mi_sig_frac"], 5e-3)

    # ---- Gate C: readout-channel ablation -----------------------------
    rows = wr["per_profile"]
    profiles = []
    for row in rows:
        if row["profile"] not in profiles:
            profiles.append(row["profile"])
    aligned, random_ = [], []
    for prof in profiles:
        pr = [r_ for r_ in rows if r_["profile"] == prof]
        for sub in ("lif", "izh"):
            aligned.append(np.mean([r_[sub]["aligned"] for r_ in pr]))
            random_.append(np.mean([r_[sub]["random"] for r_ in pr]))
    check("GateC aligned RMSE lower bound", 0.12, min(aligned), 5e-3)
    check("GateC aligned RMSE upper bound", 0.30, max(aligned), 5e-3)
    check("GateC random RMSE lower bound", 0.36, min(random_), 5e-3)
    check("GateC random RMSE upper bound", 0.41, max(random_), 5e-3)
    for row in rows:
        check(f"GateC dead-null delta ({row['profile']} s{row['seed']})",
              0.0, row["poisson"]["delta"], 5e-4)
        check_eq(f"GateC collapse direction ({row['profile']} s{row['seed']})",
                 True, row["lif"]["delta"] > 0 and row["izh"]["delta"] > 0)

    # ---- Gate G: architecture robustness (second hub) ----------------
    aB, bB = ab["architecture_A"], ab["architecture_B"]
    check_eq("GateG archB N", 4000, bB["N"])
    check_eq("GateG archB p_re", 0.02, bB["P_RE"])
    check("GateG archB LIF carried", 0.101, bB["lif_carried_info"], 5e-4)
    check("GateG archB IZH carried", 0.000, bB["izh_carried_info"], 5e-4)
    check("GateG archB null floor", -0.004, bB["null_carried_info"], 5e-4)
    check("GateG archB LIF info-over-null", 0.104,
          bB["lif_carried_info"] - bB["null_carried_info"], 5e-4)
    check("GateG archB IZH info-over-null", 0.004,
          bB["izh_carried_info"] - bB["null_carried_info"], 5e-4)
    check("GateG archA LIF info-over-null", 0.103,
          aB["lif_carried_info"] - aB["null_carried_info"], 5e-4)
    check("GateG archA IZH info-over-null", 0.001,
          aB["izh_carried_info"] - aB["null_carried_info"], 5e-4)
    check("GateG archB LIF canon RMSE", 0.165, bB["canon_rmse_lif"], 5e-4)
    check("GateG archB IZH canon RMSE", 0.242, bB["canon_rmse_izh"], 5e-4)
    check_eq("GateG ranking reproduces", True,
             ab["criteria_B"]["ranking_reproduces"])

    # ---- NC port consistency (main_v2.tex -> main_nc_v2.tex) ----------
    p_v2 = os.path.join(SB, "paper", "main_v2.tex")
    p_nc = os.path.join(SB, "paper", "main_nc_v2.tex")
    with open(p_v2, encoding="utf-8") as f:
        v2 = f.read()
    with open(p_nc, encoding="utf-8") as f:
        nc = f.read()

    def labels(s):
        return sorted(set(re.findall(r"\\label\{([^}]+)\}", s)))

    check_eq("NC port keeps all labels", labels(v2), labels(nc))
    check_eq("NC port has no \\cite", 0, nc.count("\\cite{"))
    check_eq("NC port has no \\path", 0, nc.count("\\path{"))
    check_eq("NC port has no inlined \\input", 0, nc.count("\\input{tables/"))
    check_eq("NC port has no numbered bibliography", 0,
             nc.count("\\begin{thebibliography}"))
    sys.path.insert(0, os.path.join(SB, "paper"))
    import make_nc_v2
    bib_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", v2))
    check_eq("NC port cite-map covers all bibitems",
             sorted(bib_keys), sorted(make_nc_v2.CITE))
    check_eq("NC port reference count matches bibitems",
             len(bib_keys), len(make_nc_v2.REFS))

    # ---- verdicts ------------------------------------------------------
    check_eq("Gate B verdict", "PASS", b["verdict"])
    check_eq("Gate B2 verdict", "PASS", b2["verdict"])


def main():
    A = load()
    run(A)
    n_fail = 0
    for ok, label, claimed, actual in CHECKS:
        status = "ok  " if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"[{status}] {label:44s} claimed={claimed!r:<12} actual={actual:.6g}"
              if isinstance(actual, float)
              else f"[{status}] {label:44s} claimed={claimed!r:<12} actual={actual!r}")
    print("-" * 78)
    print(f"VERIFY_PAPER_NUMBERS {'PASS' if n_fail == 0 else 'FAIL'} "
          f"({len(CHECKS) - n_fail}/{len(CHECKS)} checks passed)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
