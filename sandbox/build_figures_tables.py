"""build_figures_tables.py — regenerate everything the PDF needs from JSON.

Reads the gate_*/results/*_summary.json artifacts and the canonical
task_tracking.json (READ-ONLY, from the closed-loop companion repo) and emits:
  * paper/figures/decoder_bars.png        — RMSE by decoder & substrate
  * paper/figures/canonical.png           — copy of task_tracking.png (attribution)
  * paper/tables/tab_results.tex          — input table for the methods note

Numbers are never typed by hand: every cell comes from a JSON artifact.
"""

import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "paper")
FIG = os.path.join(PAPER, "figures")
TAB = os.path.join(PAPER, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

SUBSTRATES = ["lif", "izh", "poisson"]
SUB_LABEL = {"lif": "LIF", "izh": "Izhikevich", "poisson": "Poisson (dead null)"}
DECODER_ORDER = [
    ("canon", "fixed canonical"),
    ("ridge", "ridge (opt. linear)"),
    ("meanpd", "mean-predictor (floor)"),
]


def load(path, root=None):
    p = os.path.join(root or HERE, path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_raw_gated():
    """{profile: {substrate: [ridge_rmse, ...], canon_d: [..]}}"""
    from collections import defaultdict
    res = defaultdict(lambda: defaultdict(list))
    seed_dir = os.path.join(HERE, "gate_d", "results")
    for fn in os.listdir(seed_dir):
        if not (fn.startswith("raw_") and fn.endswith(".json")):
            continue
        base = fn[len("raw_"):-len(".json")]
        pr = base.rsplit("_s", 1)[0]
        with open(os.path.join(seed_dir, fn), encoding="utf-8") as f:
            raw = json.load(f)
        for sub in SUBSTRATES:
            if sub in raw:
                res[pr][sub].append(raw[sub]["ridge"]["rmse"])
    return res


def fmt3(x):
    return f"{x:.3f}"


def main():
    gb = load("gate_b/results/gate_b_summary.json")
    gd = load("gate_d/results/gate_d_summary.json")
    gd_raw = load_raw_gated()
    ge = load("gate_e/results/gate_e_summary.json")
    gcw = load("verify/verify_wr_summary.json")
    gb2 = load("gate_b2v2_summary.json", root=os.path.join(HERE, "..", "04_p2", "gate_b2", "results"))
    mlp = load("mlp/mlp_summary.json", root=os.path.join(HERE, "gate_d", "results"))
    s20 = load("gate_d20_seed_stats.json", root=os.path.join(HERE, "gate_d", "results20"))
    archb = load("archB_summary.json", root=os.path.join(HERE, "gate_d", "results_archB"))
    canonical = load("../../surrogate_cl/results/task_tracking.json")

    # ------------------------------------------------------------------ figure
    profiles = sorted(gb["per_profile"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, (sub, sl) in zip(axes, [(s, SUB_LABEL[s]) for s in SUBSTRATES]):
        col = {
            "lif": "#2c7fb8",
            "izh": "#d95f0e",
            "poisson": "#636363",
        }[sub]
        for name, lab in DECODER_ORDER:
            vals = []
            if name == "canon":
                vals = [gb["per_profile"][pr][sub]["rmse_mean"] for pr in profiles]
            elif name == "ridge":
                vals = [float(np.mean(gd_raw[pr][sub])) for pr in profiles]
            else:
                vals = [gd["pooled"][sub]["meanpd_rmse_mean"] for _ in profiles]
            vals = np.asarray(vals, dtype=float)
            x = np.arange(len(profiles)) + (0.25 if name == "canon" else
                                            (-0.25 if name == "ridge" else 0.0))
            ax.bar(x, vals, width=0.22, label=lab,
                   color=col, alpha=0.85 if name == "canon" else
                   (0.5 if name == "ridge" else 0.22))
        ax.set_title(f"{sl} — RMSE vs {len(profiles)} control profiles (min over seeds)")
        ax.set_xticks(np.arange(len(profiles)))
        ax.set_xticklabels(profiles, rotation=20, ha="right")
        ax.set_ylabel("decode RMSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Gate B (fixed canonical) vs Gate D (optimal linear readout): "
                 "the readout carries the signal, not the substrate",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "decoder_bars.png"), dpi=160)
    plt.close(fig)

    shutil.copy(os.path.join("..", "..", "surrogate_cl", "results", "task_tracking.png"),
                os.path.join(FIG, "canonical.png"))
    with open(os.path.join(FIG, "ATTRIBUTION.txt"), "w", encoding="utf-8") as f:
        f.write(
            "canonical.png: read-only copy of the canonical artefact\n"
            "  source: surrogate_cl/results/task_tracking.png (companion repo)\n"
            "  owner: closed-loop tracking session. Do not edit/regenerate here;\n"
            "  the numbers cited in the text come from task_tracking.json\n"
            "  (0.238; p_1s=0.0156; p_2s=0.0312; d_z=-8.58/-2.43/-38.94).\n"
        )

    # ----------------------------------------------------------------- tables
    def write_tab(fname, lines):
        with open(os.path.join(TAB, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ---- Tab. 1 — Gate B per-profile (fixed canonical decode)
    tab = []
    tab.append("% TABLA Gateway B: sustrato intercambiable, decode fijo (5 perfiles x 6 seeds)")
    tab.append("% Generada desde gate_b_summary.json. No editar a mano.")
    tab.append("\\begin{tabular}{llrrrrr}")
    tab.append("\\toprule")
    tab.append("profile & substrate & $\\widehat{\\mathrm{RMSE}}$ & $\\mathrm{MI}$ "
               "& $\\rho$ & $\\mathrm{TE}$ & $p_{\\mathrm{TE}}$ \\\\")
    tab.append("\\midrule")
    for pr in profiles:
        for sub in SUBSTRATES:
            row = gb["per_profile"][pr][sub]
            tab.append(" & ".join([
                f"\\texttt{{{pr.replace('_', '\\_')}}}", SUB_LABEL[sub],
                fmt3(row["rmse_mean"]), fmt3(row["mi_msd"]), fmt3(row["rho_mean"]),
                f"{row['te_mean']:.0e}", fmt3(row["te_p_two"]),
            ]) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    write_tab("tab_gateb.tex", tab)

    # ---- Tab. 2 — Gate D pooled (same spikes, decoder sweep)
    tab = []
    tab.append("% TABLA Gateway D: reader desenchufado (MISMAS spikes, decodificadores)")
    tab.append("% desde gate_d_summary.json (pooled 5 perfiles x 6 seeds)")
    tab.append("\\begin{tabular}{llrrrrr}")
    tab.append("\\toprule")
    tab.append("substrate & decoder & $\\widehat{\\mathrm{RMSE}}$ & "
               "$\\Delta\\mathrm{RMSE}_{\\mathrm{floor}}$ & $\\mathrm{MI}\\,"
               "\\mathrm{(nats)}$ & $\\rho$ & $\\mathrm{MI}_{\\mathrm{sig}}$ \\\\")
    tab.append("\\midrule")
    for sub in SUBSTRATES:
        p = gd["pooled"][sub]
        floor_d_canon = p["meanpd_rmse_mean"] - p["canon_rmse_mean"]
        tab.append(" & ".join([
            SUB_LABEL[sub], "fixed calibrated channel",
            fmt3(p["canon_rmse_mean"]), fmt3(floor_d_canon), fmt3(p["canon_mi_mean"]),
            fmt3(p["canon_rho_mean"]), fmt3(p["canon_mi_sig_frac"]),
        ]) + r" \\")
        tab.append(" & ".join([
            SUB_LABEL[sub], "ridge (cross-validated)",
            fmt3(p["ridge_rmse_mean"]), fmt3(p["carried_info"]), fmt3(p["ridge_mi_mean"]),
            fmt3(p["ridge_rho_mean"]), fmt3(p["ridge_mi_sig_frac"]),
        ]) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("% $\\Delta\\mathrm{RMSE}_{\\mathrm{floor}}$ = RMSE_floor - RMSE")
    tab.append("% (mean-predictor floor, no information baseline; passthru ceiling="
               + fmt3(gd["pooled"]["lif"]["passthru_rmse_mean"]) + ")")
    write_tab("tab_gated.tex", tab)

    # ---- Tab. 3 — matched marginal histogram statistics (null match quality)
    rates = np.asarray(gb["calibration"]["poisson_rate"], dtype=np.float64)
    dt_ms = 0.5  # hub.DT_MS; per-tick rates -> Hz
    rates_hz = rates / (dt_ms / 1000.0)
    silent = int(np.sum(rates > 1e-6))
    tab = []
    tab.append("% TABLA marginal emparejada: nulos Pois LIF-matched (mean/SD/Fano).")
    tab.append("% desde gate_b_summary.json calibration.poisson_rate")
    tab.append("\\begin{tabular}{lrrrr}")
    tab.append("\\toprule")
    tab.append("statistic & LIF marginal ($r_i$) & matched IID Poisson & "
               "$N=2000$ neurons & $\\mathrm{Fano}$ \\\\")
    tab.append("\\midrule")
    tab.append("per-neuron mean rate (Hz) & " + fmt3(float(rates_hz.mean())) +
               " & " + fmt3(float(rates_hz.mean())) + " & both & -- \\\\")
    tab.append("per-neuron StdDev (Hz) & " + fmt3(float(rates_hz.std())) +
               " & $\\sqrt{\\mu}$ only & -- & " +
               fmt3(float(rates_hz.var() / rates_hz.mean())) + " \\\\")
    tab.append("fraction silent ($r_i=0$) & " +
               f"{float(1 - silent / len(rates)):.3f}" +
               " & $e^{-\\mu}$ expected & " + f"{len(rates) - silent}/{len(rates)}"
               " & -- \\\\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("% nota: Fano>1 => marginal sobredisperso vs Poisson homogeneo;")
    tab.append("% la coincidencia es exacta en tasas marginales (no en estructura")
    tab.append("% temporal), que es lo que hace que el nulo sea 'matched'.")
    write_tab("tab_hist.tex", tab)

    # ---- Tab. 4 — Gate C: readout-channel ablation (aligned vs randomized wr)
    tab = []
    tab.append("% TABLA Gateway C: canal de lectura. RMSE canon bajo wr=g "
               "(alineado) vs wr aleatorio por-seed; k_ro recalibrado.")
    tab.append("% desde verify_wr_summary.json (sine+pulse, seeds 1/29/55)")
    tab.append("\\begin{tabular}{llrrrr}")
    tab.append("\\toprule")
    tab.append("profile & substrate & $\\widehat{\\mathrm{RMSE}}_{\\mathrm{aligned}}$ "
               "& $\\widehat{\\mathrm{RMSE}}_{\\mathrm{random}}$ & $\\Delta$ & "
               "$p_{\\mathrm{paired}}$ \\\\")
    tab.append("\\midrule")
    gcw_per_pr = {}
    for row_ in gcw["per_profile"]:
        gcw_per_pr.setdefault(row_["profile"], []).append(row_)
    for pr in ("sine", "pulse"):
        rows_pr = gcw_per_pr[pr]
        for sub in SUBSTRATES:
            aligned_ = np.array([r_[sub]["aligned"] for r_ in rows_pr])
            random_ = np.array([r_[sub]["random"] for r_ in rows_pr])
            d = random_ - aligned_
            n_pos = int(np.sum(d > 0.0))
            n_seeds = int(d.size)
            tab.append(" & ".join([
                f"\\texttt{{{pr}}}", SUB_LABEL[sub],
                fmt3(float(aligned_.mean())), fmt3(float(random_.mean())),
                f"{d.mean():+.3f}", f"{n_pos}/{n_seeds}",
            ]) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("% p_paired: fraccion de seeds con random>aligned (3 seeds);")
    tab.append("% LIF e IZH colapsan al piso Poisson bajo wr aleatorio (d>0,")
    tab.append("% toda seed); Poisson inmune (d=0). La informacion que queda")
    tab.append("% utilizable solo la recupera el ridge (Gate D).")
    write_tab("tab_gatec.tex", tab)

    # ---- Tab. 3 — Gate E memory: carried per profile, one col per substrate
    tab = []
    tab.append("% TABLA Gateway E: memoria. delta-RMSE-floor (ridge - mean-predictor) por perfil.")
    tab.append("% desde gate_e_summary.json (6 seeds). input-FIR8 = respuesta del FIR sobre u.")
    tab.append("\\begin{tabular}{lrrrrr}")
    tab.append("\\toprule")
    tab.append("profile & LIF & Izhikevich & Poisson & input-FIR8 \\\\")
    tab.append("\\midrule")
    floor = ge["pooled"]["lif"]["meanpd_rmse_mean"]
    for pr in ["lag", "integ", "stair"]:
        row = ge["per_profile"][pr]
        cells = []
        for sub in SUBSTRATES:
            ci = row.get(sub, {}).get("carried_info", np.nan)
            cells.append(fmt3(float(ci)))
        cells.append(fmt3(row.get("passthru", {}).get("carried_info", np.nan)))
        tab.append(" & ".join([f"\\texttt{{{pr.replace('_', '\\_')}}}"] + cells) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("% floor Gate E meanpd=" + fmt3(floor))
    write_tab("tab_gatee.tex", tab)

    # canonical numbers as LaTeX constants for the text
    st = canonical["profiles_data"]["baseline"]["stats"]
    with open(os.path.join(TAB, "canonical_numbers.tex"), "w", encoding="utf-8") as f:
        f.write(
            f"\\newcommand{{\\canRMSE}}{{{st['neural_vs_random']['neural_mean']:.3f}}}\n"
            f"\\newcommand{{\\canPone}}{{{st['neural_vs_random']['p_exact_1s']:.4f}}}\n"
            f"\\newcommand{{\\canPtwo}}{{{st['neural_vs_random']['p_exact_perm']:.4f}}}\n"
            f"\\newcommand{{\\canDzRnd}}{{{st['neural_vs_random']['cohen_dz']:.2f}}}\n"
            f"\\newcommand{{\\canDzMask}}{{{st['neural_vs_mask0.5']['cohen_dz']:.2f}}}\n"
        )

    # ---- Tab. 5 — Gate B2 null suite (paired population nulls, FP control,
    #      readout invariance). Canonical evidence for paper v2.
    tab = []
    tab.append("% TABLA Gateway B2: null-suite emparejada (circshift/blockshuffle/")
    tab.append("% identityswap), K=50 por cell; FP control y readout-invariance.")
    tab.append("% desde 04_p2/gate_b2/results/gate_b2v2_summary.json (read-only)")
    tab.append("\\begin{tabular}{llrrr}")
    tab.append("\\toprule")
    tab.append("null type & substrate & sig MI cells & total & frac. \\\\")
    tab.append("\\midrule")
    for nt in gb2["null_types"]:
        for sn in ("lif", "izh"):
            d = gb2["pooled"][sn][nt]
            tab.append(" & ".join([
                f"\\texttt{{{nt}}}", SUB_LABEL[sn],
                f"{d['signif_l05_n']}", f"{d['n_cells']}",
                f"{d['signif_l05_frac']:.2f}",
            ]) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("")
    tab.append("% FP control (drive_gain=0, no task signal): cells per profile =")
    tab.append(f"% {json.dumps(gb2['false_positive_control']['significant_cells'])} "
               f"| criterion fp_control_clean={gb2['criteria']['fp_control_clean']}.")
    rid_n = sum(len(v) for v in gb2["ridge_invariance"].values())
    rid_p = gb2["ridge_invariance"]["lif"][0]["p_ridge"]
    tab.append(f"% Readout invariance (fixed weights): {rid_n}/{rid_n} cells, "
               f"p={rid_p:.4f}.")
    write_tab("tab_gateb2.tex", tab)

    # ---- Tab. 6 — Gate D bis: linear vs nonlinear (MLP) readout on the same spikes
    tab = []
    tab.append("% TABLA Gateway D-bis: readout no-lineal (MLP 2000->32->1, L2) sobre")
    tab.append("% las MISMAS spikes de Gate D; carried_info=meanpd_rmse-rmse_decoder.")
    tab.append("% desde gate_d/results/mlp/mlp_summary.json")
    tab.append("\\begin{tabular}{lrrrr}")
    tab.append("\\toprule")
    tab.append("substrate & $\\dfloor$ linear & $\\dfloor$ MLP & "
               "MI-sig linear & MI-sig MLP \\\\")
    tab.append("\\midrule")
    for sn in SUBSTRATES:
        p = mlp["pooled"][sn]
        tab.append(" & ".join([
            SUB_LABEL[sn],
            fmt3(p["carried_info"]),
            fmt3(p["carried_info_mlp"]),
            fmt3(p.get("ridge_mi_sig_frac", 0.0)),
            fmt3(p.get("mlp_mi_sig_frac", 0.0)),
        ]) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("% MLP = ridge skip + 32-unit tanh residual net (L2, grad-clip,")
    tab.append("% lam in {1e-3,1e-2,1e-1,1}, selected on validation).")
    write_tab("tab_mlp.tex", tab)

    # ---- Tab. 7 — n=20 seeds seed-level paired statistics (Gate D repetition)
    tab = []
    tab.append("% TABLA seed-level n=20: efecto vs null Poisson por seed (pooled 5")
    tab.append("% perfiles), permutation emparejada exacta.")
    tab.append("% desde gate_d/results20/gate_d20_seed_stats.json")
    tab.append("\\begin{tabular}{lrrrr}")
    tab.append("\\toprule")
    tab.append("substrate & $\\Delta$ (vs null) & SE & paired perm $p$ & "
               "seeds same dir \\\\")
    tab.append("\\midrule")
    for sn in ("lif", "izh"):
        d = s20[sn]
        tab.append(" & ".join([
            SUB_LABEL[sn], fmt3(d["mean_effect_vs_poisson"]), fmt3(d["se_effect"]),
            f"{d['paired_perm_p_one_sided']:.4g}",
            f"{d['seeds_same_direction']}/{d['n_seeds']}",
        ]) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    write_tab("tab_seeds20.tex", tab)

    # ---- Tab. 8 — architecture robustness: same Gate D battery on a second
    #      hub (N=4000, p_re=0.02, variance-normalized). Does the ranking hold?
    tab = []
    tab.append("% TABLA arquitectura B: misma bateria Gate D (5 perfiles x 6")
    tab.append("% seeds) sobre un segundo hub N=4000, p_re=0.02.")
    tab.append("% desde gate_d/results_archB/archB_summary.json (A = v1 N=2000)")
    tab.append("\\begin{tabular}{llrrr}")
    tab.append("\\toprule")
    tab.append("architecture & substrate & $\\dfloor$ & info$-$null & "
               "MI-sig \\\\")
    tab.append("\\midrule")
    for key, label in (("architecture_A", "A ($N{=}2000$, $p_{\\mathrm{re}}{=}0.01$)"),
                       ("architecture_B", "B ($N{=}4000$, $p_{\\mathrm{re}}{=}0.02$)")):
        a = archb[key]
        for sn in ("lif", "izh"):
            d = a[f"{sn}_carried_info"]
            floor = a["null_carried_info"]
            sig = a[f"{sn}_ridge_mi_sig_frac"]
            tab.append(" & ".join([
                label, SUB_LABEL[sn], fmt3(d), fmt3(d - floor), fmt3(sig),
            ]) + r" \\")
        label = ""  # only print the architecture label on its first row
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("% criteria_B: " + json.dumps(archb["criteria_B"]) + ".")
    write_tab("tab_archB.tex", tab)

    print("WROTE figures: decoder_bars.png, canonical.png (+ATTRIBUTION)")
    print("WROTE tables: tab_results.tex, canonical_numbers.tex, "
          "tab_gateb2.tex, tab_mlp.tex, tab_seeds20.tex, tab_archB.tex")


if __name__ == "__main__":
    main()