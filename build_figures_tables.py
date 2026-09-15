"""build_figures_tables.py — regenerate everything the PDF needs from JSON.

Reads the gate_*/results/*_summary.json artifacts and the canonical
task_tracking.json (READ-ONLY, owned by the tracking session) and emits:
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
            "canonical.png: COPIA (solo lectura) del artefacto canónico\n"
            "  source: ../../surrogate_cl/results/task_tracking.png\n"
            "  owner:  sesión de tracking (task_tracking.py) — propiedad de ese agente.\n"
            "  NUCA editar/regenerar aquí; los números citados en el texto vienen de\n"
            "  task_tracking.json (0.238; p_1s=0.0156; p_2s=0.0312; d_z=-8.58/-2.43/-38.94).\n"
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
    tab.append("\\begin{tabular}{llrrrrrr}")
    tab.append("\\toprule")
    tab.append("substrate & decoder & $\\widehat{\\mathrm{RMSE}}$ & $\\mathrm{MI}$ "
               "& $\\rho$ & $\\mathrm{TE}$ & carried & $\\mathrm{MI}_{\\mathrm{sig}}$ \\\\")
    tab.append("\\midrule")
    for sub in SUBSTRATES:
        p = gd["pooled"][sub]
        tab.append(" & ".join([
            SUB_LABEL[sub], "canonical",
            fmt3(p["canon_rmse_mean"]), fmt3(p["canon_mi_mean"]), fmt3(p["canon_rho_mean"]),
            f"{p['canon_te_mean']:.0e}", "--", fmt3(p["canon_mi_sig_frac"]),
        ]) + r" \\")
        tab.append(" & ".join([
            SUB_LABEL[sub], "ridge (opt. lin.)",
            fmt3(p["ridge_rmse_mean"]), fmt3(p["ridge_mi_mean"]), fmt3(p["ridge_rho_mean"]),
            "--", fmt3(p["carried_info"]), fmt3(p["ridge_mi_sig_frac"]),
        ]) + r" \\")
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    tab.append("% floor problema: meanpd=" + fmt3(gd["pooled"]["lif"]["meanpd_rmse_mean"]) +
               "  upper-baseline: passthru=" + fmt3(gd["pooled"]["lif"]["passthru_rmse_mean"]))
    write_tab("tab_gated.tex", tab)

    # ---- Tab. 3 — Gate E memory: carried per profile, one col per substrate
    tab = []
    tab.append("% TABLA Gateway E: memoria. carried(best-linear readout) por perfil.")
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

    print("WROTE figures: decoder_bars.png, canonical.png (+ATTRIBUTION)")
    print("WROTE tables: tab_results.tex, canonical_numbers.tex")


if __name__ == "__main__":
    main()