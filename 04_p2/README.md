# 04_p2 - Second paper: robustness of the falsification protocol

Home of the follow-up to the methods note in `sandbox/paper/`. The first paper
establishes the falsification protocol (gates B-E, matched dead-substrate null,
canonical multi-seed battery). The second paper's core question is stronger:

> Does the substrate's task-carried information survive a null that is matched
> far more tightly than a Poisson rate-match -- namely, the substrate's OWN
> spike trains with their population task alignment destroyed?

The answer lives in `gate_b2/`, a **population paired null**: every neural run is
compared against a *distribution* of K=50 paired-null replicates built by
circular-shifting each neuron's real count sequence with its own random lag.
Per-neuron marginals and autocorrelation are preserved; only the synchronized,
task-locked population response is destroyed. Separation is then measured as a
z-score and a permutation p-value against that distribution.

## Status (results)

- Battery: 8 profiles (5 original + `ramp`, `perturb`, `step_pulse`) x 6 seeds
  x 2 substrates (LIF, IZH) = 96 real runs, each with K=50 paired nulls.
- VERDICT **PASS**:
  - LIF beats its own paired null on MI in 42/42 task-varying cells
    (mi_signif p_perm <= 0.05), mean z = 17.8 SD.
  - IZH beats its own paired null in 42/42 cells, mean z = 50.5 SD.
  - baseline (constant command, no task variation) shows 0/6 significant seeds
    for both substrates -- the negative control behaves.
- Per-profile: LIF z 11.1-30.5, IZH z 24.4-74.7, all 6/6 seeds significant on
  every task-varying profile.

## Contents

```
gate_b2/hub.py        dynamics, wiring and decide map (same as Gate B)
gate_b2/metrics.py    MI / TE metrics (same as Gate B)
gate_b2/run2.py       battery + paired-circular-shift null -> results/
gate_b2/results/      gate_b2_summary.json, gate_b2.png
paper/                (draft of the second paper, in preparation)
```

## Reproduce

```powershell
python gate_b2/run2.py            # full battery (8 x 6 x 2, K=50)
python gate_b2/run2.py --smoke    # quick: 2 profiles x 2 seeds, K=10
```

Runs on CPU (numpy only), a few minutes on the dev laptop for the full battery.
Requires the project-level venv (`02_cl/.venv312`); no third-party assets.
A second paper (`paper/`) will reuse the canonical read-only numbers from the
closed-loop companion battery.

## Relation to the first paper

- Gate B (sandbox/): LIF/IZH separate from a **single** Poisson null matched on
  mean rates. That null is degenerate by construction (input-independent).
- Gate B2 (this folder): separates from the **substrate's own** data. A null the
  substrate never gets to "cheat", and the strongest marginals control a
  readout-invariance protocol needs before the TNNLS-style venue reader asks.
- Both keep the calibrated `decide` map and `k_ro` fixed (single calibration on
  seed 7, u=0.5); no numbers from the companion battery are rewritten here.