# 04_p2 - Second paper: robustness of the falsification protocol

Home of the follow-up to the methods note in `sandbox/paper/`.  The first paper
establishes the falsification protocol (gates B-E, matched dead-substrate null,
canonical multi-seed battery).  The second paper's core claim is stronger:

> Does the substrate's task-carried information survive a null that is matched
> far more tightly than a Poisson rate-match -- namely, the substrate's OWN
> spike trains with their population task alignment destroyed -- and does this
> hold under a null *suite*, a *false-positive calibration*, and a different
> readout?

All three tests pass.  The answer lives in `gate_b2/`, producing
`results/gate_b2v2_summary.json`.

## Result (B2v2, canonical for the second paper)

### Null suite: 3 paired-surrogate constructions (K=50 each)

| Construction    | Preserves                                  | Destroys                   |
|-----------------|--------------------------------------------|----------------------------|
| circshift       | per-neuron marginals + autocorrelation     | cross-neuron task phase     |
| blockshuffle    | per-neuron marginals + within-block smooth  | long-range + cross-neuron   |
| identityswap    | per-window population envelope exactly     | neuron-to-readout alignment |

### Pooled task-varying cells (54 per substrate per null type)

| substrate | circshift | blockshuffle | identityswap |
|-----------|-----------|--------------|--------------|
| LIF       | 54/54     | 54/54        | 54/54        |
| IZH       | 54/54     | 54/54        | 54/54        |

All 3 null types at 100% significant (p_perm <= 0.05).

### False-positive control (DRIVE_GAIN = 0, dead-drive)

| substrate | sine | noise | ramp |
|-----------|------|-------|------|
| LIF       | 0/6  | 0/6   | 0/6  |
| IZH       | 0/6  | 0/6   | 0/6  |

Zero false positives in 18 substrate evaluations.

### Readout invariance (fixed-weights ridge, sine, K=20)

| substrate | p_ridge (6/6 seeds) | MI_obs_ridge |
|-----------|----------------------|--------------|
| LIF       | 0.0476 (6/6)         | 0.61–0.82    |
| IZH       | 0.0476 (6/6)         | 0.60–0.70    |

Ridge weights trained once on real spikes, applied to circshift nulls; same
readout, different spike statistics.

### Per-profile (circshift, n=6 seeds)

| profile     | LIF | IZH |
|-------------|-----|-----|
| baseline    | 0/6 | 0/6 |
| multi_step  | 6/6 | 6/6 |
| sine        | 6/6 | 6/6 |
| descend     | 6/6 | 6/6 |
| pulse       | 6/6 | 6/6 |
| ramp        | 6/6 | 6/6 |
| perturb     | 6/6 | 6/6 |
| step_pulse  | 6/6 | 6/6 |
| chirp       | 6/6 | 6/6 |
| noise       | 6/6 | 6/6 |

**VERDICT: PASS** on all three criteria (null suite, FP control, readout
invariance).

## Contents

```
gate_b2/hub.py        dynamics, wiring and decide map (same as Gate B)
gate_b2/metrics.py    MI / TE metrics (same as Gate B)
gate_b2/nulls.py      paired-null generators: circshift, blockshuffle, identityswap
gate_b2/run2.py       B2v1 battery (circular-shift null only, 8 profiles)
gate_b2/run3.py       B2v2 battery (null suite + FP + ridge, 10 profiles)
gate_b2/results/      gate_b2_summary.json, gate_b2.png  (B2v1)
                      gate_b2v2_summary.json, gate_b2v2.png  (B2v2)
paper/                (draft of the second paper, in preparation)
```

## Reproduce

```powershell
# full B2v2 battery  (~20-30 min)
& "C:\Proyectos\papers\Icra2027\02_cl\.venv312\Scripts\python.exe" gate_b2\run3.py

# quick smoke test  (~2 min)
& "C:\Proyectos\papers\Icra2027\02_cl\.venv312\Scripts\python.exe" gate_b2\run3.py --smoke
```

Runs on CPU (numpy only).  Requires the project-level venv (`02_cl/.venv312`);
no third-party assets.

## Relation to the first paper

- Gate B (sandbox/): LIF/IZH separate from a **single** Poisson null matched on
  mean rates.  That null is degenerate by construction (input-independent).
- Gate B2v1 (run2.py): separates from the substrate's own circular-shift null
  (K=50), 42/42 task-varying cells.
- Gate B2v2 (run3.py): separates from **three** paired null constructions that
  preserve successively larger sets of structure, with zero false positives on
  dead-drive controls and readout invariance under fixed-weight ridge -- the
  strongest controls before a venue reader asks.
- Both keep the calibrated `decide` map and `k_ro` fixed (single calibration on
  seed 7, u=0.5); no numbers from the companion battery are rewritten here.
