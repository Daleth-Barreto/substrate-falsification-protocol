# Gate B2v2 - Paired population null with null-suite, FP control, readout invariance

Extended validation of the substrate-carried task information claim, prepared for
a methods-level submission (IEEE TNNLS).  Three independent controls rule out
concealed confounds that could explain Gate B's separation:

1. **Null suite** -- three paired-surrogate constructions preserving different
   marginals, each K=50 per run.
2. **False-positive control** -- same pipeline applied to a substrate receiving
   *no task signal* (DRIVE_GAIN = 0), verifying the test does not reject where
   it should not.
3. **Readout invariance** -- separation is recomputed under the cross-validated
   *fixed-weights ridge* readout, ruling out the possibility that only one
   decoder benefits.

## Null suite (three paired surrogate constructions, per-run)

All three use the same real count matrix `C (N x N_WIN)`.  They differ in
what they preserve and what they destroy.

### circshift (per-neuron circular shift)

    C'_k[n, w] = C[n, (w - l_n[k]) mod N_WIN],   l_n ~ U{0..N_WIN-1}

Preserves per neuron: exact total count, per-window histogram, and full
autocorrelation.  Destroys cross-neuron, task-locked temporal alignment.

### blockshuffle (per-neuron block shuffle)

Split each neuron's count sequence into B = 5 window blocks, then randomly
shuffle the block order.  Preserves per-neuron marginals and within-block
short-range dynamics; destroys long-range temporal structure and cross-neuron
phase alignment across blocks.

### identityswap (per-window neuron permutation)

For each time window, randomly permute the neuron identity of the N counts.
Preserves the per-window population count envelope *exactly* (the sum over all
neurons per window is unchanged), so any global mean-activity modulation by the
task survives.  Destroys the neuron-to-readout weight alignment `wr ~ g` that
the canonical decode exploits.

If all three null types collapse the readout's MI while identityswap
preserves the population envelope, the information depends on *neuron-specific
recruitment* under task control, not on a trivial mean-rate modulation.

## False-positive control

Same pipeline applied to a **dead-drive** substrate: DRIVE_GAIN = 0, REC_SCALE =
0.0 (LIF base current only, no task-coupled input; IZH similarly).  The
substrate fires but receives no task information.  Under the circshift null the
MI is expected to be near zero for both observed and null, so p_perm ≈ 0.5
(never significant).  Any significant cell here would be a false positive.

## Readout invariance (fixed-weights ridge)

A single cross-validated ridge readout (nested tr/va/te, λ over 13 values,
same kernel trick as Gate D) is trained **once** on the real counts.  Those
fixed weights are then applied to the circshift null count matrices -- same
readout, different spike statistics.  This mirrors the canonical decode
comparison exactly: the only thing that changes is the spike timing alignment,
not the decoder.

MI under the fixed ridge is reported for each of the 6 seeds × 2 substrates
(sine profile) along with the per-cell p_perm.

## Metrics

- MI(xhat; u): bias-corrected (100 permutation null).
- RMSE(xhat; 0.75 u).
- rho: Pearson correlation(xhat; u).
- TE is reported for the observed run only (informational); it is not used for
  the null decision.
- One-sided p_perm = (1 + #{nulls >= obs}) / (K+1) for MI and rho;
  nulls LOWER than obs for RMSE.
- Cohen's d_z = (obs - mean_null) / sd_null (population effect size).

## Battery

### Main battery

10 profiles (baseline + 9 task: multi_step, sine, descend, pulse, ramp, perturb,
step_pulse, chirp, noise) x 6 seeds {1,7,13,29,55,91} x 2 substrates (LIF,
IZH) = 120 runs (110 task-varying).  Each real run x 3 null types x K=50 = 150
null replicates per run => 18,000 paired null evaluations.

### Controls

- False-positive control: 3 profiles (sine, noise, ramp) x 6 seeds x 2 substrates =
  36 dead-drive runs, each 1 null type x K=50 = 1,800 null evaluations.
- Readout invariance: sine profile x 6 seeds x 2 substrates = 12 cells, each 1
  null type x K=20 = 240 null evaluations (plus fixed ridge calibration).

## Result (canonical, B2v2)

### Pooled (all 3 null types, 54 task-varying cells each)

| substrate | circshift p<=0.05 | blockshuffle p<=0.05 | identityswap p<=0.05 |
|-----------|--------------------|-----------------------|-----------------------|
| LIF       | 54/54 (100%)       | 54/54 (100%)          | 54/54 (100%)          |
| IZH       | 54/54 (100%)       | 54/54 (100%)          | 54/54 (100%)          |

### False-positive control (dead-drive, 36 runs)

| substrate | sine | noise | ramp |
|-----------|------|-------|------|
| LIF       | 0/6  | 0/6   | 0/6  |
| IZH       | 0/6  | 0/6   | 0/6  |

Total: 0 false positives out of 18 substrate evaluations.

### Readout invariance (sine, fixed-weights ridge, K=20)

| substrate | p_ridge (6/6 seeds) |
|-----------|----------------------|
| LIF       | 0.0476 (6/6)         |
| IZH       | 0.0476 (6/6)         |

MI_obs_ridge: LIF 0.61–0.82, IZH 0.60–0.70.  Ridge MI is the cross-validated
MI of the fixed readout applied to real held-out spikes; the null MI
distribution under circshift collapses to ~0.10 (LIF) / ~0.08 (IZH).

### Per-profile (circshift, n=6 seeds)

| profile     | LIF sig | IZH sig |
|-------------|---------|---------|
| baseline    | 0/6     | 0/6     |
| multi_step  | 6/6     | 6/6     |
| sine        | 6/6     | 6/6     |
| descend     | 6/6     | 6/6     |
| pulse       | 6/6     | 6/6     |
| ramp        | 6/6     | 6/6     |
| perturb     | 6/6     | 6/6     |
| step_pulse  | 6/6     | 6/6     |
| chirp       | 6/6     | 6/6     |
| noise       | 6/6     | 6/6     |

**VERDICT: PASS** on all three criteria (null suite, FP control, readout
invariance).

## Key numbers for the manuscript

- Task-varying cells per substrate per null type: 54/54 p<=0.05 (100%).
- False-positive rate: 0/18 across 3 profiles x 6 seeds x 2 substrates.
- Fixed-weight ridge invariance: 12/12 (sine), p=0.0476 (K=20 resolution).
- identityswap null preserves per-window population envelope exactly, yet MI
  collapses -- ruling out trivial mean-rate explanation.
- Separation holds across 2 novel adversarial profiles (chirp, noise) in
  addition to the original 8 from Gate B.

## Reproduce

```powershell
# full B2v2 battery  (~20-30 min on a laptop)
& "C:\Proyectos\papers\Icra2027\02_cl\.venv312\Scripts\python.exe" run3.py

# quick smoke test  (~2 min)
& "C:\Proyectos\papers\Icra2027\02_cl\.venv312\Scripts\python.exe" run3.py --smoke
```

Outputs `results/gate_b2v2_summary.json` and `results/gate_b2v2.png`.

## Files

| file | purpose |
|------|---------|
| `run2.py` | Gate B2v1 (circular-shift null only, baseline battery) |
| `run3.py` | Gate B2v2 (this file's battery) |
| `nulls.py` | Paired-null generators: circshift, blockshuffle, identityswap |
| `hub.py`  | Substrate simulators (LIF, IZH, PoissonNull) |
| `metrics.py` | MI, TE, bias-corrected permutation tests |
