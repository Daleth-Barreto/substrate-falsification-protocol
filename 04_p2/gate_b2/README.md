# Gate B2 - population paired null (surrogate circular shift)

Follow-up to Gate B (`sandbox/gate_b/`). Gate B separated LIF/IZH from a single
Poisson null matched only on per-neuron mean rates. Gate B2 asks whether the
separation survives a **distribution** of nulls matched to the substrate's own
marginal and temporal statistics.

## Protocol

For every real run we keep the count matrix `C (N x N_WIN)`. Each of the K=50
null replicates shifts every neuron's count sequence by an independent random
lag (circular):

    C'_k[n, w] = C[n, (w - l_n[k]) mod N_WIN],   l_n ~ U{0, ..., N_WIN-1}

This preserves per neuron:
- the exact total spike count,
- the per-window count histogram (same multinomial marginals),
- the autocorrelation structure (the sequence is rotated, not resampled).

It destroys the cross-neuron, task-locked synchronization that the readout
`wr @ C` picks up. The readout, the calibrated `decide` map, and `k_ro` are
byte-identical to the real run (same seed, same wiring). K=50 paired nulls form
a per-run null distribution.

## Metrics

- MI(xhat; u): bias-corrected, as in Gate B.
- RMSE(xhat; 0.75 u), tracking rho, TE informational.
- For each metric a one-sided p_perm = (1 + #{nulls >= obs}) / (K+1) and
  z = (obs - mean_null) / sd_null (RMSE flipped: nulls LOWER than obs counts).

## Battery and verdict

8 task profiles (baseline, multi_step, sine, descend, pulse, ramp, perturb,
step_pulse) x 6 seeds {1,7,13,29,55,91} x 2 substrates (LIF, IZH) = 96 real
runs, each with K=50 nulls = 4,800 null replicates.

- PASS requires LIF MI p_perm <= 0.05 on >= 95% of task-varying cells and >= 80%
  of seeds per task profile, with the baseline (constant command) control at 0/6.

## Result (canonical for the second paper)

| substrate | MI cells p<=0.05 (n=42) | mean z | RMSE cells p<=0.05 (n=42) | baseline MI z |
|-----------|-------------------------|--------|----------------------------|----------------|
| LIF       | 42/42                   | 17.8   | 42/42                      | ~0.0 (0/6)    |
| IZH       | 42/42                   | 50.5   | 42/42                      | ~0.0 (0/6)    |

Per-profile z: LIF 11.1-30.5, IZH 24.4-74.7; 6/6 seeds significant on every
task-varying profile. VERDICT: **PASS** -- the substrate separates from a null
that keeps every single-neuron marginal and temporal statistic.

## Reproduce

```powershell
python run2.py            # full battery
python run2.py --smoke    # quick
```

Outputs `results/gate_b2_summary.json` and `results/gate_b2.png`.