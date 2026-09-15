# Gate B - substrate interchangeability probe

Decisive experiment: does the *substrate* compute, holding wiring and decode
fixed? Built in the isolated `sandbox` lane of opencode agent `opencode/big-pickle`
(2026-09-14). Zero contact with the canonical experiment (`surrogate_cl/`) or any
peer file; canonical numbers are only read, never written.

## Protocol

Three substrates share the SAME input projection `g`, the SAME sparse recurrent
wiring `w` and the SAME readout `wr`; only the neuron dynamics differ:

- **LIF**: leaky integrate-and-fire (Euler, dt = 0.5 ms, tau = 20 ms,
  threshold 1, refractory 2 ms).
- **IZH**: Izhikevich regular spiking (a = 0.02, b = 0.2, c = -65, d = 8).
- **POISSON (dead null)**: per-neuron Poisson rates matched to the LIF mean
  firing, but INDEPENDENT of the task input.

The readout channel feeds the canonical calibrated decode, reused verbatim from
AGENTS.md of the research line:

    vx = clamp((thsig - 0.0615) / 0.277, 0, 0.75) * gate * slow,  gate = slow = 1

Fairness: the readout scale `k_ro` is calibrated ONCE on wiring seed 7 at a fixed
command u = 0.5 (th sig -> 0.20), then held fixed on all seeds and profiles. The
Poisson-null rates are matched to the LIF per-neuron means from that same
calibration run.

Battery: 5 task profiles (baseline, multi_step, sine, descend, pulse) x 6 seeds
{1, 7, 13, 29, 55, 91} x 3 substrates = 90 runs, 2 s each (200 control windows).

Metrics (mini analogues of Gate A):
- MI(command; task), binned, bias-corrected by permutation, two-sided p.
- TE(channel event series -> command), bias-corrected, two-sided p -
  INFORMATIONAL: the mini-model is feed-forward, so the canonical closed-loop TE
  is not expected to be reproduced.
- RMSE vs the calibrated reference V_ref = 0.75 * u, and tracking correlation rho.

## Decisive criterion

- neural_ok: LIF and IZH both carry the task: pooled MI > 0.05 and rho > 0.3.
- poisson_dead: pooled MI <= 0.01, |rho| <= 0.1, TE ~ 0.
- PASS if both, FAIL if either direction fails, INCONCLUSIVE otherwise.

## Result (pooled, 30 runs per substrate)

| substrate | RMSE (mean +/- SE) | MI (nats, +/- SE) | MI signif (p2<0.05) | TE   | rho |
|-----------|--------------------|-------------------|----------------------|------|------|
| LIF       | 0.168 +/- 0.008    | 0.651 +/- 0.067   | 24/30               | ~0   | 0.765|
| IZH       | 0.239 +/- 0.007    | 0.385 +/- 0.045   | 24/30               | ~0   | 0.633|
| POISSON   | 0.392 +/- 0.017    | 0.0004 +/- 0.0004 | 0/30                | 0.0  | 0.000|

VERDICT: **PASS** - with wiring and decode held constant, the two neural
substrates carry the task (MI 0.39-0.65, rho 0.63-0.77) while the dead Poisson
substrate carries exactly zero task information (MI ~ 0.0004, rho ~ 0.00,
TE = 0.0). The computation lives in the substrate dynamics, not in the decode.

Consistent with, and independent from, the canonical surrogate result: the neural
hub separates from null on all 5 profiles (one-sided p = 1/64), and the dead
Poisson substrate scores exact-zero MI/TE in the closed-loop experiment.

## Files

- `hub.py` - substrates, shared wiring, canonical `decide()`.
- `metrics.py` - MI and TE with permutation bias correction.
- `run.py` - calibration + battery + verdict + figures.
- `smoke.py` - single-run sanity check and parameter tuning aid.
- `results/gate_b_summary.json` - full per-profile, per-seed and pooled numbers.
- `results/gate_b.png` - pooled bar summary.

## Reproduce

    python run.py

requires Python >= 3.12 with numpy (verified: 3.12.13, numpy 2.5.3). No other
dependencies, no network, no writes outside this directory.

## Caveats

- Mini-model, feed-forward: it isolates substrate vs decode; it is NOT a
  reimplementation of the G1 closed-loop plant, so its TE is not comparable to
  the canonical closed-loop TE numbers.
- The readout is aligned with the input projection (channel reads the
  task-modulated neurons); a zero-mean random readout decorrelates all
  substrates identically and is the wrong null for this probe.
- Driver gains were tuned on seed 7 so both neural substrates fire in the
  5-30 Hz band; deltas between LIF and IZH (e.g. RMSE 0.168 vs 0.239) reflect
  gain matching, not a substrate ranking.