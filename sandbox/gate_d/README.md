# Gate D — "Is it the readout?"

Same hub runs as Gate B (identical spikes), but the DECODER is now the variable.
Question: how much of what we measured is the readout, and how much is really
carried in the substrate's spikes?

## Protocol

For every `(profile, seed, substrate)` (5 x 6 x 3 = 90 runs) the SAME spike
counts produce four readouts:

- **canon**: `k_ro * (wr @ rates) -> decide()` — the Gate B fixed, calibrated
  readout (wr aligned with the input projection g).
- **meanpd**: predict the run-mean of `vref = 0.75*u` everywhere. The honest
  floor: any decoder that carries task information must beat this; the dead
  null should not.
- **ridge**: best L2-regularized LINEAR readout of the raw spike counts
  (all 2000 neurons), nested train/validation/test, interleaved splits so
  train and test see the same profile phases. The Woodbury identity keeps the
  fit O(n_train^3) instead of O(N^3). This is the upper bound any fixed linear
  decode could reach from these spikes.
- **passthru**: `xhat = 0.75*u` directly, i.e. no substrate at all. RMSE = 0 by
  construction — the honest ceiling for this open-loop mini (a clean command
  in a feed-forward setting does not need a substrate; the substrate's value is
  only in the closed/embodied setting, cf. the canonical battery).

Recurrence contribution is isolated by re-running LIF/IZH with `rec_scale = 0`
("nohub", recurrent wiring ignored), same canon readout.

Carried info (the decisive statistic) = `meanpd.rmse - ridge.rmse` pooled:
how much better the best linear readout is than predicting the mean.

## Pooled results (30 runs per substrate)

| substrate | canon RMSE | ridge RMSE | meanpd RMSE | carried info | ridge sig MI frac |
|-----------|-----------|-----------|-------------|--------------|-------------------|
| LIF       | 0.168      | 0.055      | 0.149       | **+0.094**   | 0.80              |
| IZH       | 0.239      | 0.157      | 0.149       | **−0.008**   | 0.77              |
| POISSON   | 0.392      | 0.158      | 0.149       | **−0.009**   | 0.03              |

Traces: `results/gate_d_summary.json`, `results/gate_d.png`,
`results/raw_<profile>_s<seed>.json`.

## Findings (the gate did its job)

1. **The dead-substrate null survives the best possible linear decoder.** Under
   ridge (upper bound), Poisson STILL tracks at the mean-predictor floor
   (carried −0.009, only 1/30 runs with significant MI). The Gate B negative
   control is not a decoder artifact: dead substrate = no information, even
   with an optimally tuned linear readout. Robust.
2. **LIF's substrate contribution is real and survives an optimal linear
   readout.** carried +0.094 ≫ null floor (Δ +0.103), ridge RMSE 0.055 below
   everything else. Not "just the readout".
3. **IZH's apparent Gate B tracking was substantially a READOUT-CHANNEL
   artifact.** Under the ridge upper bound, IZH is indistinguishable from the
   dead null on RMSE (0.157 vs 0.158, carried −0.008 ≈ −0.009). The Gate B
   IZH-vs-Poisson gap came from the fixed canonical readout channel (`wr`
   aligned with the input projection `g`), not from information the IZH spikes
   actually carry. IZH retains only weak MI (sigma 0.25 nats, significant in
   23/30).
4. **The fixed canonical readout leaves information on the table** for BOTH
   neural substrates (canon − ridge = +0.113 LIF, +0.081 IZH). A fixed `wr`
   channel is not the best weapon even when the substrate does the job.
5. **Recurrence is neutral here** (nohub ≈ canon to <0.001 RMSE for both). The
   mini's recurrent wiring adds nothing measureable; consistent with the
   caveat that this is a feed-forward command-tracking mini, not the closed
   loop (TE and loop-memory live in the canonical battery).

## Verdict for the research line

- The **negative-control methodology itself is validated**: the matched
  dead-substrate null gives exact-zero information under ANY decoder, fixed,
  mean, or optimal-linear. That is the publishable core.
- The **"substrate interchangeability" claim of Gate B is REVISED**: LIF and
  IZH do not interchange under the best linear readout — LIF genuinely carries
  the task, IZH mostly did not. Any publication based on this protocol must not
  claim that "any spiking substrate computes"; it must claim that LIF-type rate
  coding carries task information (Δcarried +0.103 vs null) while the dead null
  and a poorly-informative substrate do not, and that a fixed readout can both
  inflate (IZH) and mask (LIF vs ridge) the substrate's information.
- **Follow-up (Gate E)** tested a task that REQUIRES memory: the dead null
  stays dead, but the neural substrates also do NOT transpose past input into
  present spikes and a trivial input-FIR dominates. Result NEGATIVE (see
  gate_e/README.md): the mini cannot demonstrate functional substrate memory;
  that claim, if made at all, must live in the closed-loop canonical battery.

## Running

```
..\..\02_cl\.venv312\Scripts\python.exe run.py     # ~1 min on the laptop
```

Requires numpy + matplotlib (02_cl venv). Runtime figures excluded from git.