# Gate E — Memory-requiring tasks (negative result)

Follow-up to Gate D. Question: do the substrates implement SHORT-TERM MEMORY
in the open-loop mini? If the dead null fails hard when the task requires
integration/history while LIF/IZH track, we get a functional (not just
statistical) substrate argument.

## Protocol (same harness as Gate B/D)

Three task profiles where vref depends on the HISTORY of u, not on the
current window:

- **lag**  : answer of window w = input of window w-4 (delayed response)
- **integ**: vref = 6-window leaky integral of (u - 0.5)
- **stair**: vref steps up once per upward pulse crossing of u

For every (profile, seed, substrate) (3 x 6 x 3 = 54 runs):

- **canon**  : fixed Gate B readout (memoryless by construction)
- **meanpd** : predict the run-mean of vref (floor)
- **ridge**  : best L2-regularized LINEAR readout of PRESENT-window spike
  counts only (NO lagged spike features). To track these tasks the substrate
  must transpose the past into its present-window spikes via membrane +
  recurrent state. Dead null has no state, so it cannot.
- **passthru**: best ridge on INPUT taps u(w)..u(w-8) -> vref. The trivial
  "no substrate" bound: what a fixed FIR filter on the input alone achieves.

carried info = meanpd.rmse - ridge.rmse (how much the readout beats the floor).

## Pooled results (18 runs per substrate)

| substrate | canon RMSE | ridge RMSE | meanpd RMSE | carried info | ridge sig MI frac |
|-----------|-----------|-----------|-------------|--------------|-------------------|
| LIF       | 0.328      | 0.298      | 0.167       | **−0.131**   | 0.28              |
| IZH       | 0.419      | 0.229      | 0.167       | **−0.061**   | 0.17              |
| POISSON   | 0.422      | 0.173      | 0.167       | **−0.006**   | 0.06              |
| input FIR | —          | 0.088      | 0.167       | **+0.079**   | —                 |

Per-profile carried: lag  LIF −0.139 / IZH −0.055 / POI −0.016 / FIR +0.152;
integ LIF −0.114 / IZH −0.104 / POI −0.012 / FIR +0.144;
stair LIF −0.141 / IZH −0.025 / POI +0.010 / FIR −0.016.

Details: `results/gate_e_summary.json`, `results/gate_e.png`,
`results/raw_<profile>_s<seed>.json`.

## Findings — this is a NEGATIVE result, and that is the honest answer

1. **The dead null stays dead even when the task requires memory** (carried
   −0.006 ≈ floor, sig MI 1/18). Poisson cannot manufacture task information
   from a memory requirement.
2. **But the neural substrates do NOT transpose the past into present spikes
   either.** Under the optimal present-only linear readout, LIF/IZH carry
   NEGATIVE info vs the mean floor on all three memory tasks. Their
   present-window rate code is dominated by the CURRENT input drive; membrane
   leak (tau ~2 windows) and the mini's recurrent wiring (rec_scale 0.002)
   are too weak to hold history.
3. **A trivial FIR on the input dominates everything** (carried +0.079,
   ridge RMSE 0.088 vs 0.228-0.298 for the substrates). Where vref can be
   reconstructed from the last 8 input taps alone, a fixed linear filter
   removes any need for the substrate — exactly the "is it the readout/does
   the substrate matter" concern this gate was designed to answer.
4. **Recurrence has no effect at all** (recurrence scale 1x and 5x give the
   same carried ≈ −0.15). The mini is effectively feed-forward.

## Consequence for the research line (do NOT over-claim)

- The **mini is a decoder/readout diagnostic, not a place to demonstrate
  substrate memory**. Any claim of "the substrate contributes computation" must
  rest on the CLOSED-LOOP canonical battery (where the loop itself and
  time-locked dynamics carry TE and stability), NOT on this open-loop
  command-tracking mini.
- The methodology conclusions that DO survive: (a) matched dead-substrate null
  is exactly zero-informative under memory demands; (b) LIF's rate code
  genuinely carries instantaneous task info (Gate D carried +0.103 vs null);
  (c) fixed readout leaves information on the table.
- **Next: Gate E suggests the real test belongs in the closed loop.** The
  sandbox runs open-loop only; the companion closed-loop battery (separate repo)
  is where a functional substrate test with memory-demanding profiles
  (perturbation, terrain) should land. In-sandbox, the decisive deliverable is
  the METHOD: decoder-invariant null + honest readout ablation (Gates B/D/E).

## Running

```
..\..\02_cl\.venv312\Scripts\python.exe run.py     # <1 min
```

numpy + matplotlib required (02_cl venv). Runtime figures excluded from git.