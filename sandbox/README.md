# Falsification protocol (Gates B–E)

Empirical evaluation harness for the claim *"the substrate computes"* in a minimal
embodied spiking hub (2,000 neurons, fixed sparse wiring, one calibrated decision
map). The protocol probes the substrate with four controls:

- **Gate B — substrate swap**: living substrates (LIF, Izhikevich) are compared
  against a dead substrate (Poisson) matched in single-neuron marginal firing
  statistics (mean rate, variance, silent fraction) but statistically independent
  of the task, under a fixed readout.
- **Gate C — readout-channel ablation**: the readout projection into the decision
  map is randomized, so apparent tracking that depended on readout alignment
  collapses toward the dead floor.
- **Gate D — cross-validated ridge readout**: the *same* spikes are re-decoded
  with a linear readout re-fit per fold; task information that survives readout
  randomization and re-optimization is carried by the substrate activity rather
  than by the calibrated interface.
- **Gate E — memory gate**: more memory-demanding tasks probe whether
  present-window population states carry linearly decodable task history.

## Layout

| Path | Contents |
|---|---|
| `gate_b/` | substrate-swap battery (living vs. matched dead null), fixes, results |
| `gate_d/` | cross-validated ridge readout battery, results, summary plots |
| `gate_e/` | memory-demanding task battery (FIR8 filter) |
| `verify/` | pipeline verification: physics, dead-substrate marginals, readout fits |
| `paper/` | manuscript (ICRA-format `main.tex`, Neural-Comp.-format `main_nc.tex`) |
| `video/` | open-loop robot playback and closed-loop screen captures |
| `build_figures_tables.py` | regenerates all manuscript tables/figures from result JSONs |
| `reproduce.py` | convenience driver to run the whole protocol end-to-end |
| `closed_loop_video.py` | real closed-loop capture (hub in loop with the MuJoCo walker) |

## Reproduce

See `README.reproducible` in this directory for the environment, the exact
command lines, and the expected runtime budget. Summary JSONs live next to each
gate's results and are the canonical source for the manuscript's numbers.

The closed-loop companion (`surrogate_cl/`) lives in a separate location and is
tracked in its own repository; this sandbox consumes it read-only through the
bridge defined in `closed_loop_video.py`.

## Results summary (canonical)

- Matched dead substrate: `MI ≈ 0`, `TE = 0`, `ρ ≈ 0`, all seeds and profiles —
  task-independent by construction, measures stay at the null floor.
- LIF (Gate D, pooled): ridge RMSE `0.055`, `ΔRMSE_floor = +0.094` vs. the
  mean-predictor floor; `MI` significant in `24/30` runs.
- Izhikevich (Gate D, pooled): ridge RMSE `0.157`, `ΔRMSE_floor = -0.008`;
  `MI` significant in `23/30` runs — the apparent tracking is largely a
  readout-channel artifact.
- Gate C: randomizing the readout collapses *both* living substrates to the dead
  floor (`3/3` seeds).
- Gate E: negative for all substrates — memory is not certified in the
  present-window population state.