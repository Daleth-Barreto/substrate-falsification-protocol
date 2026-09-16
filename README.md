# Does the substrate compute? — a falsification protocol

Closed-loop, information-theoretic validation that a neuromorphic substrate
carries a load-bearing control task through a Cortical-Labs-style API contract
(`cl-sdk` semantics), applied to a simulated humanoid. This repository is the
registry of a research line with three build phases, a surrogate experiment and
an empirical **falsification protocol** for the claim *"the substrate
computes"*.

Everything runs in simulation on CPU (no GPU training, no biological hardware,
no CL1 purchase).

## Research line

| phase | path | question |
|---|---|---|
| F1 - plant | `01_snn/` | Can a spiking stack (Nengo / NEF) replace the policy and the inner loop of a Unitree H1/G1 humanoid in MuJoCo, at least at a stable trot, without training RL from scratch? |
| F2 - contract | `02_cl/` | Can a control loop written against the Cortical Labs API contract (official `cl-sdk`, free, local) drive a simulated humanoid with an in-silico spiking substrate, inside real-time constraints? |
| F3 - union | `03_union/` | Does the full stack (contract + neuromorphic substrate) reach a sprint, and what does each layer measurably contribute? |
| protocol | `sandbox/` | **Does the substrate compute?** Four falsification gates (B–E) probe substrate material, readout robustness and memory with a matched dead-substrate null. |

## Thesis

A culture substrate (Cortical Labs CL1 / DishBrain) is not trained by gradient
descent; it is only conditioned slowly and closed-loop. The same function,
however, can be trained once in-silico as a spiking neural network and deployed
verbatim over the same contract (sensors -> electrodes -> spikes -> decisions ->
stimulation). The protocol turns this substitution into a load-bearing,
falsifiable experiment.

The central methodological claim: *"the substrate computes"* is **unfalsifiable**
unless (a) the substrate is swapped for a dead substrate **matched** in
single-neuron marginal firing statistics (mean rate, variance, silent fraction)
while being statistically independent of the task, and (b) the readout is itself
controlled.

## The four gates

| gate | control | question |
|---|---|---|
| B | substrate swap under a **fixed** readout | LIF / Izhikevich vs. matched Poisson (dead) null |
| C | readout-channel ablation | randomize the readout projection into the decision map |
| D | cross-validated ridge readout | re-decode the **same** spikes with a readout re-fit per fold |
| E | memory-demanding tasks | is task history linearly decodable from present-window population state? |

## Headline results (September 2026)

- **Closed-loop baseline.** The canonical surrogate battery (5 task profiles x 5
  modes x 6 seeds) confirms the neural Nengo-LIF hub separates from the dead
  Poisson substrate on every profile (exact paired permutation, one-sided p =
  1/64; Cohen d_z −7.8…−75.0 vs random, −2.4…−3.0 vs more conservative nulls).
  Baseline neural RMSE = 0.238 ± 0.010.
- **Gate B — matched dead null stays dead.** MI ≈ 0, TE = 0, ρ ≈ 0, every seed
  and profile: the null is task-independent **by construction** and its measures
  stay at the floor under *every* readout.
- **Gate D — the ridge readout splits living and dead.** Task information that
  survives readout randomization *and* re-optimization is carried by the
  substrate activity, not by the calibrated interface: LIF `ΔRMSE_floor =
  +0.094` over the mean-predictor floor (significant MI 24/30 runs); Izhikevich
  `ΔRMSE_floor = −0.008` — its apparent tracking is largely a **readout-channel
  artifact**.
- **Gate C — readout robustness is a stronger test.** Randomizing the readout
  collapses *both* living substrates to the dead floor (3/3 seeds).
- **Gate E — memory not certified.** No substrate shows linearly decodable task
  history in the present-window population state.
- **Authority envelope.** Lateral impulse frontier 0.40 s survived / 0.45 s fell
  at 140 N; the loop closes in 25 ms but cannot defend sustained floor loads.

## Reproducibility

- Every phase pins its environment (`requirements.lock.txt`, `uv freeze`).
- The seed protocol, task profiles and analysis pipeline are documented per
  module and in `docs/REPRODUCIBILITY.md`.
- Tables and figures of the method note are **regenerated from committed JSON
  summaries**, never hand-typed: `python sandbox/build_figures_tables.py`
  rebuilds `sandbox/paper/main.pdf` (see `sandbox/README.reproducible`).
- Third-party assets (Unitree RL Gym, MuJoCo model zoo, the G1 `motion.pt`
  checkpoint; about 2 GB) are fetched by the bootstrap script and are not
  redistributed here.
- The closed-loop companion battery (`surrogate_cl/`) lives in a separate
  repository and is consumed read-only here; its canonical numbers are cited
  with attribution, never re-run.

## Repository layout

```
01_snn/                  F1 - spiking plant (Nengo/NEF + MuJoCo walker)
02_cl/                   F2 - Cortical Labs API contract loop
03_union/                F3 - full stack (contract + substrate)
sandbox/                 falsification protocol (gates B/D/E, paper, videos)
docs/REPRODUCIBILITY.md  environment manifest, data provenance, hardware notes
scripts/                 Windows / POSIX bootstrap scripts
third_party/             fetched at bootstrap time (not tracked)
```

## License

MIT-licensed. Third-party dependencies and assets keep their own licenses:
`cl-sdk` is CC BY-NC (academic use), MuJoCo is Apache-2.0, `nengo` is MIT,
`m9h/bl1` is MIT, and the G1 assets follow Unitree RL Gym's terms. This project
does not cultivate biological cells and requires no CL1 hardware.

## Citation

If you use this work, please cite the protocol note:

```bibtex
@misc{barreto2026substrate,
  title        = {Does the substrate compute? A matched dead-substrate
                  falsification protocol},
  author       = {Hernández Barreto, Alan Daleth},
  year         = {2026},
  howpublished = {\url{https://github.com/Daleth-Barreto/substrate-falsification-protocol}},
  note         = {Manuscript in preparation}
}
```

## Status

The falsification protocol (gates B–E), its artifacts, tables and figures, and
the closed-loop validation battery are complete, versioned and reproducible from
this repository (regenerated from committed JSON summaries).