# F3 - Union: CL-aware neuromorphic sprint (work in progress)

**Question:** does the full stack (**Cortical Labs contract + neuromorphic substrate**) reach a humanoid sprint (>= 3 m/s in simulation) and beat PD and ANN-RL on energy, latency and robustness, with a documented upgrade path to a real CL1?

## The gap (state of the art)

- F1 shows a spiking substrate on the plant (trot).
- F2 shows the CL contract closing a loop (real-time contract + in-silico substrate).
- **Nobody has combined the two on legged humans:** no SNN sprint, no CL <-> MuJoCo closed loop, no per-layer ablation. This phase combines them and measures the ablation.

## The strong claim lives in the surrogate line

The publication-grade claim that the substrate *computes* is validated by the falsification protocol in the `sandbox/` module of this repository (gates B–E, matched dead-substrate null) and by the closed-loop companion battery (versioned separately in `surrogate_cl/`, consumed read-only). This phase stays focused on the substrate-interchangeability question:

- **Interchangeability contract test**: Nengo-LIF hub vs an Izhikevich substrate (`m9h/bl1`, JAX, CPU) vs a dead Poisson substrate, through the *same* contract, comparing spike statistics, tracking quality and information metrics. If BL-1 runs stably on CPU JAX, the claim is "substrate-interchangeable at the contract level (Nengo <-> Izhikevich <-> Poisson)".

## Stack

- Python 3.12+ contract venv (`cl-sdk`, `nengo`, `mujoco`, `torch` CPU) -- shared with F2.
- BL-1 venv (`bl1_venv`): `jax==0.11.1` (CPU), `numpy`, `scipy`, `matplotlib`, `nengo`.
- `m9h/bl1` vendored into `vendor/bl1/` (commit 91f7c01891ebae7f17190bf81a7b7b6fe3c0cf4e, MIT; see `vendor/bl1/VENDOR.txt`). Relevant API: `bl1.core.izhikevich` (populations + JIT step), `mea.stimulation`, `loop.decoding/encoding`, `monitor.activity`, `compat.cl_sdk`.

## Repository layout

```
03_union/
+- README.md
+- requirements-bl1.lock.txt     # BL-1 substrate environment (uv freeze)
+- bl1_venv/                     # isolated JAX(CPU) environment (git-ignored)
+- vendor/bl1/                   # vendored m9h/bl1 (vendored source + licence)
+- (src/)                        # substrate adapter (F4, in progress)
```

## Experiments / baselines (tabular for reviewers)

| config | vel. 30 m | CoT | falls | spikes/tick | latency tick->stim | est. energy/step |
|---|---|---|---|---|---|---|
| classic PD (baseline) | | | | - | - | - |
| ANN-RL (checkpoint; SPRINT/KSLC reported) | | | | - | - | - |
| SNN only (F1) | | | | | - | |
| CL only (contract + BL-1, F2) | | | | | | |
| **CL + SNN (union)** | | | | | | |

The central ablation is **what each layer contributes**: spiking = sparse / energy; contract = determinism / latency / upgrade.

## Open items

- substrate adapter (F4) and Gate B (BL-1 stable on CPU JAX)
- sprint >= 3 m/s on H1 or G1 (preliminary sample; 6 m/s needs RL training, out of GPU scope)
- ablation table + figures (trajectories, spike rasters, latency histogram)
- video <= 180 s

## Close neighbours (research consolidation, September 2026)

- Nobody has closed a full humanoid (G1, 23-DoF) loop through the CL contract.
- Biological legged baselines are non-neuronal (mycelia, Physarum) or hobby 1-10 DoF; nearest neighbours are H1 NEF+SPA (arXiv:2606.11034) and the Nengo+Loihi Jaco arm (arXiv:2007.10227).
- The "non-trainable CL substrate <-> trainable SNN surrogate" line (train once in-silico, run verbatim on wetware through the same contract) is substantiated by doom-neuron, Assembloid Agency (backend-agnostic, NeurIPS 2025) and BL-1 (differentiable virtual CL1 server). The reformulated contribution: the first humanoid loop with this interchangeability.

## Notes / ethics / upgrade path

- Simulation-only and quantified: no sim-to-real promise, everything is measured in simulation.
- The biocomputation claim is limited to *architectural compatibility with the CL contract*: the same code runs on CL1 / Cortical Cloud later as future validation. This is stated, not executed, in the paper.
- Licensing: `cl-sdk` is CC BY-NC (academic only); `m9h/bl1` is MIT; energy claims require counted memory plus fixed throughput/time-sparsity (NeuroBench) and are not yet claimed.