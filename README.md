# F1 - Neuromorphic locomotion on the plant (Unitree H1 / G1, MuJoCo)

**Question:** can a spiking stack (Nengo / NEF) replace the high-level policy *and* the inner loop (PD / PID) of a humanoid (Unitree H1 / G1) in MuJoCo, reaching at least a stable trot, without training RL from scratch?

## Why (state of the art)

- SNN walking on H1 exists (ICONS 2026, Nengo/NEF, flat ground). An SNN sprint does not exist.
- Replacing a PID with spikes exists for drones/arms (Stroobants 2022, 93-neuron Loihi PID; spiking-PID NEF 2024), never for a running humanoid inner loop.
- Public H1/G1 checkpoints (`unitree_rl_lab`, `legged_gym`) allow ANN-to-SNN conversion without a training GPU.

## Out of scope

Nothing about Cortical Labs (that is F2/F3). No PPO training from scratch. No BL-1 (F2).

## Target architecture

```
MuJoCo (H1 or G1, MJCF from mujoco_menagerie) @ 200 Hz
   | state (69-dimensional, H1 flat-terrain style)
   v
SNN policy (Nengo/NEF, 50-100 Hz) -- conversion of the ANN checkpoint
   | joint targets
   v
Spiking inner loop 1 kHz -- spiking PD/PID per channel (Stroobants-style)
   | torques
   v
MuJoCo
```

## Stack (laptop-compatible, CPU)

- Python 3.11.9 (does not require 3.12 in this phase)
- `mujoco` (CPU) + MJCF assets from the model zoo (see `scripts/bootstrap.ps1`)
- `nengo` (CPU) -- NEF, LIF ensembles, least-squares decoders
- Public checkpoint: `unitree_rl_lab` (H1/G1) or Hugging Face export

## Repository layout

```
01_snn/
+- src/
|  +- deploy12.py        # Deploy12 G1 loader (LSTM + PD), unitree_rl_gym deployment
|  +- mujco_env.py       # MuJoCo environment wrapper (menagerie MJCF), perturb, reset
|  +- ann_checkpoint.py  # ANN checkpoint loading and conversion entry points
|  +- convert.py         # ANN-to-SNN conversion (NEF: max+scale, ELU-to-LIF ensembles)
|  +- spiking_pid.py     # 1 kHz spiking inner loop
|  +- run_eval.py        # runner: PD baseline vs spiking stack
+- scripts/              # diagnostics, probes, ONNX export, figures
+- exported/             # converted/exported checkpoints and datasets (committed, small)
+- results/              # summary.json, figures, videos
+- README.md
```

## Experiments / metrics

| ID | experiment | metric |
|---|---|---|
| F1-E1 | velocity command 0 -> 2.5 m/s on flat ground (H1 and G1) | mean/max velocity, distance 30 m, falls |
| F1-E2 | perturbation recovery (50-100 N push) | distance after push, stabilisation time |
| F1-E3 | robustness: sensor noise, 5 deg slope | falls per episode |
| F1-E4 | energetic cost of the spiking stack | spikes/tick, estimated energy per step |

Baselines: pure PD (ANN targets, classic PD), original ANN policy, converted SNN.

## Status (measured, September 2026)

**Baseline validated** (`src/deploy12.py`, G1 12-DoF `motion.pt` + `configs/g1.yaml`): walks 12 s (mean vx 0.47 @ cmd 0.5, height about 0.77 m). The E1-E4 battery has been executed:

- E1 sprint: PD walks 18 s (mean vx 1.12, 8.08 m, no fall); spiking PD falls at 1.08 s (sprint is too hard for NEF at 50 Hz).
- E2 push: PD is stable for 10 s; spiking PD falls at 0.6 s.
- E3 robustness (noise): 18 PD/SPD runs x 3 noise levels x 3 seeds.
- E4 spiking compute (1000 steps): 230,168 spikes, 115,084 spikes/s, wall 104 s (overhead of `run_steps(1)`).

Evidence: `results/summary.json`, `results/figures/e1_e3_summary.png`, `results/videos/e1_pd.mp4` / `e1_spd.mp4`.

**SpikingPD characterised** (`src/spiking_pid.py`): in open loop the NEF decodes well at operational magnitudes (|tau| > 20 N-m: median ratio 0.996, correlation 0.999); slices/transforms do not degrade the decode. In closed loop at 50/10 Hz the spiking stack does not sustain fast gait (falls), so the spiking inner loop below the ANN policy does not yet replace a classic PD without timing work.

**ANN-to-SNN conversion** (`src/convert.py`, G1 policy = LSTM(64) + actor 64-32-12 ELU):
- Uniform NEF head: stage-1 NRMSE 0.47 -> 0.293 (1500 -> 12000 neurons, saturation at about 0.29).
- `SpikeHeadManifold` (decoders fitted on observed activity): action NRMSE 0.579 vs 1.05 for the uniform head.
- `ESNReader` (reservoir): rollout h relative-error median 0.004;
- The hybrid loop falls at about 8 s (end height 0.349, mean vx -0.258): small fidelity errors accumulate, a documented barrier in `results/snn_debug_results.json`.

## Relation to F2/F3

The fidelity findings and the NEF speed limits in the inner loop define which substrate can carry the CL contract. The F2 demo uses a *decision* Nengo-LIF hub at high speed rather than the full inner loop, and the surrogate line (`surrogate-cl`) uses the decision hub for the load-bearing task experiment.

## Reproducibility

Environment and pins: see the project-level `docs/REPRODUCIBILITY.md` and the phase lock files. Third-party assets are fetched by the project bootstrap script.

## Publication route

Neuromorphic workshop (ICRA NFR / ICONS) or a short control-conference paper; base material for the final F3 paper.