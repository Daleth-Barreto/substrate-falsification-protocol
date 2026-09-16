# Literature consolidation — closed-loop neuromorphic control

Consolidation of three literature surveys (September 2026): SNN legged-locomotion
state of the art, the CL contract / closed-loop biocomputation, and neuromorphic
control / ANN-to-SNN conversion / energy.

> URL verification marker: `[verify]` = not confirmed in the survey session;
> confirm before citing.

---

## 1. Position

The closest prior work to the stack **G1 (23-DoF humanoid, MuJoCo) + Nengo/NEF
spiking policy + spiking inner-loop PD + CL1 contract (cl-sdk)**:

- **Unitree H1 NEF+SPA** (arm + locomotion, Nengo-Isaac Sim co-simulation) - ICONS 2026, arXiv:2606.11034.
- **Nengo+Loihi adaptive control of a Jaco 2 arm** (2.45x more accurate than PID, about 4.6x less power than CPU, 2.5-4.5 ms loop) - arXiv:2007.10227.
- **Cart-pole / cortical organoids** (real closed loop, 1-DoF) - Cell Reports 2026.
- **Pong (DishBrain) -> Doom (CL1)** - games, no actuation, no physical dynamics.

No public work combines the full stack above; the negative corpus sweep
(GitHub/arXiv/COSYNE/ICRA) should be re-run before any future claim.

---

## 2. Literature panorama (grouped map)

### 2.1 SNN policies for legged locomotion (2020-2026)

| # | Title | Source | Key claim (one line) | URL |
|---|---|---|---|---|
| a1 | Fully Spiking Neural Network for Legged Robots | Jiang et al., arXiv:2310.05022 (ICASSP 2025) | First SNN as a complete locomotion policy (A1/Cassie/MIT-Humanoid), trained end-to-end in Isaac Gym with a population-spiking actor; matches ANN on terrain tasks | https://arxiv.org/abs/2310.05022 |
| a2 | Proxy Target: Bridging Discrete SNNs and Continuous Control | Xu et al., arXiv:2505.24161 (NeurIPS 2025) | Proxy-stabilized soft-update; simple LIF SNN beats ANN on continuous benchmarks | https://arxiv.org/abs/2505.24161 |
| a3 | Synaptic Motor Adaptation (three-factor learning) | Schmidgall & Hays, arXiv:2306.01906 | Three-factor plasticity enables online adaptation (sim) matching RMA without privileged information | https://arxiv.org/abs/2306.01906 |
| a4 | Neuromorphic QP for MPC on ANYmal | Mangalore et al., arXiv:2401.14885 (IEEE RAM 2024) | Loihi 2 solves the MPC QP >100x better in energy-delay product vs CPU/GPU OSQP, <10 ms | https://arxiv.org/abs/2401.14885 |
| a5 | Astrocyte-modulated CPG hexapod on Loihi | Polykretis et al., ICONS 2020 (10.1145/3407197.3407205) | First spiking CPG with astrocyte dynamics on Loihi controlling a hexapod; robust to noise and speed changes | https://doi.org/10.1145/3407197.3407205 |
| a6 | Dopamine-modulated spiking CPG (multi-gait) | Torre et al., IEEE 2025 `[verify]` | Dopaminergic-neuromodulated spiking CPG produces several gaits and smooth transitions in a quadruped | https://ieeexplore.ieee.org/document/11338745 |
| a7 | 12-neuron spiking CPG (walk/jog/run) | Rostro-Gonzalez et al., Frontiers 2025 | Minimal 12-neuron spiking CPG yields walk/jog/run on a hexapod with smooth transitions | https://pmc.ncbi.nlm.nih.gov/articles/PMC12190837 |
| a8 | Astrocyte-regulated CPG + reward STDP | Han & Sengupta, IEEE TCDS 2026 (10.1109/TCDS.2025.3599472) | STDP + astrocyte dynamics learn a trot; 23.3x computational savings vs RL state of the art | https://doi.org/10.1109/TCDS.2025.3599472 |
| a9 | Integrated arm + locomotion on Unitree H1 | ICONS 2026, arXiv:2606.11034 | NEF/SPA bipedal locomotion + 4-DoF arm on H1 via Nengo-Isaac Sim with basal-ganglia selection | https://arxiv.org/abs/2606.11034 |
| a10 | Spiking CPG lamprey (SpiNNaker vs Loihi) | Bartolozzi et al., NCE 2022 (10.1088/2634-4386/ac1b76) | Spiking CPG (Nengo NEF) on two platforms; SpiNNaker better for real time, Loihi better for efficiency | https://doi.org/10.1088/2634-4386/ac1b76 |
| a11 | NeuroPod: SpiNNaker CPG hexapod | Gutierrez-Galan et al., Neural Networks 2020 (arXiv:1904.11243) | Spiking CPG on SpiNNaker with 3 gaits and online reconfiguration on a physical hexapod | https://arxiv.org/abs/1904.11243 |
| a12 | NEF/REACH 7-DoF arm on Loihi | DeWolf et al., NCE 2023 (10.1088/2634-4386/acb286) | Operational NEF control on Loihi: 4.13% RMSE vs analytical, about 100x less energy per inference vs GPU | https://doi.org/10.1088/2634-4386/acb286 |
| a13 | Nengo + low-power AI for embedded neurorobotics | Mayol et al., Front. Neurorobot. 2020 (arXiv:2007.10227) | Rover navigation + Jaco arm with on-chip PES on Loihi; full Nengo-to-Loihi flow | https://arxiv.org/abs/2007.10227 |
| a14 | Evolving connectivity for RSNN | Cheng et al., arXiv:2305.17650 | Connectivity evolution trains an RSNN on a 17-DoF humanoid to parity with deep RNNs | https://arxiv.org/abs/2305.17650 |
| a15 | Vision+control drone on Loihi (7-12 mW) | Paredes-Valles et al., Science Robotics 2024 (10.1126/scirobotics.adi0591) | End-to-end event-camera + SNN pipeline on Loihi: autonomous hover/landing at 7-12 mW | https://doi.org/10.1126/scirobotics.adi0591 |
| a16 | SpikeGym: SNN vs ANN in Isaac Gym | PMC11680704 (2024) `[verify]` | SNNs lag behind ANNs on deep networks (Ant task) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11680704 |
| a17 | Error amplification limits ANN-to-SNN conversion (control) | Xu et al., arXiv:2601.21778 (ICML 2026) | Conversion degrades severely in continuous control from temporally-correlated error amplification; CRPI mitigates | https://arxiv.org/abs/2601.21778 |
| a18 | Reconsidering SNN energy efficiency | Yan et al., arXiv:2409.08290 | SNNs only beat quantized ANNs above 93% sparsity (T=6); most comparisons ignore memory | https://arxiv.org/abs/2409.08290 |
| a19 | ANN vs SNN under resource constraints | Davidson et al., PMC8055931 (2021) | Most rate-coded SNNs on standard hardware are NOT more efficient than the original ANN | https://pmc.ncbi.nlm.nih.gov/articles/PMC8055931 |
| a20 | Trajectory generation on Loihi | Michaelis et al., Front. Neurorobot. 2020 (10.3389/fnbot.2020.589532) | Anisotropic network on Loihi stores/generalizes sequential motor trajectories | https://doi.org/10.3389/fnbot.2020.589532 |
| a21 | Spike-based RL hexapod (STDP) | Lele et al., AICAS 2020 (arXiv:2003.10026) | Online tripod gait learning with STDP from turning/camera input | https://arxiv.org/abs/2003.10026 |
| a22 | End-to-end model-based SNN control | Huebotter et al., NCE 2026 (arXiv:2509.05356) | Model-based spiking multi-DoF controller at parity with non-spiking baselines, fewer parameters | https://arxiv.org/abs/2509.05356 |

### 2.2 Biological closed loop / CL contract / biocomputation (2022-2026)

| # | Title | Source | Key claim (one line) | URL |
|---|---|---|---|---|
| b1 | **CL API: Real-Time Closed-Loop with Biological Neural Networks** | arXiv:2602.11632 (2026) | Defines the CL contract (transactional stim admission, deterministic ordering, explicit synchrony, sub-ms latency). **Anchor.** | https://arxiv.org/abs/2602.11632 |
| b2 | cl-sdk - CL API Simulator | GitHub Cortical-Labs/cl-sdk | Local replica of the CL1 API with custom SimulatorDataSources; pins the exact API version | https://github.com/Cortical-Labs/cl-sdk |
| b3 | cl.sim module docs | docs.corticallabs.com/cl/sim.html | Pull/push sources (`read(from_timestamp, frame_count)`), `on_stim()`, batch; defaults channel_count=64, fps=25k | https://docs.corticallabs.com/cl/sim.html |
| b4 | cl.Neurons/loop/accelerate docs | docs.corticallabs.com/cl.html | `loop()` yields spikes+stims up to 25 kHz; real CL1 jitter raises TimeoutError (not simulated); accelerated mode exists | https://docs.corticallabs.com/cl.html |
| b5 | DishBrain: neurons play Pong | Kagan et al., Neuron 110(23) 2022 | About 800 k neurons play Pong in ~5 min with structured (free-energy) feedback | https://doi.org/10.1016/j.neuron.2022.09.001 |
| b6 | DishBrain beats deep RL (sample efficiency) | Khajehnejad et al., arXiv:2405.16946; Cyborg & Bionic Syst. 6:0336 (2025) | DishBrain beats DQN/A2C/PPO in ~70 episodes (~5 min); dynamical plasticity | https://arxiv.org/abs/2405.16946 |
| b7 | CL1 plays Doom (Freedoom) | Cortical, Mar 2026 (Tom's Hardware and others) | ~200 k neurons, game state as electrical stimulation, spikes decoded to move/aim/fire; learns in ~1 week | https://www.tomshardware.com/tech-industry/artificial-intelligence/200-000-living-human-neurons-on-a-microchip-demonstrated-playing-doom-cortical-labs-cl1-video-shows-the-gameplay-and-explains-how-the-neurons-learn-the-game |
| b8 | doom-neuron repo (honest ablation) | SeanCole02/doom-neuron (GPL-3.0) | The device does not compute; PPO/encoder/decoder run in PyTorch over UDP; random/zero decoder ablations still play | https://github.com/SeanCole02/doom-neuron |
| b9 | Goal-directed learning in organoids (cart-pole) | Robbins et al., Cell Reports 45(2):116984 (2026) | Mouse cortical organoids on cart-pole (rate-coded); success 4.5% to 46% with adaptive RL; requires glutamate | https://www.cell.com/cell-reports/fulltext/S2211-1247(26)00062-8 |
| b10 | Organoid reservoir computing / OI surveys | Cai et al., Nat. Electronics 2023; Talavera & Ulmann, arXiv:2503.19770 | 3D organoid reservoirs for speech recognition and nonlinear prediction; OI surveys | https://www.nature.com/articles/s41928-023-01069-w |
| b11 | In-vitro BNNs for robot intelligence - review | Cyborg and Bionic Systems, 2024 (10.34133/cbsystems.0001) | Surveys BNN-on-MEA closed loops (Shahaf/Marom, DeMarse); 1-10 DoF loops, no commercial platform | https://spj.science.org/doi/10.34133/cbsystems.0001 |
| b12 | Neuron-connected robots (MSR + UTokyo) | Microsoft Research, 2022- | First step: simulate the neural substrate to avoid harming cultures - validates a sim-first pipeline | https://www.microsoft.com/en-us/research/project/neuron-connected-robots |
| b13 | Rat-hippocampus culture drives robot | PLOS ONE 11(10):e0165600 (2016) | Hippocampal cultures driving a mobile robot; honest report: bio-loop beaten by silicon baseline | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165600 |
| b14 | Fungal mycelia robot control | Mishra et al., Science Robotics 9(93):eadk8019 (2024) | Oyster mycelium controls a wheel and a 5-legged "starfish" with UV-triggered gait switching | https://www.science.org/doi/10.1126/scirobotics.adk8019 |
| b15 | phi-Bot: slime mould hexapod | Tsuda et al., 2009 | Physarum oscillates an insectoid's legs; historical non-neuronal legged bio-control | https://eprints.soton.ac.uk/268247/1/TsudaS08AlifeInHardware.pdf |
| b16 | Brain-inspired motor control review | Mompo Alepuz et al., Front. Neurorobot. 18:1429445 (2024) | Reviews brain-inspired controllers (SNN, cerebellum, basal ganglia); none combine living tissue + real-time contract | https://pmc.ncbi.nlm.nih.gov/articles/PMC11366706 |
| b17 | NEURON + MuJoCo co-simulation | bioRxiv 2025.06.17.660217 | NEURON neural models driving MuJoCo musculoskeletal open/closed loop - **direct methodological neighbour** | https://www.biorxiv.org/content/10.1101/2025.06.17.660217v1.full.pdf |
| b18 | **No public CL1/cl-sdk robot control found** | 2026 sweep (negative) | Every public CL closed loop is a game (Pong/Doom) or cloud demo; **the claimed gap** | - (re-verify) |
| b19 | BL-1 in-silico cortical culture (JAX/Izhikevich) | GitHub m9h/bl1 (MIT) | Realistic substitute for the SDK substrate: 64-ch virtual CL1 MEA, Pong/ViZDoom, validated against Wagenaar 2006 | https://github.com/m9h/bl1 |
| b20 | Scale-NeuroEval: LLM-designed environments for OI | arXiv:2509.04633 / NeurIPS 2025 | LLM-designed closed-loop virtual environments + plasticity evaluation for organoid agents | https://arxiv.org/html/2509.04633v1 |
| b21 | CL1 hardware/cost economics | Wikipedia + DataCenterDynamics (2026) | ~USD 35 k/unit, ~USD 300/week cloud, ~6-month culture lifetime, ~200-800 k neurons; Bio Data Centre Melbourne (120 units) | https://en.wikipedia.org/wiki/Cortical_Labs |
| b22 | Organoid-intelligence ethics | Smirnova, Nat Rev Bioeng 2024; Baltimore Declaration, Front. Sci. 2023 | OI roadmap + formal ethics | https://www.nature.com/articles/s44222-024-00200-6 |
| b23 | ICRA 2026 workshop: Neuromorphic Field Robotics | IEEE ICRA Vienna (Jun 2026) `[verify]` | Neuromorphic robotics workshop; no biocomputation track | https://2026.ieee-icra.org/workshops-and-tutorials |

### 2.3 Neuromorphic control / ANN-to-SNN conversion / energy

| # | Title | Source | Key claim (one line) | URL |
|---|---|---|---|---|
| c1 | Neuromorphic computing for embodied intelligence (survey) | arXiv:2507.18139, IEEE IOLTS 2025 | Surveys cross-layer SNN algorithms+hardware+workflows for autonomous systems | https://arxiv.org/abs/2507.18139 |
| c2 | Benchmarking framework for embodied neuromorphic agents | Nat. Mach. Intell. 8:300-312 (2026) `[verify]` | Proposes tasks/metrics/physical platform for embodied "brains" | https://doi.org/10.1038/s42256-026-01197-w |
| c3 | NeuroBench | arXiv:2304.04640, Nat. Commun. 16:1545 (2025) | Dual (algorithm+system) benchmark with standard energy/latency measurement protocols | https://arxiv.org/abs/2304.04640 |
| c4 | Toward large-scale SNNs (conversion vs direct survey) | arXiv:2409.02111 (2024) | Survey: conversion vs surrogate gradient for large networks | https://arxiv.org/abs/2409.02111 |
| c5 | Inference-scale complexity in ANN-to-SNN conversion | CVPR 2025, arXiv:2409.03368 | Local threshold balancing: ResNet-34 @90% = 622 FPS/W vs 22 (ANN), ~28x, no quantized retraining | https://arxiv.org/abs/2409.03368 |
| c6 | Differential coding (training-free conversion) | arXiv:2503.00301 | Spikes transmit rate changes; VGG-16 73.17% at ~22% of ANN power | https://arxiv.org/abs/2503.00301 |
| c7 | PASCAL: precise conversion | arXiv:2505.01730, TMLR 2025 | Conversion equivalent to quantized-ANN QCFS; 64x fewer inference timesteps | https://arxiv.org/abs/2505.01730 |
| c8 | One-timestep conversion (Scale-and-Fire) | arXiv:2510.23383 (2025) | 88.8% ImageNet-1K at T=1 using the 4.6 pJ/MAC vs 0.9 pJ/AC ratio | https://arxiv.org/abs/2510.23383 |
| c9 | Error amplification limits conversion (control) | arXiv:2601.21778, ICML 2026 | Temporally-correlated errors cause drift in MuJoCo control; CRPI (residual membrane) recovers almost all | https://arxiv.org/abs/2601.21778 |
| c10 | Reconsidering SNN energy (hardware-aware) | arXiv:2409.08290, ICASSP 2024 | Counting memory: SNNs only win with T in [5,10] and rate <6.4%; VGG-16 T=6 needs >93% sparsity | https://arxiv.org/abs/2409.08290 |
| c11 | Rethinking SNN/ANN energy (accelerator-validated) | ACM TACO 23(3), Art. 90 (2026) `[verify]` | SNN/ANN energy ratio grows with timesteps and saturates; pruning helps SNNs more | https://doi.org/10.1145/3822176 |
| c12 | Nengo+Loihi adaptive arm (vs PID) | arXiv:2007.10227, Front. Neurorobot. 2020 | Jaco 2: adaptive 2.45x more accurate than PID, ~4.6x less power than CPU, 2.5-4.5 ms loop | https://arxiv.org/abs/2007.10227 |
| c13 | NEF spiking PID (3-DoF arm) | Sensors 24(2):491 (2024) | NEF spiking PID (300 LIF) beats classic PID (6% ITAE, 30% RMSE) and fuzzy (5% ITAE) | https://doi.org/10.3390/s24020491 |
| c14 | Bioinspired smooth neuromorphic arm control | arXiv:2209.02787 (2023) | Compact joint SNN on Loihi, bell-shaped profiles, low jerk, comparable to PID on real Jaco | https://arxiv.org/abs/2209.02787 |
| c15 | ED-BioRob: spike-based PID on FPGA | Front. Neurorobot. 2020 (10.3389/fnbot.2020.590163) | 100% spiking PID (SSP building blocks), accepts dynap-SE references; BioRob 4-DoF | https://doi.org/10.3389/fnbot.2020.590163 |
| c16 | Parsimonious adjustable neuromorphic PID on Loihi | ACM IDT 2022 `[verify]` | Adjustable ~93-neuron spiking PID on Loihi | https://doi.org/10.1145/3546790.3546799 |
| c17 | NEF LQR for cart-pole | arXiv:2507.03621 (2025) | NEF/LIF implement LQR with 7 control + 7 neuromorphic metrics | https://arxiv.org/abs/2507.03621 |
| c18 | Spiking control taxonomy review | arXiv:2509.05356, NCE 2025 | Taxonomy: analytic spike coding, predictive, NEF arms (DeWolf), surrogate RL (PopSAN ~140x), conversion | https://arxiv.org/abs/2509.05356 |
| c19 | Loihi / Loihi 2 | IEEE Micro 38(1):82 2018; Proc. IEEE 109(5):911 2021 | Primary Nengo hardware; Loihi 2 adds programmable neurons/graded spikes | https://doi.org/10.1109/MM.2018.112130359 |
| c20 | Akida energy (vendor) | BrainChip user guide `[verify]` | ~23.48 mJ/frame ImageNet on AKD1000; vendor data without peer review - re-measure | https://doc.brainchipinc.com/user_guide/akida.html |
| c21 | Speck event-based vision SoC | arXiv:2304.06793, Nat. Commun. 15:4464 (2024) | DVS+SNN single chip: 3.36 us/layer, ~0.7 mW real time, <0.1 ms latency, 320 k neurons | https://arxiv.org/abs/2304.06793 |
| c22 | SENECA: digital neuromorphic RISC-V | arXiv:2303.15224 (2023) | Instruction-level energy accounting (~2.8 pJ/syn-op) enabling hardware-aware co-design | https://arxiv.org/abs/2303.15224 |
| c23 | Deng et al. energy analysis | Neural Networks 121:294-307 (2020) (10.1016/j.neunet.2019.09.005) | SNN/ANN comparison including memory | https://doi.org/10.1016/j.neunet.2019.09.005 |

---

## Vulnerability map (stack-linked)

Known facts of the stack (F1/F2): NEF fidelity degrades at dt=0.01 (identity 2.0
-> 0.68, 1.951 at dt=0.002); the hybrid ANN-to-SNN loop falls at ~8 s from
imperfect conversion; `run_steps` overhead ~100 ms/call on CPU (CL latency
budget); `read()` in 5-sample chunks; 3 nC stim charge limit.

| # | improvement | weakness it addresses | evidence | status |
|---|---|---|---|---|
| M1 | Run everything at dt=0.002 (no decimation to 0.01) | NEF identity fidelity 0.68 | 1.951@0.002; c5/c7 precise conversion | done in hub (HUB_HORIZON_STEPS=600); carries the demo |
| M2 | Replace the SDK's random spikes with a substrate with real neural dynamics | The F2 loop with SDK randomness is not "authentic" | b19 BL-1 (JAX/Izhikevich, virtual 64-ch MEA, CL1 UDP protocol); b3 `on_stim` | pending (optional F2+/F3) |
| M3 | Doom-neuron-style ablation: random/zero decoder, spike-gated action masking, fraction of control carried by neurons | Reviewer concern: "the decoder did everything" | b8 (245 stars, GPL) | done (`02_cl/src/ablate_loop.py`, `results/f2_ablation*.json`); finding: loop redundant on flat ground at 0.5 m/s -> F3 needs perturbations |
| M4 | Report loop latencies: us histograms, tick overruns, wall-clock vs accelerated; real CL1 jitter = TimeoutError | ~100 ms/call `run_steps` latency budget | b4 docs; c12 reports 2.5-4.5 ms | done: tick p50 ~47 ms (overruns ~488/499 vs 25 ms), `bridge_read` mean 30 us/tail 172 ms |
| M5 | Energy curve with counted memory, fixed T and sparsity; NeuroBench protocol | Unfounded energy claims | a18, c3, c10, c11, a19 | pending for F3 |
| M6 | Prefer end-to-end SNN / temporal coding over ANN-to-SNN conversion for the control loop | Hybrid loop falls at ~8 s (imperfect conversion) | a17/c9 (error amplification), a2 (proxy target), a1 (population) | pending (F3 or F1 improvement) |
| M7 | Endorse the spiking PD with a dedicated citation (6-30% improvement vs classic PID) | SpikingPD characterised but not connected to literature | c13 Sensors 24(2):491; c17 NEF LQR; c14/15/16 loihi/FPGA PIDs | ready to write |
| M8 | Include an honest "bio-loop vs silicon-loop" baseline | Reviewer expectations that wetware wins | b13 (honest PLOS ONE 2016), a16 (SNN loses on deep), a17 | recommended (already done with E1-E4) |
| M9 | Position as a sim-first pipeline (contract testbed, not biological ground truth) | CC BY-NC license + overselling risk | b2 license, b12 (MSR sim-first) | already our stance |

---

## Ablations and honesty (recommended in the note)

1. **Random/zero-decoder baselines** (doom-neuron protocol, b8): the humanoid
   must clearly fall or degrade if the decoder disagrees with the neural
   dynamics. Implemented in F2 as `02_cl/src/ablate_loop.py`.
2. **Spike-gated action masking**: action mask from spikes (legs only move if
   the substrate emits), quantifying the fraction of control carried by neurons.
3. **Bio-loop vs silicon-loop**: contrast the CL-contract loop with an
   equivalent ANN/PD loop (extends E1-E4) - honest precedent b13.
4. **Negatives to cite**: deep SNNs lose to ANNs (a16), conversion loses
   temporal dynamics (a17/c9), rate-coding on von Neumann machines is not more
   efficient (a19/c10). This disarms objections before they arrive.

## Legal / ethical / logical caveats

- **Licenses**: cl-sdk CC BY-NC 4.0 (non-commercial); CL API whitepaper CC
  BY-NC-SA; doom-neuron GPL-3.0; BL-1 MIT. Handle each separately in the
  reproducibility artefacts; declare the code license position.
- **Costs/hardware (reviewers will ask)**: CL1 ~USD 35 k, cloud ~USD 300/week,
  ~6-month culture lifetime. The sim-first pipeline (USD 0) is the de-risking
  argument.
- **Simulator honesty**: the SDK simulator produces "non-learning" data that
  does not respond to stimulation - therefore an `on_stim` source is necessary;
  in any paper the sim is a *contract and loop testbed*, never biological ground
  truth.
- **HW-SW gap**: TimeoutErrors from jitter and other real-CL1 behaviours are not
  reproduced in sim; specify the accelerated vs wall-clock mode used.
- **Negative as a snapshot**: "no public CL-contract robot control" is a sweep
  as of September 2026; **re-run before submission**.

---

## Paradigm: non-trainable substrate <-> trainable SNN surrogate

The real wetware (DishBrain/CL1) is **not gradient-trainable** - no backprop, no
DL-style labels; only closed-loop conditioning (reward/stim), costly and
ethically limited. Therefore the trainable part is implemented as an **in-silico
SNN** (Nengo/NEF or surrogate gradient), trained in simulation, and deployed
**without touching code** over the CL contract (same spikes<->stims I/O). **The
contract makes the swap drop-in: you train once in-silico, you run verbatim on
wetware.**

Validating precedents (none brought to a full humanoid):

| # | item | source | point it validates | URL |
|---|---|---|---|---|
| d1 | doom-neuron - train the trainable, not the wetware | SeanCole02/doom-neuron | PPO/REINFORCE over the UDP encoder/decoder; the culture is NOT trained, the decoder is; ablation shows the limit | https://github.com/SeanCole02/doom-neuron |
| d2 | **Assembloid Agency** (backend-agnostic bridge) | jennnital/UE-CL1-API, NeurIPS 2025 Creative AI Track | "The same plugin drives CL1 or a NEST/SNN/EEG stand-in"; `--organoid` Brian2/LIF simulator + `SendRewardSignal` (reward==stim, DishBrain style) | https://github.com/jennnital/UE-CL1-API |
| d3 | BL-1: differentiable virtual CL1 server | m9h/bl1 (JAX, MIT, active) | Trainable, drop-in in-silico substitute of the CL1 (64-ch, UDP): literally implements the concept | https://github.com/m9h/bl1 |
| d4 | MSR neuron-connected robots: sim-first is mandatory | Microsoft Research 2022- | Train in sim first to avoid harming cultures = standard; validates "train in-silico, validate on wetware" | https://www.microsoft.com/en-us/research/project/neuron-connected-robots |
| d5 | Organoid cart-pole: adaptation by external RL | Robbins et al., Cell Reports 45(2), 2026 | Conditioning-based learning (success 4.5% to 46%), NOT backprop - the clamp is the trainable part | https://www.cell.com/cell-reports/fulltext/S2211-1247(26)00062-8 |
| d6 | Survey on substrate-independent biocomputing (SBI) | arXiv:2604.27933 (cs.ET, 2026) | "Substrate-as-platform" interchangeable framework - cite for the contract as the abstraction layer | https://arxiv.org/abs/2604.27933 |
| d7 | MetaBOC (robot-organoids) | Tianjin/SUSTech 2024 (press) | Claim of organoid-driven robots; **control problems = press-only `[verify]`** - a comparative negative that favours us (no contract, no published decoding) | https://teqnoverse.com/metaboc/ |

**Important honest nuance:** "non-trainable" is qualified - DishBrain *does*
learn through structured feedback (Pong, ~5 min, arXiv:2405.16946). The correct
thesis is: **not trainable by DL/gradient; adaptable by conditioning ->
therefore the trainable part lives in the SNN surrogate, not the wetware.**

**Empirical status (`surrogate_cl/` companion, consumed read-only):** the G1
12-joint closed loop (MuJoCo 500 Hz, LSTM policy at 50 Hz) with a 1000-LIF Nengo
hub uses velocity-profile tracking through channel 62; the canonical multi-seed
battery (5 profiles x 5 modes x 6 seeds) shows baseline neural RMSE
0.238 +/- 0.010 and significant separation from the dead Poisson null on all
profiles (exact paired permutation, one-sided p = 1/64). The falsification
protocol (gates B-E in `sandbox/`) evaluates substrate material, readout
robustness and memory against a matched dead-substrate null.