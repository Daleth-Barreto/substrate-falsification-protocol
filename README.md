# F2 - Cortical Labs contract: closed loop

**Question:** can a control loop written against the **Cortical Labs API contract** (official `cl-sdk`, free, local) drive a simulated humanoid with an **in-silico spiking substrate** that plays the role of a DishBrain / CL1, within real-time constraints?

## CL context (verified, September 2026)

- **CL API** (arXiv:2602.11632): a formal contract for timing, ordering and synchronisation; sub-millisecond round trips; `Neurons.loop()` up to 25 kHz; transactional `stim()`.
- **`cl-sdk`** (PyPI, free, CC BY-NC): emulates the contract locally. It does **not** simulate biology (Poisson / HDF5 spikes, "does not respond to stimulation"). The official path to plug in a real substrate is a custom simulator data source (`cl.sim.set_simulator_data_source()`).
- Candidate in-silico substrates: **BL-1** (JAX, 10k Izhikevich + STDP, virtual 64-channel MEA, speaks the CL1 UDP protocol) or **Nengo/NEF** (simpler, CPU).
- No hardware: CL1 is USD 35k + ethics; Cloud is about USD 300/week. **This phase needs neither.**

## What is demonstrated

1. The **CL contract** can schedule a humanoid loop without breaking the dynamics.
2. The in-silico spiking substrate produces coherent (non-Poisson) spikes connected to the contract.
3. A simple DishBrain-style learning (predictability reward / STDP) is possible in this loop.
4. The same code is an *upgrade path* to a real CL1 / Cloud (narrative, not an experiment here).

## Target architecture

```
MuJoCo (H1 / G1) --(state)--> encoder (rate / time-to-first-spike)
                                  | stim()   v
 cl-sdk Neurons.loop(1000 Hz) <---> substrate in-silico (BL-1 or Nengo)
      | on_stim() callback        |
      v                           - spikes
 decoder (spikes -> actions) --> MuJoCo
 DataStream: (x, vel, joints) + clock --> HDF5 (RecordingView)
```

## Stack (requires Python 3.12+, installed with `uv`)

- `cl-sdk` (PyPI, local, CC BY-NC; data source registered via `set_simulator_data_source("module:factory", config=..., metadata=...)`)
- `mujoco 3.13` + G1 MJCF (Unitree RL Gym)
- `nengo 4.1` (CPU) -- decision hub + synthetic spikes in the datasource
- `torch 2.14+cpu` (only for `deploy12` in the subprocess)
- Python 3.12.13 standalone via `uv python install 3.12`

**venv**: `.venv312` with `mujoco==3.13.0, nengo==4.1.0, torch==2.14.0+cpu, cl-sdk==1.0.0`. Full pins in `requirements.lock.txt`.

## Key findings (live cl-sdk API)

- This SDK version has no `cl.open` module and no `LiveSimulatorDataSource`; `cl.open()` is a **generator** and the data source registers with `set_simulator_data_source("module:factory", ...)`.
- `neurons.loop(tps)`: the tick carries `tick.frames` (int16) and `tick.analysis.spikes`. **`analysis.spikes` does not detect synthetic frames** (it only works with replay/Poisson ground truth); the correct path is custom spike sorting over `tick.frames`. The data source can attach `DataSourceBatch(frames=..., spikes=DataSourceSpike(...))` if desired.
- `Neurons.stim(channel_set, stim_design, ...)` is **positional-only**; float designs become `StimDesign(160 us, -I, 160 us, +I)`; charge limit 3 nC -> 0..~18 uA (we encode 4 uA per m/s). Do not send `cmd ~= 0` (equal polarities raise `ValueError`).
- Nengo 4.1 `run_steps(N)` costs about 100 ms **per call** (0.256 ms/step at N=400), so the hub runs in a `HubThread` with 1.2 s simulation bursts per call and the loop reads `hub.latest`.
- **LIF fidelity**: at `dt=0.01` a NEF decodes identity(2.0) as about 0.68 (the substrate degrades); at `dt=0.002` it is about 1.95. The hub uses `dt=0.002, 600 steps/burst`.
- `read()` is invoked in chunks of 5 samples: transient generation must be by a **global temporal program** (interval proportional to 1/|sensor|), not per chunk, to avoid multiplying density.
- The datasource subprocess does not guarantee `close()` on exit, so walker telemetry is written from `_ctrl_step` every 25 control steps.

## State: working closed-loop demo

**`src/bridge_g1.py`** implements a custom simulator data source (G1DataSource) that owns a `Deploy12` inside the CL simulator subprocess: it encodes sensory state into 64 electrodes at 25 kHz (joint errors `[0:12]`, attitude `[12:15]`, height `ch23`, vx-override `ch62`, culture/hub `ch63`), advances the physics at 50 Hz boundaries, and applies microstimulation in `on_stim` (torque overlay `[0:12]`, vx override `ch62`).

**`src/demo_walk.py`** closes the full contract at 40 TPS: `walker -> sensors -> electrodes -> culture spike detection (own spike sorting on tick.frames) -> Nengo-LIF hub (NEF, 1000 neurons, decide vx) -> myo-electric stims (uA) -> on_stim -> walker`.

Result (12 s, reproducible): **h_last = 0.772 m, vx_last about 0.50 m/s, fallen = False**, hub cmd about 0.53, 50 +/- 5 adaptive stim events, about 8.8 spikes/tick. Evidence: `results/f2_demo.json` + `results/f2_demo.png`.

## Ablations (doom-neuron protocol) and latency

**`src/ablate_loop.py`** replicates the doom-neuron protocol on this loop: 4 modes (neural / zero / random / mask0.5) x 12 s, with per-tick latency metrics for the SDK loop and the bridge `read()`.

### v1 and the discovered bug

An audit (in the surrogate line) showed that in v1 the vx override **never reached the plant**: `cmd_override` only re-encoded to the channel-62 frame (what the substrate sees) but not to `deploy12.cmd[0]` (what the LSTM observes), so the physical trajectory was identical across modes (vx = 0.50 everywhere, an artefact). The fix connects the command in `_ctrl_step`; everything was regenerated as `results/f2_ablation_v2.*`. The v1 files stay intact for transparency. **Only v2 is valid for velocity-modulation claims.**

### v2 (corrected loop), mean over 12 s

| mode | mean_cmd | stims | mean_nspk | fallen | vx_last | loop p50/p95 | overruns>25ms | bridge_read mean/max |
|---|---|---|---|---|---|---|---|---|
| neural | 0.531 | 70 | 9.1 | False | **0.55** | 31/31 ms | 296/499 | 14/78 ms |
| zero | 0.000 | 0 | 7.1 | False | 0.50 | 31/32 ms | 292/499 | 16/94 ms |
| random | 0.255 | 481 | 8.9 | False | **0.28** | 31/32 ms | 292/499 | 16/63 ms |
| mask0.5 (50 % lesion) | 0.482 | 94 | 6.3 | False | 0.50 | 31/32 ms | 292/499 | 17/79 ms |

Reading (v2, revisited):

- With the loop actually closed, the decoded command does modulate physical velocity: neural walks at 0.55 m/s, the uncoupled decoder (mean command 0.255) at 0.28 m/s, zero/mask keep the default 0.50 m/s. The loop is now an observable velocity controller, not decoration.
- On flat ground at 0.5 m/s none of the modes fall (the stability redundancy persists, now with the correct cause: the base policy absorbs velocity modulation inside its envelope), exactly as in the doom-neuron result.
- Cadence measured here: tick p50 about 31 ms, p95 31-32 ms, overruns about 292/499 (>25 ms), reported in wall-clock. `bridge_read`: mean 14-17 us, tail 63-94 ms.

Honest reading (important for the paper):

- The loop is **redundant on flat ground at 0.5 m/s**: all four modes walk alike (h = 0.772, vx = 0.50). The G1 base (PD + LSTM) absorbs the modulation. This is why the load-bearing task experiment (surrogate line) uses velocity *profiles* and information-theoretic separators rather than stability alone.
- The loop metrics do separate modes (stims 0/49/481, nspk 6.0 -> 8.9), which supports the measurement claim.
- Cadence: the real tick is about 47 ms (p50) in the loaded runs, above the 25 ms control deadline (overruns about 488/499); causes: loop overhead + GIL contention with the HubThread + physics bursts. Report in wall-clock; a real CL1 lapses raise `TimeoutError`. `bridge_read`: chunks of 5 samples (67,499 calls / 337,495 samples in 12.5 s), mean 26-32 us, tail 141-219 ms (50 Hz physics block).

## Experiments / metrics

| ID | experiment | metric |
|---|---|---|
| F2-E1 | per-tick latency (`Neurons.loop` 1 kHz) | time budget, jitter, round trip |
| F2-E2 | closed-loop CL-contract: follow velocity command | tracking error, stability |
| F2-E3 | DishBrain-style learning (gait selection, target velocity) | convergence sample, spikes/actions |
| F2-E4 | BL-1 substrate vs Poisson (negative control) | behavioural divergence |

## Deliverables

- [x] `bridge_g1.py` working (cl-sdk <-> MuJoCo walker)
- [x] Reproducible closed-loop demo (12 s, walker walks) + `f2_demo.json/png`
- [x] M3 ablations (neural/zero/random/mask0.5) + M4 latency (`f2_ablation*.json/png`)
- [ ] BL-1 substrate in contract -- in progress (see `03_union` and the surrogate Gate B)
- [ ] Latency report vs the 25 ms control budget in the paper (latency already measured)

## Reproducibility

Environment and pins: see the project-level `docs/REPRODUCIBILITY.md` and `requirements.lock.txt`. Third-party assets (G1 model, menagerie MJCF, `unitree_rl_gym` deployment with `motion.pt`) are fetched by the project bootstrap script.

## Publication route

ICRA "Neuromorphic Field Robotics" workshop (active since 2026) or a biocomputing journal (Cyborg and Bionic Systems / Frontiers), first-but-at-scale real-time validation of the CL contract in closed-loop dynamics.

## Notes / ethics

Everything is **simulation**: no cells are cultivated, no CL1 is purchased. The CC BY-NC SDK license is correct for academic use. The paper is honest: the SDK provides the contract; the "biology" is the simulated spiking substrate.