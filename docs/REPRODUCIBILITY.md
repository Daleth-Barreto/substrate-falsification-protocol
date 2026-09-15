# Reproducibility report

This document is the executable provenance for the research line "Does the substrate compute?". It states the exact environment, the hardware that produced every committed artefact, the seed protocol, and the data provenance of each result file.

## Environment manifest

Host that produced the results (September 2026):

| item | value |
|---|---|
| OS | Windows 11 |
| CPU | (laptop, CPU-only) |
| GPU | RTX 3050 Laptop 4 GB (not used; everything runs on CPU) |
| Python | 3.12.13 (managed with `uv`, `uv` 0.11.32) |
| Package manager | `uv pip` (`pip` not present inside the venv) |

Pinned Python packages (from `02_cl/requirements.lock.txt`, shared with `surrogate-cl`):

| package | version | role |
|---|---|---|
| cl-sdk | 1.0.0 | Cortical Labs contract simulator (CC BY-NC) |
| nengo | 4.1.0 | NEF / LIF hub (the trained substrate) |
| mujoco | 3.13.0 | physics |
| torch | 2.14.0+cpu | G1 `Deploy12` policy runtime and checkpoint export |
| numpy | 2.5.3 | numerics |
| scipy | 1.18.1 | statistics |
| matplotlib | 3.11.2 | figures |

The BL-1 substrate environment (`03_union/bl1_venv`, `requirements-bl1.lock.txt`) additionally pins `jax==0.11.1` (CPU) and is isolated from the contract venv.

## Determinism and seeds

- The Nengo hub network is created with `seed=1` and is deterministic; per-run variance comes from the simulator RNG, the cultural spike/count RNG, and the lesion draw.
- Multi-seed protocol: `{1, 7, 13, 29, 55}` for the tracking battery; `{7, 29}` for the Gate A captures.
- Task profiles are fixed schedules injected on channel 62 (see `surrogate-cl` README).
- The exact-permutation statistical test with 5 seeds has p-floor 1/32 (0.0625); no p < 0.05 claim is made (noted in `surrogate-cl/README.md`).

## Data provenance

All timestamps reference sessions run with the corrected loop (the v1 velocity-channel bug is preserved only in the marked historical files).

### surrogate-cl (`results/`)

| artefact | produced by | seeds | notes |
|---|---|---|---|
| `task_tracking.json` / `.png` | `src/task_tracking.py` | {1,7,13,29,55} | 5 profiles x 4 modes x 5 seeds, 12 s each |
| `f1_capture_{mode}_s{seed}.json` | `src/capture_signal.py` | {7,29} | per-tick counts[64] + plant state, 5 modes |
| `f1_capture_index.json` | `src/capture_signal.py` | - | capture manifest incl. wall time |
| `f2_signal_summary.json` / `.png` | `src/signal_analysis.py` | {7,29} | MI/TE, 50-permutation shuffle null |
| `push_frontier.json` | `src/probe_frontier.py` | 7 | lateral impulse frontier, 21 rows |
| `envelope.json` / `.png` | `src/envelope.py` | 7 | authority x latency envelope |
| `perturb_sweep.json` / `.png` | `src/perturb_sweep.py` | 7 | flat-ground lateral sweep |
| `push_probe2/3.json` | `src/probe_push.py` | 7 | sustained push limits |
| `task_progress.json` | `src/task_progress.py` | 7 | flat-ground distance |

Wall-clock guidance for planning: a single 12 s, 40 TPS run takes roughly 20-55 s of wall time depending on host load; the full tracking battery (100 runs) is the dominant cost.

### 01-snn / 02-cl

See the READMEs and `results/` of each phase repository. The 02-cl `f2_ablation_v2.*` files correspond to the corrected loop; `f2_ablation.*` (v1) are preserved for transparency and are superseded.

## Known nondeterminism

- Wall-clock cadence figures (tick p50/p95, overruns, `bridge_read` tail) are host-dependent and must be reported as measured, not as platform guarantees.
- MuJoCo contact right-hand-side and the G1 `Deploy12` subprocess scheduling produce run-to-run jitter; all quantitative claims in the paper use the aggregate over the fixed seed set.

## Assets not redistributed

- `third_party/unitree_rl_gym` (Unitree RL Gym, includes `deploy/deploy_mujoco/configs/g1.yaml` and the pre-trained G1 `motion.pt`).
- `third_party/mujoco_menagerie` (Google DeepMind model zoo, G1 MJCF assets).

These are fetched by `scripts/bootstrap*` and resolve against `{LEGGED_GYM_ROOT_DIR}` as configured in the deployment YAML. They are not committed because of size (about 2 GB) and licensing; see the bootstrap script and each phase README.

## Ethical and licensing note

- `cl-sdk` is CC BY-NC: academic use only, no commercial use.
- Everything is simulated: no biological cells are cultivated, no CL1 hardware is purchased, and the "biology" is the simulated spiking substrate.
- The paper claims *architectural compatibility with the CL contract* and in-simulation metrics; sim-to-real transfer is explicitly not claimed.