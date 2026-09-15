# Does the substrate compute?

Closed-loop, information-theoretic validation that a neuromorphic substrate carries a load-bearing control task through the cortical-labs contract, applied to a simulated humanoid.

This repository is the registry of a research line with three phases and one shared thesis. It holds the paper-level documentation and the reproducibility scaffold; each phase lives in its own repository linked below. Everything runs in simulation on CPU (no GPU training, no biological hardware, no CL1 purchase).

## Research line

| phase | repository | question |
|---|---|---|
| F1 - plant | `01-snn` | Can a spiking stack (Nengo / NEF) replace the policy and the inner loop of a Unitree H1/G1 humanoid in MuJoCo, at least at a stable trot, without training RL from scratch? |
| F2 - contract | `02-cl` | Can a control loop written against the Cortical Labs API contract (official `cl-sdk`, free, local) drive a simulated humanoid with an in-silico spiking substrate, inside real-time constraints? |
| F3 - union | `03-union` | Does the full stack (contract + neuromorphic substrate) reach a sprint, and what does each layer measurably contribute? |
| surrogate | `surrogate-cl` | **Does the substrate compute?** Is the load-bearing task function trainable in-silico, deployable verbatim over the contract, and verifiable with information theory (MI / transfer entropy)? |

The three phases answer *can it* (plant, contract, union). The surrogate repository answers *does it compute*, which is the claim the paper is built on: the task information must demonstrably travel through the spikes, and an ablation must be able to destroy it.

## Thesis

A culture substrate (Cortical Labs CL1 / DishBrain) is not trained by gradient descent; it is only conditioned slowly and closed-loop. The same function, however, can be trained once in-silico as a spiking neural network and deployed verbatim over the same contract (sensors -> electrodes -> spikes -> decisions -> stimulation). The surrogate repository turns this substitution into a load-bearing experiment with honest ablations: disconnected decoder, uncoupled decoder, 50 % electrode lesion, and a dead Poisson substrate, against the intact Nengo-LIF hub.

## Headline results (September 2026)

- **The loop is a real velocity controller.** In the corrected loop the decoded command modulates the physical velocity: neural walks at about 0.55 m/s, the uncoupled decoder (mean command 0.255) at about 0.28 m/s; the zeroed and lesioned modes keep the default 0.50 m/s.
- **The neural hub is the best decoder on all five task profiles.** RMSE 0.166-0.194 vs 0.251-0.644 for the ablation conditions (5 profiles x 4 modes x 5 seeds). Baseline neural passes the task threshold (RMSE 0.182 vs threshold 0.18), and every lesion degrades the decode (+40 % to +90 %).
- **The task information demonstrably travels through the spikes.** MI(cmd; task): neural 0.037 nats vs zero/random 0.000; transfer entropy electrode-62 -> command: neural 0.0079, all other modes exactly 0.000; the dead Poisson substrate scores exactly zero channel MI and zero transfer entropy.
- **The authority envelope is honest and measured.** Lateral impulse frontier 0.40 s survived / 0.45 s fell at 140 N; the contract delivers about 30 N-m of recovery torque against a 112 N-m tipping moment, so sustained pushes fall in every mode. The loop closes in 25 ms but cannot defend sustained floor loads.

## Reproducibility

- Every phase pins its environment (`requirements.lock.txt`, generated with `uv freeze`).
- The seed protocol, the task profiles and the analysis pipeline are documented in each repository and in `docs/REPRODUCIBILITY.md`.
- `scripts/bootstrap*` creates the shared environment and prints the exact commands per phase.
- Third-party assets (Unitree RL Gym, MuJoCo model zoo, the G1 `motion.pt` checkpoint; about 2 GB) are fetched by the bootstrap script and are not redistributed here.

## Repository layout

```
docs/REPRODUCIBILITY.md    environment manifest, data provenance, hardware notes
scripts/bootstrap.ps1      Windows bootstrap (also .sh for POSIX)
third_party/               fetched at bootstrap time (not tracked)
```

Each phase repository is standalone and self-describing. Start with `surrogate-cl` for the active claim; the phase repositories hold the build evidence.

## License

This repository and the phase repositories are MIT-licensed. Third-party dependencies and assets keep their own licenses: `cl-sdk` is CC BY-NC (academic use), MuJoCo is Apache-2.0, `nengo` is MIT, `m9h/bl1` is MIT, and the G1 assets follow Unitree RL Gym's terms. This project does not cultivate biological cells and requires no CL1 hardware.

## Citation

If you use this work, please cite the project registry:

```bibtex
@misc{barreto2026substrate,
  title        = {Does the substrate compute?},
  author       = {Barreto, Daleth},
  year         = {2026},
  howpublished = {\url{https://github.com/Daleth-Barreto/icra2027}},
  note         = {In preparation (ICRA 2028 / IEEE RA-L target)}
}
```

## Publications status

- ICRA 2028 is the fixed target; IEEE RA-L (no APC) is the preferred journal route with transfer to ICRA.
- The surrogate milestone is the load-bearing piece: task tracking, information-theoretic validation and the authority envelope are complete and versioned; substrate interchangeability and the full negative sweep are in progress.