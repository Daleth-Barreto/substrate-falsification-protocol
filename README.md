# Fase 3 — Unión: CL-Aware Neuromorphic Sprint

**Pregunta:** ¿el stack completo **contrato CL + sustrato neuromórfico** logra **sprint humanoide** (≥3 m/s en sim) y supera a PD y a RL-(ANN) en energía, latencia y robustez, con una narrativa de *upgrade path* a CL1 real?

## El hueco que cierra (SOTA)

- F1 demuestra sustrato espiking en la planta (trote).
- F2 demuestra el contrato CL cerrando el lazo (tiempo real + sustrato in-silico).
- **Nadie** ha unido ambos en humanos de piernas: sin sprint-SNN, sin CL↔MuJoCo, sin medición de qué aporta cada capa. Aquí la unión los junta y **mide la ablación**.

## Arquitectura unificada

```
MuJoCo (H1/G1) 200 Hz
   ▲   │ estado
   │   ▼
   [contrato CL: Neurons.loop(100-1000 Hz)  ←── F2]
   │      │ stim() transaccional  └──► on_stim() → sustrato (Nengo o BL-1)
   │      │ DataStream + HDF5
   │      ▼ spike train
   [policy SNN NEF 50–100 Hz ←── F1]  → targets de articulación
   [lazo interno espiking 1 kHz ←── F1]  → torques
```

Flujo de trabajo: muñeco F1 (planta) re-cableado con el contrato F2 como scheduler, ablaciones capa por capa.

## Stack

- Python 3.12+ venv (cl-sdk) · nengo · BL-1 (opcional) · mujoco · numpy/scipy/jax
- Checkpoints públicos H1/G1 · configs de F1 y F2 reutilizados

## Estructura del repo (propuesta)

```
03_union/
├── README.md
├── requirements.txt
├── configs/            # ablaciones: {pd, snn-only, cl-only, cl+snn}
├── src/
│   ├── stack.py        # composición: contrato CL + policy NEF + inner spiking PID
│   ├── ablation.py     # correr las 4 configs y recoger métricas
│   ├── metrics.py      # velocidad, CoT, spikes/tick, energía est., latencia
│   └── sprint_task.py  # 30 m plano, transición walk→run, perturbaciones
└── paper/              # LaTeX IEEEtran 8pp doble-anónimo (plantilla ICRA)
```

## Experimentos / baselines cruzados (tabular para reviewers)

| Config | Vel. 30 m | CoT | Caídas | Spikes/tick | Latencia tick→stim | Energía est./paso |
|---|---|---|---|---|---|---|
| PD clásico (baseline) | | | | — | — | — |
| ANN-RL (checkpoint; SPRINT/KSLC reportado) | | | | — | — | — |
| SNN only (F1) | | | | | — | |
| CL only (contract+BL-1, F2) | | | | | | |
| **CL+SNN (unión)** | | | | | | |

Ablación central: **qué aporta cada capa** (espiking = sparse/energía; contrato = determinismo/latencia/upgrade).

## Entregables F3

- [ ] `stack.py` funcional con las 4 configs
- [ ] Sprint ≥3 m/s en H1 o G1 (muestra preliminar; 6 m/s requiere entrenar RL = fuera de alcance GPU)
- [ ] Tabla de ablación + figuras (trajectories, spike rasters, latency histogram)
- [ ] Paper **ICRA** (o IROS si se nos pasa la ventana): 8pp, keywords Neurorobotics / Humanoid and Bipedal Locomotion / Learning and Adaptive Systems
- [ ] Video ≤180 s/20 MB

## Público / fechas

- ICRA 2027 (deadline 15-sep-2026) — **demasiado cerca para F3 completo**; F3 apunta a **IROS 2027 (~mar-2027)** o **ICRA 2028 (~sep-2027)**. F1/F2 pueden salir antes como workshop/short.
- Si el deadline de ICRA 2027 se quiere aprovechar: solo con alcance F1 preliminar (+ discusión de roadmap F2/F3).

## Investigación para ICRA 2028

Ver `ICRA_2028_investigacion.md` (consolidación de 3 encuestas web, sept 2026):

- **Hueco confirmado**: nadie ha cerrado el lazo de un humanoide completo (G1, 23-DoF) vía el contrato CL. Baselines biológicos legged = CPG/no-neuronales y lentos (mycelia, Physarum) o hobby 1–10 DoF. Vecinos más cercanos: H1 NEF+SPA (arXiv:2606.11034) y Nengo+Loihi brazo Jaco (arXiv:2007.10227).
- **Nueva línea (sección 8)**: "sustrato CL no-entrenable ↔ sustituto SNN entrenable" — entrenar una vez in-silico, correr verbatim en wetware por el mismo contrato. Precedentes: doom-neuron, Assembloid Agency (backend-agnóstico, NeurIPS 2025), BL-1 (virtual CL1 server diferenciable). Aporte reformulado: primer lazo humanoidal con esta intercambiabilidad.
- **Oportunidades (sección 9)**: **ICRA 2028 = meta fija + IEEE RA-L (sin APC) como vía principal de revista**, manteniendo ICRA 2027 vivo vía RA-L→transfer (31-dic-2026). ~~ICOI 2026~~ descartado (costo/visa). Plan B: IROS 2027 (1-mar-2027). Competencia sin costo: HVAC 2027; integración: Telluride 2027; revista temática: Cyborg and Bionic Systems (APC/waiver a verificar).
- **Mejoras priorizadas**: (M3) ablación decoder random/zero estilo doom-neuron + (M4) reporte de latencia del lazo — costo bajo, impacto alto para el paper; (M2) opcional sustrato BL-1 via `on_stim`; (M6) política end-to-end SNN/NEF vs conversión. M3+M4 ya implementadas (`02_cl/src/ablate_loop.py`, `results/f2_ablation.json`) — hallazgo: lazo redundante en plano → tarea portante con perturbaciones en F3.
- **Nueva línea operativa**: `C:\Proyectos\papers\surrogate_cl\` — implementación de la sección 8 con H1 (tarea portante). Hallazgos: (1) se halló y reparó un **bug crítico** (el override de vx no llegaba a `deploy12.cmd` — la física era idéntica entre modos); (2) la planta es robusta a impulsos de 0.15 s (≥140 N) → lazo redundante en plano; (3) bajo empujes sostenidos (≥0.5 s) el lazo queda fuera de autoridad (τ≤30 N·m vs ≥112 N·m de volcamiento) y caen todos; (4) la **tarea portante es el seguimiento de un perfil de velocidad por ch62**: RMSE neural 0.279 vs zero 0.612, y una lesión del 50% de canales borra la señal (ch62=0) — casualidad vs función separadas. Evidencia: `surrogate_cl/results/{task_tracking,perturb_sweep,push_probe2,push_probe3,task_progress}.json`.
- **Caveats**: cl-sdk CC BY-NC (no comercial); reclamos de energía exigen memoria contada + T/sparsity fijas (NeuroBench); sim = testbed de contrato, no ground truth biológico; barrido negativo a re-verificar al envío.

## Notas / ética / upgrade path

- Sim-only, cuantificado (regla ICRA): no prometer sim2real; medir todo en sim.
- La afirmación de biocomputación se limita a *compatibilidad arquitectónica con el contrato CL*: el mismo código corre en CL1/Cortical Cloud (pago) como validación futura — se declara, no se ejecuta en el paper.