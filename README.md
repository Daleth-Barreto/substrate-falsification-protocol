# Fase 2 — Cortical Labs Contract: bucle cerrado

**Pregunta:** ¿puede un bucle de control escrito contra el **contrato del CL API** de Cortical Labs (librería oficial `cl-sdk`, gratis, local) controlar un humanoide simulado usando un **sustrato espiking in-silico** que imita el papel de un DishBrain/CL1, respetando latencias de tiempo real?

## Contexto CL (verificado, sep 2026)

- **CL API** (arXiv:2602.11632): contrato formal de timing/orden/sincronización; round-trip sub-ms; `Neurons.loop()` hasta 25 kHz; `stim()` transaccional.
- **`cl-sdk`** (PyPI, **gratis**, CC BY-NC): emula el contrato localmente. ⚠️ **No simula biología** (spikes Poisson / replay HDF5, "does not respond to stimulation"). La vía oficial para enchufar un sustrato real: **`cl.sim.set_simulator_data_source()`** con `LiveSimulatorDataSource` → `sink.emit_frames()` y callback **`on_stim(stim)`**.
- Sustrato in-silico candidato: **BL-1** (JAX, 10k Izhikevich + STDP + virtual MEA 64ch, habla el UDP del CL1) o **Nengo/NEF** (más simple, CPU).
- Sin hardware: CL1 = $35k + ética; Cloud = ~$300/sem. **Esta fase NO los necesita.**

## Qué demostrar

1. El **contrato CL** puede ser el scheduler del lazo de un humanoide (1 kHz) sin romper la dinámica.
2. El sustrato espiking in-silico produce spikes coherentes (no Poisson) conectado al contrato.
3. Un aprendizaje simple de tipo DishBrain (reward por predictibilidad / STDP) es posible en ese bucle sobre H1→G1.
4. Mismo código → real CL1/Cloud como *upgrade path* (narrativa del paper, no experimento aquí).

## Arquitectura target

```
MuJoCo (H1/G1) ──(estado)──► codificador (rate/time-to-first-spike)
                                  │ stim()  ▼
   cl-sdk Neurons.loop(1000 Hz)  ◄──►  Sustrato in-silico (BL-1 o Nengo)
        │ on_stim() callback              │
        ▼                                  └─ spikes
   decodificador (spikes→acciones) ──► MuJoCo
   DataStream: (x,vel,joints)+reloj ──► HDF5 (RecordingView)
```

## Stack (requiere **Python 3.12+**, instalado vía `uv`)

- `cl-sdk` (PyPI, local, CC BY-NC — vía oficial: `cl.sim.set_simulator_data_source()`)
- `mujoco 3.13` + MJCF G1 (Unitree RL Gym)
- `nengo 4.1` (CPU) — hub de decisión + spikes sintéticos en el datasource
- `torch 2.14+cpu` (solo para `deploy12` en el subproceso)
- Python 3.12.13 standalone instalado vía `uv python install 3.12` (winget/python.org hangs en este HW)

**venv**: `.venv312` con `mujoco==3.13.0, nengo==4.1.0, torch==2.14.0+cpu, cl-sdk`.

## Estructura del repo (propuesta)

```
02_cl/
├── README.md
├── requirements.txt          # py3.12 venv
├── src/
│   ├── contract.py           # Neurons.loop(1000) + record() + DataStream
│   ├── substrate/            # interfaz agnóstica de sustrato
│   │   ├── base.py           # on_stim() / produce_spikes()
│   │   ├── nengo_lif.py      # substrato LIF/NEF
│   │   └── bl1_jax.py        # substrato BL-1 (Izhikevich+STDP, UDP)
│   ├── bridge.py             # LiveSimulatorDataSource wrapper
│   ├── encoder.py / decoder.py
│   └── mujco_env.py
├── experiments/
│   ├── e1_latency.py         # latencia tick→stim→spike medida
│   ├── e2_closed_loop.py     # H1 sigue comando de velocidad
│   └── e3_learning.py        # reward-modulated STDP sobre gait
└── notebooks/
```

## Estado (realizado, sep 2026) — demo de lazo cerrado funcional

**`src/bridge_g1.py`** implementa un `SimulatorDataSource` custom (G1DataSource) que posee la `Deploy12` dentro del subproceso del simulador CL: codea el estado sensorial en 64 electrodos a 25 kHz (errores de joints `[0:12]`, actitud `[12:15]`, altura `ch23`, vx-override `ch62`, cultivo/hub `ch63`), avanza la física en fronteras de 50 Hz y aplica micro-estimulación en `on_stim` (torque overlay `[0:12]`, vx override `ch62`).

**`src/demo_walk.py`** cierra el contrato completo a 40 TPS:
`walker sensores → electrodos → detección de spikes culturales (spike-sorting propio sobre tick.frames) → hub Nengo LIF (NEF, 1000 neuronas, decide vx) → stims Myo-electric (μA) → on_stim → walker`.

Resultado (12 s, reproducible): **h_last=0.772 m, vx_last≈0.50 m/s, fallen=False**, hub cmd≈0.53, 50±5 eventos de stim adaptativos, ~8.8 spikes/tick. Evidencia: `results/f2_demo.json` + `results/f2_demo.png`.

### Hallazgos clave (API viva de cl-sdk)

- En esta versión del SDK **no hay modulo `cl.open`**, ni `LiveSimulatorDataSource`/`sink.emit_frames()`: **`cl.open()` es generador** y el datasource se registra con `set_simulator_data_source("modulo:factory", config=..., metadata=...)`.
- `neurons.loop(tps)`: el tick trae `tick.frames` (int16) y `tick.analysis.spikes`. **`analysis.spikes` NO detecta frames sintéticos** (solo funciona con ground-truth de replay/Poisson); la vía correcta es hacer spike-sorting propio sobre `tick.frames` (que sí reciben los datos del datasource). El datasource, si quiere, puede adjuntar `DataSourceBatch(frames=..., spikes=DataSourceSpike(...))`.
- `Neurons.stim(channel_set, stim_design, /, ...)` es **posicional-only**. Con float → `StimDesign(160us, −I, 160us, +I)`; límite de carga **3 nC** ⇒ 0..~18 μA (codificamos 4 μA per m/s). No enviar con cmd≈0 (polaridades iguales → ValueError).
- `run_steps(N)` de Nengo 4.1 cuesta ~100 ms **por llamada** (0.256 ms/step a N=400) ⇒ el hub corre en `HubThread` con ráfagas de 1.2 s por llamada; el lazo real lee `hub.latest`.
- **Fidelidad LIF**: con `dt=0.01` un NEF decodifica identidad de 2.0 como ~0.68 (el sustrato se degrada); con `dt=0.002` ≈ 1.95 ✓. El hub usa `dt=0.002, 600 pasos/ráfaga`.
- `read()` se invoca en chunks de 5 muestras: la generación de transitorios debe ser **por programa temporal global** (intervalo ∝ 1/|sensor|), no por chunk, para no multiplicar la densidad.
- El subproceso del datasource no garantiza `close()` al salir ⇒ la telemétrica del walker se escribe desde `_ctrl_step` cada 25 controles.

## Ablaciones M3 + latencias M4 (implementadas)

**`src/ablate_loop.py`** replica el protocolo doom-neuron sobre nuestro lazo: 4 modos (neural / zero / random / mask0.5) × 12 s, con métricas de latencia por tick (SDK loop) y de `read()` del puente (telemetría `bridge_read`). Evidencia: `results/f2_ablation.json`, `f2_ablation.png`, `f2_ablation_latency.png`.

| modo | mean_cmd | stims | mean_nspk | fallen | vx_last | loop p50/p95 | overruns>25ms | bridge read mean/max |
|---|---|---|---|---|---|---|---|---|
| neural | 0.533 | 49 | 8.9 | False | 0.50 | 47/62 ms | 488/499 | 30/172 ms |
| zero | 0.000 | 0 | 7.1 | False | 0.50 | 47/47 ms | 496/499 | 27/219 ms |
| random | 0.255 | 481 | 8.6 | False | 0.50 | 47/62 ms | 478/499 | 31/157 ms |
| mask0.5 (lesión 50%) | 0.479 | 63 | 6.0 | False | 0.50 | 47/63 ms | 499/499 | 32/141 ms |

### v2 — canal vx realmente conectado a la planta (bug reparado)

Auditoría post-M3 (`surrogate_cl`) reveló que en v1 el override de vx **nunca llegaba a la planta**: `cmd_override` solo se reflectaba al frame del canal 62 (lo que "ve" el sustrato) pero no a `deploy12.cmd[0]` (lo que observa la LSTM), por lo que **la trayectoria física era idéntica entre modos** (vx=0.50 en todos — un artefacto). El fix conecta el comando en `_ctrl_step` y se regeneró todo como `results/f2_ablation_v2.*` (los archivos v1 se conservan intactos).

| modo | mean_cmd | stims | mean_nspk | fallen | vx_last | loop p50/p95 | overruns>25ms | bridge read mean/max |
|---|---|---|---|---|---|---|---|---|
| neural | 0.531 | 70 | 9.1 | False | **0.55** | 31/31 ms | 296/499 | 14/78 ms |
| zero | 0.000 | 0 | 7.1 | False | 0.50 | 31/32 ms | 292/499 | 16/94 ms |
| random | 0.255 | 481 | 8.9 | False | **0.28** | 31/32 ms | 292/499 | 16/63 ms |
| mask0.5 (lesión 50%) | 0.482 | 94 | 6.3 | False | 0.50 | 31/32 ms | 292/499 | 17/79 ms |

**Lectura v2 (revisada):**
- Con el lazo funcionando, el comando decodificado **sí modula la velocidad real**: neural camina a 0.55 m/s, random (comando medio 0.255) a 0.28 m/s, zero/mask conservan el default 0.50 m/s. El lazo ya es un *controlador de velocidad* observable, no decorativo.
- En terreno plano y 0.5 m/s **ninguno cae** (redundancia de estabilidad se mantiene, ahora con causa correcta: la política base absorbe la modulación de velocidad dentro de su sobre), igual que doom-neuron.
- Cadencia M4 medida aquí: tick p50≈31 ms, p95≈31–32 ms, overruns ≈292/499 (>25 ms) — reportar en wall-clock. `bridge_read`: mean 14–17 µs, cola 63–94 ms.
- **Ojo historial**: los números v1 (`f2_ablation.json`) tienen el bug; solo v2 es válido para afirmaciones de modulación de velocidad. La M3 "redunda en plano" se conserva, ahora correctamente fundamentada.

**Lectura honesta (importante para el paper):**
- El lazo es **redundante en terreno plano a 0.5 m/s**: los 4 modos caminan igual (h=0.772, vx=0.50). La base G1 (PD+LSTM) absorbe la modulación — el mismo hallazgo que doom-neuron ("el decoder tiende a volverse policy head; las ablaciones aún juegan"). Esto es exactamente por qué la M3 es metodológicamente fuerte y por qué F3 necesita **perturbaciones/terreno irregular/cambios de velocidad** para que el lazo sea portante.
- Las métricas del lazo sí se diferencian (stims 0/49/481, nspk 6.0→8.9): el protocolo distingue modos, lo que respalda el claim de medición.
- **Cadencia (M4)**: tick real ~47 ms (p50), **no cumple el deadline de 25 ms** (overruns ~488/499). Causas: overhead del lazo + competición GIL con `HubThread` + bursts de física. Reportar siempre en wall-clock (el jitter del CL1 real lanza `TimeoutError`, no se simula). `bridge_read`: chunks de 5 muestras (67499 calls / 337495 samp en 12.5 s), mean 26–32 µs, tail 141–219 ms (bloque de física 50 Hz).

## Experimentos / métricas

| ID | Experimento | Métrica |
|---|---|---|
| F2-E1 | Latencia por tick (`Neurons.loop` 1 kHz) | time budget, jitter, round-trip medido |
| F2-E2 | Lazo cerrado CL-contract: H1 sigue cmd de velocidad | tracking error, estabilidad |
| F2-E3 | Aprendizaje tipo DishBrain (selección de gait, velocidad objetivo) | muestra de convergencia, spikes/acciones |
| F2-E4 | Sustrato BL-1 vs Poisson (control negativo) | divergencia de comportamiento |

## Entregables F2

- [x] `bridge_g1.py` operativo (cl-sdk ↔ walker MuJoCo)
- [x] Demo reproducible de lazo cerrado (12 s, walker camina) + `f2_demo.json/png`
- [x] **Ablaciones M3** (neural/zero/random/mask0.5, protocolo doom-neuron) + **latencias M4** (`f2_ablation.json/png`)
- [ ] Sustrato BL-1 en contrato (UDP) — pendiente/opcional
- [ ] Reporte de latencias vs presupuesto de control (argumento central) — pendiente (latencias ya medidas en `f2_ablation.json`; falta análisis vs budget 25 ms en paper)

## Publicación candidata

- Workshop **ICRA "Neuromorphic Field Robotics"** (NFR, activo desde 2026) o journal de biocomputing (Cyborg & Bionic / Frontiers) — primero-pero-a-escala trae la validación de tiempos reales del CL-contract en dinámica dinámica.

## Notas / ética

- Todo es **simulación**: no se cultivan células, no se compra CL1. Licencia CC BY-NC del SDK = uso académico correcto.
- El paper debe ser honesto: el SDK aporta el contrato; la "biología" es el sustrato espiking simulado.