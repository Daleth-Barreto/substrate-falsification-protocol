# Investigación para ICRA 2028 — Literatura y Mejoras

> Consolidación de 3 encuestas (web research, sept 2026) para reforzar el paper ICRA 2028 (~sep-2027): SNN legged locomotion SOTA, contrato CL / biocomputación closed-loop, y control neuromórfico / conversión ANN→SNN / energía.
> Código de verificación de URLs: `[verify]` = no confirmada en sesión; confirmar antes de citar en el paper.

---

## 1. Resumen ejecutivo

Nadie ha publicado el stack exacto **G1(humanoid 23-DoF, MuJoCo) + policy Nengo/NEF espiking + PD espiking en lazo interno + contrato CL1 (cl-sdk)**. Los más cercanos:

- **Unitree H1 NEF+SPA** (arm+locomotion, Nengo–Isaac Sim co-sim) — ICONS 2026, arXiv:2606.11034.
- **Nengo+Loihi control adaptativo de brazo Jaco 2** (2.45× más preciso que PID, ~4.6× menos potencia que CPU, lazo 2.5–4.5 ms) — arXiv:2007.10227.
- **Cart-pole / organoides corticales** (lazo cerrado real, 1-DoF) — Cell Reports 2026.
- **Pong (DishBrain) → Doom (CL1)** — juegos, sin actuación ni dinámica física.

**Ese hueco es nuestra contribución.** Nuestra línea narrativa "first CL-contract closed-loop legged robot (Unitree G1 at MuJoCo) via cl-sdk custom SimulatorDataSource" está respaldada por un barrido negativo de corpus (GitHub/arXiv/COSYNE/ICRA), a re-verificar antes de envío.

---

## 2. Panorama de literatura (mapa agrupado)

### 2.1 Políticas SNN para locomoción legged (2020–2026)

| # | Título | Fuente | Claim clave (1 línea) | URL |
|---|---|---|---|---|
| a1 | Fully Spiking Neural Network for Legged Robots | Jiang et al., arXiv:2310.05022 (ICASSP 2025) | Primer SNN como policy completa de locomoción (A1/Cassie/MIT-Humanoid), entrenada end-to-end en Isaac Gym con actor espiking por población; iguala a ANN en tareas de terreno | https://arxiv.org/abs/2310.05022 |
| a2 | Proxy Target: Bridging Discrete SNNs and Continuous Control | Xu et al., arXiv:2505.24161 (NeurIPS 2025) | Frame con proxy-stabilized soft-updates; simple LIF SNN supera a ANN en benchmarks continuos | https://arxiv.org/abs/2505.24161 |
| a3 | Synaptic Motor Adaptation (three-factor learning) | Schmidgall & Hays, arXiv:2306.01906 | Regla de plasticidad de 3 factores permite adaptación online (sim) igualando RMA sin información privilegiada | https://arxiv.org/abs/2306.01906 |
| a4 | Neuromorphic QP for MPC on ANYmal | Mangalore et al., arXiv:2401.14885 (IEEE RAM 2024) | Loihi 2 resuelve el QP del MPC >100× mejor en energy-delay product vs CPU/GPU OSQP, <10 ms | https://arxiv.org/abs/2401.14885 |
| a5 | Astrocyte-modulated CPG hexapod on Loihi | Polykretis et al., ICONS 2020 (10.1145/3407197.3407205) | Primer CPG espiking con astrocitos en Loihi controlando hexápodo; robusto a ruido y cambios de velocidad | https://doi.org/10.1145/3407197.3407205 |
| a6 | Dopamine-modulated spiking CPG (multi-gait) | Torre et al., IEEE 2025 `[verify]` | CPG espiking con neuromodulación dopaminérgica genera varios gaits y transiciones suaves en un cuadrúpedo | https://ieeexplore.ieee.org/document/11338745 |
| a7 | 12-neuron spiking CPG (walk/jog/run) | Rostro-Gonzalez et al., Frontiers 2025 | CPG espiking mínimo de 12 neuronas produce walk/jog/run en hexápodo con transiciones suaves | https://pmc.ncbi.nlm.nih.gov/articles/PMC12190837 |
| a8 | Astrocyte-regulated CPG + reward STDP | Han & Sengupta, IEEE TCDS 2026 (10.1109/TCDS.2025.3599472) | STDP+astrocytes aprende trote; 23.3× ahorro computacional vs RL SOTA | https://doi.org/10.1109/TCDS.2025.3599472 |
| a9 | Integrated arm+locomotion on Unitree H1 | ICONS 2026, arXiv:2606.11034 | NEF/SPA bipedal locomotion + brazo 4-DoF en H1 vía Nengo–Isaac Sim con selección basal-ganglia | https://arxiv.org/abs/2606.11034 |
| a10 | Spiking CPG lamprey (SpiNNaker vs Loihi) | Bartolozzi et al., NCE 2022 (10.1088/2634-4386/ac1b76) | CPG espiking (Nengo NEF) en dos plataformas; SpiNNaker mejor para real-time, Loihi mejor para eficiencia | https://doi.org/10.1088/2634-4386/ac1b76 |
| a11 | NeuroPod: SpiNNaker CPG hexapod | Gutierrez-Galan et al., Neural Networks 2020 (arXiv:1904.11243) | CPG espiking en SpiNNaker con 3 gaits y reconfiguración online en hexápodo físico | https://arxiv.org/abs/1904.11243 |
| a12 | NEF/REACH 7-DoF arm on Loihi | DeWolf et al., NCE 2023 (10.1088/2634-4386/acb286) | Control operacional NEF en Loihi: 4.13% RMSE vs analítico, ~100× menos energía por inferencia vs GPU | https://doi.org/10.1088/2634-4386/acb286 |
| a13 | Nengo + low-power AI for embedded neurorobotics | Mayol et al., Front. Neurorobot. 2020 (arXiv:2007.10227) | Rover nav + brazo Jaco con PES on-chip en Loihi; flujo completo Nengo→Loihi | https://arxiv.org/abs/2007.10227 |
| a14 | Evolving connectivity for RSNN | Cheng et al., arXiv:2305.17650 | Evolución de conectividad entrena RSNN en humanoide 17-DoF a paridad con RNN profundas | https://arxiv.org/abs/2305.17650 |
| a15 | Vision+control drone en Loihi (7–12 mW) | Paredes-Vallés et al., Science Robotics 2024 (10.1126/scirobotics.adi0591) | Pipeline event-camera + SNN end-to-end en Loihi: hover/landing autónomo a 7–12 mW | https://doi.org/10.1126/scirobotics.adi0591 |
| a16 | SpikeGym: SNN vs ANN en Isaac Gym | PMC11680704 (2024) `[verify]` | SNNs quedan por detrás de ANNs en redes profundas (Ant task) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11680704 |
| a17 | Error amplification limits ANN→SNN conversion (control) | Xu et al., arXiv:2601.21778 (ICML 2026) | La conversión degrada severamente en control continuo por amplificación de errores correlacionados en el tiempo; CRPI mitiga | https://arxiv.org/abs/2601.21778 |
| a18 | Reconsidering SNN energy efficiency | Yan et al., arXiv:2409.08290 | SNN solo vence al ANN cuantizado con >93% de sparsity (T=6); mayoría de comparaciones ignoran memoria | https://arxiv.org/abs/2409.08290 |
| a19 | ANN vs SNN en recursos limitados | Davidson et al., PMC8055931 (2021) | La mayoría de SNN rate-coded en hardware estándar NO son más eficientes que el ANN original | https://pmc.ncbi.nlm.nih.gov/articles/PMC8055931 |
| a20 | Trajectory generation on Loihi | Michaelis et al., Front. Neurorobot. 2020 (10.3389/fnbot.2020.589532) | Red anisotrópica en Loihi almacena/generaliza trayectorias motoras secuenciales | https://doi.org/10.3389/fnbot.2020.589532 |
| a21 | Spike-based RL hexapod (STDP) | Lele et al., AICAS 2020 (arXiv:2003.10026) | Aprendizaje online de gait tripod con STDP desde giro/cámara | https://arxiv.org/abs/2003.10026 |
| a22 | End-to-end model-based SNN control | Huebotter et al., NCE 2026 (arXiv:2509.05356) | Controlador espiking model-based multi-DoF a paridad con baselines no-espiking, menos parámetros | https://arxiv.org/abs/2509.05356 |

### 2.2 Closed-loop biológico / contrato CL / biocomputación (2022–2026)

| # | Título | Fuente | Claim clave (1 línea) | URL |
|---|---|---|---|---|
| b1 | **CL API: Real-Time Closed-Loop with Biological Neural Networks** | arXiv:2602.11632 (2026) | Define el "contrato" CL (admisión transaccional de stim, orden determinista, sincronía explícita, latencia sub-ms). **Ancla de nuestra narrativa.** | https://arxiv.org/abs/2602.11632 |
| b2 | cl-sdk — CL API Simulator | GitHub Cortical-Labs/cl-sdk | Réplica local del API CL1 con SimulatorDataSource personalizados; fija la versión exacta del API | https://github.com/Cortical-Labs/cl-sdk |
| b3 | cl.sim module docs | docs.corticallabs.com/cl/sim.html | Fuentes pull/push (`read(from_timestamp, frame_count)`), `on_stim()`, batch; defaults channel_count=64, fps=25k | https://docs.corticallabs.com/cl/sim.html |
| b4 | cl.Neurons/loop/accelerate docs | docs.corticallabs.com/cl.html | `loop()` cede spikes+stims hasta 25 kHz; jitter real en CL1 → TimeoutError (no simulable); modo acelerado existe | https://docs.corticallabs.com/cl.html |
| b5 | DishBrain: neurons play Pong | Kagan et al., Neuron 110(23) 2022 | ~800k neuronas juegan Pong en ~5 min con feedback estructurado (free-energy) | https://doi.org/10.1016/j.neuron.2022.09.001 |
| b6 | DishBrain beats deep RL (sample efficiency) | Khajehnejad et al., arXiv:2405.16946; Cyborg & Bionic Syst. 6:0336 (2025) | DishBrain supera a DQN/A2C/PPO en ~70 episodios (~5 min); plasticidad dinámica | https://arxiv.org/abs/2405.16946 |
| b7 | CL1 plays Doom (Freedoom) | Cortical, mar 2026 (Tom's Hardware etc.) | ~200k neuronas → estado de juego como estimulación eléctrica → spikes decodificados a move/aim/fire; aprende en ~1 semana | https://www.tomshardware.com/tech-industry/artificial-intelligence/200-000-living-human-neurons-on-a-microchip-demonstrated-playing-doom-cortical-labs-cl1-video-shows-the-gameplay-and-explains-how-the-neurons-learn-the-game |
| b8 | doom-neuron repo (ablación honesta) | SeanCole02/doom-neuron (GPL-3.0) | El device no computa; PPO/encoder/decoder en PyTorch sobre UDP; ablaciones random/zero decoder aún juegan | https://github.com/SeanCole02/doom-neuron |
| b9 | Goal-directed learning in organoids (cart-pole) | Robbins et al., Cell Reports 45(2):116984 (2026) | Organoides corticales de ratón en cart-pole (rate-coded); éxito 4.5%→46% con RL adaptativo; requiere glutamato | https://www.cell.com/cell-reports/fulltext/S2211-1247(26)00062-8 |
| b10 | Organoid reservoir computing / OI surveys | Cai et al., Nat. Electronics 2023; Talavera & Ulmann, arXiv:2503.19770 | Reservorios de organoides 3D para reconocimiento de voz y predicción no lineal; surveys de OI | https://www.nature.com/articles/s41928-023-01069-w |
| b11 | In-vitro BNNs for robot intelligence — review | Cyborg and Bionic Systems, 2024 (10.34133/cbsystems.0001) | Surveys de BNN-on-MEA closed loops (Shahaf/Marom, DeMarse); loops 1–10 DoF, sin plataforma comercial | https://spj.science.org/doi/10.34133/cbsystems.0001 |
| b12 | Neuron-connected robots (MSR + UTokyo) | Microsoft Research, 2022– | Primer paso: simular sustrato neuronal para no dañar cultivos — valida pipeline sim-first | https://www.microsoft.com/en-us/research/project/neuron-connected-robots |
| b13 | Rat-hippocampus culture drives robot | PLOS ONE 11(10):e0165600 (2016) | Redes hipocampales driving mobile robot; reporta honesto: bio-lazo superado por baseline silicon | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165600 |
| b14 | Fungal mycelia robot control | Mishra et al., Science Robotics 9(93):eadk8019 (2024) | Micelio de ostra controla rueda y "starfish" 5-patas con gait switching por UV | https://www.science.org/doi/10.1126/scirobotics.adk8019 |
| b15 | φ-Bot: slime mould hexapod | Tsuda et al., 2009 | Physarum oscila piernas de insectoide; legged-bio-control histórico no neuronal | https://eprints.soton.ac.uk/268247/1/TsudaS08AlifeInHardware.pdf |
| b16 | Brain-inspired motor control review | Mompó Alepuz et al., Front. Neurorobot. 18:1429445 (2024) | Revisa controladores de inspiración brain (SNN, cerebelo, basal ganglia); ninguno combina tejido vivo + contrato real-time | https://pmc.ncbi.nlm.nih.gov/articles/PMC11366706 |
| b17 | NEURON + MuJoCo co-simulation | bioRxiv 2025.06.17.660217 | Modelos neurales NEURON driving musculoesquelético MuJoCo open/closed loop — **vecino metodológico directo** | https://www.biorxiv.org/content/10.1101/2025.06.17.660217v1.full.pdf |
| b18 | **No public CL1/cl-sdk robot control found** | barrido 2026 (negativo) | Toda lazo cerrado CL público es juego (Pong/Doom) o demo cloud; **el hueco que reclamamos** | — (re-verificar al envío) |
| b19 | BL-1 in-silico cortical culture (JAX/Izhikevich) | GitHub m9h/bl1 (MIT) | Sustituto realista del sustrato SDK: 64-ch virtual CL1 MEA, Pong/ViZDoom, validado contra Wagenaar 2006 | https://github.com/m9h/bl1 |
| b20 | Scale-NeuroEval: LLM-designed environments for OI | arXiv:2509.04633 / NeurIPS 2025 | Entornos virtuales closed-loop diseñados por LLM + evaluación de plasticidad para agentes organoide | https://arxiv.org/html/2509.04633v1 |
| b21 | CL1 hardware/cost economics | Wikipedia + DataCenterDynamics (2026) | ~$35k/unit, ~$300/wk cloud, cultura ~6 meses, ~200-800k neuronas; Bio Data Centre Melbourne (120 units) | https://en.wikipedia.org/wiki/Cortical_Labs |
| b22 | Organoid-intelligence ethics | Smirnova, Nat Rev Bioeng 2024; Baltimore Declaration, Front. Sci. 2023 | Roadmap OI + ética formalizada | https://www.nature.com/articles/s44222-024-00200-6 |
| b23 | ICRA 2026 workshop: Neuromorphic Field Robotics | IEEE ICRA Vienna (jun 2026) `[verify]` | Taller de robótica neuromórfica; sin track de biocomputación — la sala que podemos ocupar | https://2026.ieee-icra.org/workshops-and-tutorials |

### 2.3 Control neuromórfico / conversión ANN→SNN / energía

| # | Título | Fuente | Claim clave (1 línea) | URL |
|---|---|---|---|---|
| c1 | Neuromorphic computing for embodied intelligence (survey) | arXiv:2507.18139, IEEE IOLTS 2025 | Survey de SNN algorithms+hardware+workflows cross-layer para sistemas autónomos | https://arxiv.org/abs/2507.18139 |
| c2 | Benchmarking framework for embodied neuromorphic agents | Nat. Mach. Intell. 8:300–312 (2026) `[verify]` | Propone tareas/métricas/plataforma física para "brains" embebidos | https://doi.org/10.1038/s42256-026-01197-w |
| c3 | NeuroBench | arXiv:2304.04640, Nat. Commun. 16:1545 (2025) | Benchmark dual (algorithm+system) con protocolos de medida estándar p/ energía/latencia | https://arxiv.org/abs/2304.04640 |
| c4 | Toward large-scale SNNs (conversion vs direct survey) | arXiv:2409.02111 (2024) | Survey: conversión vs surrogate-gradient para redes grandes | https://arxiv.org/abs/2409.02111 |
| c5 | Inference-scale complexity in ANN→SNN conversion | CVPR 2025, arXiv:2409.03368 | Threshold balancing local: ResNet-34 @90% = 622 FPS/W vs 22 (ANN), ~28×, sin retrain cuantizado | https://arxiv.org/abs/2409.03368 |
| c6 | Differential coding (training-free conversion) | arXiv:2503.00301 | Spikes transmiten cambios de tasa; VGG-16 73.17% con ~22% de potencia ANN | https://arxiv.org/abs/2503.00301 |
| c7 | PASCAL: precise conversion | arXiv:2505.01730, TMLR 2025 | Conversión equivalente a ANN cuantizado QCFS; 64× menos inference timesteps | https://arxiv.org/abs/2505.01730 |
| c8 | One-timestep conversion (Scale-and-Fire) | arXiv:2510.23383 (2025) | 88.8% ImageNet-1K a T=1 usando ratio 4.6pJ/MAC vs 0.9pJ/AC | https://arxiv.org/abs/2510.23383 |
| c9 | Error amplification limits conversion (control) | arXiv:2601.21778, ICML 2026 | Errores temporalmente correlacionados → drift en MuJoCo control; CRPI (membrana residual) recupera casi todo | https://arxiv.org/abs/2601.21778 |
| c10 | Reconsidering SNN energy (hardware-aware) | arXiv:2409.08290, ICASSP 2024 | Incluyendo memoria: SNN gana solo con T∈[5,10] y tasa<6.4%; VGG-16 T=6 necesita >93% sparsity | https://arxiv.org/abs/2409.08290 |
| c11 | Rethinking SNN/ANN energy (accelerator-validated) | ACM TACO 23(3), Art. 90 (2026) `[verify]` | Ratio SNN/ANN crece con timesteps y satura; pruning ayuda más a SNN | https://doi.org/10.1145/3822176 |
| c12 | Nengo+Loihi adaptive arm (vs PID) | arXiv:2007.10227, Front. Neurorobot. 2020 | Jaco 2: adaptativo 2.45× + preciso que PID, ~4.6× menos potencia que CPU, lazo 2.5–4.5 ms | https://arxiv.org/abs/2007.10227 |
| c13 | NEF spiking PID (3-DoF arm) | Sensors 24(2):491 (2024) | PID espiking NEF (300 LIF) supera PID clásico (6% ITAE, 30% RMSE) y fuzzy (5% ITAE) | https://doi.org/10.3390/s24020491 |
| c14 | Bioinspired smooth neuromorphic arm control | arXiv:2209.02787 (2023) | SNN compacto de joints en Loihi, perfiles bell-shaped, bajo jerk, comparable a PID en Jaco real | https://arxiv.org/abs/2209.02787 |
| c15 | ED-BioRob: spike-based PID on FPGA | Front. Neurorobot. 2020 (10.3389/fnbot.2020.590163) | PID 100% espiking (SSP building blocks), acepta refs de neuronas Dynap-SE; BioRob 4-DoF | https://doi.org/10.3389/fnbot.2020.590163 |
| c16 | Parsimonious adjustable neuromorphic PID on Loihi | ACM IDT 2022 `[verify]` | PID espiking ajustable de ~93 neuronas en Loihi | https://doi.org/10.1145/3546790.3546799 |
| c17 | NEF LQR for cart-pole | arXiv:2507.03621 (2025) | NEF/LIF implementan LQR con 7 métricas de control + 7 neuromórficas | https://arxiv.org/abs/2507.03621 |
| c18 | Spiking control taxonomy review | arXiv:2509.05356, NCE 2025 | Taxonomía: spike-coding analítico, predictivo, brazos NEF (DeWolf), RL surrogate (PopSAN ~140×), conversión | https://arxiv.org/abs/2509.05356 |
| c19 | Loihi / Loihi 2 | IEEE Micro 38(1):82 2018; Proc. IEEE 109(5):911 2021 | Hardware primario Nengo; Loihi 2 añade neuronas programables/graded spikes | https://doi.org/10.1109/MM.2018.112130359 |
| c20 | Akida energy (vendor) | BrainChip user guide `[verify]` | ~23.48 mJ/frame ImageNet en AKD1000; datos vendor sin peer-review — re-medir | https://doc.brainchipinc.com/user_guide/akida.html |
| c21 | Speck event-based vision SoC | arXiv:2304.06793, Nat. Commun. 15:4464 (2024) | DVS+SNN single-chip: 3.36 µs/layer, ~0.7 mW real-time, <0.1 ms latencia, 320k neuronas | https://arxiv.org/abs/2304.06793 |
| c22 | SENECA: digital neuromorphic RISC-V | arXiv:2303.15224 (2023) | Contabilidad energética a nivel instrucción (~2.8 pJ/syn-op) → co-design hardware-aware | https://arxiv.org/abs/2303.15224 |
| c23 | Deng et al. energy analysis | Neural Networks 121:294–307 (2020) (10.1016/j.neunet.2019.09.005) | Comparación SNN/ANN incluyendo memoria | https://doi.org/10.1016/j.neunet.2019.09.005 |

---

## 3. El hueco (claim principal del paper)

- **Nadie** ha cerrado el lazo de un humanoide completo (G1, 23-DoF, contact-rich, ~kHz control) a través del **contrato CL** con sustrato espiking.
- Línea evolutiva: DishBrain (Pong, juego) → Doom (8 acciones discretas) → cart-pole orgánico (1-DoF) → **G1/MuJoCo (nuestro eje nuevo: estabilidad, contactos, control de alto-frecuencia)**.
- Baselines biológicos legged existentes son CPG/no-neuronales y lentos (mycelia, Physarum) o hobby 1–10 DoF sin contrato documentado.
- Nuestro vecino metodológico: NEURON+MuJoCo co-sim (b17); nuestro vecino de stack: H1 NEF+SPA (a9).
- Formular la contribución como **"contrato bioinspirado"** (timing/ordering/synchronización; cita b1) con cl-sdk como test harness de cumplimiento — reproducible y revisable.

---

## 4. Mejoras concretas mapeadas a debilidades del pipeline

Hechos conocidos de nuestro stack (F1/F2): fidelidad NEF degradada a dt=0.01 (identidad 2.0→0.68), 1.951 a dt=0.002; lazo híbrido ANN→SNN cae ~8s por conversión imperfecta; overhead `run_steps` ~100 ms/llamada en CPU (budget de latencia CL); read() en chunks de 5 muestras; límite 3 nC de stim.

| # | Mejora | Debilidad que ataca | Evidencia | Estado |
|---|---|---|---|---|
| M1 | Ejecutar todo a dt=0.002 (sin decimar a 0.01) | Fidelidad NEF identity 0.68 | 1.951@0.002; c5/c7 conversión precisa | ✅ ya en hub (HUB_HORIZON_STEPS=600) · lleva a la demo |
| M2 | Sustituir spikes aleatorios del SDK por sustrato con dinámica neural real | El lazo F2 con SDK random no es "auténtico" vs crítica RD-World | b19 BL-1 (JAX/Izhikevich, 64-ch MEA virtual, habla protocolo CL1 UDP); b3 `on_stim` | pendiente (opcional F2+/F3) |
| M3 | Ablación estilo doom-neuron: decoder random/zero, spike-gated action masking, fracción de control que portan las neuronas | Reviewer: "el decoder lo hizo todo" | b8 (245★, GPL) | ✅ implementada (`02_cl/src/ablate_loop.py`, `results/f2_ablation.json`) · hallazgo: lazo redundante en plano a 0.5 m/s → F3 necesita perturbaciones |
| M4 | Reportar latencias del lazo: histogramas µs, tick overruns, wall-clock vs accelerated; jitter real CL1 = TimeoutError | Budget de latencia ~100 ms/call `run_steps`; pacing del paper | b4 docs; c12 reporta 2.5–4.5 ms | ✅ implementada: tick p50=47ms (overruns ~488/499 vs 25ms), `bridge_read` mean 30µs/tail 172ms |
| M5 | Curva de energía con memoria contada, T y sparsity fijas; protocolo NeuroBench | Claims de energía sin fundamento | a18, c3, c10, c11, a19 | pendiente para F3 |
| M6 | Política: preferir end-to-end SNN / coding temporal a conversión ANN→SNN para el lazo de control | Caída del lazo híbrido a ~8s (conversión imperfecta) | a17/c9 (error amplification), a2 (proxy target), a1 (poblacional) | pendiente (F3 o mejora F1) |
| M7 | Endorsar el PD espiking con cita dedicada (mejora 6–30% vs PID clásico) | SpikingPD caracterizado pero sin literatura en el paper | c13 Sensors 24(2):491; c17 NEF LQR; c14/15/16 PIDs loihi/FPGA | listo para escribir |
| M8 | Incluir baseline "bio-loop vs silicon loop" honesto | Expectativas de reviewers sobre wetware ganando | b13 (PLOS ONE 2016 honesto), a16 (SNN pierde en deep), a17 | **recomendado** (ya lo hacemos con E1-E4) |
| M9 | Posicionar como pipeline sim-first (contrato testbed, no ground truth biológico) | Licencia CC BY-NC + riesgo de sobre-vender | b2 licencia, b12 (MSR sim-first), caveats sección 6 | ya es nuestra postura |
| M10 | Enviar a contenido/venue correcto | ICRA 2028 full paper o workshop neuromórfico | b23 (ICRA 2026 workshop sin biocomputación) | plan IROS 2027/ICRA 2028 |

---

## 5. Ablaciones y honestidad (recomendadas en el paper)

1. **Random/zero-decoder baselines** (protocolo doom-neuron, b8): el humanoide debe caer o degradarse claramente si el decoder discrepa de la dinámica neural. Repetible en F2: misma demo con decoder aleatorio.
2. **Spike-gated action masking**: máscara de acción por spikes (las piernas solo se mueven si el sustrato emite), cuantifica fracción de control portada por neuronas.
3. **Bio-loop vs silicon-loop**: contraponer nuestro lazo CL-contrato a un lazo equivalente ANN/PD (extiende E1-E4 ya existentes) — precedente honesto b13.
4. **Negativos que citar**: SNN profundos pierden ante ANN (a16), conversión pierde dinámica temporal (a17/c9), rate-coding en von Neumann no es más eficiente (a19/c10). Esto "desarma" objeciones antes de que lleguen.

## 6. Caveats legales/éticos/lógicos

- **Licencias**: cl-sdk CC BY-NC 4.0 (no comercial); whitepaper CL API CC BY-NC-SA; doom-neuron GPL-3.0; BL-1 MIT. Manejar cada una separadamente en artefactos de reproducibilidad; declarar postura de licencia del código.
- **Costos/HW (revisores preguntarán)**: CL1 ~$35k, cloud ~$300/semana, vida de cultura ~6 meses, claims de ~30W disputados. Nuestro pipeline sim-first ($0) es el argumento de des-riesgo.
- **Honestidad del simulador**: el simulador SDK genera datos "no-learning" que no responden a stim — por eso nuestro `on_stim` source es necesario; en el paper el sim es *testbed de contrato y lazo*, nunca ground truth biológico.
- **Hueco HW-SW**: TimeoutErrors por jitter y otras conductas del CL1 real no se reproducen en sim; especificar modo accelerated vs wall-clock usado.
- **Negativo como snapshot**: "no existe CL-contract robot control público" es un barrido a sept 2026; **re-correr antes del envío**.

---

## 7. Propuesta (siguientes pasos)

1. **M3 + M4 — IMPLEMENTADAS** (`02_cl/src/ablate_loop.py` + `results/f2_ablation.json/png`; ver interpretación en `02_cl/README.md`). Hallazgo clave: el lazo es **redundante** en terreno plano a 0.5 m/s (4 modos caminan igual; igual que doom-neuron). ⇒ Para que el lazo sea portante, F3 debe usar **perturbaciones, terreno irregular o cambios de velocidad**; con eso las ablaciones random/zero mostrarán degradación y la narrativa queda reforzada.
2. **M2 (opcional)**: probar BL-1 como sustrato en `bridge_g1.py` (`on_stim`) para que F2 tenga dinámica neural documentada. Riesgo: timeout del proyecto; beneficioso para la narrativa.
3. **M10**: decidir venue (ICRA 2028 paper vs workshop neuromórfico ICRA 2026) y re-correr el barrido negativo (b18) al despertar/durante drafting.
4. Evaluar **M6**: convertir la política de F1 hacia NEF/end-to-end en vez de conversión ANN→SNN — potencialmente el cambio más fuerte para F3.

---

## 8. Nueva línea: "sustrato CL no-entrenable ↔ sustituto SNN entrenable (surrogate)"

**Paradigma reformulado:** el wetware real (DishBrain/CL1) **no es entrenable por gradiente** — no hay backprop, ni etiquetas en el sentido DL; solo condicionamiento closed-loop (reward/stim) costoso y éticamente limitado. ⇒ La parte entrenable se implementa como **SNN in-silico** (Nengo/NEF o surrogate-gradient), se entrena en simulación, y se despliega **sin tocar código** sobre el contrato CL (mismo I/O de spikes↔stims). **El contrato hace el swap drop-in: entrenas una vez in-silico, corres verbatim en wetware.** Esto convierte nuestra "debilidad" (redundancia M3) en la tesis central.

**Precedentes que validan (y que nadie ha llevado a humanoide completo):**

| # | Item | Fuente | Punto que valida | URL |
|---|---|---|---|---|
| d1 | doom-neuron — entrenar lo entrenable, no el wetware | SeanCole02/doom-neuron | PPO/REINFORCE sobre encoder/decoder UDP; la cultura NO se entrena, el decoder sí; ablación muestra el límite | https://github.com/SeanCole02/doom-neuron |
| d2 | **Assembloid Agency** (bridge backend-agnóstico) | jennnital/UE-CL1-API, NeurIPS 2025 Creative AI Track | "El mismo plugin conduce CL1 o un NEST/SNN/EEG stand-in"; `--organoid` Brian2/LIF simulator + `SendRewardSignal` (reward==stim, estilo DishBrain) | https://github.com/jennnital/UE-CL1-API |
| d3 | BL-1: virtual CL1 server diferenciable | m9h/bl1 (JAX, MIT, activo) | Sustituto in-silico entrenable y drop-in del CL1 (64-ch, UDP): implementa literalmente nuestro concepto | https://github.com/m9h/bl1 |
| d4 | MSR neuron-connected robots: sim-first obligado | Microsoft Research 2022– | Entrenar en sim primero para no dañar el cultivo = estándar; valida "entrenar in-silico, validar en wetware" | https://www.microsoft.com/en-us/research/project/neuron-connected-robots |
| d5 | Organoid cart-pole: adaptación por RL externo | Robbins et al., Cell Reports 45(2), 2026 | Aprendizaje por condicionamiento (éxito 4.5%→46%), NO backprop — el clamp es el entrenable | https://www.cell.com/cell-reports/fulltext/S2211-1247(26)00062-8 |
| d6 | Survey substrate-independent biocomputing (SBI) | arXiv:2604.27933 (cs.ET, 2026) | Marco "sustrato como platform" intercambiable — cita para el contrato como capa de abstracción | https://arxiv.org/abs/2604.27933 |
| d7 | MetaBOC (robot-organoides) | Tianjin/SUSTech 2024 (press) | Claim de robots por organoides; **problemas de control = press-only [verify]** — comparativo negativo que nos favorece (ellos sin contrato ni decodificación publicada) | https://teqnoverse.com/metaboc/ |

**Matiz honesto importante:** "no entrenable" es matizado — DishBrain *sí* aprende por feedback estructurado (Pong, ~5 min, arXiv:2405.16946). La tesis correcta es: **no entrenable por DL/gradiente; sí adaptable por condicionamiento → por eso la parte entrenable va en el surrogate SNN, no en el wetware.**

**Aporte reformulado (más fuerte que "primera demo"):**
> "Primer lazo de un humanoide completo (G1, 23-DoF) en el que el sustrato biológico no-entrenable se sustituye por un SNN entrenable intercambiable vía el mismo contrato CL: entrenar una vez en simulación, correr verbatim sobre el wetware."

**Estado empírico (sept 2026, `surrogate_cl/`):** G1 12-joint (MuJoCo 500 Hz, política
LSTM 50 Hz), hub Nengo 1000-LIF cerrando el mismo contrato (spikes↔stims). H1 final =
**seguimiento del perfil de velocidad por el canal 62** (tarea tipo telemetría de
líder/obstáculo). Tabla multi-seed (5 semillas, RMSE cmd-vs-perfil): neural
**0.191±0.008** (único modo que modula: cmd 0.55→0.63 con el segmento 0.5→0.8; ch62
0.69→1.13), random 0.336±0.003 (plano ~0.38, sin vínculo causal), lesión mask0.5
0.305±0.036 (ch62 dañado según semilla), zero 0.612 (sin comando). Hallazgos de
honestidad que sostienen la tesis: (i) en plano **los 4 modos caminan** — la
estabilidad de planta no separa, la **tarea sí**; (ii) empujes sostenidos ≥0.5 s/140 N
volcan a todos (autoridad del canal ≤30 N·m vs ~112 N·m de volcamiento) — control
negativo limpi porque es frontera del *contrato*, no de función; (iii) bug del canal
vx reparado (v1→v2) para que el comando real llegue a `deploy12.cmd[0]`.

## 9. Oportunidades (venues / competencias / programas 2026–2027)

> **Restricciones del equipo (descartan ICOI 2026):** sin presupuesto para inscripción/viaje y sin visa/becas ⇒ prioridad a **rutas sin costo**: ICRA 2028 fijo, revista top sin APC (IEEE RA-L no cobra por publicación estándar) y opciones OA con waivers **a verificar**. ICRA 2027 NO se descarta por la vía **RA-L → transferencia** (ver abajo).

| Target | Tipo | Ventana / deadline | Por qué encaja | URL |
|---|---|---|---|---|
| **IEEE RA-L** (primera opción de revista) | journal | rolling; transferencia a ICRA 2027 **1 mar–31 dic 2026** ✅verificado | **Sin APC** estándar (hybrid OA); IF 5.3 (2025); "ICRA Option" permite enviar RA-L con opción a presentar en ICRA; también a IROS 2027 (1-ago-2026 → 30-abr-2027) y Humanoids 2027. **Matiz realista**: la aceptación (~2 decisiones en ≤5 meses) debe caer antes del cierre de la ventana para presentar en ICRA 2027 | https://www.ieee-ras.org/publications/ra-l/ |
| **ICRA 2028** | conference | deadline ~**sep 2027** (meta fija) | Paper completo con tarea portante + línea surrogate | https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/ |
| IROS 2027 (Florencia) | conference | paper **1 mar 2027** | Plan B archivístico si RA-L→transfer no encaja | https://www.ieee-ras.org/event/2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-70525/ |
| ~~ICOI 2026~~ | conf. | ~~26–27 oct 2026~~ | ❌ **Descartado**: costo + visa | https://organoidintelligence.org/icoi/ |
| Cyborg and Bionic Systems | journal | rolling (IF ~18.1) | Fit temático híbrido orgánico–mecatrónico; **verificar APC/waiver por región antes de comprometer** | https://attend.ieee.org/cbs-2026/ |
| Frontiers RT 68835 | journal (RT) | acepta ahora | Ground-truth citable temprano; **APC a verificar (waiver por país)** | https://www.frontiersin.org/research-topics/68835/neuromorphic-engineering-and-brain-inspired-control-for-autonomous-robotics-bridging-neuroscience-and-ai-for-real-world-applications/articles |
| NCE SI (IOP) | journal SI | **30 nov 2026** | IOP; lazos sensor→compute→act SNN en scope; APC/waiver a verificar | https://iopscience.iop.org/collections/nce-260407-1103 |
| NFR workshop (ICRA 2028) | workshop | pre-proposal ~sep 2027 [verify] | Proponer taller del paradigma surrogate↔contrato en la misma ventana que el paper | https://nfr-icra2026.com/ |
| Telluride 2027 | programa 3 semanas | **27 jun–16 jul 2027**; apps ~feb–mar 2027 | Integración community (becas/fellowships a verificar) | https://sites.google.com/view/telluride-2026/home |
| **HVAC 2027** | competencia | ~ago–sep 2027 [verify] | Benchmark humanoide-sim para gait SNN/PD vs clásicos (sin costo) | https://ytazz.github.io/vnoid/ |
| ALIFE 2027 (Praga, 40 aniv.) | conference | **19–23 jul 2027** | Fit bio-inspired robotics; costo de viaje es el límite | https://2027.alife.org/ |

**Notas:** ICRA 2027 paper completo ya **NO** (deadline 15-sep-2026); quedan dos vías a
ICRA 2027: **RA-L "ICRA Option" + transferencia** (aceptación antes de 31-dic-2026,
✅ verificado en ieee-ras.org) y **workshop papers** (deadlines por workshop, típ.
ene–mar 2027). Presentación más realista a corto plazo: IROS 2027 (ventana RA-L hasta
30-abr-2027) o Humanoids 2027. Reclamos de energía exigen NeuroBench; cl-sdk CC BY-NC
no afecta la publicación (código académico).

## 10. Recomendación (con la nueva línea)

1. **Estrategia de publicación (sin costo)**: **IEEE RA-L primero** (submisión rolling, sin APC) → decidir transferencia a **ICRA 2027** (ventana hasta 31-dic-2026) o **IROS 2027**; **ICRA 2028 fijo** con el paper completo (tarea portante + línea surrogate + ablaciones M3/M4). Revista alternativa temática: Cyborg and Bionic Systems (verificar APC/waiver).
2. **Corto (hecho sept 2026, `surrogate_cl/`)**: la tarea portante se diseñó como
   **seguimiento del perfil de velocidad por ch62** (push + pendiente + cambios de vx se
   exploraron primero). Hallazgo decisivo: en plano **la estabilidad no separa** (los 4
   modos caminan; empuje sostenido vuelca a todos por límite de autoridad del contrato),
   pero el **tracking + lesión sí** (RMSE neural 0.191±0.008 vs 0.305–0.612). Pendiente
   para el paper: multi-perfil, terreno irregular y sustituir los transitorios del bridge
   por un sustrato entrenable real (BL-1 / doom-neuron encoder-decoder / surrogate-gradient).
3. **Medio**: decidir sustrato entrenable real: **BL-1** o el esquema encoder/decoder de doom-neuron en vez de los transitorios del bridge (refuerza "intercambiabilidad"); CL Cloud para A/B real si algún día hay presupuesto.
4. **Ventanas**: RA-L/BL-1 loop al mes 1–2; IROS 2027 (si no hay transfer) ene–feb 2027; ICRA 2028 ~ago 2027. HVAC 2027 (ago–sep) y Telluride 2027 como demostraciones/integración sin costo de inscripción.