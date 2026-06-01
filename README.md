# 🧠 gBrain — Digital Brain / Cerebro Digital

> A biologically-inspired **Spiking Neural Network** simulating 8 interconnected brain regions, real synaptic plasticity (STDP), global neuromodulation and a live 3D dashboard.
>
> Red neuronal **de impulsos (SNN)** bio-inspirada que simula 8 regiones cerebrales interconectadas, plasticidad sináptica real (STDP), neuromodulación global y un dashboard 3D en vivo.

**🔴 Live demo / Demo en vivo:** [gbrain-production-7f5c.up.railway.app](https://gbrain-production-7f5c.up.railway.app)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue.svg)
![Node](https://img.shields.io/badge/Node-%E2%89%A520-green.svg)

---

## 🇪🇸 Español

### Objetivo

gBrain **no es un LLM**. Es un intento de modelar el aprendizaje *como lo hace un cerebro real*: neuronas que disparan impulsos (spikes), sinapsis que se refuerzan o debilitan según el *timing* de esos impulsos (STDP), y neurotransmisores que modulan todo el sistema en cada instante. El objetivo es que el sistema **aprenda de verdad por experiencia** —agrupando estímulos, vinculándolos y reforzando esos vínculos con la repetición— y que ese aprendizaje sea **visible y verificable**, no una caja negra.

Tres propiedades guían el proyecto:

1. **Biológicamente plausible** — neurona de Izhikevich, STDP, homeostasis, conectoma humano.
2. **Aprendizaje real y medible** — cada capacidad (visión, memoria, lenguaje) tiene un test cuantitativo que demuestra que aprende.
3. **Observable en vivo** — dashboard 3D con la actividad de cada región y los niveles hormonales en tiempo real.

### Características principales

- **Núcleo SNN** — Neurona de **Izhikevich** (RS/FS/CH/IB), plasticidad **STDP** (LTP/LTD), homeostasis y escalado sináptico.
- **8 regiones cerebrales** que heredan de una clase base común `BrainRegion`:
  | Región | Función |
  |--------|---------|
  | Tálamo | Filtro sensorial y cuello de botella de atención |
  | Corteza Visual | Procesa imágenes (webcam), k-WTA + fatiga neuronal |
  | Corteza Auditiva | Procesa audio (micrófono), organización tonotópica |
  | Hipocampo | Memoria episódica: separación (giro dentado) y completado (CA3) de patrones |
  | Amígdala | Estado afectivo en ejes valencia/arousal |
  | Corteza Prefrontal | Centro ejecutivo, memoria de trabajo (7±2), señales top-down |
  | Wernicke | Comprensión lingüística (patrón → palabra) |
  | Broca | Producción del lenguaje (intención + emoción → palabras) |
- **Neuromodulación global** — 6 neurotransmisores (Dopamina, Serotonina, Noradrenalina, Cortisol, Acetilcolina, Oxitocina) que alteran dinámicamente el motor SNN.
- **Conectoma humano** — 19 conexiones interregionales con retrasos axonales sobre un **bus de spikes** basado en eventos.
- **Consolidación de memoria (sueño)** — motor periódico que reproduce memorias del hipocampo en la corteza para consolidarlas.
- **Persistencia binaria** — el estado sináptico completo se guarda/carga en un `.bin` (protocolo `0xBRA1N001`).
- **Léxico español** — ~188 palabras codificadas como patrones de impulsos deterministas (hash de n-gramas) compartidas entre Wernicke y Broca.

### Aprendizaje verificado (no es marketing)

Cada capacidad tiene un test con criterios cuantitativos en `tests/`:

- **Visión** (`visual-learning.test.ts`) — el engrama de un estímulo converge (estabilidad ≥0.60) y discrimina estímulos distintos.
- **Memoria** (`hippocampus.test.ts`) — completa pistas degradadas, discrimina episodios y persiste entre sesiones.
- **Lenguaje** (`language-loop.test.ts`) — el bucle texto→Wernicke→Broca es **reproducible** (84%), **discriminativo** (3% de solape entre textos distintos) y **relevante** (100% de respuestas contienen palabras del input).

> **Honestidad técnica:** la comprensión del lenguaje es **asociativa** (recupera y reordena palabras del léxico relacionadas con la entrada), no razonamiento simbólico. Es el comportamiento esperado de una SNN con léxico distribuido.

### Puesta en marcha

```bash
# Requisitos: Node.js >= 20
npm install

# Modo desarrollo (servidor + dashboard con recarga vía tsx)
npm run dev

# Producción
npm run build
npm start
```

Luego abre **http://localhost:3000** para el dashboard 3D. El servidor expone también `GET /api/state` (estado del cerebro) y un WebSocket para streaming en vivo.

```bash
# Ejecutar un test de aprendizaje (ejemplo)
npx tsx tests/language-loop.test.ts
```

### Stack tecnológico

TypeScript · Node.js (ESM) · Express-less HTTP nativo + WebSocket (`ws`) · Three.js / WebGL (dashboard) · Railway (despliegue).

---

## 🇬🇧 English

### Objective

gBrain **is not an LLM**. It is an attempt to model learning *the way a real brain does it*: neurons that fire spikes, synapses strengthened or weakened by the *timing* of those spikes (STDP), and neurotransmitters that modulate the whole system at every step. The goal is a system that **genuinely learns from experience** —grouping stimuli, linking them and reinforcing those links through repetition— with learning that is **visible and verifiable**, not a black box.

Three principles drive the project:

1. **Biologically plausible** — Izhikevich neuron, STDP, homeostasis, human connectome.
2. **Real, measurable learning** — every capability (vision, memory, language) ships a quantitative test proving it learns.
3. **Observable live** — a 3D dashboard streaming each region's activity and hormone levels in real time.

### Key features

- **SNN core** — **Izhikevich** neuron (RS/FS/CH/IB), **STDP** plasticity (LTP/LTD), homeostasis and synaptic scaling.
- **8 brain regions** extending a common `BrainRegion` base class:
  | Region | Function |
  |--------|----------|
  | Thalamus | Sensory filter and attention bottleneck |
  | Visual Cortex | Image processing (webcam), k-WTA + neuronal fatigue |
  | Auditory Cortex | Audio processing (microphone), tonotopic organization |
  | Hippocampus | Episodic memory: pattern separation (dentate gyrus) & completion (CA3) |
  | Amygdala | Affective state on valence/arousal axes |
  | Prefrontal Cortex | Executive center, working memory (7±2), top-down signals |
  | Wernicke | Language comprehension (pattern → word) |
  | Broca | Language production (intention + emotion → words) |
- **Global neuromodulation** — 6 neurotransmitters (Dopamine, Serotonin, Norepinephrine, Cortisol, Acetylcholine, Oxytocin) dynamically altering the SNN engine.
- **Human connectome** — 19 inter-region connections with axonal delays over an event-based **spike bus**.
- **Memory consolidation (sleep)** — periodic engine replaying hippocampal memories into cortex to consolidate them.
- **Binary persistence** — full synaptic state saved/loaded to a `.bin` file (`0xBRA1N001` protocol).
- **Spanish lexicon** — ~188 words encoded as deterministic spike patterns (n-gram hashing) shared between Wernicke and Broca.

### Verified learning (not marketing)

Each capability has a test with quantitative criteria under `tests/`:

- **Vision** (`visual-learning.test.ts`) — a stimulus engram converges (stability ≥0.60) and discriminates distinct stimuli.
- **Memory** (`hippocampus.test.ts`) — completes degraded cues, discriminates episodes and persists across sessions.
- **Language** (`language-loop.test.ts`) — the text→Wernicke→Broca loop is **reproducible** (84%), **discriminative** (3% overlap between distinct texts) and **relevant** (100% of responses contain input words).

> **Technical honesty:** language comprehension is **associative** (it retrieves and reorders lexicon words related to the input), not symbolic reasoning. This is the expected behaviour of an SNN with a distributed lexicon.

### Getting started

```bash
# Requirements: Node.js >= 20
npm install

# Dev mode (server + dashboard via tsx)
npm run dev

# Production
npm run build
npm start
```

Then open **http://localhost:3000** for the 3D dashboard. The server also exposes `GET /api/state` (brain state) and a WebSocket for live streaming.

```bash
# Run a learning test (example)
npx tsx tests/language-loop.test.ts
```

### Tech stack

TypeScript · Node.js (ESM) · native HTTP + WebSocket (`ws`) · Three.js / WebGL (dashboard) · Railway (deployment).

---

## 📁 Project structure / Estructura

```
src/
├── core/
│   ├── snn/              # Izhikevich neuron, synapse, network, spike train
│   ├── bus/              # Spike bus + human connectome
│   ├── neuromodulators/  # 6-neurotransmitter system
│   ├── memory/           # Working memory, sensory buffer, consolidation
│   └── persistence/      # Binary state protocol (0xBRA1N001)
├── regions/              # 8 brain regions (thalamus, cortices, hippocampus...)
├── encoders/             # text / visual / audio → spikes
├── decoders/             # spikes → text / emotion / image / speech
├── dashboard/            # Three.js 3D client (index.html + brain-viz.js)
├── brain.ts              # DigitalBrain orchestrator
└── server.ts             # HTTP + WebSocket server
tests/                    # Quantitative learning-verification tests
```

## 🔬 Scientific basis / Base científica

Izhikevich (2003) spiking model · Spike-Timing-Dependent Plasticity (Bi & Poo, 1998) · synaptic homeostasis (Turrigiano, 2008) · pattern separation/completion in the hippocampus (Marr; Treves & Rolls) · lexical access by pattern similarity (McClelland & Rumelhart, 1981).

## 📄 License / Licencia

[MIT](./LICENSE) © 2026 Gustavo
