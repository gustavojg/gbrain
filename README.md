# 🧠 gBrain — Digital Brain / Cerebro Digital

> A biologically-inspired **Spiking Neural Network** simulating 8 interconnected brain regions, real synaptic plasticity (STDP), global neuromodulation and a live dashboard.
>
> Red neuronal **de impulsos (SNN)** bio-inspirada que simula 8 regiones cerebrales interconectadas, plasticidad sináptica real (STDP), neuromodulación global y un dashboard en vivo.

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
3. **Observable en vivo** — dashboard con la actividad de cada región y los niveles hormonales en tiempo real.

### Características principales

- **Núcleo SNN** — Neurona de **Izhikevich** (RS/FS/CH/IB), plasticidad **STDP** (LTP/LTD), homeostasis y escalado sináptico.
- **10 regiones cerebrales** que heredan de una clase base común `BrainRegion`:
  | Región | Función |
  |--------|---------|
  | Tálamo | Filtro sensorial y cuello de botella de atención |
  | Corteza Visual | Procesa imágenes (webcam), k-WTA + fatiga neuronal |
  | Corteza Auditiva | Procesa audio (micrófono), organización tonotópica |
  | Hipocampo | Memoria episódica: separación (giro dentado) y completado (CA3) de patrones |
  | Amígdala | Estado afectivo en ejes valencia/arousal; vía rápida innata (tono de voz, sobresalto, lo que se acerca, caras) y condicionamiento de miedo |
  | Corteza Prefrontal | Centro ejecutivo, memoria de trabajo (7±2), señales top-down |
  | Wernicke | Comprensión lingüística (patrón → palabra) |
  | Broca | Producción del lenguaje (intención + emoción → palabras) |
  | Corteza Motora Vocal | Controla el tracto vocal; aprende el mapa audición→motor balbuceando |
  | Corteza Motora de la Mano | Dibuja en la pizarra; aprende el mapa visión→motor garabateando |
- **Neuromodulación global** — 6 neurotransmisores (Dopamina, Serotonina, Noradrenalina, Cortisol, Acetilcolina, Oxitocina) que alteran dinámicamente el motor SNN.
- **Conectoma humano** — 19 conexiones interregionales con retrasos axonales sobre un **bus de spikes** basado en eventos.
- **Consolidación de memoria (sueño)** — motor periódico que reproduce memorias del hipocampo en la corteza para consolidarlas.
- **Persistencia binaria** — el estado sináptico completo se guarda/carga en un `.bin` (protocolo `0xBRA1N001`).
- **Léxico bilingüe (ES + EN)** — ~360 palabras codificadas como patrones de impulsos deterministas (hash de n-gramas) compartidas entre Wernicke y Broca. Español e inglés conviven en el mismo espacio de representación.
- **Aprendizaje de vocabulario** — las palabras desconocidas que lee el cerebro se aprenden tras varias exposiciones (umbral configurable) y se añaden al léxico, recompensadas con dopamina + acetilcolina. El vocabulario aprendido persiste entre sesiones.
- **Pensamientos en directo** — el dashboard muestra un *stream of consciousness*: cada ~1.2 s se decodifica la activación interna del cerebro (Wernicke + Broca + traza decadente de lo último que leyó) a una frase corta enmarcada por la emoción dominante (ej. `(stressed) miedo · tristeza · anxiety`). Es una lectura *asociativa* honesta del estado de la red, no razonamiento.

### Aprendizaje verificado (no es marketing)

Cada capacidad tiene un test con criterios cuantitativos en `tests/`:

- **Visión** (`visual-learning.test.ts`) — el engrama de un estímulo converge (estabilidad ≥0.60) y discrimina estímulos distintos.
- **Memoria** (`hippocampus.test.ts`) — completa pistas degradadas, discrimina episodios y persiste entre sesiones.
- **Lenguaje** (`language-loop.test.ts`) — el bucle texto→Wernicke→Broca es **reproducible** (84%), **discriminativo** (3% de solape entre textos distintos) y **relevante** (100% de respuestas contienen palabras del input).
- **Lenguaje bilingüe** (`bilingual-lexicon.test.ts`) — comprende y responde en español e inglés (100% de relevancia).
- **Vocabulario** (`vocabulary-learning.test.ts`) — aprende palabras nuevas tras N exposiciones, las comprende después y las persiste en disco.
- **Pensamientos en directo** (`live-thoughts.test.ts`) — `think()` produce pensamientos bien formados, **reactivos** (en reposo el pensamiento está vacío; al leer se vuelve no vacío) y **discriminativos** (entradas distintas → pensamientos distintos, solape <60%).
- **Afecto, atención y neuromodulación** (`neuro-systems.test.ts`) — la **amígdala** lee una voz cálida con valencia claramente mayor que una voz áspera; la **acetilcolina** ensancha el cuello de botella atencional del tálamo (pasan más señales); la **dopamina** sube el multiplicador de plasticidad y el **cortisol** lo suprime.

- **Cerebro completo** (`whole-brain.test.ts`) — con PRNG sembrado: en **reposo** no dispara nada ni se guardan episodios; un estímulo **recorre las 8 regiones** entrando por el tálamo y después el cerebro **vuelve al reposo**; el código talámico es reproducible y discriminativo; el estado **persiste** bit a bit. Además mide y publica los **huecos conocidos** (defectos auditados aún sin corregir) sin hacer fallar la suite: cuando uno se cierra, se promueve a comprobación dura.
- **Lo innato** (`innate.test.ts`) — el cerebro no trae de serie significados (no hay diccionario de palabras afectivas): trae **reacciones a rasgos**. Lee el **tono de la voz** por una vía rápida tálamo→amígdala (suave, aguda y ascendente calma; fuerte, grave, brusca y áspera alarma; Fernald 1993), se **sobresalta** con un sonido súbito antes de que la corteza lo oiga, se alarma con **lo que se acerca**, atiende a las **caras**, aprende a **temer** en un solo emparejamiento (y a dejar de temer despacio, con recuperación bajo estrés), toma una voz cálida tras un recuerdo como **aprobación**, y **duerme por presión de sueño**. Las palabras adquieren su emoción al leerse junto a una voz que ya asusta o calma. Plan completo en [`docs/PLAN.md`](docs/PLAN.md).
- **Motivación** (`motivation.test.ts`) — la dopamina es **error de predicción de recompensa** (Schultz): lo nuevo produce una subida que se **habitúa** con cada repetición, el elogio esperado no mueve nada, el elogio que no llega es una **bajada**. La **novedad** y el **progreso de aprendizaje** son recompensas en sí mismas: elige balbucear o garabatear según lo que aún le enseña algo y abandona lo que domina. **Impulsos**: aburrido actúa antes, solo llama, y la línea base de cada neuromodulador se **adapta** a un estado sostenido.
- **El tiempo** (`sequence-learning.test.ts`) — el micrófono envía cuadros cada 200 ms mientras se habla y la **ventana del oído** guarda el **orden** de los sonidos: dos vocales en un orden y en el otro son dos sonidos distintos. Entre perceptos aprende **qué sigue a qué**: tras ver la cruz varias veces seguida de /a/, la cruz sola le hace **esperar** /a/, y lo esperado ya no sorprende. La **memoria de trabajo** tiene compuerta dopaminérgica: entra lo nuevo y lo elogiado, lo rutinario pasa de largo.
- **Sentidos que aprenden** (`sensory-learning.test.ts`) — *aprendizaje por exposición, sin etiquetas*: al enseñarle dibujos y vocales por las mismas entradas que usa el dashboard, el cerebro **forma categorías por sí solo**, **reconoce** lo que ya ha visto u oído (también con ruido o medio tapado), distingue lo nuevo y lo **recuerda tras reiniciar**.
- **Asociación entre modalidades** (`association-learning.test.ts`) — *aprender qué va con qué, por repetición*: al enseñarle un dibujo junto a su palabra (o un sonido), la asociación **crece con cada repetición** (tras una sola no se fía; tras unas pocas sí), ver el dibujo le **trae la palabra a la mente** (la piensa y la dice), leer la palabra le trae el dibujo, enseñado con frases enteras **aísla la palabra que corresponde**, la dopamina lo acelera y todo **persiste**.
- **Voz** (`vocal-learning.test.ts`) — *aprender a usar la voz balbuceando*: antes de balbucear no puede repetir ninguna vocal que oye; con la voz activada **balbucea solo, se oye a sí mismo** y aprende qué orden motora produce qué sonido; cuanto más balbucea, **más vocales repite** y con poco error (3–7 % del rango vocal; al azar sería ≈33 %); no entra en eco con su propia voz, solo responde a sonidos y la habilidad **persiste**. Y unido a la asociación: si se le enseña que un dibujo va con un sonido, **al ver el dibujo dice ese sonido**.
- **Mano** (`drawing-learning.test.ts`) — *aprender a dibujar garabateando*: antes de garabatear no puede copiar nada; con la mano activada **garabatea sola, ve las marcas que deja** y aprende qué orden pinta qué sitio; cuanto más garabatea, **mejores son sus copias** (parecido con el modelo 0 → ~0,35 → ~0,60) y cada copia se parece a su modelo y no a los otros; no entra en bucle con sus propias copias; enseñada una palabra junto a un dibujo, **al leer la palabra lo dibuja de memoria** y al ver el dibujo **escribe la palabra**; el 👍/👎 del maestro **refuerza o debilita** lo que acaba de recordar; todo **persiste**.
- **Servidor** (`server-guards.test.ts`, `server-integration.test.ts`) — entradas malformadas, orígenes ajenos, cuerpos gigantes e inundaciones contra el servidor real: nada puede envenenar (NaN), agotar o bloquear el cerebro; un archivo de estado corrupto se detecta (CRC) y se restaura la copia anterior.

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

Luego abre **http://localhost:3000** para el dashboard. El servidor expone también `GET /api/state` (estado del cerebro) y un WebSocket para streaming en vivo. El cerebro avanza 10 ticks por segundo por defecto; `BRAIN_TICK_HZ=100` lo lleva a 100 Hz sin cambiar lo que hace por segundo de reloj (todo lo que está a escala humana, cuánto dura un estímulo a la vista, cuánto espera un percepto a que llegue su nombre, está en segundos y se convierte a ticks con esa frecuencia: solo vive diez veces más tiempo neuronal por segundo). `BRAIN_SPEED=10` sí lo hace vivir 10× más deprisa (útil para sesiones de balbuceo/garabateo, no para enseñarle en vivo).

```bash
# Ejecutar un test de aprendizaje (ejemplo)
npx tsx tests/language-loop.test.ts
```

### Enseñarle (modo enseñanza)

En el dashboard, el panel **🎓 Teach** hace de maestro:

1. Dibuja algo en la pizarra y escribe su nombre (y/o elige un sonido /a/…/u/).
2. Pulsa **Teach**: el cerebro ve y lee (u oye) las dos cosas juntas las veces indicadas, y el panel muestra la **curva de aprendizaje**: cuánto recordaba antes de cada repetición (0 % la primera vez, y creciendo).
3. Pulsa **Test it**: se le muestra el dibujo solo. En *Perception → Recalls* aparece lo que le trae a la mente; si la voz y la mano están activadas, **dice** el sonido asociado y **escribe** la palabra.
4. **👍 / 👎** refuerzan o debilitan lo que acaba de recordar. Con el micrófono activo basta con **hablarle**: una voz cálida justo después de que recuerde algo vale como 👍 y una seca como 👎, y el panel *Perception → Tone* muestra cómo ha leído tu voz.
5. **Practice** le deja balbucear o garabatear ×100 en unos segundos: así aprende sus mapas motores (necesarios para imitar sonidos y copiar dibujos).

Todo ello va por `POST /api/lesson`, `/api/practice`, `/api/feedback`, `/api/voice` y `/api/hand` (o sus mensajes WebSocket), validados y con límite de frecuencia.

El dashboard funciona también en el móvil o la tablet: los paneles se apilan y la página hace scroll, se dibuja en la pizarra con el dedo, y los botones tienen tamaño de dedo.

### Stack tecnológico

TypeScript · Node.js (ESM) · Express-less HTTP nativo + WebSocket (`ws`) · Canvas 2D + WebAudio (dashboard) · Railway (despliegue).

---

## 🇬🇧 English

### Objective

gBrain **is not an LLM**. It is an attempt to model learning *the way a real brain does it*: neurons that fire spikes, synapses strengthened or weakened by the *timing* of those spikes (STDP), and neurotransmitters that modulate the whole system at every step. The goal is a system that **genuinely learns from experience** —grouping stimuli, linking them and reinforcing those links through repetition— with learning that is **visible and verifiable**, not a black box.

Three principles drive the project:

1. **Biologically plausible** — Izhikevich neuron, STDP, homeostasis, human connectome.
2. **Real, measurable learning** — every capability (vision, memory, language) ships a quantitative test proving it learns.
3. **Observable live** — a dashboard streaming each region's activity and hormone levels in real time.

### Key features

- **SNN core** — **Izhikevich** neuron (RS/FS/CH/IB), **STDP** plasticity (LTP/LTD), homeostasis and synaptic scaling.
- **10 brain regions** extending a common `BrainRegion` base class:
  | Region | Function |
  |--------|----------|
  | Thalamus | Sensory filter and attention bottleneck |
  | Visual Cortex | Image processing (webcam), k-WTA + neuronal fatigue |
  | Auditory Cortex | Audio processing (microphone), tonotopic organization |
  | Hippocampus | Episodic memory: pattern separation (dentate gyrus) & completion (CA3) |
  | Amygdala | Affective state on valence/arousal axes; innate fast route (tone of voice, startle, looming, faces) and fear conditioning |
  | Prefrontal Cortex | Executive center, working memory (7±2), top-down signals |
  | Wernicke | Language comprehension (pattern → word) |
  | Broca | Language production (intention + emotion → words) |
  | Vocal Motor Cortex | Drives the vocal tract; learns the auditory→motor map by babbling |
  | Hand Motor Cortex | Draws on the whiteboard; learns the visual→motor map by scribbling |
- **Global neuromodulation** — 6 neurotransmitters (Dopamine, Serotonin, Norepinephrine, Cortisol, Acetylcholine, Oxytocin) dynamically altering the SNN engine.
- **Human connectome** — 19 inter-region connections with axonal delays over an event-based **spike bus**.
- **Memory consolidation (sleep)** — periodic engine replaying hippocampal memories into cortex to consolidate them.
- **Binary persistence** — full synaptic state saved/loaded to a `.bin` file (`0xBRA1N001` protocol).
- **Bilingual lexicon (ES + EN)** — ~360 words encoded as deterministic spike patterns (n-gram hashing) shared between Wernicke and Broca. Spanish and English coexist in the same representational space.
- **Vocabulary learning** — unknown words the brain reads are learned after several exposures (configurable threshold) and added to the lexicon, rewarded with dopamine + acetylcholine. Learned vocabulary persists across sessions.
- **Live thoughts** — the dashboard streams a *stream of consciousness*: every ~1.2 s the brain's current internal activation (Wernicke + Broca + a decaying trace of the last thing it read) is decoded into a short phrase framed by its dominant emotion (e.g. `(stressed) miedo · tristeza · anxiety`). An honest *associative* read-out of network state, not reasoning.

### Verified learning (not marketing)

Each capability has a test with quantitative criteria under `tests/`:

- **Vision** (`visual-learning.test.ts`) — a stimulus engram converges (stability ≥0.60) and discriminates distinct stimuli.
- **Memory** (`hippocampus.test.ts`) — completes degraded cues, discriminates episodes and persists across sessions.
- **Language** (`language-loop.test.ts`) — the text→Wernicke→Broca loop is **reproducible** (84%), **discriminative** (3% overlap between distinct texts) and **relevant** (100% of responses contain input words).
- **Bilingual language** (`bilingual-lexicon.test.ts`) — comprehends and responds in Spanish and English (100% relevance).
- **Vocabulary** (`vocabulary-learning.test.ts`) — learns new words after N exposures, comprehends them afterwards, and persists them to disk.
- **Live thoughts** (`live-thoughts.test.ts`) — `think()` produces well-formed thoughts that are **reactive** (idle → empty; after reading → non-empty) and **discriminative** (distinct inputs → distinct thoughts, <60% overlap).
- **Affect, attention & neuromodulation** (`neuro-systems.test.ts`) — the **amygdala** reads a warm voice with clearly higher valence than a harsh one; **acetylcholine** widens the thalamic attentional bottleneck (more signals pass); **dopamine** raises the plasticity multiplier and **cortisol** suppresses it.

- **Whole brain** (`whole-brain.test.ts`) — with a seeded PRNG: at **rest** nothing fires and no episode is stored; a stimulus **travels through all 8 regions**, entering via the thalamus, and then the brain **returns to rest**; the thalamic code is reproducible and discriminative; state **persists** bit for bit. It also measures and prints the **known gaps** (audited defects not fixed yet) without failing the suite: once one closes, it is promoted to a hard check.
- **The innate layer** (`innate.test.ts`) — the brain brings no meanings with it (no dictionary of affective words): it brings **reactions to features**. It reads the **tone of a voice** on a fast thalamus→amygdala route (soft, high and rising comforts; loud, low, abrupt and rough alarms; Fernald 1993), **startles** at a sudden sound before the cortex has heard it, is alarmed by **looming**, attends to **faces**, learns to **fear** in one pairing (and to stop fearing slowly, with recovery under stress), takes a warm voice after a recall as **approval**, and **sleeps by sleep pressure**. Words acquire their emotion by being read alongside a voice that already frightens or comforts. Full plan in [`docs/PLAN.md`](docs/PLAN.md).
- **Motivation** (`motivation.test.ts`) — dopamine is **reward prediction error** (Schultz): a new thing is a burst that **habituates** with every repetition, expected praise moves nothing, praise withheld is a **dip**. **Novelty** and **learning progress** are rewards in themselves: it chooses babbling or scribbling by what still teaches it something and drops what it has mastered. **Drives**: bored it acts sooner, alone it calls, and each neuromodulator's baseline **adapts** to a sustained state.
- **Time** (`sequence-learning.test.ts`) — the mic streams frames every 200 ms while someone speaks and the **ear's window** keeps the **order** of the sounds: two vowels in one order and in the other are two different sounds. Between percepts it learns **what follows what**: shown the cross and then /a/ a few times, the cross alone makes it **expect** /a/, and the expected no longer surprises. **Working memory** is dopamine-gated: the new and the praised enter, the routine passes through.
- **Senses that learn** (`sensory-learning.test.ts`) — *learning by exposure, no labels*: shown drawings and played vowels through the same entry points the dashboard uses, the brain **forms categories on its own**, **recognizes** what it has seen or heard before (also noisy or half occluded), tells new things apart and **remembers across restarts**.
- **Cross-modal association** (`association-learning.test.ts`) — *learning what goes with what, by repetition*: shown a drawing together with its word (or a sound), the association **grows with every repetition** (one pairing is not acted upon; a few are), seeing the drawing **brings the word to mind** (it thinks it and says it), reading the word brings the drawing back, taught with whole sentences it **singles out the word that belongs**, dopamine speeds it up, and it all **persists**.
- **Voice** (`vocal-learning.test.ts`) — *learning to use its voice by babbling*: before babbling it cannot repeat any vowel it hears; with the voice on it **babbles on its own, hears itself** and learns which motor command makes which sound; the more it babbles, the **more vowels it repeats**, with little error (3–7 % of the vocal range; chance would be ≈33 %); it does not echo its own voice, only answers sounds, and the skill **persists**. And together with association: taught that a drawing goes with a sound, **it says that sound when it sees the drawing**.
- **Hand** (`drawing-learning.test.ts`) — *learning to draw by scribbling*: before scribbling it cannot copy anything; with the hand on it **scribbles on its own, sees the marks it leaves** and learns which command inks which place; the more it scribbles, **the better its copies** (likeness to the model 0 → ~0.35 → ~0.60), each copy looking like its own model and not the others; no loop on its own copies; taught a word with a drawing, **reading the word makes it draw it from memory** and seeing the drawing makes it **write the word**; the teacher's 👍/👎 **strengthens or weakens** what it has just recalled; it all **persists**.
- **Server** (`server-guards.test.ts`, `server-integration.test.ts`) — malformed input, foreign origins, oversized bodies and floods against the real server: nothing can poison (NaN), exhaust or stall the brain; a corrupted state file is detected (CRC) and the previous snapshot restored.

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

Then open **http://localhost:3000** for the dashboard. The server also exposes `GET /api/state` (brain state) and a WebSocket for live streaming. The brain advances 10 ticks per second by default; `BRAIN_TICK_HZ=100` drives it at 100 Hz without changing what it does per second of wall clock (everything on a human timescale, how long a stimulus stays in view, how long a percept waits for its name, is defined in seconds and converted to ticks with that rate: it just lives through ten times more neural time per second). `BRAIN_SPEED=10` does make it live 10× faster (useful for babbling/scribbling sessions, not for interactive teaching).

```bash
# Run a learning test (example)
npx tsx tests/language-loop.test.ts
```

### Teaching it (teaching mode)

In the dashboard, the **🎓 Teach** panel is the teacher:

1. Draw something on the whiteboard and type its name (and/or pick a sound /a/…/u/).
2. Press **Teach**: the brain sees and reads (or hears) both together as many times as requested, and the panel shows the **learning curve**: how much it recalled before each repetition (0 % the first time, then growing).
3. Press **Test it**: it is shown the drawing alone. *Perception → Recalls* shows what comes to its mind; with the voice and the hand on, it **says** the associated sound and **writes** the word.
4. **👍 / 👎** strengthen or weaken what it has just recalled. With the microphone on, just **talk to it**: a warm voice right after it recalls something counts as 👍 and a harsh one as 👎, and *Perception → Tone* shows how it read your voice.
5. **Practice** lets it babble or scribble ×100 in a few seconds: that is how it learns its motor maps (needed to imitate sounds and copy drawings).

All of it goes through `POST /api/lesson`, `/api/practice`, `/api/feedback`, `/api/voice` and `/api/hand` (or their WebSocket messages), validated and rate-limited.

The dashboard also works on a phone or tablet: the panels stack and the page scrolls, the whiteboard takes finger strokes, and the buttons are finger-sized.

### Tech stack

TypeScript · Node.js (ESM) · native HTTP + WebSocket (`ws`) · Canvas 2D + WebAudio (dashboard) · Railway (deployment).

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
├── dashboard/            # Canvas 2D client (index.html + brain-viz.js)
├── brain.ts              # DigitalBrain orchestrator
└── server.ts             # HTTP + WebSocket server
tests/                    # Quantitative learning-verification tests
```

## 🔬 Scientific basis / Base científica

Izhikevich (2003) spiking model · Spike-Timing-Dependent Plasticity (Bi & Poo, 1998) · synaptic homeostasis (Turrigiano, 2008) · pattern separation/completion in the hippocampus (Marr; Treves & Rolls) · lexical access by pattern similarity (McClelland & Rumelhart, 1981).

## 📄 License / Licencia

[MIT](./LICENSE) © 2026 Gustavo
