/**
 * DIGITAL BRAIN — Main Orchestrator
 * ========================================
 * Main class that unifies all brain regions,
 * the spike bus, the neuromodulators, and the memory system
 * into a single coherent system.
 *
 * The brain runs on a main perception loop:
 * 1. PERCEIVE: Receive sensory inputs (audio, image, text)
 * 2. FILTER: The thalamus selects the most relevant inputs
 * 3. PROCESS: The sensory cortices extract features
 * 4. EVALUATE: The amygdala assigns emotional valence
 * 5. REMEMBER: The hippocampus stores/retrieves memories
 * 6. DECIDE: The prefrontal cortex integrates and decides
 * 7. RESPOND: Broca/Wernicke generates language
 * 8. MODULATE: The neuromodulators adjust the whole system
 *
 * Biology: This loop mimics the perception-action cycle of the real brain,
 * where information flows from the primary sensory cortices
 * toward the association areas and the frontal lobe in ~300-500ms.
 */

import { type BrainConfiguration, DEFAULT_BRAIN_CONFIG, getTotalNeurons, estimateMemoryUsage } from './brain.config.js';
import { SpikeBus, type SpikePacket } from './core/bus/spike-bus.js';
import { Connectome } from './core/bus/connectome.js';
import { NeuromodulatorSystem, ModulatorType, type ModulationEffects } from './core/neuromodulators/modulator-system.js';
import { type BrainRegion, type RegionActivity } from './core/brain-region.js';
import { ConsolidationEngine } from './core/memory/consolidation.js';
import { BrainPersistence } from './core/persistence/binary-protocol.js';
import { VisualEncoder } from './encoders/visual-encoder.js';
import { AudioEncoder } from './encoders/audio-encoder.js';
import { TextEncoder as BrainTextEncoder } from './encoders/text-encoder.js';
import { TextDecoder as BrainTextDecoder } from './decoders/text-decoder.js';
import { EmotionDecoder, type EmotionalState, type ModulatorLevels } from './decoders/emotion-decoder.js';
import { SpeechSynthesizer } from './decoders/speech-synthesizer.js';
import { ImageGenerator } from './decoders/image-generator.js';

// --- Brain regions ---
import { Thalamus } from './regions/thalamus/thalamus.js';
import { VisualCortex } from './regions/visual-cortex/visual-cortex.js';
import { AuditoryCortex } from './regions/auditory-cortex/auditory-cortex.js';
import { Hippocampus } from './regions/hippocampus/hippocampus.js';
import { Amygdala } from './regions/amygdala/amygdala.js';
import { PrefrontalCortex } from './regions/prefrontal-cortex/prefrontal-cortex.js';
import { BrocaArea, type LanguageResponse } from './regions/broca-wernicke/broca.js';
import { WernickeArea } from './regions/broca-wernicke/wernicke.js';
import { Lexicon } from './regions/broca-wernicke/lexicon.js';
import { seedSpanishLexicon, encodeSentenceToLexiconSpace } from './regions/broca-wernicke/spanish-lexicon.js';
import { seedEnglishLexicon } from './regions/broca-wernicke/english-lexicon.js';

// ================================================================
// TYPES
// ================================================================

/** Complete state of the brain at a given moment */
export interface BrainState {
  /** Current simulation time (ms) */
  time: number;
  /** Activity of each region */
  regions: Record<string, RegionActivity>;
  /** Neuromodulator levels */
  modulators: ModulatorLevels;
  /** Current emotional state */
  emotion: EmotionalState;
  /** Number of memories in the hippocampus */
  memoriesCount: number;
  /** Spike bus traffic */
  busTraffic: Record<string, { sent: number; received: number }>;
  /** Total ticks processed */
  tickCount: number;
  /** Last response from Broca */
  broca?: { lastResponse: string; words: string[]; confidence: number };
  /** Vocabulary size */
  vocabCount: number;
  /** Visual cortex learning metrics (engram, stability, convergence). */
  learning?: {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
  };
  /** Hippocampus CA3 learning metrics (engram, stability, episodes). */
  learningHippocampus?: {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
    memoryCount: number;
  };
}

/** Result of a perception */
export interface PerceptionResult {
  /** Type of input processed */
  inputType: 'visual' | 'auditory' | 'text' | 'image';
  /** Emotional state after processing */
  emotion: EmotionalState;
  /** Generated text response (if applicable) */
  textResponse?: string;
  /** Speech parameters (if applicable) */
  speechParams?: { text: string; rate: number; pitch: number; volume: number };
  /** Regions that were activated */
  activeRegions: string[];
  /** Processing time (simulated ms) */
  processingTime: number;
}

/** Event emitted by the brain */
export interface BrainEvent {
  type: 'spike' | 'emotion' | 'memory' | 'consolidation' | 'response';
  timestamp: number;
  data: Record<string, unknown>;
}

// ================================================================
// MAIN CLASS
// ================================================================

/**
 * Digital Brain — Main system.
 * Orchestrates 7 brain regions with SNNs, neuromodulation and hierarchical memory.
 */
export class DigitalBrain {
  // --- Configuration ---
  private config: BrainConfiguration;

  // --- Core infrastructure ---
  private bus: SpikeBus;
  private connectome: Connectome;
  private modulators: NeuromodulatorSystem;
  private consolidationEngine: ConsolidationEngine;
  private persistence: BrainPersistence = new BrainPersistence();

  // --- Brain regions ---
  private regions: Map<string, BrainRegion> = new Map();

  // --- Encoders (inputs) ---
  private visualEncoder: VisualEncoder;
  private audioEncoder: AudioEncoder;
  private textEncoder: BrainTextEncoder;

  /**
   * Clean linguistic intention from the last `read()`, in lexicon space.
   * Allows `speak()` to regenerate Broca's response deterministically
   * (reproducible and discriminative), without the dilution of the recurrent loop
   * overwriting it with background noise.
   */
  private lastLinguisticIntention: Float32Array | null = null;

  // --- Decoders (outputs) ---
  private textDecoder: BrainTextDecoder;
  private emotionDecoder: EmotionDecoder;
  private speechSynthesizer: SpeechSynthesizer;
  private imageGenerator: ImageGenerator;

  // --- State ---
  private currentTime: number = 0;
  private tickCount: number = 0;
  private isRunning: boolean = false;
  private eventListeners: Map<string, Array<(event: BrainEvent) => void>> = new Map();

  // --- Automatic consolidation ---
  private lastConsolidation: number = 0;

  constructor(config: Partial<BrainConfiguration> = {}) {
    this.config = { ...DEFAULT_BRAIN_CONFIG, ...config };

    console.log(`\n🧠 ═══════════════════════════════════════════`);
    console.log(`   CEREBRO DIGITAL — Inicializando...`);
    console.log(`   Neuronas totales: ${getTotalNeurons(this.config).toLocaleString()}`);
    console.log(`   Memoria estimada: ${estimateMemoryUsage(this.config).toFixed(1)} MB`);
    console.log(`   Regiones: ${Object.keys(this.config.regions).length}`);
    console.log(`═══════════════════════════════════════════════\n`);

    // 1. Initialize spike bus
    this.bus = new SpikeBus();

    // 2. Initialize connectome
    this.connectome = new Connectome(this.config.connectome);

    // 3. Initialize neuromodulators
    this.modulators = new NeuromodulatorSystem();

    // 4. Initialize consolidation
    this.consolidationEngine = new ConsolidationEngine();

    // 5. Initialize encoders
    this.visualEncoder = new VisualEncoder({
      processWidth: 32,
      processHeight: 32,
      foveation: true,
      edgeDetection: true,
    });

    this.audioEncoder = new AudioEncoder({
      sampleRate: 16000,
      fftSize: 2048,
      numBands: 40,
      numFrames: 20,
    });

    this.textEncoder = new BrainTextEncoder({
      vectorSize: 5000,
      tokenization: 'word',
    });

    // 6. Initialize decoders
    this.textDecoder = new BrainTextDecoder(0.3);
    this.emotionDecoder = new EmotionDecoder();
    this.speechSynthesizer = new SpeechSynthesizer({ lang: 'es-ES' });
    this.imageGenerator = new ImageGenerator({ outputWidth: 32, outputHeight: 32 });

    // 7. Register regions on the bus
    this.initializeRegions();

    // 8. Configure connectome on the bus
    this.setupConnectome();

    console.log(`✅ Cerebro inicializado correctamente.`);
    console.log(`   Tick rate: ${this.config.tickRate} Hz`);
    console.log(`   Consolidación cada: ${this.config.memory.consolidationIntervalMs / 1000}s\n`);
  }

  /**
   * Initializes the brain regions and registers them on the bus.
   * Uses dynamic import to load the concrete implementations.
   */
  /** Shared Broca↔Wernicke lexicon */
  private lexicon!: Lexicon;

  private initializeRegions(): void {
    // Register the config IDs on the bus (for the connectome)
    for (const regionId of Object.keys(this.config.regions)) {
      this.bus.register(regionId);
    }

    // Register Broca and Wernicke as individual regions
    this.bus.register('broca');
    this.bus.register('wernicke');

    // Create shared lexicon with Spanish vocabulary
    // Pattern size reduced to 1000 for efficiency (matching Broca/Wernicke inputCount)
    this.lexicon = new Lexicon(1000);
    seedSpanishLexicon(this.lexicon);
    seedEnglishLexicon(this.lexicon);
    console.log(`  📚 Léxico inicializado (ES+EN): ${this.lexicon.size} palabras`);

    // ── Instantiate regions with reduced sizes ──
    // Total: ~10K neurons (vs 50K before) → smooth performance
    this.addRegion(new Thalamus({ neuronCount: 500, totalInputSize: 500, bottleneckSize: 100 }));
    this.addRegion(new VisualCortex({ neuronCount: 2000, inputCount: 1000 }));
    this.addRegion(new AuditoryCortex({ neuronCount: 1000, inputCount: 400 }));
    this.addRegion(new Hippocampus(1000, 1000));
    this.addRegion(new Amygdala(500, 500));
    this.addRegion(new PrefrontalCortex(3000, 1000));
    this.addRegion(new BrocaArea(this.lexicon, 1000, 1000));
    this.addRegion(new WernickeArea(this.lexicon, 1000, 1000));

    // Connect Broca/Wernicke to the bus as an alias of 'brocaWernicke'
    // to receive packets from the existing connectome
    this.bus.onReceive('brocaWernicke', (packet: SpikePacket) => {
      const broca = this.regions.get('broca');
      const wernicke = this.regions.get('wernicke');
      if (wernicke) wernicke.feedInput(packet.spikes, packet.timestamp);
      if (broca) broca.feedInput(packet.spikes, packet.timestamp);
    });

    // Compute the real total number of neurons
    let totalNeurons = 0;
    for (const [, region] of this.regions) {
      totalNeurons += region.neurons;
    }
    console.log(`  🧩 ${this.regions.size} regiones instanciadas (${totalNeurons.toLocaleString()} neuronas reales)`);
  }

  /**
   * Adds an implemented region to the brain.
   */
  addRegion(region: BrainRegion): void {
    this.regions.set(region.id, region);
    if (!this.bus.isRegistered(region.id)) {
      this.bus.register(region.id);
    }

    // Subscribe the region to the bus to receive spikes
    this.bus.onReceive(region.id, (packet: SpikePacket) => {
      region.feedInput(packet.spikes, packet.timestamp);
    });

    console.log(`  🧩 Región añadida: ${region.id} (${region.name})`);
  }

  /**
   * Configures the connectome's delays and weights on the spike bus.
   */
  private setupConnectome(): void {
    const connections = this.connectome.getAllConnections();
    for (const conn of connections) {
      this.bus.setDelay(conn.from, conn.to, conn.delay);
      this.bus.setWeight(conn.from, conn.to, conn.weight);
    }
    console.log(`  🔗 Conectoma configurado: ${connections.length} conexiones`);
  }

  // ================================================================
  // HIGH-LEVEL API — INPUTS
  // ================================================================

  /**
   * The brain "sees" an image.
   *
   * @param pixels - Image data (grayscale, 0-255)
   * @param width - Width
   * @param height - Height
   */
  see(pixels: number[] | Float32Array | Uint8Array, width: number, height: number): PerceptionResult {
    console.log(`👁️  Percibiendo imagen (${width}×${height})...`);

    // Encode to spikes
    const spikes = this.visualEncoder.encode(pixels, width, height, this.config.snn.dt);

    // Send to the thalamus
    this.injectSensoryInput('visual', spikes);

    // Process several ticks to propagate through the brain
    return this.processPerception('visual');
  }

  /**
   * The brain "hears" audio.
   *
   * @param audioSamples - PCM samples
   */
  hear(audioSamples: Float32Array | number[]): PerceptionResult {
    // Encode to spikes via spectrogram
    const spikes = this.audioEncoder.encode(audioSamples, this.config.snn.dt);

    // Send to the thalamus
    this.injectSensoryInput('auditory', spikes);

    return this.processPerception('auditory');
  }

  /**
   * The brain "hears" an already-computed spectrogram (from 08_microphone_interaction).
   */
  hearSpectrogram(spectrogram: number[] | Float32Array): PerceptionResult {
    const spikes = this.audioEncoder.encodeSpectrogram(spectrogram, this.config.snn.dt);
    this.injectSensoryInput('auditory', spikes);
    return this.processPerception('auditory');
  }

  /**
   * The brain "reads" text.
   *
   * @param text - Text to process
   */
  /**
   * Map of emotional words → neuromodulator effects.
   * Biology: Wernicke's area recognizes emotionally charged words
   * and activates the amygdala, which in turn releases neuromodulators.
   */
  private static readonly EMOTIONAL_WORDS: Record<string, { modulator: ModulatorType; amount: number }[]> = {
    // Positive → dopamine + serotonin
    'feliz': [{ modulator: ModulatorType.Dopamine, amount: 0.15 }, { modulator: ModulatorType.Serotonin, amount: 0.1 }],
    'alegria': [{ modulator: ModulatorType.Dopamine, amount: 0.2 }],
    'amor': [{ modulator: ModulatorType.Oxytocin, amount: 0.2 }, { modulator: ModulatorType.Dopamine, amount: 0.1 }],
    'carino': [{ modulator: ModulatorType.Oxytocin, amount: 0.15 }],
    'bien': [{ modulator: ModulatorType.Serotonin, amount: 0.1 }],
    'gracias': [{ modulator: ModulatorType.Oxytocin, amount: 0.1 }, { modulator: ModulatorType.Serotonin, amount: 0.05 }],
    'hola': [{ modulator: ModulatorType.Dopamine, amount: 0.05 }, { modulator: ModulatorType.Oxytocin, amount: 0.05 }],
    'bonito': [{ modulator: ModulatorType.Dopamine, amount: 0.1 }],
    'entusiasmo': [{ modulator: ModulatorType.Dopamine, amount: 0.2 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'esperanza': [{ modulator: ModulatorType.Serotonin, amount: 0.15 }],
    'paz': [{ modulator: ModulatorType.Serotonin, amount: 0.2 }],
    'calma': [{ modulator: ModulatorType.Serotonin, amount: 0.15 }],
    'curiosidad': [{ modulator: ModulatorType.Dopamine, amount: 0.1 }, { modulator: ModulatorType.Acetylcholine, amount: 0.1 }],
    'genial': [{ modulator: ModulatorType.Dopamine, amount: 0.15 }, { modulator: ModulatorType.Serotonin, amount: 0.1 }],
    'contento': [{ modulator: ModulatorType.Dopamine, amount: 0.1 }, { modulator: ModulatorType.Serotonin, amount: 0.1 }],
    // Negative → cortisol + norepinephrine
    'triste': [{ modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'tristeza': [{ modulator: ModulatorType.Cortisol, amount: 0.15 }],
    'miedo': [{ modulator: ModulatorType.Cortisol, amount: 0.2 }, { modulator: ModulatorType.Norepinephrine, amount: 0.15 }],
    'odio': [{ modulator: ModulatorType.Cortisol, amount: 0.15 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'ansiedad': [{ modulator: ModulatorType.Cortisol, amount: 0.2 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'estres': [{ modulator: ModulatorType.Cortisol, amount: 0.25 }],
    'enojo': [{ modulator: ModulatorType.Norepinephrine, amount: 0.2 }, { modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'mal': [{ modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'feo': [{ modulator: ModulatorType.Cortisol, amount: 0.05 }],
    'soledad': [{ modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'frustracion': [{ modulator: ModulatorType.Cortisol, amount: 0.15 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    // Activation → acetylcholine + norepinephrine
    'pensar': [{ modulator: ModulatorType.Acetylcholine, amount: 0.1 }],
    'aprender': [{ modulator: ModulatorType.Acetylcholine, amount: 0.15 }, { modulator: ModulatorType.Dopamine, amount: 0.05 }],
    'recordar': [{ modulator: ModulatorType.Acetylcholine, amount: 0.1 }],
    'atencion': [{ modulator: ModulatorType.Acetylcholine, amount: 0.15 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'dormir': [{ modulator: ModulatorType.Serotonin, amount: 0.2 }],
    'sonar': [{ modulator: ModulatorType.Serotonin, amount: 0.1 }, { modulator: ModulatorType.Dopamine, amount: 0.05 }],
  };

  read(text: string): PerceptionResult {
    console.log(`📖 Leyendo: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);

    // Encode the text in LEXICON SPACE (not with the TextEncoder's generic
    // hash, which produces vectors orthogonal to the engrams and leaves
    // Wernicke unable to recognize anything). This way Wernicke retrieves the
    // real words and Broca can produce a reproducible associative response.
    const spikes = encodeSentenceToLexiconSpace(text, this.lexicon.dimensions);

    // Store the clean intention so speak() generates a reproducible
    // response (text→Wernicke→Broca) without dilution from the recurrent loop.
    this.lastLinguisticIntention = spikes;

    // Send to the thalamus (linguistic route)
    this.injectSensoryInput('linguistic', spikes);

    // ── Emotional word detection ──
    // Biology: Wernicke recognizes emotional words → activates the amygdala
    // → the amygdala releases neuromodulators according to valence
    const words = text.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^\w\sáéíóúüñ]/g, '').split(/\s+/).filter(w => w.length > 0);
    let emotionalHits = 0;
    for (const word of words) {
      // Also compare without accents
      const effects = DigitalBrain.EMOTIONAL_WORDS[word];
      if (effects) {
        for (const effect of effects) {
          this.modulators.release(effect.modulator, effect.amount);
        }
        emotionalHits++;
      }
    }

    if (emotionalHits > 0) {
      console.log(`  💭 ${emotionalHits} palabras emocionales detectadas`);
    }

    // Novelty → norepinephrine
    this.modulators.release(ModulatorType.Norepinephrine, 0.05);

    return this.processPerception('text');
  }

  // ================================================================
  // HIGH-LEVEL API — OUTPUTS
  // ================================================================

  /**
   * The brain "speaks" — generates a text response.
   */
  speak(): { text: string; speech?: { rate: number; pitch: number; volume: number } } {
    const emotion = this.feel();

    // Get Broca's response
    const brocaRegion = this.regions.get('broca') as BrocaArea | undefined;
    if (!brocaRegion) {
      return { text: '...' };
    }

    let text: string;

    // Deterministic language route (text→Wernicke→Broca):
    // if there was a recent read, regenerate the response from the CLEAN
    // intention in lexicon space. This makes it reproducible for the same
    // text and discriminative across different texts, without the loop's
    // recurrent noise overwriting it.
    if (this.lastLinguisticIntention) {
      const wernicke = this.regions.get('wernicke') as WernickeArea | undefined;
      // Build the semantic intention from what Wernicke comprehends.
      const intention = new Float32Array(this.lexicon.dimensions);
      let understoodAny = false;
      if (wernicke) {
        const understood = wernicke.comprehend(this.lastLinguisticIntention);
        for (const m of understood) {
          const p = this.lexicon.lookup(m.word);
          if (p) {
            for (let i = 0; i < intention.length; i++) intention[i] += p[i] * m.similarity;
            understoodAny = true;
          }
        }
      }
      // If it understood nothing, use the clean encoding directly.
      const semantic = understoodAny ? intention : this.lastLinguisticIntention;
      const response = brocaRegion.generateResponse(semantic, {
        valence: emotion.valence,
        arousal: emotion.arousal,
      });
      text = response.words.length > 0 ? response.words.join(' ') : '...';
      const speechParams = this.speechSynthesizer.synthesize(text, emotion);
      return { text, speech: speechParams };
    }

    // Check whether Broca generated a response
    const lastResponse = brocaRegion.getLastResponse();

    if (lastResponse && lastResponse.words.length > 0) {
      text = lastResponse.words.join(' ');
    } else {
      // Force generation: build an intention from the prefrontal activity
      const pfcRegion = this.regions.get('prefrontalCortex');
      const pfcActivity = pfcRegion?.getActivity();
      if (pfcActivity && pfcActivity.outputSpikes) {
        const response = brocaRegion.generateResponse(
          pfcActivity.outputSpikes,
          { valence: emotion.valence, arousal: emotion.arousal }
        );
        text = response.words.length > 0 ? response.words.join(' ') : '...';
      } else {
        text = '...';
      }
    }

    // Synthesize voice parameters
    const speechParams = this.speechSynthesizer.synthesize(text, emotion);

    return { text, speech: speechParams };
  }

  /**
   * The brain "feels" — returns the current emotional state.
   */
  feel(): EmotionalState {
    const levels: ModulatorLevels = {
      dopamine: this.modulators.getLevel(ModulatorType.Dopamine),
      serotonin: this.modulators.getLevel(ModulatorType.Serotonin),
      norepinephrine: this.modulators.getLevel(ModulatorType.Norepinephrine),
      cortisol: this.modulators.getLevel(ModulatorType.Cortisol),
      acetylcholine: this.modulators.getLevel(ModulatorType.Acetylcholine),
      oxytocin: this.modulators.getLevel(ModulatorType.Oxytocin),
    };

    return this.emotionDecoder.decode(levels);
  }

  /**
   * The brain "imagines" — generates an image from the visual cortex.
   */
  imagine(): { pixels: Float32Array; width: number; height: number; ascii: string } {
    const visualRegion = this.regions.get('visualCortex');
    if (!visualRegion) {
      const emptyPixels = new Float32Array(32 * 32);
      return { pixels: emptyPixels, width: 32, height: 32, ascii: '' };
    }

    const activity = visualRegion.getActivity();
    const image = this.imageGenerator.generate(activity.outputSpikes);
    const ascii = this.imageGenerator.toAscii(image.pixels, image.width, image.height);

    return { ...image, ascii };
  }

  // ================================================================
  // INTERNAL PROCESSING
  // ================================================================

  /**
   * Injects sensory input into the thalamus.
   */
  private injectSensoryInput(type: 'visual' | 'auditory' | 'linguistic', spikes: Float32Array): void {
    const thalamus = this.regions.get('thalamus');
    if (thalamus) {
      thalamus.feedInput(spikes, this.currentTime);
    }

    // Also send directly to the appropriate cortex via the bus
    const targets = type === 'visual' ? ['visualCortex'] :
                    type === 'auditory' ? ['auditoryCortex'] :
                    ['wernicke', 'broca'];

    this.bus.send({
      source: 'thalamus',
      targets,
      spikes,
      timestamp: this.currentTime,
      metadata: { inputType: type },
    });
  }

  /**
   * Processes a perception: runs N ticks to propagate signals through the brain.
   */
  private processPerception(inputType: 'visual' | 'auditory' | 'text' | 'image'): PerceptionResult {
    const startTime = this.currentTime;
    const processingTicks = 50; // ~50ms of processing

    // Run ticks
    for (let i = 0; i < processingTicks; i++) {
      this.tick();
    }

    // Get the resulting emotional state
    const emotion = this.feel();

    // Identify active regions
    const activeRegions: string[] = [];
    for (const [id, region] of this.regions) {
      const activity = region.getActivity();
      if (activity.firingRate > 0.01) {
        activeRegions.push(id);
      }
    }

    return {
      inputType,
      emotion,
      activeRegions,
      processingTime: this.currentTime - startTime,
    };
  }

  /**
   * Runs one tick of the brain.
   * This is the main simulation loop.
   */
  tick(): void {
    const dt = this.config.snn.dt;
    this.currentTime += dt;
    this.tickCount++;

    // 1. Get neuromodulation effects
    const effects = this.modulators.getEffects();

    // 2. Process each region
    for (const [regionId, region] of this.regions) {
      const activity = region.step(dt, effects);

      // 3. Send output spikes to the bus
      if (activity.activeNeurons.length > 0) {
        const outgoing = this.connectome.getOutgoing(regionId);
        if (outgoing.length > 0) {
          this.bus.send({
            source: regionId,
            targets: outgoing.map(c => c.to),
            spikes: activity.outputSpikes,
            timestamp: this.currentTime,
          });
        }
      }
    }

    // 4. Dispatch bus packets (delivery deferred by axonal delays)
    this.bus.tick(this.currentTime);

    // 5. Neuromodulator decay
    this.modulators.decay(dt);

    // 6. Periodic consolidation ("sleep")
    if (this.currentTime - this.lastConsolidation > this.config.memory.consolidationIntervalMs) {
      this.sleep();
    }
  }

  /**
   * Consolidation process ("sleep").
   * Replay of hippocampus memories to strengthen the cortices.
   */
  sleep(): void {
    console.log(`💤 Consolidación iniciada (t=${this.currentTime.toFixed(0)}ms)...`);
    this.lastConsolidation = this.currentTime;

    // Consolidation would be delegated to the hippocampus
    const hippocampus = this.regions.get('hippocampus');
    if (hippocampus) {
      const stats = this.consolidationEngine.consolidate([], this.regions);
      console.log(`   Memorias replayed: ${stats.memoriesReplayed}`);
      console.log(`   Sinapsis fortalecidas: ${stats.synapsesStrengthened}`);
    }

    // Emit event
    this.emitEvent({
      type: 'consolidation',
      timestamp: this.currentTime,
      data: { consolidated: true },
    });
  }

  // ================================================================
  // STATE AND MONITORING
  // ================================================================

  /**
   * Returns the complete state of the brain.
   */
  getState(): BrainState {
    const regionsActivity: Record<string, RegionActivity> = {};
    for (const [id, region] of this.regions) {
      regionsActivity[id] = region.getActivity();
    }

    const busTraffic: Record<string, { sent: number; received: number }> = {};
    for (const [id, stats] of this.bus.getTraffic()) {
      busTraffic[id] = { sent: stats.packetsSent, received: stats.packetsReceived };
    }

    // Broca response data
    const brocaRegion = this.regions.get('broca') as BrocaArea | undefined;
    const brocaResponse = brocaRegion?.getLastResponse();

    // Visual cortex learning metrics
    const visualRegion = this.regions.get('visualCortex') as VisualCortex | undefined;
    const learning = visualRegion?.getLearningMetrics();

    // Hippocampus CA3 learning metrics
    const hippoRegion = this.regions.get('hippocampus') as Hippocampus | undefined;
    const learningHippocampus = hippoRegion?.getLearningMetrics();

    return {
      time: this.currentTime,
      regions: regionsActivity,
      modulators: {
        dopamine: this.modulators.getLevel(ModulatorType.Dopamine),
        serotonin: this.modulators.getLevel(ModulatorType.Serotonin),
        norepinephrine: this.modulators.getLevel(ModulatorType.Norepinephrine),
        cortisol: this.modulators.getLevel(ModulatorType.Cortisol),
        acetylcholine: this.modulators.getLevel(ModulatorType.Acetylcholine),
        oxytocin: this.modulators.getLevel(ModulatorType.Oxytocin),
      },
      emotion: this.feel(),
      memoriesCount: learningHippocampus?.memoryCount ?? 0,
      busTraffic,
      tickCount: this.tickCount,
      broca: brocaResponse ? {
        lastResponse: brocaResponse.words.join(' '),
        words: brocaResponse.words,
        confidence: brocaResponse.confidence,
      } : undefined,
      vocabCount: this.lexicon?.size ?? 0,
      learning,
      learningHippocampus,
    };
  }

  // ================================================================
  // PERSISTENCE (learning across sessions)
  // ================================================================

  /**
   * Saves the synaptic state of all regions + neuromodulators to disk.
   * This is what makes learning survive process restarts.
   */
  saveState(filePath: string): void {
    this.persistence.save(filePath, this.regions, this.modulators);
  }

  /**
   * Restores the state from a previously saved file. Applies the weights
   * to each region by id; skips (without aborting) regions whose dimensions
   * do not match the file —e.g. if neuronCount changed between versions—.
   *
   * @returns Which regions were restored and which were skipped.
   */
  loadState(filePath: string): { loaded: string[]; skipped: string[] } {
    const data = this.persistence.load(filePath);
    const loaded: string[] = [];
    const skipped: string[] = [];
    for (const [id, region] of this.regions) {
      const rd = data.regions.get(id);
      if (rd && rd.weights.length === region.getNetworkConfig().weights.length) {
        region.loadWeights(rd.weights);
        loaded.push(id);
      } else {
        skipped.push(id);
      }
    }
    if (data.modulatorState) {
      this.modulators.deserialize(data.modulatorState);
    }
    return { loaded, skipped };
  }

  /**
   * Gets the spike bus (for external monitoring).
   */
  getBus(): SpikeBus {
    return this.bus;
  }

  /**
   * Gets the neuromodulator system.
   */
  getModulators(): NeuromodulatorSystem {
    return this.modulators;
  }

  /**
   * Gets a region by ID.
   */
  getRegion(id: string): BrainRegion | undefined {
    return this.regions.get(id);
  }

  /**
   * Gets all regions.
   */
  getRegions(): Map<string, BrainRegion> {
    return this.regions;
  }

  // ================================================================
  // EVENTS
  // ================================================================

  /**
   * Registers a listener for brain events.
   */
  on(eventType: string, callback: (event: BrainEvent) => void): void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, []);
    }
    this.eventListeners.get(eventType)!.push(callback);
  }

  /**
   * Emits an event to all registered listeners.
   */
  private emitEvent(event: BrainEvent): void {
    const listeners = this.eventListeners.get(event.type) || [];
    for (const listener of listeners) {
      listener(event);
    }
    // Also emit to '*' listeners
    const allListeners = this.eventListeners.get('*') || [];
    for (const listener of allListeners) {
      listener(event);
    }
  }

  // ================================================================
  // CONFIGURATION
  // ================================================================

  /**
   * Returns the brain's current configuration.
   */
  getConfig(): BrainConfiguration {
    return { ...this.config };
  }

  /**
   * Current simulation time.
   */
  get time(): number {
    return this.currentTime;
  }

  /**
   * Total number of ticks processed.
   */
  get ticks(): number {
    return this.tickCount;
  }
}
