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
import type { Recognition } from './core/memory/prototype-memory.js';
import { AssociationMemory, type ModalCode } from './core/memory/association-memory.js';
import { ConsolidationEngine, type ConsolidationStats, type ShortTermEntry } from './core/memory/consolidation.js';
import { BrainPersistence, BACKUP_SUFFIX, writeFileAtomic } from './core/persistence/binary-protocol.js';
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
import { AFFECTIVE_LEXICON } from './regions/amygdala/affective-lexicon.js';
import { PrefrontalCortex } from './regions/prefrontal-cortex/prefrontal-cortex.js';
import { BrocaArea, type LanguageResponse } from './regions/broca-wernicke/broca.js';
import { WernickeArea } from './regions/broca-wernicke/wernicke.js';
import { Lexicon } from './regions/broca-wernicke/lexicon.js';
import { MotorCortex, type MotorOutput } from './regions/motor-cortex/motor-cortex.js';
import { synthesizeSpectrum, type VocalCommand } from './core/voice/vocal-tract.js';
import { seedSpanishLexicon, encodeSentenceToLexiconSpace, wordToPattern } from './regions/broca-wernicke/spanish-lexicon.js';
import { seedEnglishLexicon } from './regions/broca-wernicke/english-lexicon.js';
import * as fs from 'fs';

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
  /** Vocabulary-acquisition stats (learned words, pending exposures). */
  vocabulary?: {
    total: number;
    /** Most recent words learned this session (trimmed for streaming). */
    learnedThisSession: string[];
    /** Total words learned this session (not trimmed). */
    learnedCount: number;
    /** Pending words closest to the threshold (trimmed for streaming). */
    pending: Array<{ word: string; count: number }>;
    /** Total pending words (not trimmed). */
    pendingCount: number;
    threshold: number;
  };
  /**
   * What the senses recognize: outcome of the last completed presentation per
   * modality ("seen/heard before?") and how many categories each sense has
   * formed by exposure.
   */
  recognition?: {
    visual: Recognition | null;
    auditory: Recognition | null;
    visualCategories: number;
    auditoryCategories: number;
  };
  /**
   * Cross-modal association: what the last percept brought back from memory,
   * and how many multimodal events have been bound so far.
   */
  association?: { lastRecall: AssociationRecall | null; bindings: number };
  /** The voice: what is switched on, how much it has babbled, the last sound it made. */
  voice?: { babbling: boolean; imitation: boolean; babbles: number; lastVocalization: Vocalization | null };
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

/** Modalities whose percepts can be bound together by association. */
export type AssociationModality = 'visual' | 'auditory' | 'lexical';

/** What a percept brought back from memory (cross-modal recall). */
export interface AssociationRecall {
  /** The percept that acted as the cue. */
  cue: { modality: AssociationModality; label: string };
  /** How well learned the cue's association is (0–1); grows with each repetition of the pairing. */
  confidence: number;
  /** Whether the confidence is high enough for the brain to act on the recall. */
  confident: boolean;
  /** Words reinstated by the cue, best first. */
  words: Array<{ word: string; similarity: number }>;
  /** Visual category reinstated by the cue. */
  visual: { label: string; overlap: number } | null;
  /** Sound category reinstated by the cue. */
  auditory: { label: string; overlap: number } | null;
  /** Simulation time of the recall (ms). */
  timestamp: number;
}

/** A sound the brain produced with its own voice. */
export interface Vocalization {
  command: VocalCommand;
  /**
   * Spontaneous exploration, an attempt to repeat a sound it heard, or saying
   * the sound that what it perceives brings to mind.
   */
  source: 'babble' | 'imitation' | 'naming';
  /** For an imitation: how well the heard sound was known to the motor map (0–1). */
  confidence: number;
  /** How long the sound lasts when rendered (ms of real time). */
  durationMs: number;
  /** Simulation time (ms) and running number of the vocalization. */
  timestamp: number;
  serial: number;
}

/** Sensory channels of the thalamic relay. */
type SensoryModality = 'visual' | 'auditory' | 'linguistic';

/** Options shared by the perception entry points (`see`, `hear`, `read`). */
export interface PerceptionOptions {
  /**
   * Run the propagation ticks inline (default `true`). Pass `false` to only
   * inject the stimulus and let the caller drive `tick()` — the server does
   * this through the PerceptionScheduler so the event loop is never blocked —
   * then build the result with `describePerception()`.
   */
  propagate?: boolean;
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

  // ── Voice (babbling and vocal imitation) ──
  /** Whether the brain babbles on its own when nothing is going on. Off by default. */
  private babbling = false;
  private lastVocalization: Vocalization | null = null;
  private vocalizationSerial = 0;
  private ticksSinceVocalization = 0;
  /** Pause between spontaneous babbles (ticks): the utterance, its echo in the brain, a breath. */
  private static readonly BABBLE_INTERVAL_TICKS = 70;
  /** Rendered length of a vocalization (ms of real time). */
  private static readonly VOCALIZATION_MS = 350;

  // ── Cross-modal association (learning what goes with what) ──
  private associations = new AssociationMemory();
  /** Latest percept of each modality, kept available for binding (working-memory span). */
  private recentPercepts: Map<
    AssociationModality,
    { code: ModalCode; label: string; tick: number; serial: number; boundWith: Set<number> }
  > = new Map();
  private perceptSerial = 0;
  /** Percept counters of the sensory cortices already handled (see `collectPercepts`). */
  private handledPercepts = { visual: 0, auditory: 0 };
  private lastRecall: AssociationRecall | null = null;
  /** Lexical pattern reinstated by the last confident recall (drives `think()` / `speak()`). */
  private recalledLexicalPattern: Float32Array | null = null;
  private ticksSinceRecall: number = Number.MAX_SAFE_INTEGER;
  /**
   * How long (ticks) a percept stays available to be bound with the next one. At the
   * server's pace this spans the ~20–30 s it takes a person to show something
   * and then name it.
   */
  associationWindowTicks = 300;
  /**
   * Confidence from which a recall is acted upon (thought, spoken). One pairing
   * leaves an association at 25%; it takes a repetition to cross this — the
   * brain does not answer on first contact.
   */
  private static readonly RECALL_CONFIDENCE = 0.4;
  /** Below this match a cue evokes nothing at all. */
  private static readonly MIN_RECALL_MATCH = 0.2;
  /** Minimum similarity between a reinstated lexical pattern and a lexicon word to count as that word. */
  private static readonly RECALLED_WORD_MATCH = 0.6;

  /** Stimuli currently held by sensory persistence, one per modality. */
  private presentations: Map<SensoryModality, { signal: Float32Array; ticksLeft: number }> = new Map();
  /**
   * Ticks a stimulus stays available after it arrives (sensory persistence).
   * Bounded by what one pathway can carry without adapting (see SpikeBus).
   */
  static readonly PRESENTATION_TICKS = 30;

  /** Slow-decaying peak of each region's drive (what the dashboard bars show). */
  private drivePeaks: Map<string, number> = new Map();
  /** Per-tick decay of the held peak (≈ 1 s at the server's 10 Hz tick). */
  private static readonly DRIVE_PEAK_DECAY = 0.9;

  /** "from->to" of the connectome projections that modulate instead of drive. */
  private modulatoryPathways: Set<string> = new Set();

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

  // ── Vocabulary acquisition (learning new words from text) ──
  /**
   * Number of exposures an unknown word needs before it is committed to the
   * lexicon as a new engram.
   *
   * Biological basis:
   *   Robust word learning requires repeated exposure (statistical learning,
   *   Saffran et al., 1996). A single encounter rarely yields a durable
   *   engram; the brain consolidates a word once it has accrued enough
   *   evidence that it is a stable unit of the language.
   */
  private static readonly LEARN_THRESHOLD = 3;

  /**
   * Ticks run inline for a perception: the presentation window plus the first
   * stretch of its propagation through the connectome. The tail of the wave
   * (and the event boundary that encodes the episode) plays out on the
   * following regular ticks.
   */
  static readonly PERCEPTION_TICKS = 50;

  /** Cortical target of the thalamic relay, per modality (nodes of the connectome). */
  private static readonly THALAMIC_RELAY = {
    visual: 'visualCortex',
    auditory: 'auditoryCortex',
    linguistic: 'brocaWernicke',
  } as const;
  private static readonly SENSORY_RELAY_TARGETS: ReadonlySet<string> = new Set(
    Object.values(DigitalBrain.THALAMIC_RELAY),
  );

  /** Neurons of the auditory cortex (= afferents of the vocal motor cortex). */
  private static readonly AUDITORY_NEURONS = 1000;

  /** Cochlear (mel) bands × frames of the sliding spectrogram the auditory cortex reads. */
  private static readonly COCHLEAR_BANDS = 40;
  private static readonly SPECTROGRAM_FRAMES = 10;
  /** Sample rate assumed for microphone frames that do not declare one (browser default). */
  private static readonly DEFAULT_MIC_SAMPLE_RATE = 48000;

  /** Side of the (square) retinal image the visual encoder works on. */
  private static readonly RETINA_SIDE = 14;
  /** Input channels of the visual cortex. */
  private static readonly VISUAL_CORTEX_INPUTS = 1000;

  /**
   * Regions whose weights are NOT restored from legacy (protocol v1) files.
   * v1 files were written while the brain never rested and the hippocampus had
   * neither DG inhibition nor synaptic downscaling: their CA3 weights hold a
   * saturated cluster of noise engrams that would capture new episodes. (The
   * episodic index itself was never persisted, so no memory is lost.)
   */
  private static readonly RESET_ON_LEGACY_STATE: ReadonlySet<string> = new Set(['hippocampus']);

  // ── Sleep (memory consolidation) ──
  /** Cortical target of hippocampal replay (the hippocampus → PFC projection). */
  private static readonly CONSOLIDATION_TARGET = 'prefrontalCortex';
  /**
   * Episodes replayed per sleep, and replays of each. Replay runs inline, so
   * the product bounds how long a sleep can hold the event loop (~0.1 s).
   */
  private static readonly SLEEP_REPLAY_EPISODES = 8;
  private static readonly SLEEP_REPLAY_CYCLES = 3;
  /** Per-sleep downscaling of the CA3 recurrent weights (replayed engrams are re-imprinted). */
  private static readonly SLEEP_SYNAPTIC_DOWNSCALING = 0.97;
  /** Per-sleep decay of the episodic index (emotional episodes decay slower). */
  private static readonly SLEEP_EPISODIC_DECAY = 0.97;

  /** Fraction of the amygdala's commanded release that reaches the modulator pools per appraisal. */
  private static readonly PHASIC_RELEASE_GAIN = 0.6;

  /** Minimum token length to consider for acquisition (filters noise). */
  private static readonly MIN_WORD_LEN = 3;

  /** Maximum token length to consider for acquisition (filters pasted junk). */
  private static readonly MAX_WORD_LEN = 24;

  /** Only plain alphabetic tokens (already lowercased / accent-stripped) can be learned. */
  private static readonly LEARNABLE_WORD = /^[a-z]+$/;

  /**
   * Bounds on vocabulary acquisition. The brain is fed by untrusted users, so
   * every structure that grows with input must be capped.
   */
  private static readonly MAX_PENDING_VOCAB = 500;
  private static readonly MAX_LEXICON_SIZE = 5000;
  private static readonly MAX_LEARNED_HISTORY = 200;
  private static readonly MAX_WORDS_PER_READ = 100;

  /** How many pending / learned words `getState()` streams to the dashboard. */
  private static readonly STATE_PENDING_SHOWN = 6;
  private static readonly STATE_LEARNED_SHOWN = 24;

  /**
   * Unknown words seen so far → exposure count (pending acquisition).
   * Insertion-ordered by recency, so the first key is the least recently seen
   * (evicted first when the map is full).
   */
  private pendingVocab: Map<string, number> = new Map();

  /** Total words committed to the lexicon during this session. */
  private learnedCount: number = 0;

  /** Words learned (committed to the lexicon) during this session. */
  private learnedThisSession: string[] = [];

  /** Ticks elapsed since the last `read()` — fades the perception trace in `think()`. */
  private ticksSinceRead: number = Number.MAX_SAFE_INTEGER;

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
    console.log(`   DIGITAL BRAIN — Initializing...`);
    console.log(`   Total neurons: ${getTotalNeurons(this.config).toLocaleString()}`);
    console.log(`   Estimated memory: ${estimateMemoryUsage(this.config).toFixed(1)} MB`);
    console.log(`   Regions: ${Object.keys(this.config.regions).length}`);
    console.log(`═══════════════════════════════════════════════\n`);

    // 1. Initialize spike bus
    this.bus = new SpikeBus();

    // 2. Initialize connectome
    this.connectome = new Connectome(this.config.connectome);

    // 3. Initialize neuromodulators
    this.modulators = new NeuromodulatorSystem();

    // 4. Initialize consolidation
    this.consolidationEngine = new ConsolidationEngine(DigitalBrain.SLEEP_REPLAY_CYCLES);

    // 5. Initialize encoders
    // 14×14 × (intensity + 4 edge orientations) = 980 channels: the whole
    // retinal code fits the visual cortex's 1000 inputs. At 32×32 the encoder
    // emitted 5120 channels and the cortex silently kept the first 1000 — the
    // top rows of the intensity map, with every edge channel thrown away.
    this.visualEncoder = new VisualEncoder({
      processWidth: DigitalBrain.RETINA_SIDE,
      processHeight: DigitalBrain.RETINA_SIDE,
      foveation: true,
      edgeDetection: true,
    });
    if (this.visualEncoder.outputSize > DigitalBrain.VISUAL_CORTEX_INPUTS) {
      throw new Error(
        `Visual encoder emits ${this.visualEncoder.outputSize} channels but the visual cortex has ` +
          `${DigitalBrain.VISUAL_CORTEX_INPUTS} inputs`,
      );
    }

    this.audioEncoder = new AudioEncoder({
      sampleRate: 16000,
      fftSize: 2048,
      numBands: DigitalBrain.COCHLEAR_BANDS,
      numFrames: DigitalBrain.SPECTROGRAM_FRAMES,
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

    console.log(`✅ Brain initialized successfully.`);
    console.log(`   Tick rate: ${this.config.tickRate} Hz`);
    console.log(`   Consolidation every: ${this.config.memory.consolidationIntervalMs / 1000}s\n`);
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
    console.log(`  📚 Lexicon initialized (ES+EN): ${this.lexicon.size} words`);

    // ── Instantiate regions with reduced sizes ──
    // Total: ~10K neurons (vs 50K before) → smooth performance
    // The attentional bottleneck must be NARROWER than a typical stimulus, or
    // attention never selects anything: a sentence activates ~90 lexical
    // channels and an image ~50 retinal ones, so a gate of 100 let everything
    // through and ACh/NE had nothing to widen. At 40 the gate passes the ~55
    // most salient channels at baseline modulation and ~67 under high ACh.
    this.addRegion(new Thalamus({ neuronCount: 500, totalInputSize: 500, bottleneckSize: 40 }));
    this.addRegion(new VisualCortex({ neuronCount: 2000, inputCount: DigitalBrain.VISUAL_CORTEX_INPUTS }));
    this.addRegion(new AuditoryCortex({
      neuronCount: DigitalBrain.AUDITORY_NEURONS,
      inputCount: DigitalBrain.COCHLEAR_BANDS * DigitalBrain.SPECTROGRAM_FRAMES,
      numBands: DigitalBrain.COCHLEAR_BANDS,
      numFrames: DigitalBrain.SPECTROGRAM_FRAMES,
    }));
    this.addRegion(new Hippocampus(1000, 1000));
    const amygdala = new Amygdala(500, 500);
    for (const [word, emotion] of AFFECTIVE_LEXICON) {
      amygdala.conditionSemantic(wordToPattern(word, this.lexicon.dimensions), emotion);
    }
    this.addRegion(amygdala);
    this.addRegion(new PrefrontalCortex(3000, 1000));
    this.addRegion(new MotorCortex({ inputCount: DigitalBrain.AUDITORY_NEURONS }));
    this.addRegion(new BrocaArea(this.lexicon, 1000, 1000));
    this.addRegion(new WernickeArea(this.lexicon, 1000, 1000));

    // Connect Broca/Wernicke to the bus as an alias of 'brocaWernicke'
    // to receive packets from the existing connectome
    this.bus.onReceive('brocaWernicke', (packet: SpikePacket) => {
      const broca = this.regions.get('broca');
      const wernicke = this.regions.get('wernicke');
      // The prefrontal projection carries the INTENTION to speak: it drives
      // Broca (production) but is top-down for Wernicke (comprehension), whose
      // content must come from what is actually heard or read.
      const topDown = packet.source === 'prefrontalCortex';
      // Stamp with the ARRIVAL time (see addRegion).
      if (wernicke) {
        if (topDown) wernicke.feedModulation(packet.spikes);
        else wernicke.feedInput(packet.spikes, this.currentTime);
      }
      if (broca) broca.feedInput(packet.spikes, this.currentTime);
    });

    // Compute the real total number of neurons
    let totalNeurons = 0;
    for (const [, region] of this.regions) {
      totalNeurons += region.neurons;
    }
    console.log(`  🧩 ${this.regions.size} regions instantiated (${totalNeurons.toLocaleString()} real neurons)`);
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
      if (this.modulatoryPathways.has(`${packet.source}->${region.id}`)) {
        region.feedModulation(packet.spikes);
        return;
      }
      // Stamp with the ARRIVAL time, not the send time: the axonal delay has
      // already elapsed on the bus, and the region's sensory trace must start
      // when the spikes actually reach it.
      region.feedInput(packet.spikes, this.currentTime);
    });

    console.log(`  🧩 Region added: ${region.id} (${region.name})`);
  }

  /**
   * Configures the connectome's delays and weights on the spike bus.
   */
  private setupConnectome(): void {
    const connections = this.connectome.getAllConnections();
    for (const conn of connections) {
      this.bus.setDelay(conn.from, conn.to, conn.delay);
      this.bus.setWeight(conn.from, conn.to, conn.weight);
      if (conn.role === 'modulator') this.modulatoryPathways.add(`${conn.from}->${conn.to}`);
    }
    console.log(`  🔗 Connectome configured: ${connections.length} connections`);
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
  see(
    pixels: number[] | Float32Array | Uint8Array,
    width: number,
    height: number,
    options: PerceptionOptions = {},
  ): PerceptionResult {
    console.log(`👁️  Perceiving image (${width}×${height})...`);

    // Retinal features as graded rates: the image is HELD for a presentation
    // window and the visual cortex samples fresh spikes from it every tick.
    const rates = this.visualEncoder.encodeRates(pixels, width, height);

    // Send to the thalamus
    this.injectSensoryInput('visual', rates);

    // Process several ticks to propagate through the brain
    return this.processPerception('visual', options);
  }

  /**
   * The brain "hears" audio.
   *
   * @param audioSamples - PCM samples
   */
  hear(audioSamples: Float32Array | number[], options: PerceptionOptions = {}): PerceptionResult {
    // Encode to spikes via spectrogram
    const spikes = this.audioEncoder.encode(audioSamples, this.config.snn.dt);

    // Send to the thalamus
    this.injectSensoryInput('auditory', spikes);

    return this.processPerception('auditory', options);
  }

  /**
   * The brain "hears" one frame of microphone audio, as linear FFT magnitudes
   * in [0, 1] (what the dashboard's AnalyserNode produces). The frame goes
   * through the cochlear front-end — mel bands + sliding window — so the
   * auditory cortex receives the spectrogram layout it is built for.
   *
   * @param magnitudes - Linear-frequency magnitudes, bin 0 = DC
   * @param sampleRate - Sample rate of the source in Hz (default: 48 kHz)
   */
  hearFrame(
    magnitudes: number[] | Float32Array,
    sampleRate: number = DigitalBrain.DEFAULT_MIC_SAMPLE_RATE,
    options: PerceptionOptions = {},
  ): PerceptionResult {
    // Frames that arrive while the previous one is still being heard belong to
    // the same utterance; after a silence, a new sound event begins.
    const continuing = this.presentations.has('auditory');
    const spectrogram = this.audioEncoder.encodeMagnitudeFrame(magnitudes, sampleRate, continuing);
    this.injectSensoryInput('auditory', spectrogram);
    return this.processPerception('auditory', options);
  }

  /**
   * The brain "hears" an already-computed spectrogram: a flat
   * `[frame0 band0…band39, frame1 …]` array of band energies in [0, 1], oldest
   * frame first, in the auditory cortex's own layout. For raw microphone
   * frames (linear FFT bins) use `hearFrame`, which builds this layout.
   */
  hearSpectrogram(spectrogram: number[] | Float32Array, options: PerceptionOptions = {}): PerceptionResult {
    const spikes = new Float32Array(spectrogram.length);
    for (let i = 0; i < spectrogram.length; i++) {
      const v = spectrogram[i] as number;
      spikes[i] = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
    }
    this.injectSensoryInput('auditory', spikes);
    return this.processPerception('auditory', options);
  }

  /**
   * The brain "reads" text.
   *
   * @param text - Text to process
   */
  read(text: string, options: PerceptionOptions = {}): PerceptionResult {
    console.log(`📖 Reading: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);

    // Encode the text in LEXICON SPACE (not with the TextEncoder's generic
    // hash, which produces vectors orthogonal to the engrams and leaves
    // Wernicke unable to recognize anything). This way Wernicke retrieves the
    // real words and Broca can produce a reproducible associative response.
    const spikes = encodeSentenceToLexiconSpace(text, this.lexicon.dimensions);

    // Store the clean intention so speak() generates a reproducible
    // response (text→Wernicke→Broca) without dilution from the recurrent loop.
    this.lastLinguisticIntention = spikes;
    this.ticksSinceRead = 0;

    // What was read is a percept too: it can recall, and be bound to, what is
    // seen or heard around the same time.
    this.onPercept('lexical', DigitalBrain.denseToCode(spikes, 0.1), text.trim().slice(0, 40));

    // Send to the thalamus (linguistic route)
    this.injectSensoryInput('linguistic', spikes);

    // ── Affective appraisal ──
    // Biology: comprehended words reach the amygdala through the temporal
    // association cortex; words with a conditioned association evoke their
    // emotion there, and the central nucleus turns the resulting state into a
    // phasic neuromodulator release.
    const words = text.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^\w\sáéíóúüñ]/g, '').split(/\s+/).filter(w => w.length > 0);
    const amygdala = this.regions.get('amygdala') as Amygdala | undefined;
    let emotionalHits = 0;
    if (amygdala) {
      for (const word of words.slice(0, DigitalBrain.MAX_WORDS_PER_READ)) {
        if (amygdala.appraise(wordToPattern(word, this.lexicon.dimensions))) emotionalHits++;
      }
      if (emotionalHits > 0) {
        this.releaseFromAmygdala(amygdala);
        console.log(`  💭 ${emotionalHits} emotional words detected`);
      }
    }

    // Novelty → norepinephrine
    this.modulators.release(ModulatorType.Norepinephrine, 0.05);

    // Vocabulary acquisition: learn unknown words after repeated exposure.
    this.acquireVocabulary(words);

    return this.processPerception('text', options);
  }

  /**
   * Phasic neuromodulator release commanded by the amygdala's central nucleus
   * for its current affective state.
   */
  private releaseFromAmygdala(amygdala: Amygdala): void {
    const release = amygdala.produceNeuromodulators();
    const gain = DigitalBrain.PHASIC_RELEASE_GAIN;
    this.modulators.release(ModulatorType.Dopamine, release.dopamine * gain);
    this.modulators.release(ModulatorType.Serotonin, release.serotonin * gain);
    this.modulators.release(ModulatorType.Norepinephrine, release.norepinephrine * gain);
    this.modulators.release(ModulatorType.Cortisol, release.cortisol * gain);
    this.modulators.release(ModulatorType.Acetylcholine, release.acetylcholine * gain);
    this.modulators.release(ModulatorType.Oxytocin, release.oxytocin * gain);
  }

  /**
   * Learns new vocabulary from a tokenized utterance.
   *
   * Known words are reinforced (frequency effect); unknown words accumulate
   * exposures and, once they cross `LEARN_THRESHOLD`, are committed to the
   * lexicon as a fresh engram and rewarded with dopamine + acetylcholine
   * (learning is reinforcing and attention-gated).
   *
   * Biological basis:
   *   Modeled on incremental word learning: repeated exposure drives
   *   consolidation (Saffran et al., 1996), and successful acquisition
   *   recruits dopaminergic reward and cholinergic attention signals that
   *   stabilize the new representation.
   *
   * @param words - Normalized tokens from the current utterance
   */
  private acquireVocabulary(words: string[]): void {
    const dim = this.lexicon.dimensions;

    // Count one exposure per utterance (dedupe repeats within the same read).
    for (const word of new Set(words.slice(0, DigitalBrain.MAX_WORDS_PER_READ))) {
      if (word.length < DigitalBrain.MIN_WORD_LEN || word.length > DigitalBrain.MAX_WORD_LEN) continue;
      if (!DigitalBrain.LEARNABLE_WORD.test(word)) continue;

      if (this.lexicon.has(word)) {
        // Known word → reinforce its engram (bumps frequency / recency).
        this.lexicon.add(word, wordToPattern(word, dim));
        continue;
      }

      // Unknown word → accumulate exposures (delete + set keeps the map
      // ordered by recency for the LRU eviction below).
      const seen = (this.pendingVocab.get(word) ?? 0) + 1;
      this.pendingVocab.delete(word);

      if (seen >= DigitalBrain.LEARN_THRESHOLD) {
        // The lexicon is searched linearly on every comprehension; stop
        // acquiring once it is full rather than let it grow without bound.
        if (this.lexicon.size >= DigitalBrain.MAX_LEXICON_SIZE) continue;

        // Commit the new word to the lexicon.
        this.lexicon.add(word, wordToPattern(word, dim));
        this.learnedCount++;
        this.learnedThisSession.push(word);
        if (this.learnedThisSession.length > DigitalBrain.MAX_LEARNED_HISTORY) {
          this.learnedThisSession.shift();
        }
        // Reward + attention consolidation of the new engram.
        this.modulators.release(ModulatorType.Dopamine, 0.1);
        this.modulators.release(ModulatorType.Acetylcholine, 0.05);
        console.log(`  💡 Learned new word: "${word}" (lexicon: ${this.lexicon.size} words)`);
      } else {
        this.pendingVocab.set(word, seen);
        if (this.pendingVocab.size > DigitalBrain.MAX_PENDING_VOCAB) {
          const oldest = this.pendingVocab.keys().next().value;
          if (oldest !== undefined) this.pendingVocab.delete(oldest);
        }
      }
    }
  }

  // ================================================================
  // VOICE — babbling and vocal imitation
  // ================================================================

  /**
   * Switches the voice on or off.
   *
   * @param options.babble - Babble spontaneously when idle (exploration: this is how the motor map is learned)
   * @param options.imitate - Try to repeat the sounds it hears (needs a motor map, i.e. to have babbled)
   */
  setVoice(options: { babble?: boolean; imitate?: boolean }): void {
    const motor = this.regions.get('motorCortex') as MotorCortex | undefined;
    if (typeof options.babble === 'boolean') this.babbling = options.babble;
    if (typeof options.imitate === 'boolean' && motor) motor.imitate = options.imitate;
  }

  /** Produces one babble right now (what spontaneous babbling does on its own schedule). */
  babbleOnce(): Vocalization | null {
    const motor = this.regions.get('motorCortex') as MotorCortex | undefined;
    if (!motor || motor.vocalizing) return null;
    return this.vocalize(motor.babble());
  }

  /** The last sound the brain made, or `null`. */
  getLastVocalization(): Vocalization | null {
    return this.lastVocalization;
  }

  /** Names aloud: executes the motor command for a sound reinstated from memory. */
  private sayImaginedSound(auditoryUnits: number[]): void {
    const motor = this.regions.get('motorCortex') as MotorCortex | undefined;
    const output = motor?.sayImagined(auditoryUnits);
    if (output) this.vocalize(output);
  }

  private driveVoice(): void {
    const motor = this.regions.get('motorCortex') as MotorCortex | undefined;
    if (!motor) return;
    this.ticksSinceVocalization++;

    const imitation = motor.takeCommand();
    if (imitation) {
      this.vocalize(imitation);
      return;
    }
    if (
      this.babbling &&
      !motor.vocalizing &&
      !this.presentations.has('auditory') &&
      this.ticksSinceVocalization >= DigitalBrain.BABBLE_INTERVAL_TICKS
    ) {
      this.vocalize(motor.babble());
    }
  }

  /**
   * Executes a motor command: the sound goes out (event → dashboard
   * synthesizer) and the brain hears itself — the acoustic consequence of the
   * command enters through the same auditory pathway as any other sound, which
   * is what lets the motor cortex learn what its commands sound like.
   */
  private vocalize(output: MotorOutput): Vocalization {
    const vocalization: Vocalization = {
      command: output.command,
      source: output.source,
      confidence: output.confidence,
      durationMs: DigitalBrain.VOCALIZATION_MS,
      timestamp: this.currentTime,
      serial: ++this.vocalizationSerial,
    };
    this.lastVocalization = vocalization;
    this.ticksSinceVocalization = 0;

    const spectrum = synthesizeSpectrum(output.command);
    const spectrogram = this.audioEncoder.encodeMagnitudeFrame(spectrum, 48000, false);
    this.injectSensoryInput('auditory', spectrogram);

    this.emitEvent({ type: 'response', timestamp: this.currentTime, data: { kind: 'vocalization', ...vocalization } });
    return vocalization;
  }

  // ================================================================
  // CROSS-MODAL ASSOCIATION
  // ================================================================

  /** Sparse code (channels above `floor`) of a dense pattern. */
  private static denseToCode(pattern: Float32Array, floor: number): ModalCode {
    const indices: number[] = [];
    const values: number[] = [];
    for (let i = 0; i < pattern.length; i++) {
      if (pattern[i] > floor) {
        indices.push(i);
        values.push(Math.min(1, pattern[i]));
      }
    }
    return { indices, values };
  }

  /** Picks up the presentations the sensory cortices completed since the last tick. */
  private collectPercepts(): void {
    const visual = this.regions.get('visualCortex') as VisualCortex | undefined;
    if (visual && visual.percepts !== this.handledPercepts.visual) {
      this.handledPercepts.visual = visual.percepts;
      const units = Array.from(visual.getLastEngram());
      this.onPercept('visual', { indices: units, values: units.map(() => 1) }, visual.getRecognition()?.label ?? 'visual');
    }
    const auditory = this.regions.get('auditoryCortex') as AuditoryCortex | undefined;
    if (auditory && auditory.percepts !== this.handledPercepts.auditory) {
      this.handledPercepts.auditory = auditory.percepts;
      const units = Array.from(auditory.getLastEngram());
      this.onPercept('auditory', { indices: units, values: units.map(() => 1) }, auditory.getRecognition()?.label ?? 'sound');
    }
  }

  /**
   * A percept has just been completed in some modality.
   *
   * 1. RECALL — it acts as a cue: whatever has been associated with it comes
   *    back (the word for the thing seen, the thing for the word read).
   * 2. BIND — if other modalities were perceived recently, they belong to the
   *    same experience: their association is strengthened (Hebbian, gated by
   *    dopamine/cortisol like any other plasticity here).
   *
   * Recall comes first so that it reflects what had been learned BEFORE this
   * experience.
   */
  private onPercept(modality: AssociationModality, code: ModalCode, label: string): void {
    if (code.indices.length === 0) return;

    this.recallFrom(modality, code, label);

    const serial = ++this.perceptSerial;
    const percept = { code, label, tick: this.tickCount, serial, boundWith: new Set<number>() };
    this.recentPercepts.set(modality, percept);

    const event: Record<string, ModalCode> = { [modality]: code };
    for (const [other, recent] of this.recentPercepts) {
      if (other === modality) continue;
      if (this.tickCount - recent.tick > this.associationWindowTicks) continue;
      if (recent.boundWith.has(serial)) continue;
      event[other] = recent.code;
      recent.boundWith.add(serial);
      percept.boundWith.add(recent.serial);
    }
    if (Object.keys(event).length >= 2) {
      this.associations.bind(event, this.modulators.getEffects().learningRateMultiplier);
      this.emitEvent({
        type: 'memory',
        timestamp: this.currentTime,
        data: { kind: 'association', modalities: Object.keys(event), bindings: this.associations.bindings },
      });
    }
  }

  /** Cross-modal recall from a cue, decoded into words and perceptual categories. */
  private recallFrom(modality: AssociationModality, code: ModalCode, label: string): void {
    const result = this.associations.recall(modality, code);
    // A faint match is a chance overlap between codes (words share letters),
    // not a memory: a single real pairing already scores above this.
    if (!result || result.match < DigitalBrain.MIN_RECALL_MATCH) return;

    const confident = result.match >= DigitalBrain.RECALL_CONFIDENCE;
    const recall: AssociationRecall = {
      cue: { modality, label },
      confidence: result.match,
      confident,
      words: [],
      visual: null,
      auditory: null,
      timestamp: this.currentTime,
    };

    const lexical = result.recalled.lexical;
    let lexicalPattern: Float32Array | null = null;
    if (lexical) {
      lexicalPattern = new Float32Array(this.lexicon.dimensions);
      for (const [channel, value] of lexical.pattern) if (channel < lexicalPattern.length) lexicalPattern[channel] = value;
      recall.words = this.lexicon
        .findContained(lexicalPattern, 3)
        // Only words the reinstated pattern really spells out. While a word is
        // still unknown to the lexicon, its pattern merely resembles a few
        // known words (~0.35): better to stay silent than to say those.
        .filter((m) => m.similarity >= DigitalBrain.RECALLED_WORD_MATCH)
        .map((m) => ({ word: m.word, similarity: m.similarity }));
    }

    const topUnits = (pattern: Map<number, number>, k: number): number[] =>
      [...pattern].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, k).map(([unit]) => unit);

    const visual = this.regions.get('visualCortex') as VisualCortex | undefined;
    if (result.recalled.visual && visual) {
      const match = visual.matchCategory(topUnits(result.recalled.visual.pattern, 20));
      if (match && match.overlap >= 0.3) recall.visual = { label: match.label, overlap: match.overlap };
    }
    const auditory = this.regions.get('auditoryCortex') as AuditoryCortex | undefined;
    if (result.recalled.auditory && auditory) {
      const units = topUnits(result.recalled.auditory.pattern, 3);
      const match = auditory.matchCategory(units);
      if (match && match.overlap >= 0.5) {
        recall.auditory = { label: match.label, overlap: match.overlap };
        // The sound that comes to mind is said aloud — if the voice is on and
        // the motor map knows how to make it. (A heard sound is already
        // handled by imitation; this is for what it SEES or READS.)
        if (confident && modality !== 'auditory') this.sayImaginedSound(units);
      }
    }

    // Recorded even when nothing could be NAMED yet (e.g. the word that comes
    // back is not in the lexicon yet): something does come to mind.
    this.lastRecall = recall;
    if (confident && lexicalPattern && recall.words.length > 0) {
      this.recalledLexicalPattern = lexicalPattern;
      this.ticksSinceRecall = 0;
    }
  }

  /** What the last percept brought back from memory (or `null`). */
  getLastRecall(): AssociationRecall | null {
    return this.lastRecall;
  }

  /**
   * What the senses currently recognize (learning by exposure): the outcome of
   * the last completed presentation in each modality, and the number of
   * perceptual categories formed so far.
   */
  getRecognition(): NonNullable<BrainState['recognition']> {
    const visual = this.regions.get('visualCortex') as VisualCortex | undefined;
    const auditory = this.regions.get('auditoryCortex') as AuditoryCortex | undefined;
    return {
      visual: visual?.getRecognition() ?? null,
      auditory: auditory?.getRecognition() ?? null,
      visualCategories: visual?.categoryCount ?? 0,
      auditoryCategories: auditory?.categoryCount ?? 0,
    };
  }

  /**
   * Returns vocabulary-acquisition statistics for monitoring/visualization.
   */
  getVocabularyStats(): {
    total: number;
    learnedThisSession: string[];
    learnedCount: number;
    pending: Array<{ word: string; count: number }>;
    pendingCount: number;
    threshold: number;
  } {
    const pending = Array.from(this.pendingVocab.entries())
      .map(([word, count]) => ({ word, count }))
      .sort((a, b) => b.count - a.count);
    return {
      total: this.lexicon.size,
      learnedThisSession: [...this.learnedThisSession],
      learnedCount: this.learnedCount,
      pending,
      pendingCount: pending.length,
      threshold: DigitalBrain.LEARN_THRESHOLD,
    };
  }

  /** Vocabulary stats trimmed to what the dashboard displays (streamed 2×/s). */
  private getVocabularyStateSlice(): NonNullable<BrainState['vocabulary']> {
    const stats = this.getVocabularyStats();
    return {
      ...stats,
      learnedThisSession: stats.learnedThisSession.slice(-DigitalBrain.STATE_LEARNED_SHOWN),
      pending: stats.pending.slice(0, DigitalBrain.STATE_PENDING_SHOWN),
    };
  }

  /** Whether a word is currently in the lexicon (known to the brain). */
  knowsWord(word: string): boolean {
    return this.lexicon.has(word);
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
    // What to talk about: the last thing read — or, if something perceived
    // since then has brought words back from memory, those words (naming what
    // it sees or hears).
    const recalledIsFresher =
      this.recalledLexicalPattern !== null && this.ticksSinceRecall < this.ticksSinceRead;
    const intentionSource = recalledIsFresher ? this.recalledLexicalPattern : this.lastLinguisticIntention;

    if (intentionSource) {
      const wernicke = this.regions.get('wernicke') as WernickeArea | undefined;
      // Build the semantic intention from what Wernicke comprehends.
      const intention = new Float32Array(this.lexicon.dimensions);
      let understoodAny = false;
      if (wernicke) {
        const understood = wernicke.comprehend(intentionSource);
        for (const m of understood) {
          const p = this.lexicon.lookup(m.word);
          if (p) {
            for (let i = 0; i < intention.length; i++) intention[i] += p[i] * m.similarity;
            understoodAny = true;
          }
        }
      }
      // If it understood nothing, use the clean encoding directly.
      const semantic = understoodAny ? intention : intentionSource;
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
   * The brain "thinks" — decodes a live stream-of-consciousness snapshot from
   * its CURRENT internal state (not a scripted message).
   *
   * How it works:
   *   The language areas (Wernicke, Broca) spike in lexicon space, so their
   *   current activation can be projected back onto the lexicon to read out
   *   which words the brain is activating right now. We blend that with a
   *   decaying trace of the last input and frame it with the current emotion
   *   and dominant neuromodulator.
   *
   * Honesty:
   *   This is an ASSOCIATIVE decode of real spiking activity — a daydream that
   *   drifts at rest and reflects recent perception when stimulated. It is not
   *   deliberative reasoning or a language model.
   *
   * @returns A thought snapshot with text, words and affective frame.
   */
  think(): {
    text: string;
    words: string[];
    emotion: string;
    emoji: string;
    valence: number;
    arousal: number;
    color: string;
    dominantModulator: string;
  } {
    const dim = this.lexicon.dimensions;
    const mental = new Float32Array(dim);

    // Superimpose the current activation of the language areas (lexicon space).
    const addInto = (src: Float32Array | undefined, weight: number): void => {
      if (!src || src.length !== dim) return;
      for (let i = 0; i < dim; i++) mental[i] += src[i] * weight;
    };
    addInto(this.regions.get('wernicke')?.getActivity().outputSpikes, 0.5);
    addInto(this.regions.get('broca')?.getActivity().outputSpikes, 0.5);
    // Decaying trace of the last thing it read: recent perception clearly
    // dominates the thought right after reading (so distinct inputs yield
    // distinct thoughts), then fades back toward the spontaneous activity of the
    // language areas (exponential decay over ~50 ticks ≈ 5 s at the 10 Hz server tick).
    const traceWeight = 6.0 * Math.exp(-this.ticksSinceRead / 50);
    if (traceWeight > 0.01) addInto(this.lastLinguisticIntention ?? undefined, traceWeight);
    // …and of what a percept just brought back from memory: seeing the ball
    // brings the word "ball" to mind.
    const recallWeight = 6.0 * Math.exp(-this.ticksSinceRecall / 50);
    if (recallWeight > 0.01) addInto(this.recalledLexicalPattern ?? undefined, recallWeight);

    // Decode the mental state into the words the brain is currently activating.
    const matches = this.lexicon.findClosest(mental, 6).filter((m) => m.similarity > 0.05);
    const words = matches.map((m) => m.word);

    const emotion = this.feel();
    const dominant = this.dominantModulator();

    // Hybrid style: affective frame + decoded words (wordless mood if silent).
    const body = words.length > 0 ? words.join(' · ') : '…';
    const text = `(${emotion.primaryEmotion.toLowerCase()}) ${body}`;

    return {
      text,
      words,
      emotion: emotion.primaryEmotion,
      emoji: emotion.emoji,
      valence: emotion.valence,
      arousal: emotion.arousal,
      color: emotion.color,
      dominantModulator: dominant,
    };
  }

  /** Name of the neuromodulator most elevated above its baseline right now. */
  private dominantModulator(): string {
    const types: ModulatorType[] = [
      ModulatorType.Dopamine,
      ModulatorType.Serotonin,
      ModulatorType.Norepinephrine,
      ModulatorType.Cortisol,
      ModulatorType.Acetylcholine,
      ModulatorType.Oxytocin,
    ];
    let best = types[0];
    let bestLevel = -Infinity;
    for (const t of types) {
      const level = this.modulators.getLevel(t);
      if (level > bestLevel) {
        bestLevel = level;
        best = t;
      }
    }
    return best;
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
  private injectSensoryInput(type: SensoryModality, signal: Float32Array): void {
    // A stimulus is not an instantaneous volley: it stays available for a
    // presentation window (sensory persistence — iconic / echoic memory), and
    // the thalamus relays it on every tick of that window. Plasticity needs
    // this: STDP and the engram of a stimulus are defined over tens of ms of
    // sustained drive, not over a single sample.
    this.presentations.set(type, { signal, ticksLeft: DigitalBrain.PRESENTATION_TICKS });
  }

  /**
   * Relays the stimuli currently being presented (called once per tick).
   */
  private relayPresentations(effects: ModulationEffects): void {
    if (this.presentations.size === 0) return;
    const thalamus = this.regions.get('thalamus') as Thalamus | undefined;

    for (const [type, presentation] of this.presentations) {
      // The thalamus is the gateway to the cortex: what it relays is the
      // attention-filtered signal (top-K salient channels, with the bottleneck
      // and gain set by ACh/NE), not the raw sensory vector.
      let relayed = presentation.signal;
      if (thalamus) {
        thalamus.feedInput(presentation.signal, this.currentTime);
        relayed = thalamus.processAttention(presentation.signal, effects).filteredInput;
      }

      // Each modality has its own thalamic nucleus and cortical target
      // (LGN → visual, MGN → auditory, pulvinar → language areas); the relay
      // travels along that projection of the connectome (its delay and weight).
      this.bus.send({
        source: 'thalamus',
        targets: [DigitalBrain.THALAMIC_RELAY[type]],
        spikes: relayed,
        timestamp: this.currentTime,
        metadata: { inputType: type },
      });

      if (--presentation.ticksLeft <= 0) this.presentations.delete(type);
    }
  }

  /**
   * Processes a perception: runs N ticks to propagate signals through the brain.
   */
  private processPerception(
    inputType: PerceptionResult['inputType'],
    options: PerceptionOptions,
  ): PerceptionResult {
    const startTime = this.currentTime;

    if (options.propagate !== false) {
      for (let i = 0; i < DigitalBrain.PERCEPTION_TICKS; i++) {
        this.tick();
      }
    }

    return this.describePerception(inputType, startTime);
  }

  /**
   * Summarizes the brain's reaction to a perception that started at
   * `startTime` (simulation ms). Used directly by callers that inject with
   * `propagate: false` and run the ticks themselves.
   */
  describePerception(inputType: PerceptionResult['inputType'], startTime: number): PerceptionResult {
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
    if (this.ticksSinceRead < Number.MAX_SAFE_INTEGER) this.ticksSinceRead++;

    // 1. Get neuromodulation effects
    const effects = this.modulators.getEffects();

    //    Stimuli being presented keep arriving through the thalamus.
    this.relayPresentations(effects);

    //    The hippocampus stamps the episodes it encodes with the current affect
    //    (emotional episodes are forgotten more slowly).
    (this.regions.get('hippocampus') as Hippocampus | undefined)?.setAffectiveContext(this.feel().valence);

    // 2. Process each region
    for (const [regionId, region] of this.regions) {
      const activity = region.step(dt, effects);
      this.drivePeaks.set(
        regionId,
        Math.max(activity.drive, (this.drivePeaks.get(regionId) ?? 0) * DigitalBrain.DRIVE_PEAK_DECAY),
      );

      // 3. Send output spikes to the bus
      if (activity.activeNeurons.length > 0) {
        // In the connectome the language areas are one node ('brocaWernicke').
        // What leaves it toward memory is what was UNDERSTOOD, i.e. Wernicke's
        // output; Broca's output is motor (speech), not an afferent of memory.
        const nodeId = regionId === 'wernicke' ? 'brocaWernicke' : regionId;
        let outgoing = this.connectome.getOutgoing(nodeId);
        // The thalamo-cortical projections carry the attention-filtered
        // sensory signal (see injectSensoryInput). The relay neurons' own
        // population code lives in a different space: written into cortical
        // input channels it was read as a spectrogram / a retina / a sentence,
        // and made the auditory cortex "hear" text. It only travels the
        // non-sensory projections (the low road to the amygdala).
        if (regionId === 'thalamus') {
          outgoing = outgoing.filter((c) => !DigitalBrain.SENSORY_RELAY_TARGETS.has(c.to));
        }
        if (outgoing.length > 0) {
          this.bus.send({
            source: nodeId,
            targets: outgoing.map(c => c.to),
            spikes: activity.outputSpikes,
            timestamp: this.currentTime,
          });
        }
      }
    }

    //    Voice: an imitation the motor cortex has just decided on, or a babble.
    this.driveVoice();

    //    Percepts completed by the sensory cortices this tick → association.
    this.collectPercepts();
    if (this.ticksSinceRecall < Number.MAX_SAFE_INTEGER) this.ticksSinceRecall++;

    // 4. Dispatch bus packets (delivery deferred by axonal delays)
    this.bus.tick(this.currentTime);

    // 5. Neuromodulator decay
    this.modulators.decay(dt);

    // 6. Periodic consolidation ("sleep")
    //    Only at rest: sleeping in the middle of a perception would replay
    //    over live activity and cut the wave short.
    if (
      this.currentTime - this.lastConsolidation > this.config.memory.consolidationIntervalMs &&
      this.bus.pendingCount === 0
    ) {
      this.sleep();
    }
  }

  /**
   * Consolidation process ("sleep").
   *
   * Biology: during slow-wave sleep the hippocampus replays recent episodes
   * (sharp-wave ripples) toward the neocortex, which gradually absorbs them
   * (systems consolidation; McClelland et al., 1995). Here the strongest and
   * most recent episodes are reactivated in CA3 and replayed along the
   * hippocampus → prefrontal projection of the connectome, where Hebbian
   * plasticity strengthens the synapses they drive. Afterwards the episodic
   * index fades a little: what the cortex has learned no longer depends on it.
   *
   * @returns What was actually replayed and strengthened
   */
  sleep(): ConsolidationStats {
    console.log(`💤 Consolidation started (t=${this.currentTime.toFixed(0)}ms)...`);
    this.lastConsolidation = this.currentTime;

    const hippocampus = this.regions.get('hippocampus') as Hippocampus | undefined;
    const cortex = this.regions.get(DigitalBrain.CONSOLIDATION_TARGET);
    let stats: ConsolidationStats = {
      memoriesReplayed: 0,
      synapsesStrengthened: 0,
      duration: 0,
      consolidatedLabels: [],
      prunedLabels: [],
    };

    if (hippocampus && cortex) {
      const episodes = hippocampus.replay(DigitalBrain.SLEEP_REPLAY_EPISODES);
      const entries: ShortTermEntry[] = episodes.map((episode) => ({
        pattern: episode.pattern,
        label: `episode@${episode.context.timestamp.toFixed(0)}`,
        timestamp: episode.context.timestamp,
        strength: episode.strength,
        associatedRegion: DigitalBrain.CONSOLIDATION_TARGET,
        replayCount: 0,
      }));

      stats = this.consolidationEngine.consolidate(entries, this.regions);
      cortex.settle();
      hippocampus.forget(DigitalBrain.SLEEP_EPISODIC_DECAY);
      hippocampus.downscale(DigitalBrain.SLEEP_SYNAPTIC_DOWNSCALING);

      console.log(`   Memories replayed: ${stats.memoriesReplayed}`);
      console.log(`   Synapses strengthened: ${stats.synapsesStrengthened}`);
    }

    // Emit event
    this.emitEvent({
      type: 'consolidation',
      timestamp: this.currentTime,
      data: {
        consolidated: stats.consolidatedLabels.length,
        memoriesReplayed: stats.memoriesReplayed,
        synapsesStrengthened: stats.synapsesStrengthened,
      },
    });

    return stats;
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
      regionsActivity[id] = { ...region.getActivity(), drivePeak: this.drivePeaks.get(id) ?? 0 };
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
      vocabulary: this.getVocabularyStateSlice(),
      recognition: this.getRecognition(),
      association: { lastRecall: this.lastRecall, bindings: this.associations.bindings },
      voice: {
        babbling: this.babbling,
        imitation: (this.regions.get('motorCortex') as MotorCortex | undefined)?.imitate ?? false,
        babbles: (this.regions.get('motorCortex') as MotorCortex | undefined)?.babbleCount ?? 0,
        lastVocalization: this.lastVocalization,
      },
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
    // Everything learned that is not a weight matrix: each region's own state
    // (episodic index, homeostasis…), the simulation clock the episodes are
    // dated with, and the words on their way to being learned.
    const extras: Record<string, unknown> = {
      brain: {
        time: this.currentTime,
        tickCount: this.tickCount,
        pendingVocab: Array.from(this.pendingVocab.entries()),
      },
      association: this.associations.serialize(),
    };
    for (const [id, region] of this.regions) {
      const extra = region.serializeExtra();
      if (extra !== null && extra !== undefined) extras[id] = extra;
    }
    this.persistence.save(filePath, this.regions, this.modulators, extras);
    // The lexicon (incl. words learned from text) is not part of the binary
    // region format, so persist it as a JSON sidecar next to the .bin.
    try {
      writeFileAtomic(this.lexiconSidecarPath(filePath), JSON.stringify(this.lexicon.serialize()));
    } catch (err) {
      console.error(`⚠️  Could not save lexicon: ${(err as Error).message}`);
    }
  }

  /**
   * Restores the simulation clock and the pending vocabulary. The clock matters:
   * episodes are dated in simulated time, and recency (replay priority,
   * forgetting) is measured against it — a clock restarted at 0 would make
   * every restored episode look like it comes from the future.
   */
  private restoreBrainExtras(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as { time?: unknown; tickCount?: unknown; pendingVocab?: unknown };

    if (typeof d.time === 'number' && Number.isFinite(d.time) && d.time >= 0) {
      this.currentTime = d.time;
      this.lastConsolidation = d.time;
      for (const region of this.regions.values()) region.currentTime = d.time;
    }
    if (typeof d.tickCount === 'number' && Number.isInteger(d.tickCount) && d.tickCount >= 0) {
      this.tickCount = d.tickCount;
    }
    if (Array.isArray(d.pendingVocab)) {
      this.pendingVocab.clear();
      for (const entry of d.pendingVocab.slice(-DigitalBrain.MAX_PENDING_VOCAB)) {
        if (!Array.isArray(entry)) continue;
        const [word, count] = entry as [unknown, unknown];
        if (
          typeof word === 'string' && DigitalBrain.LEARNABLE_WORD.test(word) &&
          word.length <= DigitalBrain.MAX_WORD_LEN &&
          typeof count === 'number' && Number.isInteger(count) && count > 0
        ) {
          this.pendingVocab.set(word, Math.min(count, DigitalBrain.LEARN_THRESHOLD - 1));
        }
      }
    }
  }

  /** Path of the lexicon sidecar associated with a brain state file. */
  private lexiconSidecarPath(filePath: string): string {
    return `${filePath}.lexicon.json`;
  }

  /**
   * Restores the state from a previously saved file. Applies the weights
   * to each region by id; skips (without aborting) regions whose dimensions
   * do not match the file —e.g. if neuronCount changed between versions— or
   * whose weights contain non-finite values.
   *
   * @returns Which regions were restored and which were skipped.
   */
  loadState(filePath: string): { loaded: string[]; skipped: string[] } {
    // If the main file is missing or corrupted (failed CRC, truncated write),
    // fall back to the previous snapshot instead of starting from scratch.
    let data;
    try {
      data = this.persistence.load(filePath);
    } catch (err) {
      const backupPath = `${filePath}${BACKUP_SUFFIX}`;
      if (!fs.existsSync(backupPath)) throw err;
      console.warn(`⚠️  ${(err as Error).message} — restoring previous snapshot ${backupPath}`);
      data = this.persistence.load(backupPath);
    }

    const loaded: string[] = [];
    const skipped: string[] = [];
    for (const [id, region] of this.regions) {
      const rd = data.regions.get(id);
      if (rd && data.version < 2 && DigitalBrain.RESET_ON_LEGACY_STATE.has(id)) {
        console.warn(`⚠️  ${id}: weights from a legacy (v${data.version}) state are not restored — starting fresh.`);
        skipped.push(id);
        continue;
      }
      if (
        rd &&
        rd.weights.length === region.getNetworkConfig().weights.length &&
        rd.weights.every(Number.isFinite)
      ) {
        region.loadWeights(rd.weights);
        // The non-weight state belongs to these weights: restore them together.
        if (data.extras[id] !== undefined) region.deserializeExtra(data.extras[id]);
        loaded.push(id);
      } else {
        skipped.push(id);
      }
    }
    if (data.modulatorState) {
      this.modulators.deserialize(data.modulatorState);
    }
    this.restoreBrainExtras(data.extras.brain);
    this.associations.deserialize(data.extras.association);

    // Restore the persisted lexicon (incl. learned words) if present and
    // dimensionally compatible; otherwise keep the freshly seeded vocabulary.
    const sidecar = this.lexiconSidecarPath(filePath);
    if (fs.existsSync(sidecar)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(sidecar, 'utf-8'));
        if (parsed?.patternSize === this.lexicon.dimensions) {
          this.lexicon.deserialize(parsed);
          console.log(`  📚 Lexicon restored: ${this.lexicon.size} words`);
        } else {
          console.warn(`⚠️  Lexicon sidecar dim mismatch (${parsed?.patternSize} vs ${this.lexicon.dimensions}); keeping seeded vocabulary.`);
        }
      } catch (err) {
        console.error(`⚠️  Could not restore lexicon: ${(err as Error).message}`);
      }
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
