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
import { PrototypeMemory, type Recognition } from './core/memory/prototype-memory.js';
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
import { PartsCortex } from './regions/visual-cortex/parts-cortex.js';
import { AuditoryCortex } from './regions/auditory-cortex/auditory-cortex.js';
import { Hippocampus } from './regions/hippocampus/hippocampus.js';
import { Amygdala } from './regions/amygdala/amygdala.js';
import { appraiseProsody, type ProsodyAppraisal, type VoiceContour } from './regions/amygdala/prosody.js';
import { Motivation, type ActivityKind, type Drives, type RewardEvent } from './core/motivation/motivation.js';
import { HabitSystem, type Habit } from './core/motivation/habits.js';
import { ImaginationMemory, mixUnits, type ImaginationOrigin, type ImaginedCombination } from './core/imagination/imagination.js';
import type { CategoryView } from './core/memory/prototype-memory.js';
import { mulberry32 } from './core/random.js';
import { SequenceMemory, type Prediction } from './core/memory/sequence-memory.js';
import { ColorCortex } from './regions/color-cortex/color-cortex.js';
import { QuestionMemory } from './core/memory/question-memory.js';
import { PrefrontalCortex } from './regions/prefrontal-cortex/prefrontal-cortex.js';
import { BrocaArea, type LanguageResponse } from './regions/broca-wernicke/broca.js';
import { WernickeArea } from './regions/broca-wernicke/wernicke.js';
import { Lexicon } from './regions/broca-wernicke/lexicon.js';
import { MotorCortex, type MotorOutput } from './regions/motor-cortex/motor-cortex.js';
import { synthesizeFrames, UTTERANCE_FRAME_MS, type VocalCommand } from './core/voice/vocal-tract.js';
import { HandMotorCortex, type HandOutput } from './regions/motor-cortex/hand-motor-cortex.js';
import { BOARD_SIDE, GRID_SIDE, inkedCells, renderDrawing, type Drawing } from './core/hand/whiteboard.js';
import type { StrokePath } from './core/hand/strokes.js';
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
  /** What the innate layer has reacted to (tone of voice, startles, faces…) */
  innate?: InnateState;
  /** Why it acts: drives, the value of its activities, the last dopamine event */
  motivation?: MotivationState;
  /** What it expects to come next, and what it holds in mind */
  sequence?: { expectation: Prediction | null; transitions: number; lastTransition: { from: string; to: string } | null };
  workingMemory?: Array<{ label: string; priority: number; age: number }>;
  /** Questions it has learned to answer, and the last answer it gave */
  questions?: { known: number; lastAnswer: { question: string; modality: string; word: string; timestamp: number } | null };
  /** What it has imagined, awake and asleep. */
  imagination?: ImaginationState;
  /** Stimulus–response links stamped in by practice (see core/motivation/habits). */
  habits?: { count: number; links: Habit[]; lastHabit: { cue: string; response: string; timestamp: number } | null };
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
    colour: Recognition | null;
    /** What the parts cortex (V2→IT) makes of the last thing seen: the object its parts complete. */
    object: Recognition | null;
    visualCategories: number;
    auditoryCategories: number;
    colourCategories: number;
    objectCategories: number;
    /** Local parts learned by the parts cortex. */
    partsKnown: number;
  };
  /**
   * Cross-modal association: what the last percept brought back from memory,
   * and how many multimodal events have been bound so far.
   */
  association?: { lastRecall: AssociationRecall | null; bindings: number };
  /** The voice: what is switched on, how much it has babbled, the last sound it made. */
  voice?: { babbling: boolean; imitation: boolean; babbles: number; lastVocalization: Vocalization | null };
  /** The hand: what is switched on, how much it has scribbled, the last thing it drew. */
  hand?: { scribbling: boolean; copying: boolean; scribbles: number; lastDrawing: HandDrawing | null };
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
export type AssociationModality = 'visual' | 'colour' | 'auditory' | 'lexical';

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
  /** Colour category reinstated by the cue. */
  colour: { label: string; overlap: number } | null;
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
  source: 'babble' | 'imitation' | 'naming' | 'call';
  /** For an imitation: how well the heard sound was known to the motor map (0–1). */
  confidence: number;
  /** How long the sound lasts when rendered (ms of real time). */
  durationMs: number;
  /** Simulation time (ms) and running number of the vocalization. */
  timestamp: number;
  serial: number;
}

/** Something the brain drew on the whiteboard with its own hand. */
export interface HandDrawing {
  /** Inked cells of the GRID_SIDE × GRID_SIDE grid (`row * gridSide + col`). */
  cells: Drawing;
  gridSide: number;
  /** A scribble (exploration), a copy of what it has just seen, a drawing of what came to mind, or of what it imagined. */
  source: 'scribble' | 'copy' | 'from-memory' | 'imagined';
  /** The strokes it lays the cells down in, when it knows a gesture for the drawing. */
  strokes?: StrokePath[];
  confidence: number;
  timestamp: number;
  serial: number;
}

/** The last voice the innate layer appraised. */
export interface VoiceAppraisal {
  valence: number;
  arousal: number;
  kind: 'warm' | 'neutral' | 'harsh';
  startle: boolean;
  /** Whether it was taken as a verdict on what the brain had just recalled. */
  judged: boolean;
  timestamp: number;
  serial: number;
}

/** State of the innate layer (what the brain brings to the world unlearned). */
export interface InnateState {
  lastVoice: VoiceAppraisal | null;
  voicesHeard: number;
  startles: number;
  loomings: number;
  /** Face-likeness of the last image seen (0..1) and whether it counted as a face. */
  faceMatch: number;
  facesSeen: number;
  /** The last conditioned cue that evoked its emotion. */
  lastCueResponse: { modality: string; label: string; valence: number; arousal: number; timestamp: number } | null;
  conditionedCues: number;
  /** Words that have acquired an emotion by being heard alongside one. */
  affectiveWords: number;
  /** 0..1: sleep pressure; the brain sleeps when it reaches 1 (and is at rest). */
  sleepPressure: number;
}

/**
 * Something the brain made up: two things it knows, recombined and completed
 * by its own memory (see core/imagination). Tagged with its origin — never an
 * episode of the world.
 */
export interface Imagined {
  origin: ImaginationOrigin;
  /** What it was made of ('visual:Visual-1', 'colour:Colour-2'). */
  sources: string[];
  /** Words it brought to mind. */
  words: string[];
  /** Categories it evoked in each modality (what it "sees" and "hears"). */
  visual: string | null;
  colour: string | null;
  auditory: string | null;
  /** How far from anything experienced (0..1): 1 for a combination never met. */
  novelty: number;
  /** Times this combination has been imagined, this one included. */
  times: number;
  /** The image in the mind's eye (retinal pattern, 0..1), if any. */
  image: number[] | null;
  imageSide: number;
  timestamp: number;
  serial: number;
}

/** State of the imagination (see core/imagination). */
export interface ImaginationState {
  lastImagined: Imagined | null;
  /** Daydreams and dreams so far. */
  daydreams: number;
  dreams: number;
  /** Distinct combinations imagined, and how many of them later turned up for real. */
  combinations: number;
  foreseen: number;
  recent: ImaginedCombination[];
}

/** State of the motivation system (see core/motivation). */
export interface MotivationState {
  drives: Drives;
  /** Expected learning progress of each activity (what makes it choose). */
  activityValues: Record<ActivityKind, number>;
  /** The last reward prediction error and what caused it, and the few before it. */
  lastEvent: RewardEvent | null;
  recentEvents: RewardEvent[];
  events: number;
  /** Spontaneous activities taken so far, by kind. */
  chosen: Record<ActivityKind, number>;
  calls: number;
}

/** Sensory channels of the thalamic relay. */
type SensoryModality = 'visual' | 'colour' | 'auditory' | 'linguistic';

/** Options shared by the perception entry points (`see`, `hear`, `read`). */
export interface PerceptionOptions {
  /**
   * Run the propagation ticks inline (default `true`). Pass `false` to only
   * inject the stimulus and let the caller drive `tick()` — the server does
   * this through the PerceptionScheduler so the event loop is never blocked —
   * then build the result with `describePerception()`.
   */
  propagate?: boolean;
  /** Colour of the image, interleaved r, g, b per pixel (`see` only). Without it, the thing has no colour. */
  rgb?: Uint8Array | number[];
  /**
   * How the image was drawn, if someone drew it in front of the brain: the
   * strokes in order, each the points the pen went through (image pixels)
   * and how long it took. The hand learns the gesture (`see` only).
   */
  strokes?: Array<{ points: Array<[number, number]>; durationMs?: number }>;
}

/** Event emitted by the brain */
export interface BrainEvent {
  type: 'spike' | 'emotion' | 'memory' | 'consolidation' | 'response' | 'affect';
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
  /** Frames of the utterance in progress still to be heard, one per UTTERANCE_FRAME_MS. */
  private voiceFrames: Float32Array[] = [];
  private ticksSinceVoiceFrame = 0;
  /** Vocalizations counted as calls for contact. */
  private calls = 0;
  private lastCallTick = Number.MIN_SAFE_INTEGER;
  /** Rendered length of a vocalization (ms of real time). */
  private static readonly VOCALIZATION_MS = 350;

  // ── Hand (scribbling, copying and drawing from memory) ──
  private scribbling = false;
  private lastDrawing: HandDrawing | null = null;
  private drawingSerial = 0;

  // ── Motivation (why it does anything on its own) ──
  private motivation!: Motivation;
  /** Pause between spontaneous activities (real ms) when nothing pushes: an utterance or a scribble, its way back, a breath. */
  private static readonly EXPLORATION_INTERVAL_MS = 7000;
  private readonly explorationIntervalTicks: number;
  private ticksSinceExploration = 0;
  /** How much boredom shortens the pause (at full boredom the pause halves). */
  private static readonly BOREDOM_URGE = 0.5;
  /** Need for contact from which a babble becomes a call, and the least time between calls. */
  private static readonly CALL_CONTACT = 0.7;
  private static readonly CALL_INTERVAL_MS = 60_000;

  // ── Imagination (default mode) and dreams ──
  /** Alone with nothing coming in for this long, the mind starts to wander. */
  private static readonly DAYDREAM_AFTER_MS = 5000;
  private readonly daydreamAfterTicks: number;
  /** Reward of a first imagining of a combination never experienced (habituates with repetition). */
  private static readonly IMAGINATION_GAIN = 0.4;
  /** Reward when something imagined turns up for real (an imagining that proved useful). */
  private static readonly FORESEEN_REWARD = 0.5;
  /** Overlap from which what a cue reinstates is the other thing: the pair was experienced together. */
  private static readonly KNOWN_PAIR_OVERLAP = 0.3;
  /** REM: chimeras of two episodes completed in CA3 and shown to the cortex, per sleep. */
  private static readonly DREAMS_PER_SLEEP = 3;
  /** REM plasticity: the cortex sees the dream, it barely learns it. */
  private static readonly DREAM_PLASTICITY = 0.3;
  /** REM: learned things replayed as variants (shifted, noisy) so that the categories generalize. */
  private static readonly REM_VARIANTS = 4;
  private static readonly REM_CYCLES = 4;
  /** Pruning: a category met fewer times than this and unseen for this many sleeps is dropped. */
  private static readonly PRUNE_MIN_EXPOSURES = 2;
  private static readonly PRUNE_AFTER_SLEEPS = 2;
  private readonly imagination = new ImaginationMemory();

  // ── Habits ──
  /**
   * Expected outcome below which the goal-directed system withholds a
   * response (devaluation): a cue that has come to predict reprimand is not
   * answered — unless a habit answers it regardless.
   */
  private static readonly DEVALUED = -0.3;
  private readonly habits = new HabitSystem();
  private lastHabit: { cue: string; response: string; timestamp: number } | null = null;
  /** Imagination and dreams draw from their own random source: they leave every other trajectory untouched. */
  private readonly imaginationRandom = mulberry32(0x1ac1e);
  private lastImagined: Imagined | null = null;
  private imaginedSerial = 0;
  private daydreams = 0;
  private dreams = 0;
  private foreseen = 0;
  /** Ticks since the last percept of the world (own voice and drawings do not count). */
  private ticksSincePercept = Number.MAX_SAFE_INTEGER;
  private ticksSinceDaydream = 0;
  /** Dopamine released per unit of positive prediction error, and per unit of negative (the dip). */
  private static readonly DOPAMINE_BURST_GAIN = 0.5;
  private static readonly DOPAMINE_DIP_GAIN = 0.4;
  private chosen: Record<ActivityKind, number> = { babble: 0, scribble: 0, daydream: 0 };
  private motorLearnings = { babble: 0, scribble: 0 };

  // ── Questions: what a question asks for, and answering it ──
  private questions = new QuestionMemory();
  /** The last text read, in case the next one answers it. */
  private pendingQuestion: { key: string; tick: number } | null = null;
  /** Real ms within which a text read after another counts as its answer. */
  private static readonly QUESTION_WINDOW_MS = 8000;
  /** Real ms within which what a dimension of the thing in front brought to mind can be given as the answer. */
  private static readonly ANSWER_WINDOW_MS = 20_000;
  /** The last recall from each modality's percept (what each dimension of the thing in front brought to mind). */
  private recallsByModality: Map<AssociationModality, { recall: AssociationRecall; tick: number }> = new Map();
  private lastAnswer: { question: string; modality: string; word: string; timestamp: number } | null = null;
  /**
   * Cross-situational word–referent statistics (Smith & Yu 2008): how often
   * each word has been read while each category (a colour, a shape, a sound)
   * was in view. A word that goes with one colour and no other is that
   * colour's name; a word that goes with everything ("de", "que") is nobody's.
   * word → modality:label → count
   */
  private wordReferents: Map<string, Map<string, number>> = new Map();
  private static readonly MAX_WORD_REFERENTS = 5000;
  /** Co-occurrences a word needs with a referent, and how specific it must be, to be its name. */
  private static readonly REFERENT_MIN_COUNT = 2;
  private static readonly REFERENT_MIN_SPECIFICITY = 0.6;

  // ── Time: what follows what, and what is held in mind ──
  private sequences!: SequenceMemory;
  /** Real ms within which one percept counts as following another. */
  private static readonly SEQUENCE_WINDOW_MS = 10_000;
  private lastTransition: { from: string; to: string } | null = null;
  /** |prediction error| from which a percept is gated into working memory (dopamine gating; O'Reilly & Frank 2006). */
  private static readonly WORKING_MEMORY_GATE = 0.2;

  // ── Cross-modal association (learning what goes with what) ──
  private associations = new AssociationMemory();
  /** Latest percept of each modality, kept available for binding (working-memory span). */
  private recentPercepts: Map<
    AssociationModality,
    { code: ModalCode; label: string; tick: number; serial: number; boundWith: Set<number> }
  > = new Map();
  private perceptSerial = 0;
  /** Percept counters of the sensory cortices already handled (see `collectPercepts`). */
  private handledPercepts = { visual: 0, auditory: 0, colour: 0, parts: 0 };
  /**
   * Object-level units (parts cortex) share the visual modality with V1's
   * units in the association memory, offset so the two never collide.
   */
  private static readonly OBJECT_UNIT_OFFSET = 10_000;
  private lastRecall: AssociationRecall | null = null;
  /** Tick of the last recall (any confidence): a voice soon after it is a verdict on it. */
  private lastRecallTick = Number.MIN_SAFE_INTEGER;
  /** Conjunction units behind `lastRecall` (what feedback reinforces or weakens). */
  private lastRecallUnits: number[] = [];
  /** Lexical pattern reinstated by the last confident recall (drives `think()` / `speak()`). */
  private recalledLexicalPattern: Float32Array | null = null;
  private ticksSinceRecall: number = Number.MAX_SAFE_INTEGER;
  /**
   * How long (real ms) a percept stays available to be bound with the next one:
   * the ~20–30 s it takes a person to show something and then name it.
   */
  private static readonly ASSOCIATION_WINDOW_MS = 30_000;
  /** The association window in ticks (settable: tests shorten it). */
  associationWindowTicks: number;
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

  // ── The innate layer ──
  /** The speaker's usual pitch (Hz); the prosody detectors read pitch relative to it. Adapts slowly. */
  private speakerPitchHz = 150;
  private lastVoice: VoiceAppraisal | null = null;
  private voicesHeard = 0;
  private startles = 0;
  private loomings = 0;
  private faceMatch = 0;
  private facesSeen = 0;
  /** The last innate emotional event (US): what a word or a cue perceived around it is conditioned to. */
  private innateAffect: { valence: number; arousal: number; tick: number } | null = null;
  /** Real ms around an emotional event within which a word or a cue gets conditioned to it. */
  private static readonly CONDITIONING_WINDOW_MS = 5000;
  private readonly conditioningWindowTicks: number;
  /** Real ms after a recall within which a voice counts as a verdict on it. */
  private static readonly VERDICT_WINDOW_MS = 8000;
  private readonly verdictWindowTicks: number;
  /** Real ms within which a retinal image that grows is "looming". */
  private static readonly LOOMING_WINDOW_MS = 3000;
  private readonly loomingWindowTicks: number;
  /** Coverage of the retina by the last image seen, for the looming detector. */
  private lastRetina: { coverage: number; tick: number } | null = null;
  /** Correlation with the innate face template from which an image counts as a face. */
  private static readonly FACE_MATCH = 0.35;
  /** Cues (category codes) perceived recently, available for conditioning. */
  private recentCues: Map<'visual' | 'auditory', { units: number[]; label: string; tick: number }> = new Map();
  private lastCueResponse: InnateState['lastCueResponse'] = null;
  /** A conditioned cue was just perceived: unless its emotional event follows, extinction weakens it. */
  private pendingExtinction: { modality: 'visual' | 'auditory'; units: number[]; untilTick: number } | null = null;
  /** Words last read, for conditioning by a voice that follows them. */
  private lastReadWords: string[] = [];
  /**
   * Sleep pressure (adenosine): grows with time awake and with neural activity,
   * is cleared by sleep. 1 = must sleep. The wake term alone reaches 1 at the
   * configured consolidation interval; activity brings it forward.
   */
  private sleepPressure = 0;
  /** Simulated ms of full-brain activity that amount to one unit of sleep pressure. */
  private static readonly ACTIVITY_PRESSURE_MS = 200;
  /** Sleep pressure at which it sleeps even in the middle of things. */
  private static readonly EXHAUSTION_PRESSURE = 1.5;
  private totalNeurons = 1;

  /** Stimuli currently held by sensory persistence, one per modality. */
  private presentations: Map<SensoryModality, { signal: Float32Array; ticksLeft: number }> = new Map();
  /**
   * How long (real ms) a stimulus stays available after it arrives (sensory
   * persistence). Bounded by what one pathway can carry without adapting (see
   * SpikeBus).
   */
  private static readonly PRESENTATION_MS = 3000;

  // ── The moving eye ──
  /**
   * A scene with several objects is not seen at once: the eye fixates one,
   * the cortex sees it alone, and a saccade takes the eye to the next
   * (superior colliculus for where, inhibition of return so that each is
   * visited once). The pause between fixations lets the cortex close one
   * presentation before the next begins.
   */
  private static readonly SACCADE_GAP_MS = 1200;
  private static readonly MAX_FIXATIONS = 4;
  private readonly saccadeGapTicks: number;
  private fixations: Array<{ rates: Float32Array; colour: Float32Array | null; box: { x: number; y: number; w: number; h: number } }> = [];
  private fixationIndex = 0;
  private fixationCount = 0;
  private ticksSinceGaze = 0;
  private saccades = 0;
  /** The presentation window in ticks. */
  readonly presentationTicks: number;

  /** Slow-decaying peak of each region's drive (what the dashboard bars show). */
  private drivePeaks: Map<string, number> = new Map();
  /** Per-tick decay of the held peak: 0.9 per 100 ms (≈ 1 s), whatever the tick rate. */
  private readonly drivePeakDecay: number;

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
   * Real ms run inline for a perception: the presentation window plus the
   * first stretch of its propagation through the connectome. The tail of the
   * wave (and the event boundary that encodes the episode) plays out on the
   * following regular ticks.
   */
  private static readonly PERCEPTION_MS = 5000;
  /** The perception in ticks (what the server's scheduler runs per stimulus). */
  readonly perceptionTicks: number;

  /** Decay time (real ms) of the trace of what was last read or recalled, in `think()`. */
  private static readonly THOUGHT_TRACE_MS = 5000;
  private readonly thoughtTraceTicks: number;

  // ── Human-timescale intervals of the regions (real ms) ──
  /** A vocal command is held for the utterance plus the time its sound takes to come back. */
  private static readonly VOCAL_HOLD_MS = 5000;
  /** A hand command is held while the drawing stays in view and its image comes back. */
  private static readonly HAND_HOLD_MS = 5500;
  /** Silence / blank that ends a heard sound or a seen image (then the imitation or copy is issued). */
  private static readonly PLAN_GAP_MS = 1000;
  /** Longest stretch of input the hippocampus binds into one episode. */
  private static readonly EPISODE_MAX_MS = 25_000;
  /** Silence that closes an episode (event boundary). */
  private static readonly EPISODE_GAP_MS = 2500;

  /** Real milliseconds per tick (1000 / tickRate). */
  readonly msPerTick: number;

  /** Cortical target of the thalamic relay, per modality (nodes of the connectome). */
  private static readonly THALAMIC_RELAY: Record<SensoryModality, readonly string[]> = {
    // Ventral stream (what it is) and dorsal stream (how to act on it).
    visual: ['visualCortex', 'partsCortex', 'handMotorCortex'],
    // Colour parts ways with shape in the ventral stream (V4).
    colour: ['colorCortex'],
    auditory: ['auditoryCortex'],
    linguistic: ['brocaWernicke'],
  };
  private static readonly SENSORY_RELAY_TARGETS: ReadonlySet<string> = new Set(
    Object.values(DigitalBrain.THALAMIC_RELAY).flat(),
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
    if (!(this.config.tickRate > 0) || !Number.isFinite(this.config.tickRate)) {
      throw new Error(`tickRate must be a positive number of ticks per second (got ${this.config.tickRate})`);
    }
    this.msPerTick = 1000 / this.config.tickRate;
    this.presentationTicks = this.ticksFor(DigitalBrain.PRESENTATION_MS);
    this.saccadeGapTicks = this.ticksFor(DigitalBrain.SACCADE_GAP_MS);
    this.perceptionTicks = this.ticksFor(DigitalBrain.PERCEPTION_MS);
    this.associationWindowTicks = this.ticksFor(DigitalBrain.ASSOCIATION_WINDOW_MS);
    this.explorationIntervalTicks = this.ticksFor(DigitalBrain.EXPLORATION_INTERVAL_MS);
    this.daydreamAfterTicks = this.ticksFor(DigitalBrain.DAYDREAM_AFTER_MS);
    this.motivation = new Motivation({
      rewardWindowTicks: this.ticksFor(DigitalBrain.VERDICT_WINDOW_MS),
      boredomTicks: this.ticksFor(60_000),
      contactTicks: this.ticksFor(300_000),
    });
    this.sequences = new SequenceMemory(this.ticksFor(DigitalBrain.SEQUENCE_WINDOW_MS));
    this.thoughtTraceTicks = this.ticksFor(DigitalBrain.THOUGHT_TRACE_MS);
    this.drivePeakDecay = Math.pow(0.9, this.msPerTick / 100);
    this.conditioningWindowTicks = this.ticksFor(DigitalBrain.CONDITIONING_WINDOW_MS);
    this.verdictWindowTicks = this.ticksFor(DigitalBrain.VERDICT_WINDOW_MS);
    this.loomingWindowTicks = this.ticksFor(DigitalBrain.LOOMING_WINDOW_MS);

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
    console.log(`   Tick rate: ${this.config.tickRate} Hz (a stimulus stays ${this.presentationTicks} ticks, a percept waits ${this.associationWindowTicks})`);
    console.log(`   Consolidation every: ${this.config.memory.consolidationIntervalMs / 1000}s\n`);
  }

  /** Whether a sound is being heard right now (a frame arriving now continues the same utterance). */
  get isHearing(): boolean {
    return this.presentations.has('auditory');
  }

  /** Ticks that span `ms` of real time at this brain's tick rate (at least 1). */
  ticksFor(ms: number): number {
    return Math.max(1, Math.round(ms / this.msPerTick));
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
    // channels and a simple drawing ~90 retinal ones (cells + edges), so a gate
    // of 100 let everything through and ACh/NE had nothing to widen. At 60 the
    // gate passes the ~83 most salient channels at baseline modulation and
    // ~100 under high ACh — narrower than a stimulus, wide enough for a
    // drawing to be seen roughly whole.
    this.addRegion(new Thalamus({ neuronCount: 500, totalInputSize: 500, bottleneckSize: 60 }));
    this.addRegion(new VisualCortex({ neuronCount: 2000, inputCount: DigitalBrain.VISUAL_CORTEX_INPUTS }));
    this.addRegion(new AuditoryCortex({
      neuronCount: DigitalBrain.AUDITORY_NEURONS,
      inputCount: DigitalBrain.COCHLEAR_BANDS * DigitalBrain.SPECTROGRAM_FRAMES,
      numBands: DigitalBrain.COCHLEAR_BANDS,
      numFrames: DigitalBrain.SPECTROGRAM_FRAMES,
    }));
    this.addRegion(new Hippocampus(1000, 1000, 10000, {
      maxEventTicks: this.ticksFor(DigitalBrain.EPISODE_MAX_MS),
      eventGapTicks: this.ticksFor(DigitalBrain.EPISODE_GAP_MS),
    }));
    // The amygdala brings no meanings with it: what a word or a thing means
    // emotionally is learned by being perceived alongside an innate emotional
    // event (a tone of voice, a startle…). See hearVoice() and the PLAN.
    this.addRegion(new Amygdala(500, 500));
    this.addRegion(new PrefrontalCortex(3000, 1000));
    this.addRegion(new MotorCortex({
      inputCount: DigitalBrain.AUDITORY_NEURONS,
      holdTicks: this.ticksFor(DigitalBrain.VOCAL_HOLD_MS),
      planGapTicks: this.ticksFor(DigitalBrain.PLAN_GAP_MS),
      // The onset frame of an utterance: what the lips learn from and read.
      onsetTicks: this.ticksFor(UTTERANCE_FRAME_MS),
    }));
    this.addRegion(new HandMotorCortex({
      inputCount: DigitalBrain.VISUAL_CORTEX_INPUTS,
      holdTicks: this.ticksFor(DigitalBrain.HAND_HOLD_MS),
      planGapTicks: this.ticksFor(DigitalBrain.PLAN_GAP_MS),
    }));
    this.addRegion(new BrocaArea(this.lexicon, 1000, 1000));
    this.addRegion(new WernickeArea(this.lexicon, 1000, 1000));
    // Last, and with deterministic synapses: the others' seeded trajectories stay as they were.
    this.addRegion(new ColorCortex());
    // The parts cortex (V2→IT): objects as arrangements of local parts. Added
    // last, with its own random source, so the other trajectories stay put.
    this.addRegion(new PartsCortex({ retinaSide: DigitalBrain.RETINA_SIDE, channelMaps: 5, inputCount: DigitalBrain.VISUAL_CORTEX_INPUTS }));

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
    this.totalNeurons = Math.max(1, totalNeurons);
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

    // Innate visual detectors on the retinal image itself (before any cortex):
    // a face draws attention and comfort; something growing fast alarms.
    const retina = this.visualEncoder.encodeIntensity(pixels, width, height);
    this.detectFace(retina);
    this.detectLooming(retina);

    const rgb = options.rgb && options.rgb.length >= width * height * 3 ? options.rgb : null;
    const colourOf = (px: ArrayLike<number>, w: number, h: number): Float32Array | null => {
      const colour = VisualEncoder.encodeColor(px, w, h);
      let any = 0;
      for (let i = 0; i < colour.length; i++) any += colour[i];
      return any > 0 ? colour : null;
    };

    // Several objects: the eye fixates them one at a time (saccades). The
    // first now; the rest follow, each once the cortex has closed the last.
    const objects = this.visualEncoder.segment(pixels, width, height).slice(0, DigitalBrain.MAX_FIXATIONS);
    this.fixations = [];
    if (objects.length >= 2) {
      const fixations = objects.map((box) => {
        const view = VisualEncoder.fixate(pixels, width, height, box);
        const viewRgb = rgb ? VisualEncoder.fixate(rgb, width, height, box, 3) : null;
        return {
          rates: this.visualEncoder.encodeRates(view.pixels, view.width, view.height),
          colour: viewRgb ? colourOf(viewRgb.pixels, viewRgb.width, viewRgb.height) : null,
          box,
        };
      });
      this.fixationCount = fixations.length;
      this.fixationIndex = 0;
      this.fixations = fixations.slice(1);
      this.gaze(fixations[0]);
      return this.processPerception('visual', options);
    }

    // Send to the thalamus
    this.injectSensoryInput('visual', rates);
    // The hand notes what is in view (a copy is a copy of it) and, if someone
    // drew it in front of the brain, watches how (the gesture).
    (this.regions.get('handMotorCortex') as HandMotorCortex | undefined)?.lookingAt(inkedCells(pixels, width, height));
    if (options.strokes && options.strokes.length > 0) this.watchStrokes(options.strokes, width, height);
    // …and, if the image has colour, its colour goes its own way (V4).
    if (rgb) {
      const colour = colourOf(rgb, width, height);
      if (colour) this.injectSensoryInput('colour', colour);
    }

    // Process several ticks to propagate through the brain
    return this.processPerception('visual', options);
  }

  /** The strokes of a drawing made in view, as paths of whiteboard cells, for the hand to learn the gesture. */
  private watchStrokes(strokes: NonNullable<PerceptionOptions['strokes']>, width: number, height: number): void {
    const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
    if (!hand) return;
    const paths: StrokePath[] = [];
    for (const stroke of strokes) {
      const cells: number[] = [];
      const visit = (x: number, y: number): void => {
        const col = Math.max(0, Math.min(GRID_SIDE - 1, Math.floor((x * GRID_SIDE) / width)));
        const row = Math.max(0, Math.min(GRID_SIDE - 1, Math.floor((y * GRID_SIDE) / height)));
        const cell = row * GRID_SIDE + col;
        if (cells[cells.length - 1] !== cell) cells.push(cell);
      };
      // The pen moves continuously between the points reported: walk the line.
      let last: [number, number] | null = null;
      for (const [x, y] of stroke.points) {
        if (last) {
          const steps = Math.max(1, Math.ceil(Math.max(Math.abs(x - last[0]), Math.abs(y - last[1]))));
          for (let i = 1; i <= steps; i++) visit(last[0] + ((x - last[0]) * i) / steps, last[1] + ((y - last[1]) * i) / steps);
        } else visit(x, y);
        last = [x, y];
      }
      if (cells.length > 0) paths.push({ cells, durationMs: stroke.durationMs ?? 0 });
    }
    if (paths.length > 0) hand.observeStrokes(paths);
  }

  /** The eye lands on an object: the cortex sees it (and its colour) alone. */
  private gaze(fixation: { rates: Float32Array; colour: Float32Array | null; box: { x: number; y: number; w: number; h: number } }): void {
    this.fixationIndex++;
    this.saccades++;
    this.ticksSinceGaze = 0;
    this.injectSensoryInput('visual', fixation.rates);
    if (fixation.colour) this.injectSensoryInput('colour', fixation.colour);
    this.emitEvent({
      type: 'affect',
      timestamp: this.currentTime,
      data: { kind: 'saccade', index: this.fixationIndex, count: this.fixationCount, box: fixation.box, remaining: this.fixations.length },
    });
  }

  /** The rest of an utterance reaches the ear frame by frame, as the mouth moves. */
  private speakOn(): void {
    if (this.voiceFrames.length === 0) return;
    if (++this.ticksSinceVoiceFrame < this.ticksFor(UTTERANCE_FRAME_MS)) return;
    this.ticksSinceVoiceFrame = 0;
    const frame = this.voiceFrames.shift() as Float32Array;
    this.injectSensoryInput('auditory', this.audioEncoder.encodeMagnitudeFrame(frame, 48000, true));
  }

  /** Saccades pending: after the cortex has closed the current fixation, the eye moves on. */
  private moveEye(): void {
    if (this.fixations.length === 0) return;
    if (this.presentations.has('visual')) {
      this.ticksSinceGaze = 0;
      return;
    }
    if (++this.ticksSinceGaze < this.saccadeGapTicks) return;
    const next = this.fixations.shift();
    if (next) this.gaze(next);
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
    const lexicalCode = DigitalBrain.denseToCode(spikes, 0.1);
    this.onPercept('lexical', lexicalCode, text.trim().slice(0, 40));
    this.noteWordReferents(text);
    this.considerQuestion(`lexical:${text.trim().slice(0, 40)}`, lexicalCode);

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
      // Classical conditioning: words that arrive while an innate emotional
      // event is still hot (a harsh voice, a startle, a warm voice) acquire its
      // emotion. Aversive events condition in one pairing; pleasant ones take a few.
      this.lastReadWords = Array.from(new Set(words.slice(0, DigitalBrain.MAX_WORDS_PER_READ)));
      this.conditionWords(amygdala, this.lastReadWords);
    }

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
        // A new word is a novelty (its dopamine is a prediction error, see
        // reward()); attention consolidates the new engram.
        this.reward(this.motivation.novelty(`word:${word}`, this.tickCount));
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
    const imitation = motor.takeCommand();
    if (imitation) this.vocalize(imitation);
  }

  /**
   * Spontaneous activity — what it does when nothing is asked of it. Which
   * activity (babbling, scribbling) is chosen by its learned value: the
   * learning progress it has been bringing (see core/motivation). Boredom
   * shortens the pause between activities; the need for contact turns a
   * babble into a call.
   */
  private driveExploration(): void {
    this.ticksSinceExploration++;
    const motor = this.regions.get('motorCortex') as MotorCortex | undefined;
    const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
    const available: ActivityKind[] = [];
    if (this.babbling && motor && !motor.vocalizing && !this.presentations.has('auditory')) available.push('babble');
    if (this.scribbling && hand && !hand.drawing && !this.presentations.has('visual')) available.push('scribble');
    // Nothing coming in for a while: the default mode — the mind wanders. A
    // mental activity, on its own clock: it takes no turn from the voice or
    // the hand. Boredom brings it forward; the value it has earned, too.
    this.ticksSinceDaydream++;
    if (this.presentations.size === 0 && !this.isHearing && this.ticksSincePercept >= this.daydreamAfterTicks) {
      const drives = this.motivation.drives(this.tickCount);
      const pause = this.explorationIntervalTicks * (1 - DigitalBrain.BOREDOM_URGE * drives.boredom) * (1.2 - this.motivation.activityValues.daydream);
      if (this.ticksSinceDaydream >= Math.max(1, Math.round(pause))) {
        this.ticksSinceDaydream = 0;
        this.chosen.daydream++;
        this.imagineOnce('daydream');
      }
    }
    if (available.length === 0) return;

    const drives = this.motivation.drives(this.tickCount);
    // Alone for a while: a babble goes out as a call.
    if (
      motor && available.includes('babble') &&
      drives.contact >= DigitalBrain.CALL_CONTACT &&
      this.tickCount - this.lastCallTick >= this.ticksFor(DigitalBrain.CALL_INTERVAL_MS)
    ) {
      this.lastCallTick = this.tickCount;
      this.calls++;
      this.ticksSinceExploration = 0;
      this.vocalize({ ...motor.babble(), source: 'call' });
      return;
    }

    const pause = Math.max(1, Math.round(this.explorationIntervalTicks * (1 - DigitalBrain.BOREDOM_URGE * drives.boredom)));
    if (this.ticksSinceExploration < pause) return;
    const kind = this.motivation.choose(available);
    if (!kind) return;
    this.ticksSinceExploration = 0;
    this.chosen[kind]++;
    if (kind === 'babble' && motor) this.vocalize(motor.babble());
    else if (kind === 'scribble' && hand) this.draw(hand.scribble());
  }

  // ================================================================
  // IMAGINATION — the default mode, awake and asleep
  // ================================================================

  /** Everything it knows in the kinds it can recombine. */
  private knownThings(): Array<{ modality: AssociationModality; category: CategoryView }> {
    const things: Array<{ modality: AssociationModality; category: CategoryView }> = [];
    for (const c of (this.regions.get('visualCortex') as VisualCortex | undefined)?.categories() ?? []) things.push({ modality: 'visual', category: c });
    for (const c of (this.regions.get('colorCortex') as ColorCortex | undefined)?.categories() ?? []) things.push({ modality: 'colour', category: c });
    for (const c of (this.regions.get('auditoryCortex') as AuditoryCortex | undefined)?.categories() ?? []) things.push({ modality: 'auditory', category: c });
    return things;
  }

  /**
   * One imagining. Two things it knows, taken at random, are recombined into
   * a cue — half the units of each when they are of one kind (a chimera), the
   * two side by side when they are of different kinds (a colour on a shape).
   * The association memory completes the cue into words and into the other
   * kinds, as it completes any partial cue (constructive episodic simulation;
   * Schacter & Addis 2007), and the visual cortex renders it in the mind's
   * eye through its feedback synapses (Kosslyn). Nothing of it is perceived:
   * it founds no category, is bound to nothing and becomes no episode —
   * reality monitoring (Johnson & Raye 1981) by construction.
   *
   * Awake (a daydream), imagining something never experienced is rewarding —
   * dopamine the brain makes for itself when the world brings none — less
   * each time the same imagining repeats, so it does not loop on one. Asleep
   * (a dream) it is just what the sleeping cortex shows itself.
   */
  imagineOnce(origin: ImaginationOrigin, random: () => number = this.imaginationRandom): Imagined | null {
    const things = this.knownThings();
    if (things.length < 2) {
      if (origin === 'daydream') this.reward(this.motivation.activityRewarded('daydream', 0, this.tickCount));
      return null;
    }
    const ai = Math.floor(random() * things.length);
    let bi = Math.floor(random() * (things.length - 1));
    if (bi >= ai) bi++;
    const a = things[ai];
    const b = things[bi];
    const sources = [`${a.modality}:${a.category.label}`, `${b.modality}:${b.category.label}`];

    const visual = this.regions.get('visualCortex') as VisualCortex | undefined;
    const colour = this.regions.get('colorCortex') as ColorCortex | undefined;
    const auditory = this.regions.get('auditoryCortex') as AuditoryCortex | undefined;
    const cortices = { visual, colour, auditory } as const;
    const code = (units: readonly number[]): ModalCode => ({ indices: [...units], values: units.map(() => 1) });

    const cue: Partial<Record<AssociationModality, ModalCode>> = {};
    const labels: Record<'visual' | 'colour' | 'auditory', string | null> = { visual: null, colour: null, auditory: null };
    let novelty: number;
    if (a.modality === b.modality) {
      const mixed = mixUnits(a.category.units, b.category.units, random);
      cue[a.modality] = code(mixed);
      const nearest = a.modality === 'lexical' ? null : cortices[a.modality]?.matchCategory(mixed) ?? null;
      novelty = 1 - (nearest?.overlap ?? 0);
      if (a.modality !== 'lexical') labels[a.modality] = nearest ? `${nearest.label}~` : null;
    } else {
      cue[a.modality] = code(a.category.units);
      cue[b.modality] = code(b.category.units);
      if (a.modality !== 'lexical') labels[a.modality] = a.category.label;
      if (b.modality !== 'lexical') labels[b.modality] = b.category.label;
      // Experienced together already? Then what one brings back of the other's kind IS the other.
      const from = this.associations.recall(a.modality, cue[a.modality]!);
      const reinstated = from?.recalled[b.modality];
      const overlap = reinstated ? PrototypeMemory.overlap(DigitalBrain.topUnits(reinstated.pattern, b.category.units.length), b.category.units) : 0;
      novelty = overlap >= DigitalBrain.KNOWN_PAIR_OVERLAP ? 0.2 : 1;
    }

    // Completion: what the cue brings to mind — words, and the kinds not in the cue.
    const words = new Set<string>();
    let lexicalPattern: Float32Array | null = null;
    let imageUnits: readonly number[] | null = cue.visual?.indices ?? null;
    for (const [modality, c] of Object.entries(cue) as Array<[AssociationModality, ModalCode]>) {
      const result = this.associations.recall(modality, c);
      if (!result || result.match < DigitalBrain.MIN_RECALL_MATCH) continue;
      if (result.recalled.lexical) {
        const read = this.readWords(result.recalled.lexical.pattern);
        // The word that comes to mind for each thing: the one it spells best.
        if (read.words.length > 0) words.add(read.words[0].word);
        if (!lexicalPattern) lexicalPattern = read.pattern;
        else for (let i = 0; i < lexicalPattern.length; i++) lexicalPattern[i] = Math.max(lexicalPattern[i], read.pattern[i]);
      }
      for (const other of ['visual', 'colour', 'auditory'] as const) {
        if (other === modality || cue[other] || !result.recalled[other]) continue;
        const raw = DigitalBrain.topUnits(result.recalled[other]!.pattern, other === 'visual' ? 28 : other === 'colour' ? 6 : 3);
        const units = other === 'visual' ? DigitalBrain.splitVisualUnits(raw).v1 : raw;
        const match = units.length > 0 ? cortices[other]?.matchCategory(units) : null;
        if (match && match.overlap >= (other === 'auditory' ? 0.2 : 0.3)) {
          labels[other] = match.label;
          if (other === 'visual' && !imageUnits) imageUnits = units;
        }
      }
    }
    const image = visual && imageUnits ? visual.imagine(imageUnits) : null;

    const record = this.imagination.imagined(sources, this.tickCount, novelty >= 0.5);
    const imagined: Imagined = {
      origin,
      sources,
      words: [...words],
      visual: labels.visual,
      colour: labels.colour,
      auditory: labels.auditory,
      novelty,
      times: record.count,
      image: image ? Array.from(image, (v) => Math.round(v * 100) / 100) : null,
      imageSide: image ? Math.round(Math.sqrt(image.length)) : 0,
      timestamp: this.currentTime,
      serial: ++this.imaginedSerial,
    };
    this.lastImagined = imagined;
    if (origin === 'daydream') this.daydreams++;
    else this.dreams++;

    // The words come to mind (the thought stream shows them; on waking, the dream is remembered).
    if (lexicalPattern) {
      this.recalledLexicalPattern = lexicalPattern;
      this.ticksSinceRecall = 0;
    }
    if (origin === 'daydream') {
      // Its own reward: new, and less each time it is the same imagining.
      const reward = (DigitalBrain.IMAGINATION_GAIN * novelty) / record.count;
      this.reward(this.motivation.activityRewarded('daydream', reward, this.tickCount, `imagined:${sources.join('+')}`));
      // If the hand is on and knows how, it draws what it imagined.
      const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
      // Drawn once, the first time it imagines something never experienced.
      const output = image && hand?.copy && novelty >= 0.5 && record.count === 1 ? hand.drawImagined(image) : null;
      if (output) this.draw({ ...output, source: 'imagined' });
    }
    this.emitEvent({ type: 'response', timestamp: this.currentTime, data: { kind: 'imagination', ...imagined } });
    return imagined;
  }

  /**
   * REM, generative replay of what it knows: each well-met visual category is
   * rendered from its engram, shifted and roughened, and shown to the cortex
   * with plasticity low — variants it never saw, so that the category comes
   * to cover them (Hoel 2021: dreams as regularization against overfitting).
   * Shown as its own imagery: no percept, no category.
   */
  private replayVariants(effects: ModulationEffects, random: () => number = this.imaginationRandom): number {
    const visual = this.regions.get('visualCortex') as VisualCortex | undefined;
    if (!visual) return 0;
    const categories = visual.categories()
      .filter((c) => c.exposures >= DigitalBrain.PRUNE_MIN_EXPOSURES)
      .sort((x, y) => y.lastSeen - x.lastSeen)
      .slice(0, DigitalBrain.REM_VARIANTS);
    if (categories.length === 0) return 0;
    const silence = new Float32Array(visual.inputs);
    for (const c of categories) {
      const image = visual.imagine(c.units);
      const side = Math.round(Math.sqrt(image.length));
      const variant = new Float32Array(image.length);
      if (side * side === image.length) {
        const dx = Math.floor(random() * 3) - 1;
        const dy = Math.floor(random() * 3) - 1;
        for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < side && ny >= 0 && ny < side) variant[ny * side + nx] = image[y * side + x];
        }
      } else variant.set(image);
      for (let i = 0; i < variant.length; i++) variant[i] = Math.max(0, Math.min(1, variant[i] * (0.8 + 0.4 * random())));
      visual.suppressNextPercept();
      for (let k = 0; k < DigitalBrain.REM_CYCLES; k++) visual.reactivate(variant, effects);
      // Silence closes the presentation as the brain's own imagery.
      for (let k = 0; k < 12; k++) visual.reactivate(silence, effects);
    }
    visual.settle();
    return categories.length;
  }

  /** What it has imagined, awake and asleep. */
  getImagination(): ImaginationState {
    return {
      lastImagined: this.lastImagined,
      daydreams: this.daydreams,
      dreams: this.dreams,
      combinations: this.imagination.size,
      foreseen: this.foreseen,
      recent: this.imagination.recent(6),
    };
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

    // Its own voice is not a sound of the world: the auditory cortex learns
    // from it (that is how the motor map is built) but it founds no category
    // and is bound to nothing. (When an imitation counted as a percept, a
    // drawing taught with a vowel got bound to the vowel AND to the brain's
    // own slightly-off repetition of it, and the association split in two.)
    (this.regions.get('auditoryCortex') as AuditoryCortex | undefined)?.suppressNextPercept();
    // An utterance unfolds in time: the onset (a murmur, a burst) now, the
    // vowel a frame later, into the same window of the ear.
    const frames = synthesizeFrames(output.command);
    this.injectSensoryInput('auditory', this.audioEncoder.encodeMagnitudeFrame(frames[0], 48000, false));
    this.voiceFrames = frames.slice(1);
    this.ticksSinceVoiceFrame = 0;

    this.emitEvent({ type: 'response', timestamp: this.currentTime, data: { kind: 'vocalization', ...vocalization } });
    return vocalization;
  }

  // ================================================================
  // HAND — scribbling, copying and drawing from memory
  // ================================================================

  /**
   * Switches the hand on or off.
   *
   * @param options.scribble - Scribble spontaneously when idle (exploration: this is how the visuomotor map is learned)
   * @param options.copy - Draw what it sees or what comes to its mind (needs a map, i.e. to have scribbled)
   */
  setHand(options: { scribble?: boolean; copy?: boolean }): void {
    const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
    if (typeof options.scribble === 'boolean') this.scribbling = options.scribble;
    if (typeof options.copy === 'boolean' && hand) hand.copy = options.copy;
  }

  /** Produces one scribble right now (what spontaneous scribbling does on its own schedule). */
  scribbleOnce(): HandDrawing | null {
    const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
    if (!hand || hand.drawing) return null;
    return this.draw(hand.scribble());
  }

  /** The last thing the brain drew, or `null`. */
  getLastDrawing(): HandDrawing | null {
    return this.lastDrawing;
  }

  /** Draws from memory: executes the hand command for an image reinstated by a word or a sound. */
  private drawImaginedImage(retinalPattern: Float32Array): void {
    const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
    const output = hand?.drawImagined(retinalPattern);
    if (output) this.draw(output);
  }

  private driveHand(): void {
    const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
    if (!hand) return;
    const copy = hand.takeCommand();
    if (copy) this.draw(copy);
  }

  /**
   * Executes a drawing command: the marks go onto the whiteboard (event →
   * dashboard) and the brain SEES what it has drawn — the image enters through
   * the same visual pathway as anything else it looks at, which is what lets
   * the hand motor cortex learn what its commands look like.
   */
  private draw(output: HandOutput): HandDrawing {
    const drawing: HandDrawing = {
      cells: output.cells,
      gridSide: GRID_SIDE,
      source: output.source,
      confidence: output.confidence,
      timestamp: this.currentTime,
      serial: ++this.drawingSerial,
    };
    if (output.strokes) drawing.strokes = output.strokes;
    this.lastDrawing = drawing;

    // Its own drawing is not an object of the world (see `vocalize`): the hand
    // learns from it, but it founds no category and is not bound to anything.
    (this.regions.get('visualCortex') as VisualCortex | undefined)?.suppressNextPercept();
    (this.regions.get('partsCortex') as PartsCortex | undefined)?.suppressNextPercept();
    const rates = this.visualEncoder.encodeRates(renderDrawing(output.cells), BOARD_SIDE, BOARD_SIDE);
    this.injectSensoryInput('visual', rates);

    this.emitEvent({ type: 'response', timestamp: this.currentTime, data: { kind: 'drawing', ...drawing } });
    return drawing;
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
    const parts = this.regions.get('partsCortex') as PartsCortex | undefined;
    if (visual && visual.percepts !== this.handledPercepts.visual) {
      this.handledPercepts.visual = visual.percepts;
      const units = Array.from(visual.getLastEngram());
      // The object level's engram joins V1's in the same visual code: what a
      // partial view completes up there reaches memory alongside what V1 saw.
      let recallCode: ModalCode | undefined;
      if (parts && parts.percepts !== this.handledPercepts.parts) {
        this.handledPercepts.parts = parts.percepts;
        const objectUnits = Array.from(parts.getLastEngram(), (u) => u + DigitalBrain.OBJECT_UNIT_OFFSET);
        units.push(...objectUnits);
        // Two routes to memory — the whole (V1) and the parts (the object
        // level). The whole is the more specific and comes first; the parts
        // complete what the whole has never seen (a new bracket that is half
        // a square) — a whole V1 has never seen would only dilute their cue.
        const v1Units = units.slice(0, units.length - objectUnits.length);
        const v1Code: ModalCode = { indices: v1Units, values: v1Units.map(() => 1) };
        const objectCode: ModalCode = { indices: objectUnits, values: objectUnits.map(() => 1) };
        const byWhole = this.associations.recall('visual', v1Code)?.match ?? 0;
        if (byWhole >= DigitalBrain.MIN_RECALL_MATCH) recallCode = v1Code;
        else if ((this.associations.recall('visual', objectCode)?.match ?? 0) > byWhole) recallCode = objectCode;
      }
      this.onPercept('visual', { indices: units, values: units.map(() => 1) }, visual.getRecognition()?.label ?? 'visual', visual.getRecognition()?.surprise ?? 1, recallCode);
    } else if (parts && parts.percepts !== this.handledPercepts.parts) {
      this.handledPercepts.parts = parts.percepts;
    }
    const auditory = this.regions.get('auditoryCortex') as AuditoryCortex | undefined;
    if (auditory && auditory.percepts !== this.handledPercepts.auditory) {
      this.handledPercepts.auditory = auditory.percepts;
      const units = Array.from(auditory.getLastEngram());
      this.onPercept('auditory', { indices: units, values: units.map(() => 1) }, auditory.getRecognition()?.label ?? 'sound', auditory.getRecognition()?.surprise ?? 1);
    }
    const colour = this.regions.get('colorCortex') as ColorCortex | undefined;
    if (colour && colour.percepts !== this.handledPercepts.colour) {
      this.handledPercepts.colour = colour.percepts;
      const units = Array.from(colour.getLastEngram());
      this.onPercept('colour', { indices: units, values: units.map(() => 1) }, colour.getRecognition()?.label ?? 'colour', colour.getRecognition()?.surprise ?? 1);
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
  private onPercept(modality: AssociationModality, code: ModalCode, label: string, surprise: number = 1, recallCode: ModalCode = code): void {
    if (code.indices.length === 0) return;
    this.ticksSincePercept = 0;

    this.recallFrom(modality, recallCode, label);
    const key = `${modality}:${label}`;
    // A stamped-in habit answers the cue directly — faster, without attention,
    // and whether or not the recall behind it still holds.
    const habit = this.habits.habitFor(key);
    if (habit) this.respondByHabit(habit, label);
    // Order: was this expected to follow what came before? Then learn the
    // transition, and expect what usually follows this.
    const expectedness = this.sequences.expectedness(key, this.tickCount);
    const step = this.sequences.observe(key, this.tickCount);
    if (step.learned) this.lastTransition = step.learned;
    if (step.prediction) {
      this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'expectation', after: key, ...step.prediction } });
    }
    if (modality === 'visual' || modality === 'auditory' || modality === 'colour') {
      if (modality !== 'colour') this.appraiseCue(modality, code.indices, label);
      const novelty = this.motivation.perceive(key, this.tickCount, expectedness, surprise);
      this.reward(novelty);
      this.gateIntoWorkingMemory(label, novelty.error);
      // Surprise — the cortex's own prediction error — is what attention runs on:
      // an input the category did not predict well recruits alertness and focus.
      if (surprise >= 0.3) {
        this.modulators.release(ModulatorType.Norepinephrine, 0.1 * surprise);
        this.modulators.release(ModulatorType.Acetylcholine, 0.05 * surprise);
      }
    }

    const serial = ++this.perceptSerial;
    const percept = { code, label, tick: this.tickCount, serial, boundWith: new Set<number>() };
    this.recentPercepts.set(modality, percept);

    const event: Record<string, ModalCode> = { [modality]: code };
    const experienced = [key];
    for (const [other, recent] of this.recentPercepts) {
      if (other === modality) continue;
      if (this.tickCount - recent.tick > this.associationWindowTicks) continue;
      if (recent.boundWith.has(serial)) continue;
      event[other] = recent.code;
      experienced.push(`${other}:${recent.label}`);
      recent.boundWith.add(serial);
      percept.boundWith.add(recent.serial);
    }
    if (Object.keys(event).length >= 2) {
      this.associations.bind(event, this.modulators.getEffects().learningRateMultiplier);
      // Something it had imagined, met for real: the imagining proved useful.
      const foreseen = this.imagination.foresaw(experienced);
      if (foreseen) {
        this.foreseen++;
        this.reward(this.motivation.activityRewarded('daydream', DigitalBrain.FORESEEN_REWARD, this.tickCount, `foreseen:${foreseen.sources.join('+')}`));
        this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'foreseen', sources: foreseen.sources, imagined: foreseen.count } });
      }
      this.emitEvent({
        type: 'memory',
        timestamp: this.currentTime,
        data: { kind: 'association', modalities: Object.keys(event), bindings: this.associations.bindings },
      });
    }
  }

  /** V1's units and the object level's (offset in the association's visual code), apart. */
  private static splitVisualUnits(units: number[]): { v1: number[]; objects: number[] } {
    const v1: number[] = [];
    const objects: number[] = [];
    for (const u of units) {
      if (u >= DigitalBrain.OBJECT_UNIT_OFFSET) objects.push(u - DigitalBrain.OBJECT_UNIT_OFFSET);
      else v1.push(u);
    }
    return { v1, objects };
  }

  /** The `k` strongest channels of a reinstated pattern. */
  private static topUnits(pattern: Map<number, number>, k: number): number[] {
    return [...pattern].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, k).map(([unit]) => unit);
  }

  /** Reads a reinstated lexical pattern out as the words it really spells. */
  private readWords(reinstated: Map<number, number>): { pattern: Float32Array; words: Array<{ word: string; similarity: number }> } {
    const pattern = new Float32Array(this.lexicon.dimensions);
    for (const [channel, value] of reinstated) if (channel < pattern.length) pattern[channel] = value;
    const contained = this.lexicon
      .findContained(pattern, 10)
      // Only words the reinstated pattern really spells out. While a word is
      // still unknown to the lexicon, its pattern merely resembles a few
      // known words (~0.35): better to stay silent than to say those.
      .filter((m) => m.similarity >= DigitalBrain.RECALLED_WORD_MATCH);
    // A word that is a piece of a better-matching word ("ver" in "verde",
    // "o" in "coche") is that word's shadow in the pattern, not a word recalled.
    const words = contained
      .filter((m) => m.word.length >= 3)
      .filter((m, i) => !contained.some((other, j) => j !== i && other.word !== m.word && other.word.includes(m.word)))
      .slice(0, 6)
      .map((m) => ({ word: m.word, similarity: m.similarity }));
    return { pattern, words };
  }

  /** Cross-modal recall from a cue, decoded into words and perceptual categories. */
  private recallFrom(modality: AssociationModality, code: ModalCode, label: string): void {
    const result = this.associations.recall(modality, code);
    // A faint match is a chance overlap between codes (words share letters),
    // not a memory: a single real pairing already scores above this.
    if (!result || result.match < DigitalBrain.MIN_RECALL_MATCH) return;

    const confident = result.match >= DigitalBrain.RECALL_CONFIDENCE;
    // Getting better at recalling this is learning progress: rewarding in itself.
    this.reward(this.motivation.progress(`${modality}:${label}`, result.match, this.tickCount));
    const recall: AssociationRecall = {
      cue: { modality, label },
      confidence: result.match,
      confident,
      words: [],
      visual: null,
      colour: null,
      auditory: null,
      timestamp: this.currentTime,
    };

    const lexical = result.recalled.lexical;
    let lexicalPattern: Float32Array | null = null;
    if (lexical) {
      const read = this.readWords(lexical.pattern);
      lexicalPattern = read.pattern;
      recall.words = read.words;
    }

    const topUnits = DigitalBrain.topUnits;

    const visual = this.regions.get('visualCortex') as VisualCortex | undefined;
    const parts = this.regions.get('partsCortex') as PartsCortex | undefined;
    if (result.recalled.visual && visual) {
      const { v1, objects } = DigitalBrain.splitVisualUnits(topUnits(result.recalled.visual.pattern, 28));
      const match = v1.length > 0 ? visual.matchCategory(v1) : null;
      const objectMatch = parts && objects.length > 0 ? parts.matchCategory(objects) : null;
      if (match && match.overlap >= 0.3) {
        recall.visual = { label: match.label, overlap: match.overlap };
        // What comes to the mind's eye is drawn — if the hand is on and knows
        // how. (Something SEEN is already handled by copying; this is for what
        // it reads or hears.)
        if (confident && modality !== 'visual') this.drawImaginedImage(visual.imagine(v1));
      } else if (objectMatch && objectMatch.overlap >= 0.3) {
        // V1 has no whole-image match, but the object level does: the parts complete it.
        recall.visual = { label: objectMatch.label, overlap: objectMatch.overlap };
        if (confident && modality !== 'visual' && parts) this.drawImaginedImage(parts.imagine(objects));
      }
    }
    const colourCortex = this.regions.get('colorCortex') as ColorCortex | undefined;
    if (result.recalled.colour && colourCortex) {
      const units = topUnits(result.recalled.colour.pattern, 6);
      const match = colourCortex.matchCategory(units);
      if (match && match.overlap >= 0.3) recall.colour = { label: match.label, overlap: match.overlap };
    }
    const auditory = this.regions.get('auditoryCortex') as AuditoryCortex | undefined;
    if (result.recalled.auditory && auditory) {
      // Sound engrams are tiny (3 neurons) and drift a little between
      // hearings, so the reinstated units are matched leniently to name the
      // category; saying the sound does not depend on the name at all — the
      // motor map is driven by the units themselves.
      const units = topUnits(result.recalled.auditory.pattern, 3);
      const match = auditory.matchCategory(units);
      if (match && match.overlap >= 0.2) recall.auditory = { label: match.label, overlap: match.overlap };
      // The sound that comes to mind is said aloud — if the voice is on and
      // the motor map knows how to make it. (A heard sound is already
      // handled by imitation; this is for what it SEES or READS.)
      if (confident && modality !== 'auditory') this.sayImaginedSound(units);
    }

    this.lastRecallUnits = result.units;
    // Recorded even when nothing could be NAMED yet (e.g. the word that comes
    // back is not in the lexicon yet): something does come to mind.
    this.lastRecall = recall;
    this.lastRecallTick = this.tickCount;
    this.recallsByModality.set(modality, { recall, tick: this.tickCount });
    // Goal-directed: it writes the word that came to mind — if that is worth
    // doing (a cue that has come to predict reprimand is left unanswered:
    // outcome devaluation) and no habit answers this cue already (a habit,
    // once stamped in, takes over; see `onPercept`).
    const cueKey = `${modality}:${label}`;
    if (
      confident && lexicalPattern && recall.words.length > 0 &&
      this.habits.habitFor(cueKey) === null && this.motivation.expectation(cueKey) >= DigitalBrain.DEVALUED
    ) {
      this.recalledLexicalPattern = lexicalPattern;
      this.ticksSinceRecall = 0;
      const word = recall.words[0].word;
      // Every execution stamps the stimulus–response link in a little.
      this.habits.practice(cueKey, word);
      // It WRITES the word that came to mind (the dashboard has a text area for it).
      this.emitEvent({
        type: 'response',
        timestamp: this.currentTime,
        data: { kind: 'writing', text: word, cue: label, confidence: result.match, habit: false },
      });
    }
  }

  /**
   * A habit answers the cue: the response runs off the stimulus itself,
   * whatever the memory behind it now says and whatever the outcome is now
   * worth (Dickinson; Yin & Knowlton). It is still judged — praise stamps it
   * in, reprimand wears it down, slowly.
   */
  private respondByHabit(habit: Habit, label: string): void {
    this.habits.practice(habit.cue, habit.response);
    this.recalledLexicalPattern = wordToPattern(habit.response, this.lexicon.dimensions);
    this.ticksSinceRecall = 0;
    this.lastRecallTick = this.tickCount;
    this.lastHabit = { cue: habit.cue, response: habit.response, timestamp: this.currentTime };
    this.emitEvent({
      type: 'response',
      timestamp: this.currentTime,
      data: { kind: 'writing', text: habit.response, cue: label, confidence: habit.strength, habit: true },
    });
  }

  /**
   * Feedback from the teacher on what the brain has just recalled (named,
   * said or drawn): "yes, that's it" / "no, that's not it".
   *
   * Biology: reward is a dopamine burst, its opposite a dip plus stress; the
   * synapses that produced the judged response are still eligible and are
   * strengthened or weakened accordingly (three-factor learning; Schultz, 1998).
   * The neuromodulators released also colour whatever is learned next.
   *
   * @returns Whether there was a recent recall to apply the feedback to
   */
  giveFeedback(positive: boolean): boolean {
    if (positive) {
      this.modulators.release(ModulatorType.Serotonin, 0.05);
    } else {
      this.modulators.release(ModulatorType.Cortisol, 0.12);
      this.modulators.release(ModulatorType.Norepinephrine, 0.08);
    }
    // The dopamine of a 👍 is a prediction error too: expected praise moves nothing.
    this.reward(this.motivation.external(positive ? 0.6 : -0.6, this.tickCount));
    this.markInnateAffect(positive ? { valence: 0.6, arousal: 0.5 } : { valence: -0.6, arousal: 0.6 });
    // The verdict is on the last response, whenever it was (the button, unlike a voice, is explicit).
    this.habits.credit(positive ? 0.6 : -0.6);
    if (this.lastRecallUnits.length === 0 || this.lastRecall === null) return false;
    this.associations.reinforce(this.lastRecallUnits, positive ? 0.6 : -0.8);
    return true;
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
    const colour = this.regions.get('colorCortex') as ColorCortex | undefined;
    const parts = this.regions.get('partsCortex') as PartsCortex | undefined;
    return {
      visual: visual?.getRecognition() ?? null,
      auditory: auditory?.getRecognition() ?? null,
      colour: colour?.getRecognition() ?? null,
      visualCategories: visual?.categoryCount ?? 0,
      object: parts?.getRecognition() ?? null,
      auditoryCategories: auditory?.categoryCount ?? 0,
      colourCategories: colour?.categoryCount ?? 0,
      objectCategories: parts?.categoryCount ?? 0,
      partsKnown: parts?.partsKnown ?? 0,
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
    // language areas (exponential decay over ~5 s of real time).
    const traceWeight = 6.0 * Math.exp(-this.ticksSinceRead / this.thoughtTraceTicks);
    if (traceWeight > 0.01) addInto(this.lastLinguisticIntention ?? undefined, traceWeight);
    // …and of what a percept just brought back from memory: seeing the ball
    // brings the word "ball" to mind.
    const recallWeight = 6.0 * Math.exp(-this.ticksSinceRecall / this.thoughtTraceTicks);
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
    this.presentations.set(type, { signal, ticksLeft: this.presentationTicks });
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
        targets: [...DigitalBrain.THALAMIC_RELAY[type]],
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
      for (let i = 0; i < this.perceptionTicks; i++) {
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
    this.moveEye();
    this.speakOn();

    //    The hippocampus stamps the episodes it encodes with the current affect
    //    (emotional episodes are forgotten more slowly).
    (this.regions.get('hippocampus') as Hippocampus | undefined)?.setAffectiveContext(this.feel().valence);

    // 2. Process each region
    let activeNeurons = 0;
    for (const [regionId, region] of this.regions) {
      const activity = region.step(dt, effects);
      activeNeurons += activity.activeNeurons.length;
      this.drivePeaks.set(
        regionId,
        Math.max(activity.drive, (this.drivePeaks.get(regionId) ?? 0) * this.drivePeakDecay),
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

    //    Voice: an imitation the motor cortex has just decided on.
    this.driveVoice();
    //    Hand: a copy the hand motor cortex has just decided on.
    this.driveHand();
    //    Otherwise: whatever it feels like doing (motivation).
    this.driveExploration();
    //    Expected reward that did not come: a dopamine dip.
    this.reward(this.motivation.resolveOmission(this.tickCount));
    //    A motor map that has just learned: is the activity still teaching it something?
    this.creditActivityProgress();

    //    Percepts completed by the sensory cortices this tick → association.
    this.collectPercepts();
    if (this.ticksSinceRecall < Number.MAX_SAFE_INTEGER) this.ticksSinceRecall++;
    if (this.ticksSincePercept < Number.MAX_SAFE_INTEGER) this.ticksSincePercept++;

    // 4. Dispatch bus packets (delivery deferred by axonal delays)
    this.bus.tick(this.currentTime);

    // 5. Neuromodulator decay
    this.modulators.decay(dt);

    //    A conditioned cue was perceived and nothing followed it: extinction.
    if (this.pendingExtinction && this.tickCount >= this.pendingExtinction.untilTick) {
      (this.regions.get('amygdala') as Amygdala | undefined)?.extinguishCue(this.pendingExtinction.modality, this.pendingExtinction.units);
      this.pendingExtinction = null;
    }

    // 6. Sleep pressure (adenosine): time awake and neural activity add up;
    //    sleep clears it. Only at rest: sleeping in the middle of a perception
    //    would replay over live activity and cut the wave short.
    this.sleepPressure +=
      dt / this.config.memory.consolidationIntervalMs +
      (activeNeurons / this.totalNeurons) * (dt / DigitalBrain.ACTIVITY_PRESSURE_MS);
    const atRest = this.presentations.size === 0 && activeNeurons === 0 && this.bus.pendingCount === 0;
    // Exhaustion: kept awake long past the point, it falls asleep anyway.
    if (this.sleepPressure >= 1 && (atRest || this.sleepPressure >= DigitalBrain.EXHAUSTION_PRESSURE)) {
      this.sleep();
    }
  }

  // ================================================================
  // THE INNATE LAYER — what the brain brings to the world unlearned
  // ================================================================

  /**
   * The brain hears a VOICE: not its words (those go through the cochlea and
   * the auditory cortex like any sound) but its tone — the envelope and pitch
   * track of an utterance, read by innate detectors on the fast thalamus →
   * amygdala route. Warm, high, smooth and unhurried comforts; loud, low,
   * abrupt and rough alarms (Fernald 1993; Arnal 2015). A sudden loud onset
   * startles.
   *
   * The tone is also how a person tells the brain whether it did well: a warm
   * voice right after a recall strengthens what it recalled, a harsh one
   * weakens it (social referencing) — the innate form of the 👍 / 👎 buttons.
   * And whatever was perceived or read around the voice is conditioned to
   * the emotion it evoked: that is how a word, or a thing, comes to mean
   * something.
   */
  hearVoice(contour: VoiceContour): ProsodyAppraisal | null {
    const appraisal = appraiseProsody(contour, this.speakerPitchHz);
    if (!appraisal) return null;
    const { valence, arousal, features } = appraisal;

    // The detectors read pitch against the speaker's usual pitch: adapt to it slowly.
    if (features.voiced >= 0.3) {
      const meanHz = this.speakerPitchHz * Math.pow(2, features.pitchHeight / 12);
      this.speakerPitchHz += (Math.max(60, Math.min(500, meanHz)) - this.speakerPitchHz) * 0.1;
    }

    const amygdala = this.regions.get('amygdala') as Amygdala | undefined;
    const evoked = { valence, arousal };
    if (amygdala) {
      amygdala.appraiseInnate(evoked);
      this.releaseFromAmygdala(amygdala);
    }
    // A voice is contact: oxytocin, more so for a warm one.
    this.modulators.release(ModulatorType.Oxytocin, 0.03 + 0.05 * Math.max(0, valence));
    this.motivation.heardVoice(this.tickCount);
    // Its warmth or harshness is a reward or a punishment — for what was just
    // perceived, and against what that thing had led to expect.
    if (Math.abs(valence) >= 0.25) this.reward(this.motivation.external(valence, this.tickCount));
    if (appraisal.startle) this.startle('voice');

    // The voice as a verdict on what was just recalled.
    let judged = false;
    if (Math.abs(valence) >= 0.3 && this.tickCount - this.lastRecallTick <= this.verdictWindowTicks) this.habits.credit(valence);
    if (Math.abs(valence) >= 0.3 && this.tickCount - this.lastRecallTick <= this.verdictWindowTicks && this.lastRecallUnits.length > 0 && this.lastRecall) {
      this.associations.reinforce(this.lastRecallUnits, valence > 0 ? 0.6 * valence : -0.8 * -valence);
      judged = true;
    }

    // Whatever came with the voice takes on its emotion.
    this.markInnateAffect(evoked);
    if (amygdala && this.ticksSinceRead <= this.conditioningWindowTicks) this.conditionWords(amygdala, this.lastReadWords);

    this.lastVoice = {
      valence,
      arousal,
      kind: valence >= 0.25 ? 'warm' : valence <= -0.25 ? 'harsh' : 'neutral',
      startle: appraisal.startle,
      judged,
      timestamp: this.currentTime,
      serial: ++this.voicesHeard,
    };
    this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'voice', voice: this.lastVoice, features } });
    return appraisal;
  }

  /**
   * The acoustic startle reflex: a sudden loud sound fires alarm (norepinephrine,
   * cortisol, orienting acetylcholine) and freezes ongoing action, before any
   * cortex has classified the sound.
   */
  private startle(source: string): void {
    this.startles++;
    this.modulators.release(ModulatorType.Norepinephrine, 0.2);
    this.modulators.release(ModulatorType.Cortisol, 0.1);
    this.modulators.release(ModulatorType.Acetylcholine, 0.08);
    const evoked = { valence: -0.4, arousal: 0.9 };
    (this.regions.get('amygdala') as Amygdala | undefined)?.appraiseInnate(evoked);
    // Freezing: spontaneous activity waits a full pause again.
    this.ticksSinceExploration = 0;
    this.markInnateAffect(evoked);
    this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'startle', source, startles: this.startles } });
  }

  /**
   * An innate emotional event has just happened: it is the unconditioned
   * stimulus for whatever was perceived around it (a cue seen or heard, a word
   * read). Aversive, arousing events condition in one pairing (fear learning is
   * fast); pleasant ones need a few.
   */
  private markInnateAffect(evoked: { valence: number; arousal: number }): void {
    this.innateAffect = { valence: evoked.valence, arousal: evoked.arousal, tick: this.tickCount };
    if (Math.abs(evoked.valence) < 0.3 && evoked.arousal < 0.6) return;
    const amygdala = this.regions.get('amygdala') as Amygdala | undefined;
    if (!amygdala) return;
    const strength = evoked.valence < 0 && evoked.arousal >= 0.5 ? 1.0 : 0.6;
    for (const [modality, cue] of this.recentCues) {
      if (this.tickCount - cue.tick > this.conditioningWindowTicks) continue;
      amygdala.conditionCue(modality, cue.units, evoked, strength);
      // The event followed the cue: no extinction this time.
      if (this.pendingExtinction?.modality === modality) this.pendingExtinction = null;
    }
  }

  /** Conditions `words` to the innate emotional event still hot, if any. */
  private conditionWords(amygdala: Amygdala, words: string[]): void {
    const hot = this.innateAffect;
    if (!hot || this.tickCount - hot.tick > this.conditioningWindowTicks) return;
    if (Math.abs(hot.valence) < 0.3 && hot.arousal < 0.6) return;
    const strength = hot.valence < 0 ? 1.0 : 0.6;
    for (const word of words) {
      amygdala.conditionSemantic(wordToPattern(word, this.lexicon.dimensions), { valence: hot.valence, arousal: hot.arousal }, strength);
    }
  }

  /**
   * A category has just been perceived: if it was conditioned, its emotion
   * comes back (fear on seeing the figure that was there when the loud noise
   * came). Stress restores an extinguished memory. If nothing follows the cue,
   * extinction weakens it a little.
   */
  private appraiseCue(modality: 'visual' | 'auditory', units: number[], label: string): void {
    this.recentCues.set(modality, { units: units.slice(), label, tick: this.tickCount });
    const amygdala = this.regions.get('amygdala') as Amygdala | undefined;
    if (!amygdala) return;
    const stress = Math.max(0, (this.modulators.getLevel(ModulatorType.Cortisol) - 0.3) / 0.7);
    const evoked = amygdala.appraiseCue(modality, units, stress);
    if (!evoked) return;
    this.releaseFromAmygdala(amygdala);
    this.lastCueResponse = { modality, label, valence: evoked.valence, arousal: evoked.arousal, timestamp: this.currentTime };
    this.pendingExtinction = { modality, units: units.slice(), untilTick: this.tickCount + this.conditioningWindowTicks };
    this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'conditioned', ...this.lastCueResponse } });
  }

  /**
   * Innate face detector (Johnson & Morton's CONSPEC): two blobs above one,
   * matched within the bounding box of what is drawn, so position and size do
   * not matter. The features themselves span the box: eyes in its top corners,
   * mouth along its bottom middle. (A head outline around them is not covered
   * yet; that is a job for the visual hierarchy of block 3.) A face draws
   * attention (acetylcholine) and comfort (oxytocin).
   */
  private detectFace(rates: Float32Array): void {
    const side = DigitalBrain.RETINA_SIDE;
    const cells = side * side;
    let minR = side, maxR = -1, minC = side, maxC = -1;
    for (let i = 0; i < cells; i++) {
      if (rates[i] < 0.5) continue;
      const r = Math.floor(i / side), c = i % side;
      if (r < minR) minR = r; if (r > maxR) maxR = r; if (c < minC) minC = c; if (c > maxC) maxC = c;
    }
    this.faceMatch = 0;
    if (maxR - minR < 3 || maxC - minC < 3) return;
    const h = maxR - minR + 1, w = maxC - minC + 1;
    // A face is blobs on a ground, not a filled patch: a solid shape correlates
    // with any template through its soft edges and is not a face.
    let inked = 0;
    for (let r = minR; r <= maxR; r++) for (let c = minC; c <= maxC; c++) if (rates[r * side + c] >= 0.5) inked++;
    if (inked > 0.5 * h * w) return;
    const blob = (r: number, c: number, r0: number, c0: number, sr: number, sc: number): number =>
      Math.exp(-(((r - r0) / sr) ** 2 + ((c - c0) / sc) ** 2) / 2);
    let sx = 0, st = 0, sxx = 0, stt = 0, sxt = 0;
    const n = h * w;
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const t = Math.max(
          blob(r, c, 0, 0.1 * (w - 1), 0.15 * h + 0.5, 0.15 * w + 0.5),
          blob(r, c, 0, 0.9 * (w - 1), 0.15 * h + 0.5, 0.15 * w + 0.5),
          blob(r, c, h - 1, 0.5 * (w - 1), 0.12 * h + 0.5, 0.3 * w + 0.5),
        );
        const x = rates[(minR + r) * side + minC + c];
        sx += x; st += t; sxx += x * x; stt += t * t; sxt += x * t;
      }
    }
    const cov = sxt - (sx * st) / n;
    const vx = sxx - (sx * sx) / n, vt = stt - (st * st) / n;
    const corr = vx > 0 && vt > 0 ? cov / Math.sqrt(vx * vt) : 0;
    this.faceMatch = Math.max(0, corr);
    if (corr < DigitalBrain.FACE_MATCH) return;
    this.facesSeen++;
    this.modulators.release(ModulatorType.Acetylcholine, 0.06);
    this.modulators.release(ModulatorType.Oxytocin, 0.04);
    (this.regions.get('amygdala') as Amygdala | undefined)?.appraiseInnate({ valence: 0.2, arousal: 0.4 });
    this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'face', match: this.faceMatch, facesSeen: this.facesSeen } });
  }

  /**
   * Innate looming detector: an image that covers much more of the retina than
   * the previous one, moments ago, is something approaching fast. Alarm.
   */
  private detectLooming(rates: Float32Array): void {
    const cells = DigitalBrain.RETINA_SIDE * DigitalBrain.RETINA_SIDE;
    let lit = 0;
    for (let i = 0; i < cells; i++) if (rates[i] >= 0.5) lit++;
    const coverage = lit / cells;
    const previous = this.lastRetina;
    this.lastRetina = { coverage, tick: this.tickCount };
    if (!previous || this.tickCount - previous.tick > this.loomingWindowTicks) return;
    if (previous.coverage < 0.02 || coverage < 0.2 || coverage < 2 * previous.coverage) return;
    this.loomings++;
    this.modulators.release(ModulatorType.Norepinephrine, 0.15);
    this.modulators.release(ModulatorType.Cortisol, 0.08);
    const evoked = { valence: -0.3, arousal: 0.8 };
    (this.regions.get('amygdala') as Amygdala | undefined)?.appraiseInnate(evoked);
    this.markInnateAffect(evoked);
    this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'looming', coverage, loomings: this.loomings } });
  }

  // ================================================================
  // MOTIVATION — dopamine as prediction error, curiosity, drives
  // ================================================================

  /**
   * Turns a reward prediction error into dopamine: a burst for more than
   * expected, a dip for less (Schultz). What was expected releases nothing.
   */
  private reward(event: RewardEvent | null): void {
    if (!event) return;
    if (event.kind === 'external' && event.key) this.gateIntoWorkingMemory(event.key.replace(/^[a-z]+:/, ''), event.error);
    if (event.error > 0) this.modulators.release(ModulatorType.Dopamine, DigitalBrain.DOPAMINE_BURST_GAIN * event.error);
    else if (event.error < 0) this.modulators.release(ModulatorType.Dopamine, DigitalBrain.DOPAMINE_DIP_GAIN * event.error);
    if (Math.abs(event.error) >= 0.05) {
      this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'dopamine', event } });
    }
  }

  /**
   * Dopamine gating of working memory (O'Reilly & Frank, 2006): what enters
   * and stays in mind is what was surprising or rewarding — the size of the
   * prediction error opens the gate and sets the slot's priority. The routine
   * and the expected pass through without being held. (The top-down mask the
   * prefrontal cortex builds from its slots awaits block 3's feature-based
   * attention; until then a slot is a label held with a priority.)
   */
  private gateIntoWorkingMemory(label: string, error: number): void {
    const salience = Math.abs(error);
    if (salience < DigitalBrain.WORKING_MEMORY_GATE) return;
    const pfc = this.regions.get('prefrontalCortex') as PrefrontalCortex | undefined;
    pfc?.attendTo(new Float32Array(0), label, Math.min(1, salience));
  }

  /** The motor maps learned from their last activity: credit its progress to that activity. */
  private creditActivityProgress(): void {
    const motor = this.regions.get('motorCortex') as MotorCortex | undefined;
    if (motor && motor.learnings !== this.motorLearnings.babble) {
      this.motorLearnings.babble = motor.learnings;
      this.reward(this.motivation.activityLearned('babble', motor.knowledge, this.tickCount));
    }
    const hand = this.regions.get('handMotorCortex') as HandMotorCortex | undefined;
    if (hand && hand.learnings !== this.motorLearnings.scribble) {
      this.motorLearnings.scribble = hand.learnings;
      this.reward(this.motivation.activityLearned('scribble', hand.knowledge, this.tickCount));
    }
  }

  /** The motivation system's state, for the dashboard and tests. */
  getMotivation(): MotivationState {
    return {
      drives: this.motivation.drives(this.tickCount),
      activityValues: { ...this.motivation.activityValues },
      lastEvent: this.motivation.lastEvent,
      recentEvents: [...this.motivation.recentEvents],
      events: this.motivation.events,
      chosen: { ...this.chosen },
      calls: this.calls,
    };
  }

  /** What a cue leads the brain to expect, and how many times it has seen it (for tests). */
  expectationOf(key: string): { expected: number; seen: number } {
    return { expected: this.motivation.expectation(key), seen: this.motivation.seen(key) };
  }

  // ================================================================
  // QUESTIONS — a question is a signal for a dimension of what is in front
  // ================================================================

  /**
   * A text has just been read. If the text before it (a question) is still
   * fresh and this one is a word bound to some dimension of things (a colour,
   * a shape, a sound), the question is learned to ask for that dimension.
   * And if this text is a question already learned, it is answered with what
   * that dimension of the thing in front has just brought to mind.
   */
  private considerQuestion(key: string, code: ModalCode): void {
    const questionWindow = this.ticksFor(DigitalBrain.QUESTION_WINDOW_MS);
    if (this.pendingQuestion && this.pendingQuestion.key !== key && this.tickCount - this.pendingQuestion.tick <= questionWindow) {
      const modality = this.modalityOfAnswer(key.slice('lexical:'.length)) ?? this.modalityOfWord(code);
      if (modality) {
        this.questions.observe(this.pendingQuestion.key, modality);
        this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'question-learned', question: this.pendingQuestion.key, modality, known: this.questions.size } });
      }
    }
    this.pendingQuestion = { key, tick: this.tickCount };
    const target = this.questions.refersTo(key);
    if (target) this.answer(key, target.modality as AssociationModality);
  }

  private restoreWordReferents(data: unknown): void {
    if (!Array.isArray(data)) return;
    this.wordReferents = new Map();
    for (const entry of data.slice(0, DigitalBrain.MAX_WORD_REFERENTS)) {
      if (!Array.isArray(entry) || entry.length !== 2 || typeof entry[0] !== 'string' || !Array.isArray(entry[1])) continue;
      const byReferent = new Map<string, number>();
      for (const pair of entry[1].slice(0, 500)) {
        if (!Array.isArray(pair) || typeof pair[0] !== 'string' || !Number.isInteger(pair[1]) || pair[1] <= 0) continue;
        byReferent.set(String(pair[0]).slice(0, 80), Math.min(1e6, pair[1] as number));
      }
      if (byReferent.size > 0) this.wordReferents.set(entry[0].slice(0, 80), byReferent);
    }
  }

  /** Each word of the text read goes on record with every category in view right now. */
  private noteWordReferents(text: string): void {
    const words = text.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^\w\s]/g, '').split(/\s+/).filter((w) => w.length >= 2);
    if (words.length === 0) return;
    const referents: string[] = [];
    for (const [modality, recent] of this.recentPercepts) {
      if (modality === 'lexical' || this.tickCount - recent.tick > this.associationWindowTicks) continue;
      referents.push(`${modality}:${recent.label}`);
    }
    if (referents.length === 0) return;
    for (const word of new Set(words.slice(0, DigitalBrain.MAX_WORDS_PER_READ))) {
      let byReferent = this.wordReferents.get(word);
      if (!byReferent) {
        if (this.wordReferents.size >= DigitalBrain.MAX_WORD_REFERENTS) break;
        byReferent = new Map();
        this.wordReferents.set(word, byReferent);
      }
      for (const referent of referents) byReferent.set(referent, (byReferent.get(referent) ?? 0) + 1);
    }
  }

  /**
   * The name of a category: the word that has gone with it most specifically —
   * often enough, and mostly with it rather than with everything.
   */
  private nameOf(referent: string, exclude: (word: string) => boolean): { word: string; specificity: number } | null {
    const modality = referent.slice(0, referent.indexOf(':'));
    let best: { word: string; specificity: number; count: number } | null = null;
    for (const [word, byReferent] of this.wordReferents) {
      const count = byReferent.get(referent) ?? 0;
      if (count < DigitalBrain.REFERENT_MIN_COUNT || exclude(word)) continue;
      let total = 0;
      for (const [other, n] of byReferent) if (other.startsWith(`${modality}:`)) total += n;
      const specificity = count / Math.max(1, total);
      if (specificity < DigitalBrain.REFERENT_MIN_SPECIFICITY) continue;
      if (!best || specificity > best.specificity || (specificity === best.specificity && count > best.count)) best = { word, specificity, count };
    }
    return best ? { word: best.word, specificity: best.specificity } : null;
  }

  /**
   * The dimension an answer belongs to, from the word–referent record: the
   * modality in which its words are most specific (a colour word goes with
   * one colour and many shapes). A word equally specific to two dimensions
   * says nothing yet.
   */
  private modalityOfAnswer(text: string): AssociationModality | null {
    const words = text.toLowerCase().split(/\s+/).filter((w) => w.length >= 2);
    const votes = new Map<string, number>();
    for (const word of words) {
      const byReferent = this.wordReferents.get(word);
      if (!byReferent) continue;
      const perModality = new Map<string, { best: number; total: number }>();
      for (const [referent, count] of byReferent) {
        const modality = referent.slice(0, referent.indexOf(':'));
        const stat = perModality.get(modality) ?? { best: 0, total: 0 };
        stat.total += count;
        if (count > stat.best) stat.best = count;
        perModality.set(modality, stat);
      }
      let top: [string, number] | null = null;
      let tie = false;
      for (const [modality, stat] of perModality) {
        if (stat.best < DigitalBrain.REFERENT_MIN_COUNT) continue;
        const specificity = stat.best / stat.total;
        if (!top || specificity > top[1]) { top = [modality, specificity]; tie = false; }
        else if (specificity === top[1]) tie = true;
      }
      if (top && !tie && top[1] >= DigitalBrain.REFERENT_MIN_SPECIFICITY) votes.set(top[0], (votes.get(top[0]) ?? 0) + 1);
    }
    let best: [string, number] | null = null;
    for (const entry of votes) if (!best || entry[1] > best[1]) best = entry;
    return best ? (best[0] as AssociationModality) : null;
  }

  /** The dimension a word belongs to: the modality its association memory binds it to most. */
  private modalityOfWord(code: ModalCode): AssociationModality | null {
    const result = this.associations.recall('lexical', code);
    if (!result || result.match < DigitalBrain.MIN_RECALL_MATCH) return null;
    // Compared by the strength of what comes back, not by how many channels a
    // modality has (a colour code is 6 units, a shape 20).
    let best: AssociationModality | null = null;
    let bestStrength = 0;
    for (const [modality, recalled] of Object.entries(result.recalled)) {
      if (modality === 'lexical') continue;
      const top = [...recalled.pattern.values()].sort((x, y) => y - x).slice(0, 6);
      const strength = top.reduce((sum, v) => sum + v, 0) / Math.max(1, top.length);
      if (strength > bestStrength) {
        bestStrength = strength;
        best = modality as AssociationModality;
      }
    }
    return best;
  }

  /**
   * Answers a question: attention turns to the dimension it asks for, and
   * what that dimension of the thing in front brought to mind — its colour
   * word, its name — is written and said.
   */
  private answer(question: string, modality: AssociationModality): void {
    // The thing in front, in that dimension: the category last perceived there.
    const region = { visual: 'visualCortex', colour: 'colorCortex', auditory: 'auditoryCortex', lexical: '' }[modality];
    const cortex = region ? (this.regions.get(region) as { getRecognition(): Recognition | null } | undefined) : undefined;
    const seen = cortex?.getRecognition() ?? null;
    if (!seen || (this.currentTime - seen.timestamp) / this.config.snn.dt > this.ticksFor(DigitalBrain.ANSWER_WINDOW_MS)) return;
    // Its name: the word that has gone with this colour (this shape, this
    // sound) and not with the others. Not a word of the question itself,
    // which goes with everything it is asked about. Failing a record, what the
    // association memory brought back for it.
    const held = this.recallsByModality.get(modality);
    const name = this.nameOf(`${modality}:${seen.label}`, (w) => question.includes(w));
    const word = name?.word ?? held?.recall.words.map((w) => w.word).find((w) => !question.includes(w));
    if (!word) return;
    this.lastAnswer = { question, modality, word, timestamp: this.currentTime };
    this.recalledLexicalPattern = wordToPattern(word, this.lexicon.dimensions);
    this.ticksSinceRecall = 0;
    this.emitEvent({ type: 'response', timestamp: this.currentTime, data: { kind: 'writing', text: word, cue: `answer:${modality}`, confidence: held?.recall.confidence ?? name?.specificity ?? 0 } });
    this.emitEvent({ type: 'affect', timestamp: this.currentTime, data: { kind: 'answer', question, modality, word } });
  }

  /** The innate layer's state, for the dashboard and tests. */
  getInnate(): InnateState {
    const amygdala = this.regions.get('amygdala') as Amygdala | undefined;
    return {
      lastVoice: this.lastVoice,
      voicesHeard: this.voicesHeard,
      startles: this.startles,
      loomings: this.loomings,
      faceMatch: this.faceMatch,
      facesSeen: this.facesSeen,
      lastCueResponse: this.lastCueResponse,
      conditionedCues: amygdala?.conditionedCueCount ?? 0,
      affectiveWords: amygdala?.semanticAssociationCount ?? 0,
      sleepPressure: Math.min(1, this.sleepPressure),
    };
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
    this.sleepPressure = 0;
    this.motivation.sleep();

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

      // REM. Acetylcholine high, norepinephrine and serotonin low, no
      // prefrontal control (Hobson & Pace-Schott 2002): CA3 completes
      // chimeras of two episodes and the cortex is shown them with plasticity
      // low — generative replay (van de Ven et al. 2020) — and what it knows
      // comes back recombined as dreams, and as variants of itself.
      const dreamEffects: ModulationEffects = {
        learningRateMultiplier: DigitalBrain.DREAM_PLASTICITY,
        thresholdMultiplier: 0.8,
        attentionGain: 0.2,
        consolidationRate: 0.5,
        spikeGainMultiplier: 0.7,
        socialWeightBoost: 1.0,
      };
      this.modulators.release(ModulatorType.Acetylcholine, 0.1);
      let chimeras = 0;
      for (let d = 0; d < DigitalBrain.DREAMS_PER_SLEEP; d++) {
        const chimera = hippocampus.dream(this.imaginationRandom);
        if (!chimera || chimera.code.length !== cortex.inputs) break;
        cortex.reactivate(chimera.code, dreamEffects);
        chimeras++;
      }
      const dreamed: string[] = [];
      for (let d = 0; d < DigitalBrain.DREAMS_PER_SLEEP; d++) {
        const dream = this.imagineOnce('dream');
        if (!dream) break;
        dreamed.push([...dream.words, dream.visual, dream.colour, dream.auditory].filter((x): x is string => x !== null).join(' '));
      }
      const variants = this.replayVariants(dreamEffects);
      stats.dreams = dreamed.length;
      stats.dreamed = dreamed;

      cortex.settle();
      hippocampus.forget(DigitalBrain.SLEEP_EPISODIC_DECAY);
      hippocampus.downscale(DigitalBrain.SLEEP_SYNAPTIC_DOWNSCALING);

      console.log(`   Memories replayed: ${stats.memoriesReplayed}`);
      console.log(`   Synapses strengthened: ${stats.synapsesStrengthened}`);
      console.log(`   REM: ${chimeras} chimeras in CA3, ${dreamed.length} dreams, ${variants} things replayed as variants`);
      for (const d of dreamed) console.log(`   🌙 ${d || '(wordless)'}`);
    }

    // Pruning: what was met once and never again is not worth its neurons.
    const prunedCategories: string[] = [];
    const motor = this.regions.get('motorCortex') as MotorCortex | undefined;
    for (const [id, modality] of [['visualCortex', 'visual'], ['colorCortex', 'colour'], ['auditoryCortex', 'auditory'], ['partsCortex', 'object']] as const) {
      const region = this.regions.get(id) as { pruneCategories(minExposures: number, afterSleeps: number, inUse: (unit: number) => boolean): string[] } | undefined;
      if (!region) continue;
      const inUse = (unit: number): boolean =>
        modality === 'object'
          ? this.associations.usesChannel('visual', unit + DigitalBrain.OBJECT_UNIT_OFFSET)
          : this.associations.usesChannel(modality, unit) || (modality === 'auditory' && motor !== undefined && motor.knowsHeard(unit));
      prunedCategories.push(...region.pruneCategories(DigitalBrain.PRUNE_MIN_EXPOSURES, DigitalBrain.PRUNE_AFTER_SLEEPS, inUse));
    }
    stats.prunedCategories = prunedCategories;
    if (prunedCategories.length > 0) console.log(`   Pruned: ${prunedCategories.join(', ')}`);

    // Emit event
    this.emitEvent({
      type: 'consolidation',
      timestamp: this.currentTime,
      data: {
        consolidated: stats.consolidatedLabels.length,
        memoriesReplayed: stats.memoriesReplayed,
        synapsesStrengthened: stats.synapsesStrengthened,
        dreams: stats.dreams ?? 0,
        dreamed: stats.dreamed ?? [],
        pruned: stats.prunedCategories ?? [],
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
      innate: this.getInnate(),
      motivation: this.getMotivation(),
      sequence: { expectation: this.sequences.expectation, transitions: this.sequences.size, lastTransition: this.lastTransition },
      workingMemory: ((this.regions.get('prefrontalCortex') as PrefrontalCortex | undefined)?.getWorkingMemory() ?? [])
        .map((slot) => ({ label: slot.label, priority: slot.priority, age: slot.age })),
      questions: { known: this.questions.size, lastAnswer: this.lastAnswer },
      imagination: this.getImagination(),
      habits: { count: this.habits.size, links: this.habits.list().slice(0, 6), lastHabit: this.lastHabit },
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
      hand: {
        scribbling: this.scribbling,
        copying: (this.regions.get('handMotorCortex') as HandMotorCortex | undefined)?.copy ?? false,
        scribbles: (this.regions.get('handMotorCortex') as HandMotorCortex | undefined)?.scribbleCount ?? 0,
        lastDrawing: this.lastDrawing,
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
        speakerPitchHz: this.speakerPitchHz,
      },
      association: this.associations.serialize(),
      motivation: this.motivation.serialize(),
      sequence: this.sequences.serialize(),
      questions: this.questions.serialize(),
      wordReferents: Array.from(this.wordReferents.entries()).map(([w, m]) => [w, Array.from(m.entries())]),
      imagination: { combinations: this.imagination.serialize(), daydreams: this.daydreams, dreams: this.dreams, foreseen: this.foreseen },
      habits: this.habits.serialize(),
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

  private restoreImagination(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as { combinations?: unknown; daydreams?: unknown; dreams?: unknown; foreseen?: unknown };
    this.imagination.deserialize(d.combinations);
    const count = (x: unknown): number => (typeof x === 'number' && Number.isInteger(x) && x >= 0 ? Math.min(x, 1e9) : 0);
    this.daydreams = count(d.daydreams);
    this.dreams = count(d.dreams);
    this.foreseen = count(d.foreseen);
  }

  /**
   * Restores the simulation clock and the pending vocabulary. The clock matters:
   * episodes are dated in simulated time, and recency (replay priority,
   * forgetting) is measured against it — a clock restarted at 0 would make
   * every restored episode look like it comes from the future.
   */
  private restoreBrainExtras(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as { time?: unknown; tickCount?: unknown; pendingVocab?: unknown; speakerPitchHz?: unknown };
    if (typeof d.speakerPitchHz === 'number' && Number.isFinite(d.speakerPitchHz)) {
      this.speakerPitchHz = Math.max(60, Math.min(500, d.speakerPitchHz));
    }

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
    this.motivation.deserialize(data.extras.motivation);
    this.sequences.deserialize(data.extras.sequence);
    this.questions.deserialize(data.extras.questions);
    this.restoreWordReferents(data.extras.wordReferents);
    this.restoreImagination(data.extras.imagination);
    this.habits.deserialize(data.extras.habits);

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
