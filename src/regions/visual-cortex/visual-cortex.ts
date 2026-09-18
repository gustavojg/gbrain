/**
 * VISUAL CORTEX — Visual pattern processing (real biological core)
 * ===========================================================================
 * Reference region for Path A: Izhikevich membrane dynamics executed
 * tick by tick, k-WTA lateral inhibition, refractory period,
 * background current, excitability homeostasis and STDP plasticity
 * based on pre/post traces.
 *
 * Biological basis:
 *   - V1 neurons (Hubel & Wiesel, 1962) respond to edges/orientations.
 *   - Sparse coding via GABAergic lateral inhibition (k-WTA).
 *   - STDP (Bi & Poo, 1998): a pre that fires BEFORE the post potentiates
 *     the synapse (LTP); a pre that fires AFTER depresses it (LTD).
 *   - Homeostasis (Turrigiano, 1998/2008): synaptic scaling + adjustment of
 *     intrinsic excitability to maintain a target firing rate.
 *
 * STDP implementation:
 *   The TRACE-based formulation (pair-based, nearest-neighbour) is used, more
 *   numerically stable than comparing raw lastSpikeTime values when pre and post
 *   fire in the same tick. Each input keeps a pre trace that decays
 *   with τ+, and each neuron a post trace that decays with τ-:
 *     - When the post fires:  Δw_i = +A+ · preTrace_i      (LTP)
 *     - When the pre fires :  Δw_i = -A- · postTrace_n     (LTD)
 */

import type { SpikePacket } from '../../core/bus/spike-bus.js';
import { SpikeBus } from '../../core/bus/spike-bus.js';
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { SpikingNeuron, createNeuronPopulation } from '../../core/snn/neuron.js';
import type { NeuronTypeName } from '../../core/snn/neuron.js';
import { rateCoding } from '../../core/snn/spike-train.js';
import { packArray, unpackFloat32, unpackInt32 } from '../../core/persistence/binary-protocol.js';
import { PERCEPTUAL_NARROWING, PresentationTracker, PrototypeMemory, RELEASE_KEEP, type CategoryView, type Recognition } from '../../core/memory/prototype-memory.js';

/**
 * Minimum engram overlap (Jaccard) for a stimulus to count as a re-encounter of
 * a known visual category. Two unrelated engrams of 20 among 2000 neurons
 * share ~0 units; a maturing engram of the SAME image keeps well over a third.
 */
const VISUAL_CATEGORY_MATCH = 0.35;

/** Minimum cosine match between an image and a tuned neuron's weights for the neuron to compete on equal terms. */
const VISUAL_VIGILANCE = 0.3;
/** Handicap (in cosine units) of a tuned neuron whose match is below the vigilance. */
const MISMATCH_PENALTY = 0.3;

/** Labelled memories restored from disk are capped (the list is unbounded in memory). */
const MAX_PERSISTED_MEMORIES = 2000;

// ====================================================================
// Visual Cortex types
// ====================================================================

/** Visual memory: pattern of active neurons associated with a label. */
export interface VisualMemory {
  pattern: Int32Array;
  label: string;
  strength: number;
  createdAt: number;
}

/** Result of visual processing. */
export interface VisualProcessingResult {
  winners: Int32Array;
  potentials: Float32Array;
  predictions: string[];
  activity: number;
}

/** Result of presenting a stimulus during a temporal window. */
export interface PresentationResult {
  /** Engram: indices of the neurons that fired most during the window. */
  engram: Int32Array;
  /** Spike count per neuron during the window. */
  spikeCounts: Int32Array;
  /** Total magnitude of the weight change in the window (Σ|Δw|). */
  weightChange: number;
  /** Mean fraction of cortical activity during the window. */
  activity: number;
}

/** Visual cortex configuration. */
export interface VisualCortexConfig {
  neuronCount: number;
  inputCount: number;
  /** Number of winning neurons in k-WTA (engram size). */
  kWinners: number;
  /** Homeostatic fatigue factor (penalty for accumulated wins). */
  fatigueFactor: number;
  /** Base learning rate (global scale of Δw). */
  learningRate: number;
  /** STDP potentiation amplitude (LTP). */
  aPlus: number;
  /** STDP depression amplitude (LTD). */
  aMinus: number;
  /** Time constant of the LTP trace (ms). */
  tauPlus: number;
  /** Time constant of the LTD trace (ms). */
  tauMinus: number;
  /** Maximum synaptic weight budget per neuron (normalization). */
  maxWeightBudget: number;
  /** Allowed weight range [min, max]. */
  weightRange: [number, number];
  /** Gain of the synaptic input current. */
  inputGain: number;
  /** Background current (keeps neurons close to threshold). */
  backgroundCurrent: number;
  /** Membrane current ceiling (avoids Izhikevich numerical instability). */
  maxCurrent: number;
  /** Amplitude of the background synaptic noise. */
  noiseAmplitude: number;
  /** Absolute refractory period (ms). */
  refractoryMs: number;
  /** Simulation ticks per stimulus presentation. */
  presentationTicks: number;
  /** Target firing rate per neuron (fraction, for homeostasis). */
  targetRate: number;
  /** Speed of intrinsic excitability homeostasis. */
  homeostasisRate: number;
  /** Overlap threshold for recognition (out of k neurons). */
  recognitionThreshold: number;
  /** Cortical neuron type. */
  neuronType: NeuronTypeName;
}

const DEFAULT_VISUAL_CONFIG: VisualCortexConfig = {
  neuronCount: 2000,
  inputCount: 1000,
  kWinners: 20,
  fatigueFactor: 0.05,
  learningRate: 1.0,
  aPlus: 0.02,
  aMinus: 0.012,
  tauPlus: 20.0,
  tauMinus: 20.0,
  maxWeightBudget: 40.0,
  weightRange: [-1.0, 4.0],
  inputGain: 8.0,
  backgroundCurrent: 2.0,
  maxCurrent: 20.0,
  noiseAmplitude: 0.6,
  refractoryMs: 2.0,
  presentationTicks: 60,
  targetRate: 0.02,
  homeostasisRate: 0.01,
  recognitionThreshold: 6,
  neuronType: 'RegularSpiking',
};

// ====================================================================
// VisualCortex class
// ====================================================================

export class VisualCortex extends BrainRegion {
  private config: VisualCortexConfig;

  /** Local spiking neurons (Izhikevich model with real dynamics). */
  private localNeurons: SpikingNeuron[];

  /** Presynaptic trace per input (decays with τ+). */
  private preTrace: Float32Array;
  /** Postsynaptic trace per neuron (decays with τ-). */
  private postTrace: Float32Array;
  /** Intrinsic excitability bias per neuron (homeostasis). */
  private homeostaticBias: Float32Array;
  /** Moving average of activity per neuron (for homeostasis). */
  private avgActivity: Float32Array;
  /** Fatigue trace per neuron (decays each tick; transient adaptation). */
  private winCounts: Float32Array;
  /** Continuous synaptic excitation per neuron (Σ w·rate), reused. */
  private excBuf: Float32Array;
  /** Synaptic current buffer per neuron (reused). */
  private currentBuf: Float32Array;
  /** Indices for the k-WTA (reused). */
  private sortIdx: Int32Array;

  /** Stored visual memories (labeled engrams). */
  /** Segments the live activity into presentations and extracts their engrams. */
  private presentation!: PresentationTracker;
  /** Perceptual categories learned by mere exposure (no labels). */
  private prototypes!: PrototypeMemory;
  /** Outcome of the last completed presentation. */
  private lastRecognition: Recognition | null = null;
  private suppressPercept = false;
  /** 1 for neurons that are part of a learned engram (see vigilance in dynamicsTick). */
  private tuned!: Int32Array;
  /** Entrenchment (0..1) of each neuron's categories: relaxes its vigilance (perceptual narrowing). */
  private entrenched!: Float32Array;
  private lastEngram: Int32Array = new Int32Array(0);
  private perceptCount = 0;

  private memories: VisualMemory[] = [];

  /** Reference to the spike bus. */
  private spikeBus: SpikeBus | null = null;

  /** Last winners (for external queries). */
  private lastWinners: Int32Array = new Int32Array(0);

  /** dt of the last step (captured from the base class). */
  private _dt: number = 1.0;

  // --- Live learning metrics (for the dashboard) ---
  /** Leaky spike count: defines a stable engram despite per-tick noise. */
  private recentSpikeCounts: Float32Array;
  /** Live engram of the previous tick (to measure stability). */
  private prevLiveEngram: Int32Array = new Int32Array(0);
  /** Current live engram (top-k of recentSpikeCounts). */
  private liveEngram: Int32Array = new Int32Array(0);
  /** EMA of Σ|Δw| per tick → decays as learning converges. */
  private weightChangeEMA = 0;
  /** EMA of the engram_t vs engram_{t-1} overlap → rises as it consolidates. */
  private engramStabilityEMA = 0;
  /** EMA of the fraction of active neurons per tick. */
  private liveActivityEMA = 0;
  /** Σ|Δw| accumulated since startup (total learning energy). */
  private cumWeightChange = 0;

  constructor(config: Partial<VisualCortexConfig> = {}) {
    const cfg = { ...DEFAULT_VISUAL_CONFIG, ...config };
    super('visualCortex', 'Corteza Visual', cfg.neuronCount, cfg.inputCount);

    this.config = cfg;
    this.localNeurons = createNeuronPopulation(cfg.neuronCount, cfg.neuronType);

    this.preTrace = new Float32Array(cfg.inputCount);
    this.postTrace = new Float32Array(cfg.neuronCount);
    this.homeostaticBias = new Float32Array(cfg.neuronCount);
    this.avgActivity = new Float32Array(cfg.neuronCount);
    this.winCounts = new Float32Array(cfg.neuronCount);
    this.excBuf = new Float32Array(cfg.neuronCount);
    this.currentBuf = new Float32Array(cfg.neuronCount);
    this.sortIdx = new Int32Array(cfg.neuronCount);
    this.recentSpikeCounts = new Float32Array(cfg.neuronCount);
    this.tuned = new Int32Array(cfg.neuronCount);
    this.entrenched = new Float32Array(cfg.neuronCount);
    this.seenInput = new Float32Array(cfg.inputCount);
    this.presentation = new PresentationTracker(cfg.neuronCount, cfg.kWinners);
    this.prototypes = new PrototypeMemory({
      labelPrefix: 'Visual',
      matchThreshold: VISUAL_CATEGORY_MATCH,
      unitCount: cfg.neuronCount,
    });

    this.initializeVisualWeights();
  }

  /** Initializes small, random synaptic weights (synaptogenesis). */
  private initializeVisualWeights(): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = Math.random() * 0.1;
    }
  }

  connectBus(bus: SpikeBus): void {
    this.spikeBus = bus;
    bus.register(this.id);
  }

  /** Captures dt and delegates to the base class logic. */
  override step(dt: number, modulationEffects: ModulationEffects) {
    this._dt = dt;
    return super.step(dt, modulationEffects);
  }

  /**
   * One tick of cortical dynamics: integrates currents, runs Izhikevich,
   * applies k-WTA lateral inhibition and (if learn) trace-based STDP plasticity.
   *
   * @param rates - Input rate vector (0-1); sampled with Poisson per tick.
   * @param dt - Time step (ms).
   * @param t - Simulation time (ms).
   * @param learn - Whether to apply STDP this tick.
   * @param lrMul - Learning rate multiplier (neuromodulation).
   * @param gain - Spike gain multiplier (neuromodulation).
   * @returns Total Δw applied this tick (Σ|Δw|).
   */
  private dynamicsTick(
    rates: Float32Array,
    dt: number,
    t: number,
    learn: boolean,
    lrMul: number,
    gain: number,
  ): number {
    const n = this.neuronCount;
    const m = this.inputCount;
    const cfg = this.config;

    // --- 0. Active input indices + decay pre trace + sample Poisson ---
    const decayPre = Math.exp(-dt / cfg.tauPlus);
    for (let i = 0; i < m; i++) this.preTrace[i] *= decayPre;

    const activeInputs: number[] = [];
    for (let i = 0; i < m; i++) if (rates[i] > 0) activeInputs.push(i);

    // Input spikes for this tick (Poisson rate coding) → temporal STDP realism.
    const firedInputs: number[] = [];
    for (let a = 0; a < activeInputs.length; a++) {
      const i = activeInputs[a];
      if (rateCoding(rates[i], 200, dt) > 0.5) {
        firedInputs.push(i);
        this.preTrace[i] = 1.0; // nearest-neighbour: resets the trace on spike
      }
    }

    let rateNorm2 = 0;
    for (let a = 0; a < activeInputs.length; a++) rateNorm2 += rates[activeInputs[a]] * rates[activeInputs[a]];
    const rateNorm = Math.sqrt(rateNorm2);

    // --- 1. CONTINUOUS excitation (Σ w·rate) and cosine match SCORE ---
    // excBuf = w·rate (magnitude, for the membrane current).
    // scoreBuf = (w·rate)/‖w‖ → cosine similarity with the pattern: measures how well
    // the neuron's weights "point" to the current stimulus, not their raw magnitude.
    // Without this normalization, a few shared synapses saturated to maxW
    // would make A's engram leak into B (false match).
    // Sparse: only the active channels contribute to the excitation, and each
    // neuron's weight norm is cached (kept up to date by the STDP below) instead
    // of being recomputed over every synapse on every tick.
    const score = this.currentBuf;
    const norm2 = this.weightNorm2();
    for (let nn = 0; nn < n; nn++) {
      const offset = nn * m;
      let exc = 0;
      for (let a = 0; a < activeInputs.length; a++) {
        const i = activeInputs[a];
        exc += this.weights[offset + i] * rates[i];
      }
      this.excBuf[nn] = exc;
      score[nn] = exc / (Math.sqrt(norm2[nn]) + 1e-6);

      // Vigilance: a neuron already TUNED to some image competes at a
      // disadvantage for an image that does not match what it is tuned to.
      // Otherwise its large learned weights win any image that shares a few
      // channels with its own, STDP then re-tunes it toward that image, and
      // what it had learned is eroded (a cross seen after 150 unrelated
      // drawings was no longer recognized). A poor match leaves the field to
      // untuned neurons, which get recruited (adaptive resonance; Grossberg, 1987).
      // Perceptual narrowing: a neuron of a well-worn category is lenient — it captures nearby inputs.
      if (this.tuned[nn] === 1 && score[nn] < VISUAL_VIGILANCE * (1 - PERCEPTUAL_NARROWING.gain * this.entrenched[nn]) * rateNorm) score[nn] -= MISMATCH_PENALTY * rateNorm;
    }

    // --- 2. k-WTA lateral inhibition by cosine score (+ homeostatic bias) ---
    // The bias enters with a SMALL weight (BIAS_GAIN): it is a fairness nudge
    // so that chronically silent neurons win ties, NOT a term
    // that can override the match with the stimulus (that would collapse all
    // patterns to the same engram of "least active neurons").
    const BIAS_GAIN = 0.05;
    for (let i = 0; i < n; i++) this.sortIdx[i] = i;
    const bias = this.homeostaticBias;
    this.sortIdx.sort((a, b) => score[b] + BIAS_GAIN * bias[b] - (score[a] + BIAS_GAIN * bias[a]));
    const k = cfg.kWinners;
    const isWinner = new Uint8Array(n);
    for (let i = 0; i < k && i < n; i++) isWinner[this.sortIdx[i]] = 1;

    // --- 3. Izhikevich membrane dynamics (real) ---
    // Winners: sustained supra-threshold current (background + gain·exc).
    // Non-winners: only sub-threshold background → their membrane evolves but
    // (except for a homeostatic push) does not cross the threshold. Models the
    // GABAergic lateral inhibition without resetting the competitors' membrane.
    const decayPost = Math.exp(-dt / cfg.tauMinus);
    const winnersFired: number[] = [];

    for (let nn = 0; nn < n; nn++) {
      this.postTrace[nn] *= decayPost;

      // Absolute refractory
      if (t - this.localNeurons[nn].lastSpikeTime < cfg.refractoryMs) {
        this.localNeurons[nn].fired = false;
        this.spikes[nn] = 0;
        continue;
      }

      // The background is SUB-threshold for everyone → a non-winner never fires from
      // background alone. Firing is strictly governed by the k-WTA: the homeostatic
      // bias does NOT enter the current of non-winners (if it did,
      // a high bias would make them fire ignoring the stimulus and the engram would
      // become independent of the pattern). The bias only influences WHO wins
      // (via BIAS_GAIN in the WTA) and slightly modulates the winners.
      const noise = (Math.random() - 0.5) * 2 * cfg.noiseAmplitude;
      let I = cfg.backgroundCurrent + noise;
      if (isWinner[nn]) {
        const fatigue = Math.min(this.winCounts[nn] * cfg.fatigueFactor, 3.0);
        I += this.homeostaticBias[nn] + cfg.inputGain * this.excBuf[nn] * gain - fatigue;
      }
      // Current ceiling: the excitation of a trained engram (w→maxW) can
      // reach hundreds; with dt=1 ms Izhikevich's 0.04v² term becomes
      // numerically unstable and u explodes, making the neuron fire in
      // a loop ignoring the stimulus. The clamp keeps the integration stable.
      if (I > cfg.maxCurrent) I = cfg.maxCurrent;
      const fired = this.localNeurons[nn].step(I, dt, t);

      this.spikes[nn] = fired ? 1 : 0;
      if (fired) {
        this.postTrace[nn] = 1.0; // post trace for future LTD
        if (isWinner[nn]) winnersFired.push(nn);
      }
    }

    // --- 4. Trace-based STDP plasticity (only if learn) ---
    let weightChange = 0;
    if (learn) {
      const lr = cfg.learningRate * lrMul;
      const [minW, maxW] = cfg.weightRange;

      // LTP: the post fired → potentiate synapses with an active pre trace.
      // SOFT bound: Δw ∝ (maxW − w) → potentiation slows as it approaches the
      // ceiling, driving the synapse to a stable fixed point (Δw convergence).
      for (let w = 0; w < winnersFired.length; w++) {
        const nn = winnersFired[w];
        this.winCounts[nn]++;
        const offset = nn * m;
        for (let i = 0; i < m; i++) {
          const pre = this.preTrace[i];
          if (pre > 1e-4) {
            const w0 = this.weights[offset + i];
            const dw = lr * cfg.aPlus * pre * (maxW - w0);
            this.weights[offset + i] = w0 + dw;
            norm2[nn] += (2 * w0 + dw) * dw;
            weightChange += Math.abs(dw);
          }
        }
      }

      // LTD: the pre fired → depress synapses toward neurons that fired
      // BEFORE (active post trace) but NOT now or recently. Excluded are:
      //   - the winners (isWinner): protected, their active synapse is causal;
      //   - those firing this tick (spikes===0 already guarantees it);
      //   - the refractory ones (they just fired → the silence is an artifact).
      // Thus LTD decorrelates the losers without undoing the engrams' LTP.
      for (let f = 0; f < firedInputs.length; f++) {
        const i = firedInputs[f];
        for (let nn = 0; nn < n; nn++) {
          if (isWinner[nn] || this.spikes[nn] !== 0) continue;
          if (t - this.localNeurons[nn].lastSpikeTime < cfg.refractoryMs) continue;
          const post = this.postTrace[nn];
          if (post > 1e-4) {
            const offset = nn * m;
            // Symmetric soft bound: Δw ∝ (w − minW) → depression slows
            // near the floor, avoiding oscillations and giving a stable fixed point.
            const w0 = this.weights[offset + i];
            const dw = lr * cfg.aMinus * post * (w0 - minW);
            this.weights[offset + i] = w0 - dw;
            norm2[nn] += (dw - 2 * w0) * dw;
            weightChange += Math.abs(dw);
          }
        }
      }
    }

    // --- 5. Intrinsic excitability homeostasis (only during learning) ---
    // Adjusts the bias to bring each neuron's mean rate closer to targetRate and
    // decays the fatigue (transient adaptation). The bias is clamped to [-2, 2] so
    // that it never silences a supra-threshold winner (engram stability).
    // Probes (learn=false) do NOT mutate the homeostatic state → clean measurement.
    if (learn) {
      const a = cfg.homeostasisRate;
      for (let nn = 0; nn < n; nn++) {
        this.avgActivity[nn] = (1 - a) * this.avgActivity[nn] + a * this.spikes[nn];
        let b = this.homeostaticBias[nn] + a * (cfg.targetRate - this.avgActivity[nn]);
        if (b > 1) b = 1;
        else if (b < -1) b = -1;
        this.homeostaticBias[nn] = b;
        this.winCounts[nn] *= 0.98; // leaky fatigue
      }
    }

    return weightChange;
  }

  /**
   * Processes a visual input: runs ONE tick of dynamics with live learning.
   * Called by the base class on each tick of the brain.
   *
   * @param spikes - Input vector (treated as rates 0-1).
   * @param modulationEffects - Neuromodulation effects.
   * @returns Output spike vector (1.0 = fired, 0.0 = silent).
   */
  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    // Nothing on the retina → nothing to compute. The cortex is silent at rest
    // anyway (the background current is sub-threshold); skipping the dense
    // synaptic pass makes an idle tick ~30% cheaper for the whole brain, and
    // plasticity and homeostasis only ever run on actual experience.
    let driven = false;
    for (let i = 0; i < spikes.length; i++) {
      if (spikes[i] > 0) {
        driven = true;
        break;
      }
    }
    if (!driven) {
      this.spikes.fill(0);
      this.lastWinners = new Int32Array(0);
      this.closePresentation(this.presentation.tick(false, this.lastWinners));
      if (!this.presentation.active) this.resetSeen();
      return new Float32Array(this.neuronCount);
    }
    for (let i = 0; i < spikes.length && i < this.seenInput.length; i++) this.seenInput[i] += spikes[i];
    this.seenTicks++;

    const lrMul = modulationEffects.learningRateMultiplier ?? 1.0;
    const gain = modulationEffects.spikeGainMultiplier ?? 1.0;
    // The brain's own scribbles are motor exploration, not objects: the hand
    // learns from them (dorsal stream), this cortex does not.
    const plastic = !this.suppressPercept;
    const dw = this.dynamicsTick(spikes, this._dt, this.currentTime, plastic, lrMul, gain);

    // Instantaneous engram: neurons that fired this tick
    const winners: number[] = [];
    const out = new Float32Array(this.neuronCount);
    for (let nn = 0; nn < this.neuronCount; nn++) {
      if (this.spikes[nn] > 0) {
        out[nn] = 1.0;
        winners.push(nn);
      }
    }
    this.lastWinners = Int32Array.from(winners);
    // Surprise: on the first tick of a presentation, how far the image is from
    // what the neurons that answered it stand for — before they learn from it.
    // (This cortex learns live, tick by tick; measured at the end, its own
    // learning would have erased the surprise.)
    // (The membrane takes a tick or two to fire: measured on the first tick
    // with winners, not on the first tick with input.)
    if (this.seenSurprise < 0 && this.lastWinners.length > 0) {
      const expected = this.imagine(this.lastWinners);
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < spikes.length && i < expected.length; i++) {
        dot += spikes[i] * expected[i];
        na += spikes[i] * spikes[i];
        nb += expected[i] * expected[i];
      }
      this.seenSurprise = na > 0 && nb > 0 ? Math.max(0, Math.min(1, 1 - dot / Math.sqrt(na * nb))) : 1;
    }
    this.presentation.tick(true, this.lastWinners);
    this.updateLearningMetrics(dw, winners.length);
    return out;
  }

  /**
   * End of a presentation: its engram is matched against the perceptual
   * categories learned so far ("have I seen this before?").
   */
  private closePresentation(engram: Int32Array | null): void {
    if (!engram) return;
    // The incremental norm updates drift a little: resynchronize once per presentation.
    this.norm2Cache = null;
    // Self-generated exploration (a babble, a scribble) is not an object of
    // the world: it founds no category and is not offered for association.
    if (this.suppressPercept) {
      this.suppressPercept = false;
      return;
    }
    // Commitment of freshly recruited neurons (adaptive resonance): a neuron
    // that has just been tuned drops its random initial synapses off the
    // image that recruited it. Left in place, that random background — as
    // large as one exposure's worth of learning — pointed the neuron at every
    // image alike (its cosine with an unrelated drawing was as high as with
    // its own), and categories merged and drifted.
    const surprise = this.seenSurprise < 0 ? 1 : this.seenSurprise;
    if (this.seenTicks > 0) {
      const m = this.inputCount;
      for (let k = 0; k < engram.length; k++) {
        const nn = engram[k];
        if (this.tuned[nn] === 1) continue;
        const offset = nn * m;
        for (let i = 0; i < m; i++) {
          if (this.seenInput[i] / this.seenTicks < VisualCortex.COMMIT_FLOOR) this.weights[offset + i] *= VisualCortex.COMMIT_KEEP;
        }
      }
    }
    this.resetSeen();
    for (let i = 0; i < engram.length; i++) this.tuned[engram[i]] = 1;
    const recognition = this.prototypes.observe(engram, this.currentTime);
    if (recognition) {
      recognition.surprise = surprise;
      this.lastRecognition = recognition;
      this.lastEngram = engram;
      this.perceptCount++;
      this.refreshEntrenchment();
    }
  }

  /** Mean input over the presentation in progress (what a recruited neuron commits to). */
  private seenInput: Float32Array;
  private seenTicks = 0;
  /** Surprise measured on the first responding tick of the presentation in progress (−1 = not yet). */
  private seenSurprise = -1;
  /** Input level below which a recruited neuron's synapse is pruned, and what is left of it. */
  private static readonly COMMIT_FLOOR = 0.1;
  private static readonly COMMIT_KEEP = 0.1;

  private resetSeen(): void {
    this.seenInput.fill(0);
    this.seenTicks = 0;
    this.seenSurprise = -1;
  }

  /** Squared weight norm per neuron, cached (see `dynamicsTick`). */
  private norm2Cache: Float32Array | null = null;

  private weightNorm2(): Float32Array {
    if (!this.norm2Cache) {
      const m = this.inputCount;
      const norms = new Float32Array(this.neuronCount);
      for (let nn = 0; nn < this.neuronCount; nn++) {
        const offset = nn * m;
        let sum = 0;
        for (let i = 0; i < m; i++) sum += this.weights[offset + i] * this.weights[offset + i];
        norms[nn] = sum;
      }
      this.norm2Cache = norms;
    }
    return this.norm2Cache;
  }

  override loadWeights(weights: Float32Array): void {
    super.loadWeights(weights);
    this.norm2Cache = null;
  }

  /**
   * Mental imagery: the retinal pattern a set of cortical neurons stands for.
   *
   * Runs the cortex backwards — from an engram (e.g. one reinstated from memory
   * by a word) to the input its neurons are tuned to: the mean of their
   * afferent weights, minus what the average neuron would give (so only what
   * is specific to the engram remains), scaled to 0–1.
   *
   * Biology: visual imagery reactivates early visual cortex through feedback
   * connections, with the retinotopy of the imagined object (Kosslyn et al., 1995).
   *
   * @param units - Neurons of this cortex (an engram)
   * @returns Imagined retinal pattern (length = inputCount), all zeros if the
   *   neurons are not tuned to anything in particular
   */
  imagine(units: ArrayLike<number>): Float32Array {
    const m = this.inputCount;
    const image = new Float32Array(m);
    if (units.length === 0) return image;

    const baseline = new Float32Array(m);
    for (let nn = 0; nn < this.neuronCount; nn++) {
      const offset = nn * m;
      for (let i = 0; i < m; i++) baseline[i] += this.weights[offset + i];
    }
    let peak = 0;
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let u = 0; u < units.length; u++) sum += this.weights[units[u] * m + i];
      const specific = sum / units.length - baseline[i] / this.neuronCount;
      image[i] = specific > 0 ? specific : 0;
      if (image[i] > peak) peak = image[i];
    }
    if (peak > 0) for (let i = 0; i < m; i++) image[i] /= peak;
    return image;
  }

  /**
   * The presentation that is starting is the brain's own motor exploration:
   * learn from it as usual, but do not treat it as a percept.
   */
  suppressNextPercept(): void {
    this.suppressPercept = true;
  }

  /** Engram of the last completed presentation (what `getRecognition()` refers to). */
  getLastEngram(): Int32Array {
    return this.lastEngram;
  }

  /** Completed presentations so far — changes exactly when a new percept is available. */
  get percepts(): number {
    return this.perceptCount;
  }


  /** The perceptual categories learned so far. */
  categories(): CategoryView[] {
    return this.prototypes.list();
  }

  /** Vigilance of a neuron: relaxed by how entrenched its categories are (perceptual narrowing). */
  private refreshEntrenchment(): void {
    this.entrenched = this.prototypes.entrenchment(this.neuronCount, PERCEPTUAL_NARROWING.tau);
  }

  /**
   * A sleep passed: categories age, and those met fewer than `minExposures`
   * times and unseen for `afterSleeps` sleeps are pruned; their neurons, if no
   * surviving category — nor anything else (`inUse`) — runs on them, are
   * released for recruitment.
   *
   * @returns Labels of the pruned categories
   */
  pruneCategories(minExposures: number, afterSleeps: number, inUse: (unit: number) => boolean = () => false): string[] {
    this.prototypes.age();
    const pruned = this.prototypes.prune(minExposures, afterSleeps);
    if (pruned.length === 0) return [];
    const used = this.prototypes.unitsInUse();
    // A neuron is released only if nothing else runs on it: no surviving
    // category, no association, no motor map (synapses that carry something
    // are not the ones pruned).
    for (const c of pruned) for (const u of c.units) if (!used.has(u) && !inUse(u)) this.release(u);
    this.refreshEntrenchment();
    return pruned.map((c) => c.label);
  }

  private release(n: number): void {
    this.tuned[n] = 0;
    this.winCounts[n] = 0;
    const offset = n * this.inputCount;
    for (let i = 0; i < this.inputCount; i++) this.weights[offset + i] *= RELEASE_KEEP;
    this.onReleased();
  }

  private onReleased(): void {
    this.norm2Cache = null;
  }

  /** Names a code reinstated from memory: the learned category it overlaps most. */
  matchCategory(units: ArrayLike<number>): { id: number; label: string; overlap: number; exposures: number } | null {
    return this.prototypes.match(units);
  }

  /** Outcome of the last completed presentation, or `null` if none yet. */
  getRecognition(): Recognition | null {
    return this.lastRecognition;
  }

  /** Number of visual categories learned by exposure. */
  get categoryCount(): number {
    return this.prototypes.size;
  }

  /**
   * Updates the live learning metrics after a tick.
   * The instantaneous engram (what fired THIS tick) is noisy due to the Poisson
   * sampling and the refractory period, so we keep a leaky spike count
   * (recentSpikeCounts) whose top-k defines a stable engram. On it we measure:
   *   - stability: overlap with the previous tick's engram (rises on consolidation);
   *   - weight change (EMA of Σ|Δw|): falls as learning converges;
   *   - activity: fraction of active neurons.
   */
  private updateLearningMetrics(dw: number, firedCount: number): void {
    const n = this.neuronCount;
    const decay = 0.85; // accumulator leak → effective window ~6-7 ticks
    for (let nn = 0; nn < n; nn++) {
      this.recentSpikeCounts[nn] = this.recentSpikeCounts[nn] * decay + this.spikes[nn];
    }

    // Stable engram = top-kWinners of recentSpikeCounts (only counts > 0).
    for (let i = 0; i < n; i++) this.sortIdx[i] = i;
    const counts = this.recentSpikeCounts;
    this.sortIdx.sort((a, b) => counts[b] - counts[a]);
    const k = this.config.kWinners;
    const engram: number[] = [];
    for (let i = 0; i < k && i < n; i++) {
      if (counts[this.sortIdx[i]] > 0.05) engram.push(this.sortIdx[i]);
    }
    this.prevLiveEngram = this.liveEngram;
    this.liveEngram = Int32Array.from(engram.sort((a, b) => a - b));

    const stability = VisualCortex.engramOverlap(this.prevLiveEngram, this.liveEngram);
    this.engramStabilityEMA = 0.9 * this.engramStabilityEMA + 0.1 * stability;
    this.weightChangeEMA = 0.95 * this.weightChangeEMA + 0.05 * dw;
    this.cumWeightChange += dw;
    this.liveActivityEMA = 0.95 * this.liveActivityEMA + 0.05 * (firedCount / n);
  }

  /**
   * Live learning metrics for external monitoring (dashboard).
   * Does not mutate state: it is safe to call on every broadcast.
   */
  getLearningMetrics(): {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
  } {
    return {
      engram: Array.from(this.liveEngram),
      engramSize: this.liveEngram.length,
      stability: this.engramStabilityEMA,
      weightChange: this.weightChangeEMA,
      cumWeightChange: this.cumWeightChange,
      activity: this.liveActivityEMA,
      neuronCount: this.neuronCount,
    };
  }

  /**
   * Presents a stimulus during a temporal window and returns the resulting
   * engram (the neurons that fired most). It is the reference API
   * for deterministic learning experiments.
   *
   * @param rates - Input rate vector (0-1).
   * @param opts - Options: ticks, learn, modulation.
   */
  present(
    rates: Float32Array,
    opts: { ticks?: number; learn?: boolean; lrMul?: number; gain?: number } = {},
  ): PresentationResult {
    const ticks = opts.ticks ?? this.config.presentationTicks;
    const learn = opts.learn ?? true;
    const lrMul = opts.lrMul ?? 1.0;
    const gain = opts.gain ?? 1.0;

    const spikeCounts = new Int32Array(this.neuronCount);
    let totalWeightChange = 0;
    let totalActive = 0;

    for (let tick = 0; tick < ticks; tick++) {
      this.currentTime += this._dt;
      const dw = this.dynamicsTick(rates, this._dt, this.currentTime, learn, lrMul, gain);
      totalWeightChange += dw;
      for (let nn = 0; nn < this.neuronCount; nn++) {
        if (this.spikes[nn] > 0) {
          spikeCounts[nn]++;
          totalActive++;
        }
      }
    }

    // Engram = top-kWinners by spike count in the window
    for (let i = 0; i < this.neuronCount; i++) this.sortIdx[i] = i;
    const counts = spikeCounts;
    this.sortIdx.sort((a, b) => counts[b] - counts[a]);
    const k = this.config.kWinners;
    const engram: number[] = [];
    for (let i = 0; i < k && i < this.neuronCount; i++) {
      if (counts[this.sortIdx[i]] > 0) engram.push(this.sortIdx[i]);
    }
    const engramArr = Int32Array.from(engram.sort((a, b) => a - b));
    this.lastWinners = engramArr;

    return {
      engram: engramArr,
      spikeCounts,
      weightChange: totalWeightChange,
      activity: ticks > 0 ? totalActive / (ticks * this.neuronCount) : 0,
    };
  }

  /**
   * Overlap (Jaccard index) between two engrams. 1.0 = identical, 0 = disjoint.
   */
  static engramOverlap(a: Int32Array, b: Int32Array): number {
    if (a.length === 0 && b.length === 0) return 1;
    if (a.length === 0 || b.length === 0) return 0;
    const setA = new Set(a);
    let inter = 0;
    for (let i = 0; i < b.length; i++) if (setA.has(b[i])) inter++;
    const union = a.length + b.length - inter;
    return union > 0 ? inter / union : 0;
  }

  /**
   * Learns a pattern by associating it with a label (wrapper around present()).
   * Maintains compatibility with the previous labeled flow.
   */
  learn(
    input: Float32Array,
    label: string,
    _dt: number,
    timestamp: number,
    modulationEffects?: ModulationEffects,
  ): VisualProcessingResult {
    const lrMul = modulationEffects?.learningRateMultiplier ?? 1.0;
    const gain = modulationEffects?.spikeGainMultiplier ?? 1.0;
    const result = this.present(input, { learn: true, lrMul, gain });

    this.memories.push({
      pattern: new Int32Array(result.engram),
      label,
      strength: 1,
      createdAt: timestamp,
    });

    return {
      winners: result.engram,
      potentials: new Float32Array(this.neuronCount),
      predictions: this.compareWithMemories(result.engram),
      activity: result.activity,
    };
  }

  /** Predicts without learning (inference only). */
  predict(input: Float32Array, _dt: number, _timestamp: number): VisualProcessingResult {
    const result = this.present(input, { learn: false });
    return {
      winners: result.engram,
      potentials: new Float32Array(this.neuronCount),
      predictions: this.compareWithMemories(result.engram),
      activity: result.activity,
    };
  }

  /** Compares an activation pattern with the stored memories. */
  compareWithMemories(currentWinners: Int32Array): string[] {
    const matches: string[] = [];
    const seenLabels = new Set<string>();
    const current = new Set(currentWinners);

    for (const mem of this.memories) {
      let overlap = 0;
      for (let m = 0; m < mem.pattern.length; m++) {
        if (current.has(mem.pattern[m])) overlap++;
      }
      if (overlap >= this.config.recognitionThreshold && !seenLabels.has(mem.label)) {
        matches.push(mem.label);
        seenLabels.add(mem.label);
      }
    }
    return matches;
  }

  /** Sends the current cortical representation to the spike bus. */
  emitToDownstream(timestamp: number, targets: string[] = ['hippocampus', 'amygdala']): void {
    if (!this.spikeBus) return;
    const outSpikes = new Float32Array(this.neuronCount);
    for (let i = 0; i < this.lastWinners.length; i++) {
      outSpikes[this.lastWinners[i]] = 1.0;
    }
    this.spikeBus.send({
      source: this.id,
      targets,
      spikes: outSpikes,
      timestamp,
      metadata: { winnerCount: this.lastWinners.length, activity: this.getLocalActivity() },
    });
  }

  // ----------------------------------------------------------------
  // Persistence of the non-weight learned state
  // ----------------------------------------------------------------

  /**
   * Homeostatic state and labelled memories. The weights alone are not the
   * whole skill: they were learned under a given homeostatic bias and fatigue,
   * and without those the same stimulus recruits different winners after a restart.
   */
  override serializeExtra(): unknown {
    return {
      homeostaticBias: packArray(this.homeostaticBias),
      avgActivity: packArray(this.avgActivity),
      winCounts: packArray(this.winCounts),
      tuned: packArray(this.tuned),
      categories: this.prototypes.serialize(),
      memories: this.memories.map((m) => ({
        pattern: Array.from(m.pattern),
        label: m.label,
        strength: m.strength,
        createdAt: m.createdAt,
      })),
    };
  }

  override deserializeExtra(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as Record<string, unknown>;
    const n = this.neuronCount;
    const bias = unpackFloat32(d.homeostaticBias, n);
    const avg = unpackFloat32(d.avgActivity, n);
    const wins = unpackFloat32(d.winCounts, n);
    if (bias) this.homeostaticBias.set(bias);
    if (avg) this.avgActivity.set(avg);
    if (wins) this.winCounts.set(wins);
    const tuned = unpackInt32(d.tuned, n);
    if (tuned) this.tuned.set(tuned);
    this.prototypes.deserialize(d.categories);
    this.refreshEntrenchment();

    const memories = Array.isArray(d.memories) ? d.memories : [];
    this.memories = [];
    for (const raw of memories.slice(-MAX_PERSISTED_MEMORIES)) {
      const m = raw as { pattern?: unknown; label?: unknown; strength?: unknown; createdAt?: unknown };
      if (!Array.isArray(m?.pattern) || typeof m.label !== 'string') continue;
      if (!m.pattern.every((i) => Number.isInteger(i) && i >= 0 && i < this.neuronCount)) continue;
      this.memories.push({
        pattern: Int32Array.from(m.pattern as number[]),
        label: m.label.slice(0, 80),
        strength: typeof m.strength === 'number' && Number.isFinite(m.strength) ? m.strength : 1,
        createdAt: typeof m.createdAt === 'number' && Number.isFinite(m.createdAt) ? m.createdAt : 0,
      });
    }
  }

  get memoryCount(): number {
    return this.memories.length;
  }

  getLastWinners(): Int32Array {
    return this.lastWinners;
  }

  /** Fraction of neurons that fired in the last tick. */
  getLocalActivity(): number {
    let active = 0;
    for (let i = 0; i < this.neuronCount; i++) if (this.spikes[i] > 0) active++;
    return active / this.neuronCount;
  }

  getStats(): {
    memoryCount: number;
    averageWeight: number;
    maxFatigue: number;
    activity: number;
    meanBias: number;
  } {
    let weightSum = 0;
    for (let i = 0; i < this.weights.length; i++) weightSum += this.weights[i];
    let maxFatigue = 0;
    for (let i = 0; i < this.winCounts.length; i++) {
      if (this.winCounts[i] > maxFatigue) maxFatigue = this.winCounts[i];
    }
    let biasSum = 0;
    for (let i = 0; i < this.homeostaticBias.length; i++) biasSum += this.homeostaticBias[i];
    return {
      memoryCount: this.memories.length,
      averageWeight: weightSum / this.weights.length,
      maxFatigue,
      activity: this.getLocalActivity(),
      meanBias: biasSum / this.homeostaticBias.length,
    };
  }

  /** Resets the homeostatic fatigue (effect of sleep). */
  resetFatigue(): void {
    this.winCounts.fill(0);
  }
}
