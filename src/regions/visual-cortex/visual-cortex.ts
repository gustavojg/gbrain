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

    // --- 1. CONTINUOUS excitation (Σ w·rate) and cosine match SCORE ---
    // excBuf = w·rate (magnitude, for the membrane current).
    // scoreBuf = (w·rate)/‖w‖ → cosine similarity with the pattern: measures how well
    // the neuron's weights "point" to the current stimulus, not their raw magnitude.
    // Without this normalization, a few shared synapses saturated to maxW
    // would make A's engram leak into B (false match).
    const score = this.currentBuf;
    for (let nn = 0; nn < n; nn++) {
      const offset = nn * m;
      let exc = 0;
      let norm2 = 0;
      for (let i = 0; i < m; i++) {
        const w = this.weights[offset + i];
        norm2 += w * w;
        if (rates[i] > 0) exc += w * rates[i];
      }
      this.excBuf[nn] = exc;
      score[nn] = exc / (Math.sqrt(norm2) + 1e-6);
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
            const dw = lr * cfg.aPlus * pre * (maxW - this.weights[offset + i]);
            this.weights[offset + i] += dw;
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
            const dw = lr * cfg.aMinus * post * (this.weights[offset + i] - minW);
            this.weights[offset + i] -= dw;
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
    const lrMul = modulationEffects.learningRateMultiplier ?? 1.0;
    const gain = modulationEffects.spikeGainMultiplier ?? 1.0;
    const dw = this.dynamicsTick(spikes, this._dt, this.currentTime, true, lrMul, gain);

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
    this.updateLearningMetrics(dw, winners.length);
    return out;
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
