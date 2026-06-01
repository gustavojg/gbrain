/**
 * HIPPOCAMPUS — Episodic Memory Center (real biological core)
 * ================================================================
 * Models the hippocampal formation of the medial temporal lobe. Unlike
 * the visual cortex (Izhikevich+STDP feedforward map), the natural
 * computation of the hippocampus is a RECURRENT AUTOASSOCIATIVE MEMORY: an
 * attractor network that reconstructs a complete pattern from a partial
 * or degraded cue (pattern completion).
 *
 * Modeled trisynaptic circuit:
 *   Entorhinal cortex → DG (separation) → CA3 (autoassociation) → output
 *
 *   - Dentate gyrus (DG): pattern separation. FIXED sparse projection
 *     (mossy fibers, non-plastic) + k-WTA competition that orthogonalizes
 *     similar inputs into ultra-sparse codes (~2% active). The projection
 *     is DETERMINISTIC (seeded with a constant seed) so that the same
 *     stimulus ALWAYS generates the same code, even after reloading weights.
 *
 *   - CA3: pattern completion. Autoassociative network with plastic
 *     recurrent connections (N×N matrix = this.weights). Each episode is
 *     imprinted by Hebbian learning (coincidence rule, outer product
 *     of the sparse code) with a soft bound. Recall is NOT a
 *     lookup over stored patterns: it emerges from the ATTRACTOR DYNAMICS
 *     (recurrent iteration h = W·s + k-WTA until convergence).
 *
 *   Key consequence: the memory lives in the SYNAPSES (this.weights), so
 *   the binary persistence protocol serializes it automatically.
 *   Learning survives reloads and redeploys.
 *
 * References: O'Reilly & McClelland (1994), "Hippocampal conjunctive
 *   encoding, storage, and recall"; Treves & Rolls (1994), CA3 attractors;
 *   Marr (1971), archicortex theory.
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';

// ==================================================================
// Interfaces (stable public API)
// ==================================================================

/** Context associated with an episodic memory. */
export interface EpisodicContext {
  /** Timestamp of the encoding moment (ms) */
  timestamp: number;
  /** Emotional valence of the event (-1 negative … +1 positive). */
  emotionalValence: number;
  /** Brain region the pattern originated from (e.g.: 'visualCortex') */
  sourceRegion: string;
}

/**
 * Episodic index: lightweight metadata of an encoded event.
 * The PATTERN here is the DG sparse code (the "key" of the attractor); the
 * actual reconstruction is produced by CA3 from the weights, not this record.
 */
export interface EpisodicMemory {
  /** DG sparse code associated with the event (engram-index). */
  pattern: Float32Array;
  /** Context associated with the event (when, valence, source). */
  context: EpisodicContext;
  /** Trace strength (0–1); decays with forgetting, rises with replay. */
  strength: number;
}

/** Result of a recall operation. */
export interface RecallResult {
  /** Recovered memory (metadata + context). */
  memory: EpisodicMemory;
  /** Overlap [0,1] between the recovered attractor and the event's code. */
  similarity: number;
}

/** Hippocampus configuration. */
export interface HippocampusConfig {
  /** Sparsity of the CA3/DG code (fraction of active units). */
  sparsity: number;
  /** Input connections per DG unit (mossy fiber fan-in). */
  dgFanIn: number;
  /** Hebbian learning rate (scale of Δw per coincidence). */
  learnRate: number;
  /** Recurrent weight ceiling (soft LTP bound). */
  maxWeight: number;
  /** Iterations of the attractor dynamics in completion. */
  attractorIterations: number;
  /** Minimum input energy to consider that there is a stimulus. */
  inputEnergyThreshold: number;
  /**
   * Novelty threshold: if the input's DG code overlaps the last
   * encoded one above this, it is NOT imprinted again (event-driven
   * encoding, not per-tick → avoids saturating and flooding the index).
   */
  noveltyOverlapThreshold: number;
  /** Deterministic seed of the DG projection (fixed connectivity). */
  dgSeed: number;
}

const DEFAULT_HIPPO_CONFIG: HippocampusConfig = {
  sparsity: 0.02,
  dgFanIn: 32,
  learnRate: 0.5,
  maxWeight: 1.0,
  attractorIterations: 6,
  inputEnergyThreshold: 1.0,
  noveltyOverlapThreshold: 0.9,
  dgSeed: 0x1d0c_a3e5,
};

// ==================================================================
// Deterministic PRNG (mulberry32) — reproducible DG projection
// ==================================================================

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ==================================================================
// Hippocampus class — autoassociative CA3
// ==================================================================

export class Hippocampus extends BrainRegion {
  private readonly cfg: HippocampusConfig;

  /** Active units per code (k of the k-WTA). */
  private readonly kActive: number;

  /** Maximum capacity of the episodic index (metadata). */
  private readonly maxCapacity: number;

  /** Episodic index (context metadata; the actual memory is in the weights). */
  private episodicMemories: EpisodicMemory[] = [];

  /** Deterministic DG projection: input indices per unit (N×fanIn). */
  private readonly dgIdx: Int32Array;
  /** Deterministic DG projection: ±1 sign per connection (N×fanIn). */
  private readonly dgSign: Float32Array;

  /** Last DG code imprinted (for novelty-driven encoding). */
  private lastStoredCode: Float32Array;

  // --- Reused buffers (avoid per-tick allocations) ---
  private readonly actBuf: Float32Array;
  private readonly stateBuf: Float32Array;
  private readonly nextBuf: Float32Array;
  private readonly sortIdx: Int32Array;

  // --- Live learning metrics (parity with the visual cortex) ---
  private prevEngram: Float32Array;
  private lastEngram: Float32Array;
  private weightChangeEMA = 0;
  private engramStabilityEMA = 0;
  private activityEMA = 0;
  private cumWeightChange = 0;

  /**
   * @param neuronCount - CA3 units (default: 1000)
   * @param inputCount - Cortical input dimension (default: 1000)
   * @param maxCapacity - Capacity of the episodic index (default: 10000)
   * @param config - Configuration overrides
   */
  constructor(
    neuronCount: number = 1000,
    inputCount: number = 1000,
    maxCapacity: number = 10000,
    config: Partial<HippocampusConfig> = {},
  ) {
    super('hippocampus', 'Hipocampo — Memoria Episódica', neuronCount, inputCount);

    this.cfg = { ...DEFAULT_HIPPO_CONFIG, ...config };
    this.maxCapacity = maxCapacity;
    this.kActive = Math.max(1, Math.floor(neuronCount * this.cfg.sparsity));

    // CA3 starts blank: the base initialized random sparse weights, we
    // zero them out so that Hebbian learning is the only source of
    // the recurrent connections (clean memory, no spurious attractors).
    this.weights.fill(0);

    // Deterministic DG projection (fixed mossy fiber connectivity).
    const fanIn = this.cfg.dgFanIn;
    this.dgIdx = new Int32Array(neuronCount * fanIn);
    this.dgSign = new Float32Array(neuronCount * fanIn);
    const rng = mulberry32(this.cfg.dgSeed);
    for (let u = 0; u < neuronCount; u++) {
      const base = u * fanIn;
      for (let f = 0; f < fanIn; f++) {
        this.dgIdx[base + f] = Math.floor(rng() * inputCount);
        this.dgSign[base + f] = rng() < 0.5 ? -1 : 1;
      }
    }

    this.actBuf = new Float32Array(neuronCount);
    this.stateBuf = new Float32Array(neuronCount);
    this.nextBuf = new Float32Array(neuronCount);
    this.sortIdx = new Int32Array(neuronCount);
    this.lastStoredCode = new Float32Array(neuronCount);
    this.prevEngram = new Float32Array(neuronCount);
    this.lastEngram = new Float32Array(neuronCount);
  }

  // ----------------------------------------------------------------
  // Dentate Gyrus: pattern separation (deterministic)
  // ----------------------------------------------------------------

  /**
   * Pattern separation (DG): fixed sparse projection + ReLU + k-WTA.
   * Orthogonalizes similar inputs into reproducible ultra-sparse codes.
   *
   * @param input - Cortical input pattern (Float32Array, dim inputCount)
   * @returns Binary sparse code (Float32Array dim neuronCount; 1 = active)
   */
  patternSeparation(input: Float32Array): Float32Array {
    const out = new Float32Array(this.neuronCount);
    this.dgEncodeInto(input, out);
    return out;
  }

  /** Encodes DG into a destination buffer (without allocating). */
  private dgEncodeInto(input: Float32Array, out: Float32Array): void {
    const n = this.neuronCount;
    const fanIn = this.cfg.dgFanIn;
    const act = this.actBuf;

    // Fixed random projection + ReLU.
    for (let u = 0; u < n; u++) {
      const base = u * fanIn;
      let sum = 0;
      for (let f = 0; f < fanIn; f++) {
        const idx = this.dgIdx[base + f];
        const v = idx < input.length ? input[idx] : 0;
        sum += this.dgSign[base + f] * v;
      }
      act[u] = sum > 0 ? sum : 0;
    }

    // k-WTA: only the kActive most-excited units survive (→ binary).
    out.fill(0);
    const winners = this.topKIndices(act, this.kActive);
    for (let i = 0; i < winners.length; i++) out[winners[i]] = 1;
  }

  // ----------------------------------------------------------------
  // CA3: autoassociative storage (Hebbian) and completion (attractor)
  // ----------------------------------------------------------------

  /**
   * Imprints an episode in CA3 by Hebbian learning (outer product of the
   * sparse code) with a soft bound. The memory remains in the recurrent weights.
   *
   * @param pattern - Cortical pattern to encode
   * @param context - Associated context (timestamp, valence, source)
   * @returns Σ|Δw| applied (learning energy of this event)
   */
  store(pattern: Float32Array, context: Partial<EpisodicContext> = {}): number {
    const code = this.patternSeparation(pattern);
    const dw = this.imprint(code, this.cfg.learnRate);

    const fullContext: EpisodicContext = {
      timestamp: context.timestamp ?? this.currentTime,
      emotionalValence: context.emotionalValence ?? 0,
      sourceRegion: context.sourceRegion ?? 'unknown',
    };
    const memory: EpisodicMemory = { pattern: code, context: fullContext, strength: 1.0 };

    if (this.episodicMemories.length >= this.maxCapacity) {
      // Replace the weakest trace (interference/forgetting by competition).
      let weakestIdx = 0;
      for (let i = 1; i < this.episodicMemories.length; i++) {
        if (this.episodicMemories[i].strength < this.episodicMemories[weakestIdx].strength) {
          weakestIdx = i;
        }
      }
      this.episodicMemories[weakestIdx] = memory;
    } else {
      this.episodicMemories.push(memory);
    }

    this.lastStoredCode.set(code);
    return dw;
  }

  /**
   * Hebbian outer product of the sparse code over the recurrent weights.
   * It only traverses active pairs (kActive²), so it is cheap despite being N×N.
   * Soft bound: Δw ∝ (maxW − w) → stable fixed point, without hard saturation.
   */
  private imprint(code: Float32Array, lr: number): number {
    const n = this.neuronCount;
    const active: number[] = [];
    for (let i = 0; i < n; i++) if (code[i] > 0) active.push(i);

    const maxW = this.cfg.maxWeight;
    let dwTotal = 0;
    for (let a = 0; a < active.length; a++) {
      const i = active[a];
      const row = i * n;
      for (let b = 0; b < active.length; b++) {
        if (a === b) continue; // no self-connection
        const j = active[b];
        const idx = row + j;
        const dw = lr * (maxW - this.weights[idx]);
        this.weights[idx] += dw;
        dwTotal += Math.abs(dw);
      }
    }
    return dwTotal;
  }

  /**
   * Pattern completion (CA3): recurrent attractor dynamics.
   * Starting from the degraded cue (via DG), it iterates h = W·s followed by k-WTA, and
   * converges to the nearest attractor (episode) in the cue's basin.
   *
   * @param partialInput - Partial or degraded pattern (dim inputCount)
   * @returns Reconstructed sparse code (Float32Array dim neuronCount)
   */
  patternCompletion(partialInput: Float32Array): Float32Array {
    const n = this.neuronCount;
    let state = this.stateBuf;
    let next = this.nextBuf;

    // Initial state = DG code of the cue (may be incomplete/noisy).
    this.dgEncodeInto(partialInput, state);

    for (let iter = 0; iter < this.cfg.attractorIterations; iter++) {
      // h[i] = Σ_j W[i,j] · state[j], traversing only active j (sparse).
      const activeJ: number[] = [];
      for (let j = 0; j < n; j++) if (state[j] > 0) activeJ.push(j);

      // No learning yet → return the DG code itself (graceful).
      if (activeJ.length === 0) break;

      for (let i = 0; i < n; i++) {
        const row = i * n;
        let h = 0;
        for (let a = 0; a < activeJ.length; a++) h += this.weights[row + activeJ[a]];
        this.actBuf[i] = h;
      }

      next.fill(0);
      const winners = this.topKIndices(this.actBuf, this.kActive);
      // If the weights produce no signal (all 0), preserve the current state.
      let anySignal = false;
      for (let i = 0; i < winners.length; i++) {
        if (this.actBuf[winners[i]] > 0) {
          next[winners[i]] = 1;
          anySignal = true;
        }
      }
      if (!anySignal) break;

      // Converged? (stable state → attractor reached)
      if (Hippocampus.overlapBinary(state, next) >= 0.999) {
        const tmp = state;
        state = next;
        next = tmp;
        break;
      }
      const tmp = state;
      state = next;
      next = tmp;
    }

    return new Float32Array(state);
  }

  // ----------------------------------------------------------------
  // Recall, replay and forgetting (over the episodic index)
  // ----------------------------------------------------------------

  /**
   * Recovers the K memories whose code overlaps most with the attractor evoked
   * by the cue. Recall is grounded in the CA3 dynamics (weights),
   * not in comparing the raw cue with stored patterns.
   */
  recall(cue: Float32Array, topK: number = 5): RecallResult[] {
    if (this.episodicMemories.length === 0) return [];
    const completed = this.patternCompletion(cue);

    const results: RecallResult[] = [];
    for (const memory of this.episodicMemories) {
      const sim = Hippocampus.overlapBinary(completed, memory.pattern);
      const adjusted = sim * (0.5 + 0.5 * memory.strength);
      results.push({ memory, similarity: adjusted });
    }
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }

  /**
   * Hippocampal replay: reactivates (and reinforces) the most recent and
   * strongest episodes. Each reactivation re-imprints the code in CA3 (consolidation).
   */
  replay(count: number = 10): EpisodicMemory[] {
    if (this.episodicMemories.length === 0) return [];
    const candidates = [...this.episodicMemories]
      .sort((a, b) => {
        const sa = a.context.timestamp * 0.0001 + a.strength;
        const sb = b.context.timestamp * 0.0001 + b.strength;
        return sb - sa;
      })
      .slice(0, count);

    for (const memory of candidates) {
      memory.strength = Math.min(1.0, memory.strength + 0.05);
      this.imprint(memory.pattern, this.cfg.learnRate); // engram reinforcement
    }
    return candidates;
  }

  /**
   * Gradual forgetting of the episodic index. Emotional memories decay more
   * slowly (β-adrenergic modulation of consolidation, McGaugh 2004).
   * It does not erase recurrent weights: the attractor persists even if the index is lost.
   */
  forget(decayFactor: number): void {
    for (let i = this.episodicMemories.length - 1; i >= 0; i--) {
      const memory = this.episodicMemories[i];
      const emotionalProtection = 1.0 - Math.abs(memory.context.emotionalValence) * 0.3;
      memory.strength *= decayFactor * emotionalProtection;
      if (memory.strength < 0.01) this.episodicMemories.splice(i, 1);
    }
  }

  // ----------------------------------------------------------------
  // Main processing (one tick of the brain)
  // ----------------------------------------------------------------

  /**
   * Processes cortical spikes: encodes the episode (if novel),
   * completes the pattern by attractor and emits the reconstructed engram.
   */
  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    let inputEnergy = 0;
    for (let i = 0; i < spikes.length; i++) inputEnergy += spikes[i];

    if (inputEnergy < this.cfg.inputEnergyThreshold) {
      this.spikes.fill(0);
      return new Float32Array(this.neuronCount);
    }

    const input = this.adaptInput(spikes);
    const code = this.patternSeparation(input);

    // Novelty-driven encoding: only imprint events different from the
    // last one (avoids imprinting the same pattern every tick and flooding the index).
    const novelty = 1 - Hippocampus.overlapBinary(code, this.lastStoredCode);
    let dw = 0;
    if (novelty > 1 - this.cfg.noveltyOverlapThreshold) {
      const lrMul = modulationEffects.learningRateMultiplier ?? 1.0;
      dw = this.storeWithGain(code, lrMul, {
        timestamp: this.currentTime,
        emotionalValence: 0,
        sourceRegion: 'sensory',
      });
    }

    // Complete via attractor dynamics → reconstructed engram.
    const completed = this.patternCompletion(input);

    const gain = modulationEffects.spikeGainMultiplier ?? 1.0;
    const out = new Float32Array(this.neuronCount);
    for (let i = 0; i < this.neuronCount; i++) {
      const v = completed[i] > 0 ? gain : 0;
      out[i] = v;
      this.spikes[i] = completed[i] > 0 ? 1 : 0;
    }

    this.updateLearningMetrics(dw, completed);
    return out;
  }

  /** Variant of store that scales the Hebbian rate by neuromodulation. */
  private storeWithGain(
    code: Float32Array,
    lrMul: number,
    context: EpisodicContext,
  ): number {
    const dw = this.imprint(code, this.cfg.learnRate * lrMul);

    const memory: EpisodicMemory = { pattern: code, context, strength: 1.0 };
    if (this.episodicMemories.length >= this.maxCapacity) {
      let weakestIdx = 0;
      for (let i = 1; i < this.episodicMemories.length; i++) {
        if (this.episodicMemories[i].strength < this.episodicMemories[weakestIdx].strength) {
          weakestIdx = i;
        }
      }
      this.episodicMemories[weakestIdx] = memory;
    } else {
      this.episodicMemories.push(memory);
    }
    this.lastStoredCode.set(code);
    return dw;
  }

  // ----------------------------------------------------------------
  // Live learning metrics (dashboard)
  // ----------------------------------------------------------------

  private updateLearningMetrics(dw: number, engram: Float32Array): void {
    this.prevEngram = this.lastEngram;
    this.lastEngram = engram;

    const stability = Hippocampus.overlapBinary(this.prevEngram, this.lastEngram);
    this.engramStabilityEMA = 0.9 * this.engramStabilityEMA + 0.1 * stability;
    this.weightChangeEMA = 0.95 * this.weightChangeEMA + 0.05 * dw;
    this.cumWeightChange += dw;

    let active = 0;
    for (let i = 0; i < engram.length; i++) if (engram[i] > 0) active++;
    this.activityEMA = 0.95 * this.activityEMA + 0.05 * (active / this.neuronCount);
  }

  /** Learning metrics (same shape as the visual cortex). */
  getLearningMetrics(): {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
    memoryCount: number;
  } {
    const engram: number[] = [];
    for (let i = 0; i < this.lastEngram.length; i++) {
      if (this.lastEngram[i] > 0) engram.push(i);
    }
    return {
      engram,
      engramSize: engram.length,
      stability: this.engramStabilityEMA,
      weightChange: this.weightChangeEMA,
      cumWeightChange: this.cumWeightChange,
      activity: this.activityEMA,
      neuronCount: this.neuronCount,
      memoryCount: this.episodicMemories.length,
    };
  }

  // ----------------------------------------------------------------
  // Internal utilities
  // ----------------------------------------------------------------

  /** Adapts an input vector to inputCount (truncates or pads with zeros). */
  private adaptInput(input: Float32Array): Float32Array {
    if (input.length === this.inputCount) return input;
    const adapted = new Float32Array(this.inputCount);
    adapted.set(input.subarray(0, Math.min(input.length, this.inputCount)));
    return adapted;
  }

  /** Indices of the k largest values (> 0) of an array. */
  private topKIndices(arr: Float32Array, k: number): Int32Array {
    const n = arr.length;
    for (let i = 0; i < n; i++) this.sortIdx[i] = i;
    this.sortIdx.sort((a, b) => arr[b] - arr[a]);
    const out: number[] = [];
    for (let i = 0; i < k && i < n; i++) {
      if (arr[this.sortIdx[i]] > 0) out.push(this.sortIdx[i]);
    }
    return Int32Array.from(out);
  }

  /** Jaccard overlap between two binary sparse codes (>0 = active). */
  static overlapBinary(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let inter = 0;
    let ca = 0;
    let cb = 0;
    for (let i = 0; i < len; i++) {
      const ai = a[i] > 0 ? 1 : 0;
      const bi = b[i] > 0 ? 1 : 0;
      ca += ai;
      cb += bi;
      inter += ai & bi;
    }
    if (ca === 0 && cb === 0) return 1;
    const union = ca + cb - inter;
    return union > 0 ? inter / union : 0;
  }

  // ----------------------------------------------------------------
  // Public properties
  // ----------------------------------------------------------------

  /** Current number of indexed episodes. */
  get memoryCount(): number {
    return this.episodicMemories.length;
  }

  /** Maximum capacity of the episodic index. */
  get capacity(): number {
    return this.maxCapacity;
  }

  /** Occupancy [0,1] of the episodic index. */
  get occupancy(): number {
    return this.episodicMemories.length / this.maxCapacity;
  }
}
