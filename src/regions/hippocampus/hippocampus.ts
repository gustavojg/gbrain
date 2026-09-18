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
   * Familiarity threshold: if an event's DG code overlaps a stored episode
   * above this, the event is a re-experience of that episode (it is
   * reinforced) instead of a new one.
   */
  noveltyOverlapThreshold: number;
  /**
   * Strength of the cue's persistent drive during completion, as a fraction of
   * the recurrent input a fully supported unit receives (kActive × maxWeight).
   * It must stay below the support that a few correct units give to a missing
   * one (3 units × a single imprint of 0.5 = 1.5), or a degraded cue could no
   * longer be completed; and above the support from a chance overlap of 1–2
   * units with some other episode (≤ 1.0), or novel cues would drift.
   */
  cueDriveFraction: number;
  /**
   * Minimum recurrent input for a unit OUTSIDE the cue to join the recalled
   * pattern, as a fraction of kActive × maxWeight. At 0.15 (3.0 for k = 20) a
   * unit needs the converging support of 4–6 active units of one engram;
   * chance overlaps between codes (1–2 units) never reach it.
   */
  joinSupportFraction: number;
  /** Minimum overlap between an event's own DG code and the episode CA3 completes it to, for it to count as a re-experience. */
  reexperienceOverlap: number;
  /** Longest stretch of input bound into a single episode (ticks). */
  maxEventTicks: number;
  /**
   * Silence that marks an event boundary (ticks). Longer than the longest
   * axonal delay of the connectome, so the brief gaps between the volleys of
   * one perception wave do not split it into several episodes.
   */
  eventGapTicks: number;
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
  cueDriveFraction: 0.0625,
  joinSupportFraction: 0.15,
  reexperienceOverlap: 0.3,
  maxEventTicks: 250,
  eventGapTicks: 25,
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

  /** Last DG code imprinted. */
  private lastStoredCode: Float32Array;

  // --- Event segmentation (one episode per event, not per tick) ---
  /** Input accumulated over the event in progress. */
  private readonly eventInput: Float32Array;
  /** Scratch buffer: running mean of the event in progress. */
  private readonly eventMeanBuf: Float32Array;
  /** Ticks with input in the event in progress (0 = no event). */
  private eventTicks: number = 0;
  /** Consecutive silent ticks since the last input of the event in progress. */
  private silentTicks: number = 0;
  /** Plasticity gain (neuromodulation) seen during the event in progress. */
  private eventLearningGain: number = 1.0;
  /** Affective context reported by the rest of the brain (valence, −1…+1). */
  private affectiveValence: number = 0;

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
    this.eventInput = new Float32Array(inputCount);
    this.eventMeanBuf = new Float32Array(inputCount);
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

    // Feedforward inhibition: remove the common mode of the input.
    // Biology: perforant-path afferents also excite DG interneurons, which
    // subtract the overall level of cortical activity from every granule cell
    // (Ewell & Jones, 2010). Without it, a dense input drives the same granule
    // cells — those whose afferents happen to be mostly excitatory — whatever
    // the pattern is, and different events get heavily overlapping codes.
    let mean = 0;
    for (let i = 0; i < input.length; i++) mean += input[i];
    mean /= Math.max(1, input.length);

    // Fixed random projection + ReLU.
    for (let u = 0; u < n; u++) {
      const base = u * fanIn;
      let sum = 0;
      for (let f = 0; f < fanIn; f++) {
        const idx = this.dgIdx[base + f];
        const v = idx < input.length ? input[idx] - mean : 0;
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
    this.insertEpisode({ pattern: code, context: fullContext, strength: 1.0 });
    return dw;
  }

  /**
   * Adds an episode to the index. At capacity it replaces the weakest trace
   * (interference/forgetting by competition); among equally weak traces, the
   * OLDEST one — otherwise ties always resolve to slot 0 and the index freezes.
   */
  private insertEpisode(memory: EpisodicMemory): void {
    if (this.episodicMemories.length >= this.maxCapacity) {
      let victim = 0;
      for (let i = 1; i < this.episodicMemories.length; i++) {
        const candidate = this.episodicMemories[i];
        const current = this.episodicMemories[victim];
        if (
          candidate.strength < current.strength ||
          (candidate.strength === current.strength &&
            candidate.context.timestamp < current.context.timestamp)
        ) {
          victim = i;
        }
      }
      this.episodicMemories[victim] = memory;
    } else {
      this.episodicMemories.push(memory);
    }
    this.lastStoredCode.set(memory.pattern);
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
    // Initial state = DG code of the cue (may be incomplete/noisy).
    this.dgEncodeInto(partialInput, this.stateBuf);
    return this.completeFromCode(this.stateBuf);
  }

  /**
   * Attractor dynamics from a DG code. `cueCode` may be `this.stateBuf`.
   *
   * The cue keeps driving CA3 while the recurrent dynamics settle (the mossy
   * fibre "detonator" input does not vanish after the first ms): cue units get
   * a constant bonus on top of their recurrent input. A cue inside the basin
   * of a stored episode is still completed — the missing units receive far
   * more recurrent support than the wrong ones receive from the cue — but a
   * NOVEL cue, with no recurrent support, stays itself instead of drifting
   * into whatever attractor happens to share a couple of units with it.
   */
  private completeFromCode(cueCode: Float32Array): Float32Array {
    const n = this.neuronCount;
    const cue = new Float32Array(cueCode);
    const cueDrive = this.cfg.cueDriveFraction * this.kActive;
    const joinSupport = this.cfg.joinSupportFraction * this.kActive;
    let state = this.stateBuf;
    let next = this.nextBuf;
    if (cueCode !== state) state.set(cueCode);

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
        // A unit outside the cue joins only with recurrent support from several
        // active units at once. Two codes share a unit or two by chance; with
        // strong (rehearsed) weights that alone used to outweigh the cue drive
        // and drag a NOVEL cue into a stored episode — false recognition, more
        // likely the more episodes are stored.
        if (cue[i] === 0 && h < joinSupport) h = 0;
        this.actBuf[i] = h + cueDrive * cue[i];
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
      .sort((a, b) => this.replayPriority(b) - this.replayPriority(a))
      .slice(0, count);

    for (const memory of candidates) {
      memory.strength = Math.min(1.0, memory.strength + 0.05);
      this.imprint(memory.pattern, this.cfg.learnRate); // engram reinforcement
    }
    return candidates;
  }

  // ----------------------------------------------------------------
  // Persistence of the episodic index
  // ----------------------------------------------------------------

  /**
   * The episodic index (the CA3 weights are persisted separately, as weights).
   * Each DG code is stored as the list of its active units.
   */
  override serializeExtra(): unknown {
    return {
      episodes: this.episodicMemories.map((memory) => {
        const active: number[] = [];
        for (let i = 0; i < memory.pattern.length; i++) if (memory.pattern[i] > 0) active.push(i);
        return {
          active,
          timestamp: memory.context.timestamp,
          valence: memory.context.emotionalValence,
          source: memory.context.sourceRegion,
          strength: memory.strength,
        };
      }),
    };
  }

  override deserializeExtra(data: unknown): void {
    const episodes = (data as { episodes?: unknown } | null)?.episodes;
    if (!Array.isArray(episodes)) return;

    this.episodicMemories = [];
    for (const raw of episodes.slice(-this.maxCapacity)) {
      const e = raw as { active?: unknown; timestamp?: unknown; valence?: unknown; source?: unknown; strength?: unknown };
      if (!Array.isArray(e?.active) || e.active.length === 0 || e.active.length > this.neuronCount) continue;
      if (!e.active.every((i) => Number.isInteger(i) && i >= 0 && i < this.neuronCount)) continue;
      if (typeof e.strength !== 'number' || !Number.isFinite(e.strength)) continue;

      const pattern = new Float32Array(this.neuronCount);
      for (const i of e.active as number[]) pattern[i] = 1;
      this.episodicMemories.push({
        pattern,
        context: {
          timestamp: typeof e.timestamp === 'number' && Number.isFinite(e.timestamp) ? e.timestamp : 0,
          emotionalValence:
            typeof e.valence === 'number' && Number.isFinite(e.valence) ? Math.max(-1, Math.min(1, e.valence)) : 0,
          sourceRegion: typeof e.source === 'string' ? e.source.slice(0, 40) : 'unknown',
        },
        strength: Math.max(0, Math.min(1, e.strength)),
      });
    }
    const last = this.episodicMemories[this.episodicMemories.length - 1];
    if (last) this.lastStoredCode.set(last.pattern);
  }

  /**
   * Sleep-dependent synaptic downscaling of the CA3 recurrent weights.
   *
   * Biology: synaptic homeostasis (Tononi & Cirelli, 2014) — wakefulness only
   * potentiates, sleep renormalizes. The Hebbian imprint here only ever
   * increases weights, so without downscaling every engram eventually
   * saturates and the episodes merge into one giant attractor that completes
   * any cue to the same pattern. Episodes that are replayed are re-imprinted
   * and survive; the ones never rehearsed fade along with their index entry.
   *
   * @param factor - Multiplicative factor applied to every weight (0–1)
   */
  downscale(factor: number): void {
    if (!(factor > 0 && factor < 1)) return;
    for (let i = 0; i < this.weights.length; i++) this.weights[i] *= factor;
  }

  /**
   * Replay priority: trace strength plus a recency bonus that fades with the
   * episode's age (a raw timestamp term would grow without bound and drown
   * the strength after a few seconds of simulation).
   */
  private replayPriority(memory: EpisodicMemory): number {
    const age = Math.max(0, this.currentTime - memory.context.timestamp);
    return memory.strength + Math.exp(-age / Hippocampus.RECENCY_TAU_MS);
  }

  /** Time constant of the recency bonus in replay selection (simulated ms). */
  private static readonly RECENCY_TAU_MS = 3000;

  /**
   * Gradual forgetting of the episodic index. Emotional memories decay more
   * slowly (β-adrenergic modulation of consolidation, McGaugh 2004).
   * It does not erase recurrent weights: the attractor persists even if the index is lost.
   */
  forget(decayFactor: number): void {
    for (let i = this.episodicMemories.length - 1; i >= 0; i--) {
      const memory = this.episodicMemories[i];
      // Emotional episodes decay SLOWER: up to 30% of the decay is spared.
      const protection = 1 - 0.3 * Math.abs(memory.context.emotionalValence);
      memory.strength *= Math.pow(decayFactor, protection);
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
      // Sustained silence after input = event boundary → encode the episode.
      if (this.eventTicks > 0 && ++this.silentTicks >= this.cfg.eventGapTicks) this.closeEvent();
      this.spikes.fill(0);
      return new Float32Array(this.neuronCount);
    }

    const input = this.adaptInput(spikes);

    // Accumulate the event in progress. The episode is encoded ONCE, at the
    // event boundary, from everything that arrived during it — not tick by
    // tick (the cortical traffic changes every ms, so per-tick encoding stored
    // one stimulus as dozens of near-duplicate episodes).
    for (let i = 0; i < input.length; i++) this.eventInput[i] += input[i];
    this.eventTicks++;
    this.silentTicks = 0;
    this.eventLearningGain = modulationEffects.learningRateMultiplier ?? 1.0;
    let dw = 0;
    if (this.eventTicks >= this.cfg.maxEventTicks) dw = this.closeEvent();

    // Complete via attractor dynamics → reconstructed engram. The cue is the
    // event SO FAR (running mean), not the instantaneous volley: hippocampal
    // activity integrates over the event, so its output is stable while the
    // event unfolds and — if the event is a re-experience — settles on the
    // stored episode, which was imprinted from the same kind of code.
    const eventSoFar = this.eventMeanBuf;
    const ticks = Math.max(1, this.eventTicks);
    for (let i = 0; i < eventSoFar.length; i++) eventSoFar[i] = this.eventInput[i] / ticks;
    const completed = this.eventTicks > 0 ? this.patternCompletion(eventSoFar) : this.patternCompletion(input);

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

  /**
   * Reports the brain's current affective state, stamped on the episodes
   * encoded from now on (emotional episodes are forgotten more slowly).
   */
  setAffectiveContext(valence: number): void {
    if (Number.isFinite(valence)) this.affectiveValence = Math.max(-1, Math.min(1, valence));
  }

  /**
   * Closes the event in progress and encodes it as ONE episode.
   *
   * Biological basis:
   *   Hippocampal encoding is concentrated at event boundaries: activity
   *   peaks when an event ends and binds what happened during it into one
   *   episodic trace (Ben-Yakov & Henson, 2018). If the event matches a stored
   *   episode it is a re-experience: the existing trace is strengthened
   *   (reconsolidation) rather than duplicated.
   *
   * @returns Σ|Δw| applied
   */
  private closeEvent(): number {
    const mean = new Float32Array(this.eventInput.length);
    for (let i = 0; i < mean.length; i++) mean[i] = this.eventInput[i] / this.eventTicks;
    this.eventInput.fill(0);
    this.eventTicks = 0;
    this.silentTicks = 0;

    const code = this.patternSeparation(mean);

    // Familiarity is judged on what CA3 RECALLS from the event's code, not on
    // the raw code: a re-experience never reproduces the first code exactly
    // (the DG amplifies small differences — that is its job), but it falls in
    // the stored episode's basin and completes to it.
    const recalled = this.completeFromCode(code);
    let best: EpisodicMemory | null = null;
    let bestOverlap = 0;
    for (const memory of this.episodicMemories) {
      const overlap = Hippocampus.overlapBinary(recalled, memory.pattern);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = memory;
      }
    }

    // …and the event's own code must resemble that episode too: a novel code
    // that merely falls into a well-rehearsed basin is a new experience, not a
    // re-experience.
    if (
      best &&
      bestOverlap >= this.cfg.noveltyOverlapThreshold &&
      Hippocampus.overlapBinary(code, best.pattern) >= this.cfg.reexperienceOverlap
    ) {
      best.strength = Math.min(1.0, best.strength + 0.1);
      best.context.timestamp = this.currentTime;
      return this.imprint(best.pattern, this.cfg.learnRate * this.eventLearningGain * 0.5);
    }

    const dw = this.imprint(code, this.cfg.learnRate * this.eventLearningGain);
    this.insertEpisode({
      pattern: code,
      context: {
        timestamp: this.currentTime,
        emotionalValence: this.affectiveValence,
        sourceRegion: 'sensory',
      },
      strength: 1.0,
    });
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
