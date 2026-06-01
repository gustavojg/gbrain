/**
 * THALAMUS — Attentional Filter and Sensory Relay
 * =============================================
 * Refactored from 05_attention_scale/attention_brain_system.ts
 *
 * Biological basis:
 *   The thalamus is the brain's central "relay station". Almost all
 *   sensory information (except smell) passes through the thalamus before
 *   reaching the cortex. It is not a passive relay: it actively filters signals,
 *   selecting the most salient ones according to the attentional state.
 *
 *   The thalamic reticular nucleus (TRN) acts as an inhibitory gate,
 *   letting only the strongest or most relevant signals pass. The
 *   acetylcholine from the basal nucleus of Meynert modulates this gate:
 *   more ACh → more permissive gate → more inputs pass to the cortex.
 *
 *   We implement this as a top-K filter with a noise threshold,
 *   where K (the bottleneck) is modulated by acetylcholine levels.
 *
 * Scalability:
 *   Designed to handle 50,000+ sensory inputs and filter them
 *   down to ~200 salient signals using a threshold pass + partial sort,
 *   avoiding the O(N log N) cost in the average case.
 */

// ====================================================================
// Core imports (existing in the project)
// ====================================================================

import type { SpikePacket } from '../../core/bus/spike-bus.js';
import { SpikeBus } from '../../core/bus/spike-bus.js';
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { SpikingNeuron, createNeuronPopulation } from '../../core/snn/neuron.js';
import type { NeuronTypeName } from '../../core/snn/neuron.js';

// ====================================================================
// Thalamus types
// ====================================================================

/**
 * Sensory signal type for routing.
 * The thalamus routes signals to different cortices according to modality.
 */
export type SensoryModality = 'visual' | 'auditory' | 'linguistic';

/**
 * Result of the thalamic processing.
 * Contains the salient indices and the filtered vector.
 */
export interface ThalamusOutput {
  /** Indices of the most salient selected inputs */
  salientIndices: Int32Array;
  /** Filtered vector with only the salient values (rest at 0) */
  filteredInput: Float32Array;
  /** Number of inputs that exceeded the noise threshold */
  candidateCount: number;
  /** Effective bottleneck size (modulated by ACh) */
  effectiveBottleneck: number;
}

/**
 * Thalamus-specific configuration.
 */
export interface ThalamusConfig {
  /** Number of thalamic neurons */
  neuronCount: number;
  /** Total size of the sensory input vector */
  totalInputSize: number;
  /** Maximum number of signals that pass the attentional filter */
  bottleneckSize: number;
  /** Background noise threshold (signals below are ignored) */
  noiseFloor: number;
  /** Neuron type (default: FastSpiking for fast response) */
  neuronType: NeuronTypeName;
  /** Encoding sparsity (fraction of active neurons) */
  sparsity: number;
}

const DEFAULT_THALAMUS_CONFIG: ThalamusConfig = {
  neuronCount: 3000,
  totalInputSize: 50000,
  bottleneckSize: 200,
  noiseFloor: 0.1,
  neuronType: 'FastSpiking',
  sparsity: 0.05,
};

// ====================================================================
// Thalamus class
// ====================================================================

/**
 * Thalamus — Attentional filter and sensory relay of the digital brain.
 *
 * Biological basis:
 *   The thalamus receives ~50,000 simultaneous sensory inputs and filters them
 *   down to ~200 salient signals that are sent to the specialized cortices.
 *   This filtering is modulated by:
 *
 *   1. **Acetylcholine** (ACh): Increases the bottleneck → more information passes.
 *      Biologically, ACh from the basal nucleus facilitates thalamic transmission
 *      by depolarizing relay neurons and inhibiting the TRN.
 *
 *   2. **Attentional gain** (NE + ACh): Amplifies the salient signals.
 *      Norepinephrine from the locus coeruleus increases the signal/noise ratio.
 *
 *   3. **Noise threshold**: Filters subliminal signals (< 0.1).
 *      Models the firing threshold of the TRN.
 *
 * Performance:
 *   For 50K inputs, it uses an O(N) threshold pass followed by an O(M log M) sort
 *   where M << N (only suprathreshold candidates), avoiding a full sort.
 */
export class Thalamus extends BrainRegion {
  /** Base maximum number of signals that pass the filter */
  private attentionBottleneck: number;

  /** Noise threshold — signals below are considered background noise */
  private noiseFloor: number;

  /** Sparsity of the internal encoding */
  private thalamusSparsity: number;

  /** Reference to the spike bus to route signals to cortices */
  private spikeBus: SpikeBus | null = null;

  /** Last computed output (for external queries) */
  private lastOutput: ThalamusOutput | null = null;

  /** Local spiking neurons of the thalamus (Izhikevich model) */
  private localNeurons: SpikingNeuron[];

  /**
   * Creates a new thalamus instance.
   *
   * @param config - Partial configuration (merged with defaults)
   */
  constructor(config: Partial<ThalamusConfig> = {}) {
    const cfg = { ...DEFAULT_THALAMUS_CONFIG, ...config };
    super('thalamus', 'Tálamo', cfg.neuronCount, cfg.totalInputSize);

    this.attentionBottleneck = cfg.bottleneckSize;
    this.noiseFloor = cfg.noiseFloor;
    this.thalamusSparsity = cfg.sparsity;
    this.localNeurons = createNeuronPopulation(cfg.neuronCount, cfg.neuronType);

    this.initializeThalamusWeights();
  }

  /**
   * Initializes the synaptic weights with sparse connectivity.
   *
   * Biological basis:
   *   Thalamic relay neurons are not connected to ALL inputs,
   *   rather each neuron receives afferents from a subset of sensory
   *   fibers. We initialize ~1000 random connections per neuron.
   */
  private initializeThalamusWeights(): void {
    const INITIAL_CONNECTIONS = Math.min(1000, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      const offset = n * this.inputCount;
      for (let k = 0; k < INITIAL_CONNECTIONS; k++) {
        const randIdx = Math.floor(Math.random() * this.inputCount);
        this.weights[offset + randIdx] = Math.random() * 0.1;
      }
    }
  }

  /**
   * Connects the thalamus to the spike bus for inter-regional routing.
   *
   * @param bus - Instance of the central SpikeBus
   */
  connectBus(bus: SpikeBus): void {
    this.spikeBus = bus;
    bus.register(this.id);
  }

  /**
   * Filters a massive sensory input vector down to the top-K salient signals.
   *
   * Biological basis:
   *   The thalamic reticular nucleus (TRN) inhibits relay neurons
   *   whose inputs are weak, letting only the signals that
   *   exceed a threshold pass. Acetylcholine modulates this threshold:
   *   - High ACh → lower threshold → more signals pass (broad attention)
   *   - Low ACh → higher threshold → fewer signals (focused attention)
   *
   *   This process is analogous to lateral inhibition in the TRN,
   *   implemented here as thresholding + top-K selection.
   *
   * @param rawInputs - Complete sensory input vector (50K+ values, 0.0-1.0)
   * @param modulationEffects - Current neuromodulation effects
   * @returns Result of the attentional filter
   */
  processAttention(rawInputs: Float32Array, modulationEffects?: ModulationEffects): ThalamusOutput {
    // --- Bottleneck modulation by acetylcholine ---
    // High ACh → high attentionGain → larger K → more inputs pass
    const achGain = modulationEffects?.attentionGain ?? 1.0;
    const effectiveK = Math.floor(this.attentionBottleneck * achGain);

    // --- Pass 1: Noise threshold (O(N), filters ~90% of the inputs) ---
    // Structure for candidates: index + value for sort
    const candidateIndices = new Int32Array(rawInputs.length);
    const candidateValues = new Float32Array(rawInputs.length);
    let candidateCount = 0;

    for (let i = 0; i < rawInputs.length; i++) {
      if (rawInputs[i] > this.noiseFloor) {
        candidateIndices[candidateCount] = i;
        candidateValues[candidateCount] = rawInputs[i];
        candidateCount++;
      }
    }

    // --- Pass 2: Top-K selection (partial sort of M candidates) ---
    // Create an index array for sorting (only the M candidates)
    const sortIndices = new Int32Array(candidateCount);
    for (let i = 0; i < candidateCount; i++) sortIndices[i] = i;

    // Descending sort by value
    sortIndices.sort((a, b) => candidateValues[b] - candidateValues[a]);

    // --- Build the filtered output ---
    const k = Math.min(effectiveK, candidateCount);
    const salientIndices = new Int32Array(k);
    const filteredInput = new Float32Array(rawInputs.length);

    for (let i = 0; i < k; i++) {
      const candidateIdx = sortIndices[i];
      const originalIdx = candidateIndices[candidateIdx];
      salientIndices[i] = originalIdx;

      // Scale by attentional gain (NE + ACh amplify salient signals)
      const spikeGain = modulationEffects?.spikeGainMultiplier ?? 1.0;
      filteredInput[originalIdx] = rawInputs[originalIdx] * spikeGain;
    }

    this.lastOutput = {
      salientIndices,
      filteredInput,
      candidateCount,
      effectiveBottleneck: effectiveK,
    };

    return this.lastOutput;
  }

  /**
   * Determines which cortical region to route the signal to according to its modality.
   *
   * Biological basis:
   *   The thalamus has specialized nuclei for each modality:
   *   - Lateral geniculate nucleus (LGN) → Visual cortex V1
   *   - Medial geniculate nucleus (MGN) → Auditory cortex A1
   *   - Ventral posterior nucleus → Somatosensory cortex
   *   - Pulvinar → Association areas (here: Broca-Wernicke)
   *
   * @param inputType - Modality of the sensory signal
   * @returns Identifier of the destination region
   */
  routeToRegion(inputType: SensoryModality): string {
    switch (inputType) {
      case 'visual':     return 'visualCortex';
      case 'auditory':   return 'auditoryCortex';
      case 'linguistic': return 'brocaWernicke';
      default:           return 'visualCortex';
    }
  }

  /**
   * Processes a complete sensory input: filters and routes.
   *
   * Biological basis:
   *   Complete pipeline of the thalamic relay:
   *   1. Receives raw sensory input (retina, cochlea, etc.)
   *   2. Applies attentional filter (TRN + ACh modulation)
   *   3. Processes through relay neurons (internal SNN)
   *   4. Sends the resulting spikes to cortices via axonal tracts
   *
   * @param input - Raw sensory input vector
   * @param dt - Time step (ms)
   * @param timestamp - Current simulation time (ms)
   * @returns Vector filtered by attention
   */
  processInput(spikes: Float32Array, _modulationEffects: ModulationEffects): Float32Array {
    // 1. Apply attentional filter (without modulation for now)
    const attentionResult = this.processAttention(spikes);

    // 2. Process through thalamic neurons (sparse computation)
    const localPotentials = new Float32Array(this.neuronCount);

    for (let n = 0; n < this.neuronCount; n++) {
      let excitation = 0;
      const offset = n * this.inputCount;

      // Sparse dot product: only compute for salient indices
      for (let i = 0; i < attentionResult.salientIndices.length; i++) {
        const idx = attentionResult.salientIndices[i];
        excitation += this.weights[offset + idx] * attentionResult.filteredInput[idx];
      }

      // Neuronal noise (intrinsic variability)
      excitation += Math.random() * 0.01;
      localPotentials[n] = excitation;
    }

    // 3. Winner-Take-All (lateral inhibition via TRN)
    const k = Math.max(1, Math.floor(this.neuronCount * this.thalamusSparsity));
    const indices = new Int32Array(this.neuronCount);
    for (let i = 0; i < this.neuronCount; i++) indices[i] = i;
    indices.sort((a, b) => localPotentials[b] - localPotentials[a]);

    // Activate only the top-k neurons
    const activeSpikes = new Float32Array(this.neuronCount);
    for (let i = 0; i < k; i++) {
      const winnerIdx = indices[i];
      this.localNeurons[winnerIdx].fired = true;
      activeSpikes[winnerIdx] = 1.0;
    }

    // Reset non-winners
    for (let i = k; i < this.neuronCount; i++) {
      this.localNeurons[indices[i]].fired = false;
    }

    return activeSpikes;
  }

  /**
   * Processes and sends the filtered signal to a specific cortex via the spike bus.
   *
   * @param rawInput - Raw sensory input
   * @param modality - Signal type (visual, auditory, linguistic)
   * @param dt - Time step
   * @param timestamp - Simulation time
   * @param modulationEffects - Optional neuromodulator effects
   */
  processAndRoute(
    rawInput: Float32Array,
    modality: SensoryModality,
    dt: number,
    timestamp: number,
    modulationEffects?: ModulationEffects,
  ): ThalamusOutput {
    // 1. Attentional filter with modulation
    const attentionResult = this.processAttention(rawInput, modulationEffects);

    // 2. Process thalamic neurons
    const defaultMod: ModulationEffects = {
      learningRateMultiplier: 1.0,
      thresholdMultiplier: 1.0,
      attentionGain: 1.0,
      spikeGainMultiplier: 1.0,
      consolidationRate: 1.0,
      socialWeightBoost: 1.0,
    };
    const activeSpikes = this.processInput(rawInput, modulationEffects ?? defaultMod);

    // 3. Route via spike bus
    if (this.spikeBus) {
      const targetRegion = this.routeToRegion(modality);

      const packet: SpikePacket = {
        source: this.id,
        targets: [targetRegion],
        spikes: attentionResult.filteredInput,
        timestamp,
        metadata: {
          modality,
          salientCount: attentionResult.salientIndices.length,
          effectiveBottleneck: attentionResult.effectiveBottleneck,
        },
      };

      this.spikeBus.send(packet);
    }

    return attentionResult;
  }

  /**
   * Sends a broadcast to multiple cortices simultaneously.
   *
   * Biological basis:
   *   Some stimuli (e.g. a sudden loud sound) activate
   *   multiple thalamic pathways simultaneously, including the
   *   fast thalamus→amygdala pathway for an immediate emotional response.
   *
   * @param rawInput - Sensory input
   * @param targets - List of destination regions
   * @param timestamp - Simulation time
   */
  broadcastToRegions(
    rawInput: Float32Array,
    targets: string[],
    timestamp: number,
  ): void {
    if (!this.spikeBus) return;

    const attentionResult = this.processAttention(rawInput);

    const packet: SpikePacket = {
      source: this.id,
      targets,
      spikes: attentionResult.filteredInput,
      timestamp,
      metadata: {
        broadcast: true,
        salientCount: attentionResult.salientIndices.length,
      },
    };

    this.spikeBus.send(packet);
  }

  /**
   * Gets the last result of the attentional filter.
   */
  getLastOutput(): ThalamusOutput | null {
    return this.lastOutput;
  }

  /**
   * Gets statistics of the thalamus for monitoring.
   */
  /**
   * Gets the fraction of local activity (0-1) based on the spiking neurons.
   */
  getLocalActivity(): number {
    let active = 0;
    for (let i = 0; i < this.localNeurons.length; i++) {
      if (this.localNeurons[i].fired) active++;
    }
    return active / this.localNeurons.length;
  }

  getStats(): {
    bottleneck: number;
    noiseFloor: number;
    lastCandidateCount: number;
    lastEffectiveK: number;
    neuronActivity: number;
  } {
    return {
      bottleneck: this.attentionBottleneck,
      noiseFloor: this.noiseFloor,
      lastCandidateCount: this.lastOutput?.candidateCount ?? 0,
      lastEffectiveK: this.lastOutput?.effectiveBottleneck ?? this.attentionBottleneck,
      neuronActivity: this.getLocalActivity(),
    };
  }
}
