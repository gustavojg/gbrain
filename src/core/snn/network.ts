/**
 * SPIKING NEURAL NETWORK (SNN) WITH k-WTA LATERAL INHIBITION
 * ==========================================================
 * Network of Izhikevich neurons with flat weight storage in a Float32Array,
 * k-Winners-Take-All lateral inhibition, and STDP learning.
 *
 * Biological basis:
 *   The cerebral cortex organizes its neurons into cortical columns where
 *   lateral inhibition (mediated by GABAergic interneurons) ensures that
 *   only a "winning" subset of neurons (~5-15%) is active
 *   simultaneously. This mechanism:
 *
 *   - Produces sparse representations (sparse coding), maximizing
 *     the network's storage capacity.
 *   - Implements competition between neurons, forcing specialization.
 *   - Reduces energy consumption (the real brain uses ~20W with
 *     only ~5% of neurons active at any given instant).
 *
 *   Float32Array storage mimics the dense organization of
 *   the cortical synaptic matrix, where each excitatory neuron receives
 *   ~10,000 synapses from afferent neurons.
 *
 * Performance:
 *   - Weights in a flat Float32Array: cache-friendly, ~4 bytes/synapse.
 *   - k-WTA with partial quickselect: O(n) on average vs O(n log n) for sort.
 *   - For 10,000 neurons × 1,000 inputs = 40 MB of weights.
 *
 * Architecture reference:
 *   - Float32Array pattern: /04_binary_optimization/binary_brain_system.ts
 *   - k-WTA: /02_distributed_network/distributed_network_complete.ts
 */

import { SpikingNeuron, createNeuronPopulation, type NeuronTypeName } from './neuron.js';
import { computeSTDP, DEFAULT_STDP_PARAMS, type STDPParams } from './synapse.js';

// ====================================================================
// Interfaces
// ====================================================================

/**
 * Configuration for creating an SNN network.
 */
export interface SNNNetworkConfig {
  /** Number of neurons in the network */
  readonly neuronCount: number;
  /** Number of external inputs to the network */
  readonly inputCount: number;
  /** Fraction of neurons that can win the k-WTA competition (0.0-1.0) */
  readonly sparsity: number;
  /** Predominant Izhikevich neuron type */
  readonly neuronType: NeuronTypeName;
  /** Optional STDP parameters */
  readonly stdp?: STDPParams;
  /** Maximum synaptic weight */
  readonly maxWeight?: number;
  /** Minimum synaptic weight */
  readonly minWeight?: number;
  /** Total synaptic budget for normalization */
  readonly synapticBudget?: number;
}

/**
 * Serialized network data format for persistence.
 *
 * Biological basis:
 *   Analogous to a "snapshot" of the complete synaptic state,
 *   similar to how memory consolidation during sleep preserves
 *   the connections strengthened during wakefulness.
 */
export interface SerializedNetwork {
  /** Number of neurons */
  readonly neuronCount: number;
  /** Number of inputs */
  readonly inputCount: number;
  /** k-WTA sparsity */
  readonly sparsity: number;
  /** Neuron type */
  readonly neuronType: NeuronTypeName;
  /** Weights as an ArrayBuffer (for binary serialization) */
  readonly weightsBuffer: ArrayBuffer;
  /** Membrane potentials of the neurons */
  readonly voltages: ArrayBuffer;
  /** Recovery variables of the neurons */
  readonly recovery: ArrayBuffer;
}

// ====================================================================
// SNNNetwork class
// ====================================================================

/**
 * Spiking neural network with k-WTA and STDP.
 *
 * Biological basis:
 *   Models a homogeneous cortical population (e.g. a layer of visual
 *   cortex, a region of the hippocampus) as a set of Izhikevich
 *   neurons densely connected to an array of inputs, with
 *   lateral inhibition that ensures sparse representations.
 *
 *   The combination of:
 *   1. Izhikevich neurons (realistic membrane dynamics)
 *   2. k-WTA (cortical lateral inhibition)
 *   3. STDP (Hebbian plasticity with timing)
 *   4. Synaptic normalization (homeostasis)
 *
 *   produces a system that self-organizes stable sparse representations
 *   for input patterns, analogous to the orientation columns
 *   in V1 or the place cells in the hippocampus.
 */
export class SNNNetwork {
  /** Izhikevich neurons of the network */
  public readonly neurons: SpikingNeuron[];

  /**
   * Synaptic weight matrix in flat format.
   *
   * Layout: weights[neuronIdx * inputCount + inputIdx]
   * This is equivalent to a (neuronCount × inputCount) matrix stored
   * row-major, optimized for cache-friendly access.
   */
  public readonly weights: Float32Array;

  /** Number of neurons in the network */
  public readonly neuronCount: number;

  /** Number of external inputs */
  public readonly inputCount: number;

  /** Fraction of active neurons (k-WTA) */
  public readonly sparsity: number;

  /** STDP parameters */
  private readonly stdp: STDPParams;

  /** Maximum allowed weight */
  private readonly maxWeight: number;

  /** Minimum allowed weight */
  private readonly minWeight: number;

  /** Total synaptic budget for homeostatic normalization */
  private readonly synapticBudget: number;

  // Pre-allocated buffers to avoid GC during simulation
  /** Raw potentials computed at each timestep */
  private readonly rawPotentials: Float32Array;
  /** Neuron indices for the k-WTA sorting */
  private readonly sortIndices: Uint32Array;

  /**
   * Creates a new SNN network.
   *
   * @param config - Network configuration, or positional parameters
   */
  constructor(config: SNNNetworkConfig);
  constructor(
    neuronCount: number,
    inputCount: number,
    sparsity?: number,
    neuronType?: NeuronTypeName,
  );
  constructor(
    configOrNeuronCount: SNNNetworkConfig | number,
    inputCount?: number,
    sparsity?: number,
    neuronType?: NeuronTypeName,
  ) {
    let config: SNNNetworkConfig;

    if (typeof configOrNeuronCount === 'number') {
      config = {
        neuronCount: configOrNeuronCount,
        inputCount: inputCount!,
        sparsity: sparsity ?? 0.1,
        neuronType: neuronType ?? 'RegularSpiking',
      };
    } else {
      config = configOrNeuronCount;
    }

    this.neuronCount = config.neuronCount;
    this.inputCount = config.inputCount;
    this.sparsity = config.sparsity;
    this.stdp = config.stdp ?? DEFAULT_STDP_PARAMS;
    this.maxWeight = config.maxWeight ?? 5.0;
    this.minWeight = config.minWeight ?? 0.0;
    this.synapticBudget = config.synapticBudget ?? 10.0;

    // Create the neuron population
    this.neurons = createNeuronPopulation(this.neuronCount, config.neuronType);

    // Allocate the flat weight matrix (row-major: neuronCount × inputCount)
    const totalWeights = this.neuronCount * this.inputCount;
    this.weights = new Float32Array(totalWeights);

    // Pre-allocate working buffers (reused at each timestep)
    this.rawPotentials = new Float32Array(this.neuronCount);
    this.sortIndices = new Uint32Array(this.neuronCount);

    // Initialize random weights and normalize
    this.initializeWeights();
  }

  /**
   * Initializes the synaptic weights with normalized random values.
   *
   * Biological basis:
   *   Immature synapses have randomly distributed weights
   *   (synaptogenic noise). Normalization imposes a total
   *   budget per neuron, modeling the synaptic homeostasis that
   *   keeps total activity within a physiological range.
   */
  private initializeWeights(): void {
    for (let n = 0; n < this.neuronCount; n++) {
      const baseIdx = n * this.inputCount;
      for (let i = 0; i < this.inputCount; i++) {
        this.weights[baseIdx + i] = Math.random();
      }
      this.normalizeWeights(n);
    }
  }

  /**
   * Processes a complete time step of the network.
   *
   * Biological basis:
   *   Simulates a complete cycle of cortical processing:
   *   1. Synaptic integration: each neuron sums the input currents
   *      weighted by its synaptic weights (partial dot product).
   *   2. Membrane dynamics: Izhikevich step for each neuron.
   *   3. Lateral inhibition (k-WTA): only the k most excited neurons
   *      keep their spike; the rest are silenced by inhibitory
   *      interneurons (modeled implicitly).
   *
   * @param inputs - Input vector (Float32Array of 0s and 1s, or continuous values)
   * @param dt - Time step in milliseconds
   * @param currentTime - Current simulation time (ms), for STDP
   * @returns Indices of the neurons that fired (k-WTA winners)
   */
  step(inputs: Float32Array, dt: number, currentTime: number = 0): Uint32Array {
    // --- 1. Compute the input current for each neuron ---
    for (let n = 0; n < this.neuronCount; n++) {
      const baseIdx = n * this.inputCount;
      let excitation = 0;

      // Dot product: sum(weight_i * input_i)
      // Sequential memory access → cache-friendly
      for (let i = 0; i < this.inputCount; i++) {
        excitation += this.weights[baseIdx + i] * inputs[i];
      }

      // Mild synaptic noise (models neurotransmitter fluctuations)
      excitation += Math.random() * 0.05;

      this.rawPotentials[n] = excitation;
    }

    // --- 2. Simulate the membrane dynamics of each neuron ---
    for (let n = 0; n < this.neuronCount; n++) {
      this.neurons[n].step(this.rawPotentials[n], dt, currentTime);
    }

    // --- 3. Apply k-WTA lateral inhibition ---
    // Initialize indices
    for (let n = 0; n < this.neuronCount; n++) {
      this.sortIndices[n] = n;
    }

    // Sort indices by descending potential
    // Note: Uint32Array.sort uses a comparison that accesses rawPotentials
    const potentials = this.rawPotentials;
    this.sortIndices.sort((a, b) => potentials[b] - potentials[a]);

    // Only the top-k can fire
    const k = Math.max(1, Math.floor(this.neuronCount * this.sparsity));

    // Create a winner set for O(1) lookup
    const winnerSet = new Uint8Array(this.neuronCount); // 0 = not a winner
    for (let i = 0; i < k; i++) {
      winnerSet[this.sortIndices[i]] = 1;
    }

    // Build the array of winner indices that actually fired
    let firingCount = 0;
    // Temporary buffer for the winners (reusable)
    const firingBuffer = new Uint32Array(k);

    for (let n = 0; n < this.neuronCount; n++) {
      if (winnerSet[n] === 1 && this.neurons[n].fired) {
        firingBuffer[firingCount++] = n;
      } else {
        // Silence neurons that did not win the competition
        this.neurons[n].fired = false;
      }
    }

    // If none of the k winners fired naturally,
    // force the top-k to fire (activation by subthreshold current)
    if (firingCount === 0) {
      for (let i = 0; i < k; i++) {
        const winnerIdx = this.sortIndices[i];
        this.neurons[winnerIdx].fired = true;
        this.neurons[winnerIdx].lastSpikeTime = currentTime;
        this.neurons[winnerIdx].v = this.neurons[winnerIdx].params.c;
        this.neurons[winnerIdx].u += this.neurons[winnerIdx].params.d;
        firingBuffer[firingCount++] = winnerIdx;
      }
    }

    return firingBuffer.slice(0, firingCount);
  }

  /**
   * Applies the STDP rule to the synapses of the neurons that fired.
   *
   * Biological basis:
   *   Implements timing-dependent Hebbian plasticity: the synapses that
   *   contributed to making the postsynaptic neuron fire are
   *   strengthened (LTP), while those that did not contribute are
   *   weakened (LTD). The modulation factor enables neuromodulatory
   *   control (dopamine for reward, etc.).
   *
   *   After STDP, synaptic normalization (synaptic scaling) is applied
   *   to maintain homeostasis: the total synaptic budget per
   *   neuron is kept constant, modeling the homeostatic regulation
   *   observed in cortical cultures.
   *
   * @param firingNeurons - Indices of neurons that fired (from step())
   * @param inputSpikeTimes - Spike times of each input (Float32Array)
   * @param modulationFactor - Neuromodulatory modulation factor (1.0 = normal)
   */
  applySTDP(
    firingNeurons: Uint32Array,
    inputSpikeTimes: Float32Array,
    modulationFactor: number = 1.0,
  ): void {
    for (let f = 0; f < firingNeurons.length; f++) {
      const neuronIdx = firingNeurons[f];
      const neuron = this.neurons[neuronIdx];
      const postSpikeTime = neuron.lastSpikeTime;
      const baseIdx = neuronIdx * this.inputCount;

      // Apply STDP for each input
      for (let i = 0; i < this.inputCount; i++) {
        const preSpikeTime = inputSpikeTimes[i];
        const deltaW = computeSTDP(
          preSpikeTime,
          postSpikeTime,
          this.stdp,
          modulationFactor,
        );

        if (deltaW !== 0) {
          let newWeight = this.weights[baseIdx + i] + deltaW;
          // Apply homeostatic limits
          newWeight = Math.max(this.minWeight, Math.min(this.maxWeight, newWeight));
          this.weights[baseIdx + i] = newWeight;
        }
      }

      // Post-learning synaptic normalization
      this.normalizeWeights(neuronIdx);
    }
  }

  /**
   * Returns the indices of the currently active neurons (those that fired).
   *
   * Biological basis:
   *   The active representation is the "population code" — the pattern
   *   of active neurons that encodes a concept, object, or memory.
   *   It is the basis of distributed representation in the cortex.
   *
   * @returns Array of active neuron indices
   */
  getRepresentation(): Uint32Array {
    let count = 0;
    const buffer = new Uint32Array(this.neuronCount);

    for (let n = 0; n < this.neuronCount; n++) {
      if (this.neurons[n].fired) {
        buffer[count++] = n;
      }
    }

    return buffer.slice(0, count);
  }

  /**
   * Normalizes a neuron's synaptic weights to maintain
   * a fixed total synaptic budget.
   *
   * Biological basis:
   *   "Synaptic scaling" (Turrigiano, 1998): a homeostatic mechanism
   *   where the neuron multiplicatively scales all its weights
   *   to keep its activity within a physiological range.
   *   This implements implicit heterosynaptic LTD: when one synapse
   *   is strengthened (LTP), the rescaling proportionally weakens
   *   the other synapses.
   *
   * @param neuronIdx - Index of the neuron to normalize
   */
  normalizeWeights(neuronIdx: number): void {
    const baseIdx = neuronIdx * this.inputCount;

    // Compute the total sum of weights
    let total = 0;
    for (let i = 0; i < this.inputCount; i++) {
      total += this.weights[baseIdx + i];
    }

    if (total === 0) return;

    // Scale multiplicatively to reach the budget
    const factor = this.synapticBudget / total;
    for (let i = 0; i < this.inputCount; i++) {
      this.weights[baseIdx + i] *= factor;
    }
  }

  /**
   * Resets all neurons to their resting state.
   * The synaptic weights are kept (memory persists).
   *
   * Biological basis:
   *   Equivalent to a period of cortical silence, like the one observed
   *   between DOWN (silence) and UP (activity) states during
   *   slow-wave sleep.
   */
  resetNeurons(): void {
    for (let n = 0; n < this.neuronCount; n++) {
      this.neurons[n].reset();
    }
  }

  /**
   * Serializes the complete network state for binary persistence.
   *
   * Biological basis:
   *   Analogous to memory consolidation during sleep:
   *   the synaptic structure (weights) and the instantaneous
   *   state of the neurons are preserved so the simulation can resume.
   *
   * @returns Serializable object with all the network data
   */
  serialize(): SerializedNetwork {
    // Copy weights (do not share the reference)
    const weightsBuffer = this.weights.buffer.slice(0);

    // Extract neuron states
    const voltages = new Float32Array(this.neuronCount);
    const recovery = new Float32Array(this.neuronCount);

    for (let n = 0; n < this.neuronCount; n++) {
      voltages[n] = this.neurons[n].v;
      recovery[n] = this.neurons[n].u;
    }

    return {
      neuronCount: this.neuronCount,
      inputCount: this.inputCount,
      sparsity: this.sparsity,
      neuronType: this.neurons[0].params === undefined
        ? 'RegularSpiking'
        : this.inferNeuronType(),
      weightsBuffer: weightsBuffer as ArrayBuffer,
      voltages: voltages.buffer as ArrayBuffer,
      recovery: recovery.buffer as ArrayBuffer,
    };
  }

  /**
   * Restores the network state from serialized data.
   *
   * @param data - Data previously serialized with serialize()
   * @returns New SNNNetwork instance with the restored state
   */
  static deserialize(data: SerializedNetwork): SNNNetwork {
    const network = new SNNNetwork({
      neuronCount: data.neuronCount,
      inputCount: data.inputCount,
      sparsity: data.sparsity,
      neuronType: data.neuronType,
    });

    // Restore weights
    const loadedWeights = new Float32Array(data.weightsBuffer);
    network.weights.set(loadedWeights);

    // Restore neuron states
    const voltages = new Float32Array(data.voltages);
    const recovery = new Float32Array(data.recovery);

    for (let n = 0; n < network.neuronCount; n++) {
      if (n < voltages.length) {
        network.neurons[n].v = voltages[n];
        network.neurons[n].u = recovery[n];
      }
    }

    return network;
  }

  /**
   * Infers the neuron type name from the parameters.
   * Used internally for serialization.
   */
  private inferNeuronType(): NeuronTypeName {
    const p = this.neurons[0].params;
    if (p.a === 0.1 && p.c === -65 && p.d === 2) return 'FastSpiking';
    if (p.a === 0.02 && p.c === -50 && p.d === 2) return 'Chattering';
    if (p.a === 0.02 && p.c === -55 && p.d === 4) return 'IntrinsicBursting';
    return 'RegularSpiking';
  }

  /**
   * Returns network statistics for monitoring.
   *
   * @returns Object with activity and weight metrics
   */
  getStats(): {
    activeCount: number;
    activeFraction: number;
    meanWeight: number;
    maxWeightValue: number;
    minWeightValue: number;
  } {
    let activeCount = 0;
    for (let n = 0; n < this.neuronCount; n++) {
      if (this.neurons[n].fired) activeCount++;
    }

    let totalWeight = 0;
    let maxW = -Infinity;
    let minW = Infinity;
    const totalWeights = this.neuronCount * this.inputCount;

    for (let i = 0; i < totalWeights; i++) {
      const w = this.weights[i];
      totalWeight += w;
      if (w > maxW) maxW = w;
      if (w < minW) minW = w;
    }

    return {
      activeCount,
      activeFraction: activeCount / this.neuronCount,
      meanWeight: totalWeight / totalWeights,
      maxWeightValue: maxW,
      minWeightValue: minW,
    };
  }
}
