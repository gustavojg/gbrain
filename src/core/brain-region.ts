/**
 * Abstract Base Class for Brain Regions
 * ================================================
 * Defines the common interface for all functional regions of the digital brain.
 *
 * Biology: Each brain region (visual cortex, hippocampus, amygdala, etc.)
 * contains a network of specialized neurons with particular connectivity
 * patterns and dynamics. However, they all share a common architecture:
 * they receive afferent signals, process them through a local SNN, and emit
 * efferent signals. This abstract class captures that shared structure.
 *
 * The concrete subclasses (VisualCortex, Hippocampus, etc.) implement the
 * processInput() method with the specific logic of each region.
 *
 * Note: This class references SNNNetwork as a generic type. The concrete
 * implementations may use BinaryBiologicalNetwork, LargeScaleNetwork, or any
 * compatible variant.
 */

import { SensoryBuffer, type SensoryEntry } from './memory/sensory-buffer.js';
import type { ModulationEffects } from './neuromodulators/modulator-system.js';

/**
 * Instantaneous activity of a brain region.
 * Result of processing one simulation step.
 */
export interface RegionActivity {
  /** Region identifier */
  id: string;
  /** Indices of active neurons (that fired a spike) in this step */
  activeNeurons: number[];
  /** Average firing rate (Hz) of the region */
  firingRate: number;
  /**
   * Perceived activation level (EMA, 0..1). Unlike `firingRate`
   * —which with sparse k-WTA coding is constant (= sparsity) and therefore
   * uninformative— this value reflects how much signal the region is actually
   * receiving (input drive energy) smoothed over time. It is what the
   * dashboard uses for the "ACTIVIDAD POR REGIÓN" bars.
   */
  drive: number;
  /**
   * Novelty of the firing pattern (EMA, 0..1). ~0 when the region repeats the
   * same pattern (rest / stable attractor) and rises when a stimulus changes
   * WHICH neurons fire. It is the reactive signal of the activity panel:
   * unlike `firingRate` (constant with k-WTA), it responds to interaction.
   */
  novelty: number;
  /** Timestamp of the simulation step */
  timestamp: number;
  /** Output spikes to transmit to other regions */
  outputSpikes: Float32Array;
}

/**
 * Neural network configuration for a brain region.
 * Minimal interface that any SNN used as a local network must satisfy.
 */
export interface SNNNetworkConfig {
  /** Total number of neurons in the network */
  neuronCount: number;
  /** Number of inputs to the network */
  inputCount: number;
  /** Synaptic weights (may be a flat Float32Array or a complex structure) */
  weights: Float32Array;
}

/**
 * Serialized data of a brain region.
 */
export interface SerializedRegion {
  /** Region identifier */
  id: string;
  /** Descriptive name */
  name: string;
  /** Neural network configuration */
  network: SNNNetworkConfig;
  /** Current simulation time */
  currentTime: number;
}

/**
 * Abstract base class for brain regions of the digital brain.
 *
 * Each region contains:
 * 1. A spiking neural network (SNN) with its own synaptic weights
 * 2. A circular sensory buffer to store recent inputs
 * 3. Activation and timing state
 *
 * The concrete subclasses must implement processInput() with the
 * region's specific processing logic.
 */
export abstract class BrainRegion {
  /** Unique region identifier (e.g.: 'visualCortex') */
  public readonly id: string;

  /** Human-readable descriptive name (e.g.: 'Corteza Visual Primaria') */
  public readonly name: string;

  /** Number of neurons in this region */
  protected readonly neuronCount: number;

  /** Number of inputs to this region's neural network */
  protected readonly inputCount: number;

  /** Synaptic weights of the local neural network (flat Float32Array) */
  protected weights: Float32Array;

  /** Circular sensory memory buffer for recent inputs */
  protected sensoryBuffer: SensoryBuffer;

  /** Membrane potentials of all neurons */
  protected potentials: Float32Array;

  /** Spike state of each neuron (1.0 = spike, 0.0 = silence) */
  protected spikes: Float32Array;

  /** Whether the region is active and processing signals */
  public isActive: boolean = true;

  /** Current simulation time (ms) */
  public currentTime: number = 0;

  /** Base learning rate (modifiable by neuromodulators) */
  protected baseLearningRate: number = 0.1;

  /** Base firing threshold in mV (modifiable by neuromodulators) */
  protected baseThreshold: number = -55;

  /** Fraction of active neurons in the last step (k-WTA sparsity) */
  protected sparsity: number = 0.1;

  /**
   * Perceived activation level (EMA of the input drive, 0..1).
   * Reflects how much signal the region is actually receiving, smoothed over
   * time. Used for the activity panel: unlike `firingRate`, it
   * varies continuously with interaction (text/webcam/microphone).
   */
  protected driveEMA: number = 0;

  /**
   * Moving average of the firing pattern (which neurons tend to win). Used to
   * measure NOVELTY: how much the current pattern deviates from its usual regime.
   * Allocated lazily on the first `step` (it needs neuronCount).
   */
  protected spikeAvg: Float32Array | null = null;

  /**
   * Pattern novelty (EMA, 0..1). ~0 when the region repeats the same pattern
   * (rest / stable attractor) and rises when an input changes WHICH neurons
   * fire. Unlike `firingRate` (constant with k-WTA) and `drive`
   * (dominated by the recurrent background), this signal DOES react to
   * interaction: it is what lights up the activity panel bars.
   */
  protected noveltyEMA: number = 0;

  /**
   * Creates a new brain region.
   *
   * @param id - Unique identifier (e.g.: 'hippocampus')
   * @param name - Descriptive name (e.g.: 'Hipocampo - Formación de Memorias')
   * @param neuronCount - Number of neurons in the region
   * @param inputCount - Number of inputs (dimension of the input vector)
   * @param sensoryCapacity - Sensory buffer capacity (default: 100 entries)
   */
  constructor(
    id: string,
    name: string,
    neuronCount: number,
    inputCount: number,
    sensoryCapacity: number = 100
  ) {
    this.id = id;
    this.name = name;
    this.neuronCount = neuronCount;
    this.inputCount = inputCount;

    // Initialize synaptic weights (neuronCount * inputCount weights)
    this.weights = new Float32Array(neuronCount * inputCount);
    this.initializeWeights();

    // Initialize state vectors
    this.potentials = new Float32Array(neuronCount);
    this.potentials.fill(-70); // Resting potential
    this.spikes = new Float32Array(neuronCount);

    // Sensory buffer for recent inputs
    this.sensoryBuffer = new SensoryBuffer(sensoryCapacity, inputCount);
  }

  /**
   * Initializes the synaptic weights with small random values.
   *
   * Biology: The initial synaptic connections are weak and
   * partially random, refining themselves through experience and
   * synaptic plasticity (STDP, Hebbian learning).
   */
  protected initializeWeights(): void {
    const initialConnections = Math.min(
      1000,
      this.inputCount
    );

    // Sparse initialization: only connect a subset
    for (let n = 0; n < this.neuronCount; n++) {
      const baseOffset = n * this.inputCount;
      for (let k = 0; k < initialConnections; k++) {
        const inputIdx = Math.floor(Math.random() * this.inputCount);
        this.weights[baseOffset + inputIdx] = Math.random() * 0.1;
      }
    }
  }

  /**
   * Processes an input spike vector and produces output spikes.
   *
   * Abstract method that each concrete region must implement with
   * its specific processing logic.
   *
   * @param spikes - Input spike vector (Float32Array)
   * @param modulationEffects - Current neuromodulation effects
   * @returns Output spike vector (Float32Array)
   */
  abstract processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects
  ): Float32Array;

  /**
   * Runs one simulation step of the region.
   *
   * 1. Retrieves recent inputs from the sensory buffer
   * 2. Applies modulation to the network parameters
   * 3. Processes the input through the local SNN
   * 4. Returns the resulting activity
   *
   * @param dt - Time step (ms)
   * @param modulationEffects - Current neuromodulation effects
   * @returns Region activity in this step
   */
  step(dt: number, modulationEffects: ModulationEffects): RegionActivity {
    this.currentTime += dt;

    if (!this.isActive) {
      // No activity: drive and novelty decay smoothly toward 0.
      this.driveEMA *= 0.85;
      this.noveltyEMA *= 0.85;
      return {
        id: this.id,
        activeNeurons: [],
        firingRate: 0,
        drive: this.driveEMA,
        novelty: this.noveltyEMA,
        timestamp: this.currentTime,
        outputSpikes: new Float32Array(this.neuronCount),
      };
    }

    // Retrieve recent inputs from the sensory buffer
    const recentEntries = this.sensoryBuffer.getRecent(dt * 2);
    let inputSpikes: Float32Array;

    if (recentEntries.length > 0) {
      // Average the recent inputs
      inputSpikes = this.averageEntries(recentEntries);
    } else {
      inputSpikes = new Float32Array(this.inputCount);
    }

    // Update the perceived drive level (EMA). We measure the fraction of
    // input channels with signal: 0 = region at rest, rises during interaction.
    let driven = 0;
    for (let i = 0; i < inputSpikes.length; i++) {
      if (inputSpikes[i] > 0) driven++;
    }
    const instDrive =
      inputSpikes.length > 0 ? driven / inputSpikes.length : 0;
    this.driveEMA = this.driveEMA * 0.85 + instDrive * 0.15;

    // Apply modulation to parameters
    this.modulateBy(modulationEffects);

    // Process through the region's specific implementation
    const outputSpikes = this.processInput(inputSpikes, modulationEffects);

    // Compute activity
    const activeNeurons: number[] = [];
    for (let i = 0; i < outputSpikes.length; i++) {
      if (outputSpikes[i] > 0) {
        activeNeurons.push(i);
      }
    }

    const firingRate =
      this.neuronCount > 0 ? activeNeurons.length / this.neuronCount : 0;

    // Persist the output into this.spikes so getActivity() is consistent
    // across ALL regions (some did not update this.spikes on their own,
    // and the panel showed them frozen at 0%).
    if (outputSpikes.length === this.spikes.length) {
      this.spikes.set(outputSpikes);
    }

    // Pattern novelty: L1 distance between the current firing and its moving
    // average, normalized by the number of active neurons. Measures whether the
    // input changed WHICH neurons win (≠ how many). Rises during interaction, ~0 in a stable regime.
    if (!this.spikeAvg || this.spikeAvg.length !== outputSpikes.length) {
      this.spikeAvg = new Float32Array(outputSpikes.length);
    }
    const avg = this.spikeAvg;
    let l1 = 0;
    for (let i = 0; i < outputSpikes.length; i++) {
      const cur = outputSpikes[i] > 0 ? 1 : 0;
      l1 += Math.abs(cur - avg[i]);
      avg[i] = avg[i] * 0.9 + cur * 0.1; // EMA of the pattern
    }
    // Normalize: with k active, a full pattern turnover gives L1≈2k.
    const denom = Math.max(1, 2 * activeNeurons.length);
    const instNovelty = Math.min(1, l1 / denom);
    this.noveltyEMA = this.noveltyEMA * 0.7 + instNovelty * 0.3;

    return {
      id: this.id,
      activeNeurons,
      firingRate,
      drive: this.driveEMA,
      novelty: this.noveltyEMA,
      timestamp: this.currentTime,
      outputSpikes,
    };
  }

  /**
   * Averages multiple recent sensory inputs.
   *
   * @param entries - Entries from the sensory buffer
   * @returns Averaged vector
   */
  private averageEntries(entries: SensoryEntry[]): Float32Array {
    const avg = new Float32Array(this.inputCount);
    for (const entry of entries) {
      for (let i = 0; i < this.inputCount; i++) {
        avg[i] += entry.data[i];
      }
    }
    if (entries.length > 1) {
      const invCount = 1.0 / entries.length;
      for (let i = 0; i < this.inputCount; i++) {
        avg[i] *= invCount;
      }
    }
    return avg;
  }

  /**
   * Returns the region's current activity.
   *
   * @returns Activity snapshot with active neurons and firing rate
   */
  getActivity(): RegionActivity {
    const activeNeurons: number[] = [];
    for (let i = 0; i < this.spikes.length; i++) {
      if (this.spikes[i] > 0) {
        activeNeurons.push(i);
      }
    }

    return {
      id: this.id,
      activeNeurons,
      firingRate:
        this.neuronCount > 0 ? activeNeurons.length / this.neuronCount : 0,
      drive: this.driveEMA,
      novelty: this.noveltyEMA,
      timestamp: this.currentTime,
      outputSpikes: new Float32Array(this.spikes),
    };
  }

  /**
   * Applies neuromodulation effects to the network parameters.
   *
   * Biology: The diffuse neuromodulators (dopamine, serotonin, etc.)
   * globally modify neuronal excitability, synaptic
   * plasticity, and signal gain in the region.
   *
   * @param effects - Modulation effects to apply
   */
  modulateBy(effects: ModulationEffects): void {
    // Adjust firing threshold: serotonin and cortisol ↑ → threshold ↑
    const modulatedThreshold = this.baseThreshold * effects.thresholdMultiplier;

    // Adjust sparsity: more attention → more neurons can activate
    this.sparsity = Math.min(
      0.3,
      Math.max(0.02, 0.1 * effects.attentionGain)
    );

    // The modulated threshold is used internally in processInput
    // (stored so subclasses can access it)
    this._modulatedThreshold = modulatedThreshold;
    this._modulatedLearningRate =
      this.baseLearningRate * effects.learningRateMultiplier;
  }

  /** Firing threshold after modulation */
  protected _modulatedThreshold: number = -55;
  /** Learning rate after modulation */
  protected _modulatedLearningRate: number = 0.1;

  /**
   * Feeds the sensory buffer with a new input vector.
   *
   * @param data - Sensory data vector
   * @param timestamp - Timestamp of the input
   */
  feedInput(data: Float32Array, timestamp?: number): void {
    // Regions may receive spikes of a different size than their inputCount
    // (e.g., the thalamus has 3000 neurons but the visual cortex expects 5000 inputs).
    // We adapt the size: truncate if larger, pad with zeros if smaller.
    let adapted: Float32Array;
    if (data.length === this.inputCount) {
      adapted = data;
    } else if (data.length > this.inputCount) {
      adapted = data.subarray(0, this.inputCount);
    } else {
      adapted = new Float32Array(this.inputCount);
      adapted.set(data);
    }
    this.sensoryBuffer.push(adapted, timestamp ?? this.currentTime);
  }

  /**
   * Serializes the region's state for persistence.
   *
   * @returns Binary buffer with the region's complete state
   */
  serialize(): Buffer {
    // Format: id(string,len-prefixed) + neuronCount(4) + inputCount(4) + weights(Float32Array)
    const idBytes = Buffer.from(this.id, 'utf-8');
    const idLenBuf = Buffer.alloc(4);
    idLenBuf.writeUInt32LE(idBytes.length, 0);

    const metaBuf = Buffer.alloc(8);
    metaBuf.writeUInt32LE(this.neuronCount, 0);
    metaBuf.writeUInt32LE(this.inputCount, 4);

    const weightsBuf = Buffer.from(this.weights.buffer);

    return Buffer.concat([idLenBuf, idBytes, metaBuf, weightsBuf]);
  }

  /**
   * Restores the region's state from a serialized buffer.
   *
   * @param data - Buffer with previously serialized data
   */
  deserialize(data: Buffer): void {
    let offset = 0;

    // Read id (length-prefixed)
    const idLen = data.readUInt32LE(offset);
    offset += 4;
    // Skip the id (we already have it)
    offset += idLen;

    // Read neuronCount and inputCount
    const neuronCount = data.readUInt32LE(offset);
    offset += 4;
    const inputCount = data.readUInt32LE(offset);
    offset += 4;

    // Check compatibility
    if (neuronCount !== this.neuronCount || inputCount !== this.inputCount) {
      throw new Error(
        `[BrainRegion] Incompatibilidad de dimensiones al deserializar '${this.id}': ` +
          `esperado ${this.neuronCount}x${this.inputCount}, ` +
          `recibido ${neuronCount}x${inputCount}`
      );
    }

    // Restore weights
    const weightsByteLen = neuronCount * inputCount * 4;
    const weightsSlice = data.subarray(offset, offset + weightsByteLen);
    this.weights = new Float32Array(
      weightsSlice.buffer,
      weightsSlice.byteOffset,
      neuronCount * inputCount
    );
  }

  /**
   * Returns the region's neural network configuration.
   */
  getNetworkConfig(): SNNNetworkConfig {
    return {
      neuronCount: this.neuronCount,
      inputCount: this.inputCount,
      weights: this.weights,
    };
  }

  /**
   * Replaces the synaptic weights (restoration from persistence).
   * Rejects incompatible sizes to avoid corrupting the network if the region's
   * dimensions changed between saved versions.
   */
  loadWeights(weights: Float32Array): void {
    if (weights.length !== this.weights.length) {
      throw new Error(
        `[${this.id}] tamaño de pesos incompatible: ${weights.length} vs ${this.weights.length}`,
      );
    }
    this.weights.set(weights);
  }

  /**
   * Returns the number of neurons in the region.
   */
  get neurons(): number {
    return this.neuronCount;
  }

  /**
   * Returns the number of inputs to the network.
   */
  get inputs(): number {
    return this.inputCount;
  }
}
