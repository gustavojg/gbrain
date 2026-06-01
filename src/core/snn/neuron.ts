/**
 * SPIKING NEURON — Izhikevich Model
 * =========================================
 * High-performance implementation of the Izhikevich neuron model (2003).
 *
 * Biological basis:
 *   The Izhikevich model captures the membrane dynamics of real neurons
 *   using only two ordinary differential equations:
 *
 *     dv/dt = 0.04v² + 5v + 140 - u + I   (membrane voltage dynamics)
 *     du/dt = a(bv - u)                     (slow recovery variable)
 *
 *   With the reset rule on firing:
 *     if v ≥ 30 mV → v = c, u = u + d
 *
 *   The parameters (a, b, c, d) determine the firing type:
 *   - Regular Spiking (RS): cortical excitatory pyramidal neurons
 *   - Fast Spiking (FS): inhibitory interneurons (basket cells)
 *   - Chattering (CH): neurons that fire rhythmic bursts
 *   - Intrinsic Bursting (IB): neurons with intrinsic bursts
 *
 *   This model is ~100× more efficient than Hodgkin-Huxley and reproduces
 *   more than 20 experimentally observed neocortical firing patterns.
 *
 * Reference: Izhikevich, E.M. (2003). "Simple Model of Spiking Neurons."
 *             IEEE Trans. Neural Networks, 14(6):1569-1572.
 */

// ====================================================================
// Interfaces and types
// ====================================================================

/**
 * Izhikevich model parameters.
 *
 * Biological basis:
 *   - a: recovery time scale (larger = faster recovery)
 *   - b: sensitivity of u to the subthreshold v (u-v coupling)
 *   - c: reset potential after the spike (mV)
 *   - d: increment of u after the spike (adaptation current)
 */
export interface IzhikevichParams {
  /** Time scale of the recovery variable u (typically 0.02-0.1) */
  readonly a: number;
  /** Sensitivity of the recovery variable to the subthreshold potential (typically 0.2-0.25) */
  readonly b: number;
  /** Post-spike reset potential (mV, typically -65 to -50) */
  readonly c: number;
  /** Post-spike recovery increment (typically 2-8) */
  readonly d: number;
}

/**
 * Predefined neuron types based on electrophysiological observations.
 */
export type NeuronTypeName = 'RegularSpiking' | 'FastSpiking' | 'Chattering' | 'IntrinsicBursting';

// ====================================================================
// Predefined parameters per neuron type
// ====================================================================

/**
 * Predefined Izhikevich model parameters for each cell type.
 *
 * Biological basis:
 *   - RegularSpiking: the most common excitatory neuron in cortex (>80%).
 *     Progressively adapts its firing rate.
 *   - FastSpiking: GABAergic interneurons (basket/chandelier cells).
 *     Fast response without adaptation, mediating lateral inhibition.
 *   - Chattering: superficial-layer neurons that produce rhythmic
 *     bursts, possibly involved in gamma synchronization.
 *   - IntrinsicBursting: layer V neurons that produce an initial
 *     burst followed by regular spikes, important in cortico-cortical
 *     signaling.
 */
export const NEURON_PRESETS: Record<NeuronTypeName, IzhikevichParams> = {
  RegularSpiking:    { a: 0.02, b: 0.2, c: -65, d: 8 },
  FastSpiking:       { a: 0.1,  b: 0.2, c: -65, d: 2 },
  Chattering:        { a: 0.02, b: 0.2, c: -50, d: 2 },
  IntrinsicBursting: { a: 0.02, b: 0.2, c: -55, d: 4 },
} as const;

// ====================================================================
// SpikingNeuron class
// ====================================================================

/**
 * Individual spiking neuron based on the Izhikevich model.
 *
 * Biological basis:
 *   Models an individual cortical neuron with its membrane potential (v),
 *   recovery variable (u), and the ability to generate action
 *   potentials (spikes) when v crosses the ~30 mV threshold.
 *
 *   The u variable models the slow ionic currents (slow-activating K+
 *   and inactivating Na+) that produce frequency adaptation
 *   and refractory periods.
 *
 * Performance:
 *   Float32Array is used where possible. The scalar fields (v, u)
 *   are kept as native numbers for efficient individual access.
 */
export class SpikingNeuron {
  /** Current membrane potential (mV). Typical resting: -65 mV */
  public v: number;

  /** Recovery variable. Models slow ionic currents. */
  public u: number;

  /**
   * Time of the last spike in ms (relative to the start of the simulation).
   * -Infinity if it has never fired. Used by STDP to compute Δt.
   */
  public lastSpikeTime: number;

  /** Indicates whether the neuron fired in the last simulation step */
  public fired: boolean;

  /** Parameters (a, b, c, d) that define the firing type */
  public readonly params: IzhikevichParams;

  /**
   * Creates a new spiking neuron.
   *
   * @param params - Izhikevich parameters, or the name of a preset
   */
  constructor(params: IzhikevichParams | NeuronTypeName = 'RegularSpiking') {
    if (typeof params === 'string') {
      this.params = NEURON_PRESETS[params];
    } else {
      this.params = params;
    }

    this.v = -65.0;           // Resting potential (mV)
    this.u = this.params.b * this.v;  // Steady state of u
    this.lastSpikeTime = -Infinity;
    this.fired = false;
  }

  /**
   * Runs a simulation step of the Izhikevich model.
   *
   * Biological basis:
   *   Integrates the membrane differential equations using the Euler method
   *   with step dt. The input current I models the sum of synaptic
   *   currents (excitatory + inhibitory) + background currents.
   *
   *   2 Euler sub-steps of dt/2 each are used to improve
   *   numerical stability (the 0.04v² term can be unstable with
   *   large steps).
   *
   * @param I - Total input current (synaptic + external) in arbitrary units
   * @param dt - Time step in milliseconds (typically 0.5-1.0 ms)
   * @param currentTime - Current simulation time in ms (to record spike times)
   * @returns true if the neuron fired an action potential in this step
   */
  step(I: number, dt: number, currentTime: number): boolean {
    const { a, b, c, d } = this.params;
    this.fired = false;

    // --- Numerical integration with 2 Euler sub-steps (stability) ---
    // Sub-step 1: dt/2
    const halfDt = dt * 0.5;
    this.v += halfDt * (0.04 * this.v * this.v + 5.0 * this.v + 140.0 - this.u + I);
    // Sub-step 2: dt/2
    this.v += halfDt * (0.04 * this.v * this.v + 5.0 * this.v + 140.0 - this.u + I);
    // Update the recovery variable
    this.u += dt * a * (b * this.v - this.u);

    // --- Spike detection ---
    if (this.v >= 30.0) {
      this.v = c;         // Reset of the membrane potential
      this.u += d;         // Post-spike recovery increment
      this.fired = true;
      this.lastSpikeTime = currentTime;
    }

    return this.fired;
  }

  /**
   * Resets the neuron to its resting state.
   *
   * Biological basis:
   *   Equivalent to a prolonged period of silence where the neuron
   *   returns to its basal electrochemical state.
   */
  reset(): void {
    this.v = -65.0;
    this.u = this.params.b * this.v;
    this.lastSpikeTime = -Infinity;
    this.fired = false;
  }
}

// ====================================================================
// Utilities for bulk neuron creation
// ====================================================================

/**
 * Creates an array of neurons of the same type.
 * Optimized for bulk instantiation in brain regions.
 *
 * @param count - Number of neurons to create
 * @param neuronType - Neuron type (preset or custom parameters)
 * @returns Array of initialized neurons
 */
export function createNeuronPopulation(
  count: number,
  neuronType: IzhikevichParams | NeuronTypeName = 'RegularSpiking',
): SpikingNeuron[] {
  const params = typeof neuronType === 'string' ? NEURON_PRESETS[neuronType] : neuronType;
  const neurons: SpikingNeuron[] = new Array(count);

  for (let i = 0; i < count; i++) {
    neurons[i] = new SpikingNeuron(params);
  }

  return neurons;
}

/**
 * Compact state structure for serialization/bulk transfer.
 * Stores v, u, lastSpikeTime of N neurons in contiguous arrays.
 *
 * Biological basis:
 *   Efficient storage format analogous to how functional
 *   imaging techniques (fMRI, EEG) store the states of thousands of
 *   voxels/channels in contiguous arrays.
 */
export interface NeuronStateBuffer {
  /** Membrane potentials of all the neurons */
  readonly voltages: Float32Array;
  /** Recovery variables of all the neurons */
  readonly recovery: Float32Array;
  /** Last spike times (ms) */
  readonly lastSpikeTimes: Float64Array;
}

/**
 * Extracts the state of an array of neurons into compact buffers.
 *
 * @param neurons - Array of neurons
 * @returns Compact buffers with the state of all the neurons
 */
export function extractNeuronStates(neurons: readonly SpikingNeuron[]): NeuronStateBuffer {
  const n = neurons.length;
  const voltages = new Float32Array(n);
  const recovery = new Float32Array(n);
  const lastSpikeTimes = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    voltages[i] = neurons[i].v;
    recovery[i] = neurons[i].u;
    lastSpikeTimes[i] = neurons[i].lastSpikeTime;
  }

  return { voltages, recovery, lastSpikeTimes };
}

/**
 * Restores the state of neurons from compact buffers.
 *
 * @param neurons - Array of neurons to restore
 * @param state - Compact buffers with the saved state
 */
export function restoreNeuronStates(
  neurons: SpikingNeuron[],
  state: NeuronStateBuffer,
): void {
  const n = Math.min(neurons.length, state.voltages.length);

  for (let i = 0; i < n; i++) {
    neurons[i].v = state.voltages[i];
    neurons[i].u = state.recovery[i];
    neurons[i].lastSpikeTime = state.lastSpikeTimes[i];
    neurons[i].fired = false;
  }
}
