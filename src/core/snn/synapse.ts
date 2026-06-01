/**
 * SYNAPSE WITH STDP — Spike-Timing-Dependent Plasticity
 * ==================================================================
 * Implementation of the STDP synaptic learning rule
 * (Spike-Timing-Dependent Plasticity).
 *
 * Biological basis:
 *   STDP is the most studied synaptic plasticity mechanism in
 *   experimental neuroscience. Discovered by Markram (1997) and
 *   Bi & Poo (1998), it states that:
 *
 *   - If the presynaptic neuron fires BEFORE the postsynaptic one (Δt > 0):
 *     → Long-Term Potentiation (LTP): the synapse is strengthened.
 *     → Δw = A+ · exp(-Δt / τ+)
 *     → Interpretation: "the pre CAUSED the post" → reinforce the connection.
 *
 *   - If the postsynaptic neuron fires BEFORE the presynaptic one (Δt < 0):
 *     → Long-Term Depression (LTD): the synapse is weakened.
 *     → Δw = -A- · exp(Δt / τ-)
 *     → Interpretation: "the pre did NOT cause the post" → weaken.
 *
 *   The asymmetric temporal window (τ ≈ 20ms) matches the duration
 *   of the excitatory postsynaptic potentials (EPSPs) in cortex.
 *
 *   The modulation factor allows neuromodulators (dopamine, etc.)
 *   to scale the magnitude of the synaptic change, thus implementing
 *   "three-factor learning rules" (eligibility × reward × STDP).
 *
 * References:
 *   - Bi, G. & Poo, M. (1998). J. Neuroscience, 18(24):10464-10472.
 *   - Markram, H. et al. (1997). Science, 275(5297):213-215.
 */

// ====================================================================
// Interfaces and types
// ====================================================================

/**
 * Configuration of an individual synapse.
 *
 * Biological basis:
 *   - weight: synaptic efficacy (EPSP/IPSP amplitude).
 *   - maxWeight: upper limit from AMPA/NMDA receptor saturation.
 *   - minWeight: lower limit (silent synapse, not eliminated).
 *   - delay: axonal conduction delay in ms.
 */
export interface SynapseConfig {
  /** Initial synaptic weight (transmission efficacy) */
  weight: number;
  /** Maximum reachable weight (receptor saturation) */
  maxWeight: number;
  /** Minimum weight (silent synapse, but structurally preserved) */
  minWeight: number;
  /** Axonal conduction delay in milliseconds */
  delay: number;
}

/**
 * Parameters of the STDP temporal window.
 *
 * Biological basis:
 *   The default values are calibrated according to experimental data
 *   from glutamatergic cortical synapses:
 *   - A+ < A- ensures that synaptic competition tends toward
 *     stability (more LTD than LTP on average).
 *   - τ ≈ 20ms corresponds to the typical duration of an EPSP.
 */
export interface STDPParams {
  /** Maximum potentiation amplitude (LTP). Typical value: 0.01 */
  readonly aPlus: number;
  /** Maximum depression amplitude (LTD). Typical value: 0.012 */
  readonly aMinus: number;
  /** Time constant of the LTP window in ms. Typical value: 20 */
  readonly tauPlus: number;
  /** Time constant of the LTD window in ms. Typical value: 20 */
  readonly tauMinus: number;
}

/** Default STDP parameters based on experimental cortical data */
export const DEFAULT_STDP_PARAMS: STDPParams = {
  aPlus: 0.01,
  aMinus: 0.012,
  tauPlus: 20.0,
  tauMinus: 20.0,
} as const;

// ====================================================================
// STDPSynapse class
// ====================================================================

/**
 * Synapse with spike-timing-dependent plasticity (STDP).
 *
 * Biological basis:
 *   Models an individual synaptic connection between two neurons, with
 *   the ability to modify its efficacy (weight) according to the STDP rule.
 *
 *   The synapse includes:
 *   - A weight that determines the amplitude of the injected current
 *   - An axonal conduction delay
 *   - Homeostatic limits to prevent pathological weights
 *   - An STDP rule modulable by neuromodulators
 *
 * Performance:
 *   For large networks (50K+), it is recommended to use the SNNNetwork
 *   class, which stores weights in a flat Float32Array. This class is useful for
 *   individual inter-regional connections or for debugging.
 */
export class STDPSynapse {
  /** Current synaptic weight (transmission efficacy) */
  private weight: number;
  /** Maximum allowed weight (homeostasis) */
  private readonly maxWeight: number;
  /** Minimum allowed weight (homeostasis) */
  private readonly minWeight: number;
  /** Axonal conduction delay (ms) */
  public readonly delay: number;
  /** Parameters of the STDP window */
  private readonly stdp: STDPParams;

  /**
   * Creates a new STDP synapse.
   *
   * @param config - Synapse configuration (weight, limits, delay)
   * @param stdpParams - Optional parameters of the STDP window
   */
  constructor(
    config: SynapseConfig,
    stdpParams: STDPParams = DEFAULT_STDP_PARAMS,
  ) {
    this.weight = config.weight;
    this.maxWeight = config.maxWeight;
    this.minWeight = config.minWeight;
    this.delay = config.delay;
    this.stdp = stdpParams;
  }

  /**
   * Applies the STDP rule based on the pre- and postsynaptic spike times.
   *
   * Biological basis:
   *   Computes the weight change Δw according to the temporal difference:
   *     Δt = t_post - t_pre
   *
   *   - Δt > 0 (pre before post): LTP → the synapse is strengthened
   *     because the presynaptic activity predicted/caused the post firing.
   *     Δw = A+ · exp(-Δt / τ+)
   *
   *   - Δt < 0 (post before pre): LTD → the synapse is weakened
   *     because the presynaptic activity did NOT predict the post firing.
   *     Δw = -A- · exp(Δt / τ-)   (note: Δt < 0 here)
   *
   *   The modulationFactor scales Δw, allowing neuromodulators
   *   (dopamine = reward, norepinephrine = alertness) to control
   *   how much is learned. This implements the "three-factor rule":
   *     Δw_final = STDP(Δt) × modulation
   *
   * @param preSpikeTime - Time of the presynaptic spike (ms)
   * @param postSpikeTime - Time of the postsynaptic spike (ms)
   * @param modulationFactor - Neuromodulatory modulation factor (default: 1.0)
   * @returns The applied weight change (Δw)
   */
  applySTDP(
    preSpikeTime: number,
    postSpikeTime: number,
    modulationFactor: number = 1.0,
  ): number {
    // No learning if either neuron has never fired
    if (!isFinite(preSpikeTime) || !isFinite(postSpikeTime)) {
      return 0;
    }

    const deltaT = postSpikeTime - preSpikeTime;

    let deltaW = 0;

    if (deltaT > 0) {
      // Pre fires before post → LTP (potentiation)
      deltaW = this.stdp.aPlus * Math.exp(-deltaT / this.stdp.tauPlus);
    } else if (deltaT < 0) {
      // Post fires before pre → LTD (depression)
      deltaW = -this.stdp.aMinus * Math.exp(deltaT / this.stdp.tauMinus);
    }
    // If deltaT === 0, no change is applied (simultaneous, ambiguous event)

    // Apply neuromodulatory modulation (three-factor rule)
    deltaW *= modulationFactor;

    // Update weight with homeostatic limits
    this.weight = Math.max(
      this.minWeight,
      Math.min(this.maxWeight, this.weight + deltaW),
    );

    return deltaW;
  }

  /**
   * Returns the current synaptic weight.
   *
   * @returns Synaptic weight (transmission efficacy)
   */
  getWeight(): number {
    return this.weight;
  }

  /**
   * Sets the synaptic weight directly (respecting limits).
   * Useful for initialization or restoration from persistence.
   *
   * @param newWeight - New synaptic weight
   */
  setWeight(newWeight: number): void {
    this.weight = Math.max(this.minWeight, Math.min(this.maxWeight, newWeight));
  }

  /**
   * Resets the synaptic weight to the midpoint value between min and max.
   *
   * Biological basis:
   *   Simulates a massive synaptic depotentiation, like the one observed
   *   during slow-wave sleep, where the
   *   synapses are globally rescaled to basal levels.
   */
  reset(): void {
    this.weight = (this.maxWeight + this.minWeight) * 0.5;
  }

  /**
   * Computes the transmitted synaptic current (weight × presynaptic spike).
   * In practice: returns the weight if there is a spike, 0 otherwise.
   *
   * @param presynapticSpike - true if the presynaptic neuron fired
   * @returns Current injected into the postsynaptic neuron
   */
  transmit(presynapticSpike: boolean): number {
    return presynapticSpike ? this.weight : 0;
  }
}

// ====================================================================
// Utility functions for network-level STDP
// ====================================================================

/**
 * Computes the STDP weight change between a pair of neurons without applying it.
 * Pure function for use in networks that store weights in a Float32Array.
 *
 * Biological basis:
 *   Functional version of the STDP computation, designed for integration with
 *   the SNNNetwork class that uses flat weight storage.
 *
 * @param preSpikeTime - Time of the presynaptic spike (ms)
 * @param postSpikeTime - Time of the postsynaptic spike (ms)
 * @param stdp - Parameters of the STDP window
 * @param modulationFactor - Neuromodulatory modulation factor
 * @returns Computed Δw (without applying limits)
 */
export function computeSTDP(
  preSpikeTime: number,
  postSpikeTime: number,
  stdp: STDPParams = DEFAULT_STDP_PARAMS,
  modulationFactor: number = 1.0,
): number {
  if (!isFinite(preSpikeTime) || !isFinite(postSpikeTime)) {
    return 0;
  }

  const deltaT = postSpikeTime - preSpikeTime;

  let deltaW = 0;

  if (deltaT > 0) {
    deltaW = stdp.aPlus * Math.exp(-deltaT / stdp.tauPlus);
  } else if (deltaT < 0) {
    deltaW = -stdp.aMinus * Math.exp(deltaT / stdp.tauMinus);
  }

  return deltaW * modulationFactor;
}
