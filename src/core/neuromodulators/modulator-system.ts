/**
 * Neuromodulation System
 * ===========================
 * Models the 6 main neuromodulator systems of the human brain.
 *
 * Biology: Neuromodulators are chemical substances released by subcortical
 * nuclei that globally affect neural processing. Unlike
 * point-to-point synaptic neurotransmission, neuromodulation acts
 * diffusely ("volumetrically") over entire regions, altering parameters
 * such as neuronal excitability, synaptic plasticity and signal
 * gain.
 *
 * Modeled systems:
 * 1. Dopamine (VTA/SNc): Reward, motivation, reinforcement learning
 * 2. Serotonin (Raphe nuclei): Emotional regulation, stability
 * 3. Norepinephrine (Locus coeruleus): Alertness, attention, stress response
 * 4. Cortisol (HPA axis): Chronic stress, suppression of consolidation
 * 5. Acetylcholine (Basal nucleus): Focused attention, memory consolidation
 * 6. Oxytocin (Hypothalamus): Social cognition, trust, bonding
 */

/**
 * Types of neuromodulators implemented in the system.
 */
export enum ModulatorType {
  /** Dopamine: reward signal and prediction error */
  Dopamine = 'dopamine',
  /** Serotonin: mood regulation and behavioral stability */
  Serotonin = 'serotonin',
  /** Norepinephrine: alertness, wakefulness and fight-or-flight response */
  Norepinephrine = 'norepinephrine',
  /** Cortisol: response to sustained stress */
  Cortisol = 'cortisol',
  /** Acetylcholine: focused attention and memory formation */
  Acetylcholine = 'acetylcholine',
  /** Oxytocin: social cognition and affective bonds */
  Oxytocin = 'oxytocin',
}

/**
 * Instantaneous state of a neuromodulator.
 */
export interface ModulatorState {
  /** Current level of the modulator (0.0 - 1.0) */
  level: number;
  /** Baseline level the modulator tends toward at rest */
  baseline: number;
  /** Decay rate toward the baseline (fraction per ms) */
  decayRate: number;
  /** Timestamp of the last update (ms) */
  lastUpdate: number;
}

/**
 * Combined effects of all neuromodulators on the SNN parameters.
 * These multipliers are applied globally to all brain regions.
 */
export interface ModulationEffects {
  /** Learning rate multiplier (STDP/Hebbian). >1 = more plasticity */
  learningRateMultiplier: number;
  /** Firing threshold multiplier. >1 = neurons harder to activate */
  thresholdMultiplier: number;
  /** Attentional gain. >1 = amplified sensory signals */
  attentionGain: number;
  /** Memory consolidation rate. >1 = memories strengthen faster */
  consolidationRate: number;
  /** Spike gain multiplier. >1 = amplified neural activity */
  spikeGainMultiplier: number;
  /** Boost of weights related to social processing */
  socialWeightBoost: number;
}

/**
 * Serializable state of the complete neuromodulation system.
 */
export interface NeuromodulatorSnapshot {
  /** State of each modulator */
  modulators: Record<ModulatorType, ModulatorState>;
  /** Timestamp of the capture */
  timestamp: number;
}

/**
 * Central neuromodulation system of the digital brain.
 *
 * Manages the levels of 6 neuromodulators and computes their combined
 * effects on the global parameters of the neural network.
 * Each modulator decays exponentially toward its baseline level,
 * modeling biological reuptake and enzymatic degradation.
 */
export class NeuromodulatorSystem {
  /** Internal state of all modulators */
  private readonly modulators: Map<ModulatorType, ModulatorState>;

  /**
   * Creates the neuromodulation system with default baseline levels.
   *
   * Biology: The baseline levels represent the tonic (constant)
   * activity of each neuromodulator system in a quiet wakeful
   * state. The values range between 0.3 and 0.5 reflecting
   * moderate activity.
   */
  constructor() {
    this.modulators = new Map<ModulatorType, ModulatorState>([
      [
        ModulatorType.Dopamine,
        { level: 0.4, baseline: 0.4, decayRate: 0.0005, lastUpdate: 0 },
      ],
      [
        ModulatorType.Serotonin,
        { level: 0.5, baseline: 0.5, decayRate: 0.0003, lastUpdate: 0 },
      ],
      [
        ModulatorType.Norepinephrine,
        { level: 0.3, baseline: 0.3, decayRate: 0.0008, lastUpdate: 0 },
      ],
      [
        ModulatorType.Cortisol,
        { level: 0.3, baseline: 0.3, decayRate: 0.0001, lastUpdate: 0 },
      ],
      [
        ModulatorType.Acetylcholine,
        { level: 0.4, baseline: 0.4, decayRate: 0.0006, lastUpdate: 0 },
      ],
      [
        ModulatorType.Oxytocin,
        { level: 0.3, baseline: 0.3, decayRate: 0.0004, lastUpdate: 0 },
      ],
    ]);
  }

  /**
   * Releases an amount of neuromodulator (increases its level).
   *
   * Biology: Models the phasic (burst) release of neuromodulator
   * from the subcortical nuclei in response to relevant events.
   * Example: An unexpected reward causes a dopamine burst.
   *
   * @param type - Type of neuromodulator to release
   * @param amount - Amount to release (0.0 - 1.0). Added to the current level.
   */
  release(type: ModulatorType, amount: number): void {
    const state = this.modulators.get(type);
    if (!state) return;

    // Clamp to the range [0, 1]
    state.level = Math.min(1.0, Math.max(0, state.level + amount));
    state.lastUpdate = Date.now();
  }

  /**
   * Applies temporal decay to all modulators.
   *
   * Biology: Neuromodulators are taken up by presynaptic
   * transporters and degraded enzymatically (e.g. MAO for dopamine,
   * AChE for acetylcholine). This process makes the levels return
   * exponentially to the tonic baseline.
   *
   * @param dt - Time elapsed since the last tick (ms)
   */
  decay(dt: number): void {
    for (const [, state] of this.modulators) {
      // Exponential decay toward the baseline
      const diff = state.level - state.baseline;
      state.level = state.baseline + diff * Math.exp(-state.decayRate * dt);
    }
  }

  /**
   * Computes the combined effects of all neuromodulators.
   *
   * Biology: Neuromodulators interact in complex ways. For example,
   * high cortisol inhibits hippocampal consolidation and counteracts the
   * potentiating effects of dopamine on learning. Acetylcholine
   * and norepinephrine act synergistically to amplify attention.
   *
   * @returns Combined modulation effects to apply to the SNN
   */
  getEffects(): ModulationEffects {
    const da = this.getLevel(ModulatorType.Dopamine);
    const ser = this.getLevel(ModulatorType.Serotonin);
    const ne = this.getLevel(ModulatorType.Norepinephrine);
    const cort = this.getLevel(ModulatorType.Cortisol);
    const ach = this.getLevel(ModulatorType.Acetylcholine);
    const oxy = this.getLevel(ModulatorType.Oxytocin);

    return {
      // Dopamine ↑ → learning ↑ | Cortisol ↑ → learning ↓
      // DA potentiates LTP at corticostriatal synapses; cortisol blocks hippocampal NMDA
      learningRateMultiplier: (1.0 + da * 0.8) * (1.0 - cort * 0.4),

      // Serotonin ↑ → threshold ↑ (more stability, less impulsivity)
      // Cortisol ↑ → threshold ↑ (inhibition due to stress)
      // 5-HT raises the firing threshold of pyramidal neurons via 5-HT1A receptors
      thresholdMultiplier: 1.0 + ser * 0.3 + cort * 0.2,

      // Norepinephrine ↑ → attention ↑ | Acetylcholine ↑ → attention ↑
      // NE from the locus coeruleus modulates the cortical signal/noise ratio
      // ACh from the basal nucleus selectively amplifies sensory afferents
      attentionGain: 1.0 + ne * 0.6 + ach * 0.5,

      // Acetylcholine ↑ → consolidation ↑ | Cortisol ↑ → consolidation ↓
      // ACh facilitates hippocampal LTP; chronic cortisol atrophies CA3 dendrites
      consolidationRate: (1.0 + ach * 0.7) * (1.0 - cort * 0.3),

      // Norepinephrine ↑ → spike gain ↑ (more general activation)
      // NE increases neuronal excitability via β-adrenergic receptors
      spikeGainMultiplier: 1.0 + ne * 0.5,

      // Oxytocin ↑ → social boost (face processing, empathy, trust)
      // OXT potentiates neurons in the superior temporal sulcus and amygdala
      socialWeightBoost: 1.0 + oxy * 0.6,
    };
  }

  /**
   * Gets the current level of a specific neuromodulator.
   *
   * @param type - Type of neuromodulator
   * @returns Current level (0.0 - 1.0)
   */
  getLevel(type: ModulatorType): number {
    return this.modulators.get(type)?.level ?? 0;
  }

  /**
   * Gets the complete state of all modulators.
   *
   * @returns Map with the state of each neuromodulator
   */
  getState(): Map<ModulatorType, ModulatorState> {
    // Return a deep copy to avoid external mutation
    const copy = new Map<ModulatorType, ModulatorState>();
    for (const [type, state] of this.modulators) {
      copy.set(type, { ...state });
    }
    return copy;
  }

  /**
   * Serializes the complete state of the neuromodulation system.
   *
   * @returns Serializable snapshot of the current state
   */
  serialize(): NeuromodulatorSnapshot {
    const modulators = {} as Record<ModulatorType, ModulatorState>;
    for (const [type, state] of this.modulators) {
      modulators[type] = { ...state };
    }
    return {
      modulators,
      timestamp: Date.now(),
    };
  }

  /**
   * Restores the system state from a serialized snapshot.
   *
   * @param snapshot - Previously serialized state
   */
  deserialize(snapshot: NeuromodulatorSnapshot): void {
    for (const [typeStr, state] of Object.entries(snapshot.modulators)) {
      const type = typeStr as ModulatorType;
      if (this.modulators.has(type)) {
        this.modulators.set(type, { ...state });
      }
    }
  }

  /**
   * Resets all modulators to their baseline levels.
   *
   * Biology: Equivalent to the return to the homeostatic state after
   * a prolonged period of rest.
   */
  reset(): void {
    for (const [, state] of this.modulators) {
      state.level = state.baseline;
    }
  }
}
