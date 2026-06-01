/**
 * DIGITAL BRAIN CONFIGURATION
 * ==================================
 * Fully parameterizable global parameters to scale
 * the brain according to the available hardware.
 *
 * Inspired by real human neuroanatomy:
 * - Neuron proportions based on real cortical densities
 * - Axonal delays based on nerve conduction velocities
 * - Neuromodulators with biologically plausible dynamics
 */

// ==================================================================
// CONFIGURATION TYPES
// ==================================================================

export interface RegionConfig {
  /** Number of neurons in this region */
  neurons: number;
  /** Number of inputs this region receives */
  inputs: number;
  /** Fraction of neurons that fire simultaneously (sparsity) */
  sparsity: number;
  /** Predominant neuron type */
  neuronType: 'RegularSpiking' | 'FastSpiking' | 'Chattering' | 'IntrinsicBursting';
}

export interface SNNConfig {
  /** Neuron model to use */
  model: 'izhikevich' | 'lif';
  /** Simulation time step (ms) */
  dt: number;
  /** Base learning rate for STDP */
  baseLearningRate: number;
  /** Time constants for STDP (ms) */
  stdp: {
    tauPlus: number;
    tauMinus: number;
    aPlus: number;
    aMinus: number;
  };
}

export interface MemoryConfig {
  /** Duration of the sensory buffer (ms) */
  sensoryBufferMs: number;
  /** Working memory slots (7±2) */
  workingMemorySlots: number;
  /** Maximum capacity of episodic memories in the hippocampus */
  hippocampalCapacity: number;
  /** Interval between automatic consolidations — "sleep" (ms) */
  consolidationIntervalMs: number;
  /** Number of replays per consolidation */
  consolidationReplays: number;
}

export interface ModulatorConfig {
  /** Base level (0-1) — where it naturally decays toward */
  baseline: number;
  /** Decay rate toward the baseline (per second) */
  decayRate: number;
  /** Maximum amount it can reach */
  max: number;
}

export interface ConnectionConfig {
  /** Source region */
  from: string;
  /** Destination region */
  to: string;
  /** Connection weight (0-1) — scales the amplitude of transmitted spikes */
  weight: number;
  /** Axonal delay (ms) — conduction time between regions */
  delay: number;
  /** Whether the weight can be modulated by neuromodulators */
  modulatable: boolean;
}

export interface BrainConfiguration {
  regions: Record<string, RegionConfig>;
  snn: SNNConfig;
  memory: MemoryConfig;
  modulators: Record<string, ModulatorConfig>;
  connectome: ConnectionConfig[];
  /** HTTP server port */
  serverPort: number;
  /** WebSocket port */
  wsPort: number;
  /** Frequency of the brain's main tick (Hz) */
  tickRate: number;
  /** Path for brain state persistence */
  persistencePath: string;
}

// ==================================================================
// DEFAULT CONFIGURATION (~50K neurons)
// ==================================================================

export const DEFAULT_BRAIN_CONFIG: BrainConfiguration = {

  // --- BRAIN REGIONS ---
  regions: {
    thalamus: {
      neurons: 3000,
      inputs: 50000,
      sparsity: 0.05,
      neuronType: 'FastSpiking',  // The thalamus responds fast
    },
    visualCortex: {
      neurons: 10000,
      inputs: 5000,
      sparsity: 0.1,
      neuronType: 'RegularSpiking',
    },
    auditoryCortex: {
      neurons: 5000,
      inputs: 800,   // 40 bands × 20 timesteps
      sparsity: 0.1,
      neuronType: 'RegularSpiking',
    },
    hippocampus: {
      neurons: 5000,
      inputs: 5000,
      sparsity: 0.05,  // Hippocampus uses very sparse encoding
      neuronType: 'RegularSpiking',
    },
    amygdala: {
      neurons: 2000,
      inputs: 3000,
      sparsity: 0.15,  // The amygdala is more reactive
      neuronType: 'IntrinsicBursting',  // Burst responses
    },
    prefrontalCortex: {
      neurons: 15000,
      inputs: 10000,
      sparsity: 0.08,
      neuronType: 'RegularSpiking',
    },
    brocaWernicke: {
      neurons: 10000,
      inputs: 5000,
      sparsity: 0.1,
      neuronType: 'Chattering',  // Rhythmic linguistic processing
    },
  },

  // --- SNN CONFIGURATION ---
  snn: {
    model: 'izhikevich',
    dt: 1.0,
    baseLearningRate: 0.01,
    stdp: {
      tauPlus: 20.0,   // LTP time window (ms)
      tauMinus: 20.0,  // LTD time window (ms)
      aPlus: 0.01,     // LTP amplitude
      aMinus: 0.012,   // LTD amplitude (slightly larger for stability)
    },
  },

  // --- MEMORY CONFIGURATION ---
  memory: {
    sensoryBufferMs: 250,          // Iconic/echoic sensory buffer
    workingMemorySlots: 7,          // Miller's 7±2
    hippocampalCapacity: 10000,     // Maximum episodic memories
    consolidationIntervalMs: 300_000,  // "Sleep" every 5 minutes
    consolidationReplays: 10,       // Replays per memory during consolidation
  },

  // --- NEUROMODULATORS ---
  modulators: {
    dopamine: {
      baseline: 0.3,
      decayRate: 0.1,  // Decays slowly
      max: 1.0,
    },
    serotonin: {
      baseline: 0.5,   // Normally high (stability)
      decayRate: 0.05,  // Very slow
      max: 1.0,
    },
    norepinephrine: {
      baseline: 0.2,
      decayRate: 0.15,  // Decays moderately fast
      max: 1.0,
    },
    cortisol: {
      baseline: 0.15,   // Normally low
      decayRate: 0.08,
      max: 1.0,
    },
    acetylcholine: {
      baseline: 0.4,
      decayRate: 0.12,
      max: 1.0,
    },
    oxytocin: {
      baseline: 0.2,
      decayRate: 0.05,  // Very slow
      max: 1.0,
    },
  },

  // --- CONNECTOME (Inter-regional connections) ---
  // Inspired by real white matter tracts
  connectome: [
    // === FEEDFORWARD (Sensory → Associative → Executive) ===

    // Thalamus → Sensory cortices (thalamic relay)
    { from: 'thalamus', to: 'visualCortex',   weight: 1.0, delay: 5,  modulatable: true },
    { from: 'thalamus', to: 'auditoryCortex',  weight: 1.0, delay: 5,  modulatable: true },
    { from: 'thalamus', to: 'brocaWernicke',   weight: 0.8, delay: 8,  modulatable: true },

    // Sensory cortices → Hippocampus (memory formation)
    { from: 'visualCortex',  to: 'hippocampus', weight: 0.9, delay: 10, modulatable: true },
    { from: 'auditoryCortex', to: 'hippocampus', weight: 0.9, delay: 10, modulatable: true },

    // Sensory cortices → Amygdala (fast emotional evaluation)
    { from: 'visualCortex',  to: 'amygdala',    weight: 0.7, delay: 8,  modulatable: true },
    { from: 'auditoryCortex', to: 'amygdala',    weight: 0.7, delay: 8,  modulatable: true },

    // Direct thalamus-amygdala route (ultra-fast emotional response, "low road")
    { from: 'thalamus', to: 'amygdala',         weight: 0.5, delay: 3,  modulatable: false },

    // Hippocampus + Amygdala → Prefrontal Cortex
    { from: 'hippocampus', to: 'prefrontalCortex', weight: 0.8, delay: 15, modulatable: true },
    { from: 'amygdala',    to: 'prefrontalCortex', weight: 0.9, delay: 10, modulatable: true },

    // Prefrontal Cortex → Broca/Wernicke (language production)
    { from: 'prefrontalCortex', to: 'brocaWernicke', weight: 1.0, delay: 12, modulatable: true },

    // Broca/Wernicke → Hippocampus (memorize what is said/understood)
    { from: 'brocaWernicke', to: 'hippocampus', weight: 0.6, delay: 10, modulatable: true },

    // === FEEDBACK (Top-Down — Executive control) ===

    // Prefrontal Cortex → Thalamus (top-down attentional control)
    { from: 'prefrontalCortex', to: 'thalamus',      weight: 0.5, delay: 15, modulatable: true },

    // Prefrontal Cortex → Visual Cortex (imagination, expectations)
    { from: 'prefrontalCortex', to: 'visualCortex',   weight: 0.4, delay: 12, modulatable: true },

    // Prefrontal Cortex → Auditory Cortex (auditory selective attention)
    { from: 'prefrontalCortex', to: 'auditoryCortex',  weight: 0.4, delay: 12, modulatable: true },

    // Amygdala → Thalamus (emotional modulation of attention)
    { from: 'amygdala', to: 'thalamus',               weight: 0.6, delay: 8,  modulatable: true },

    // Hippocampus → Cortices (memory reactivation)
    { from: 'hippocampus', to: 'visualCortex',         weight: 0.3, delay: 12, modulatable: true },
    { from: 'hippocampus', to: 'auditoryCortex',        weight: 0.3, delay: 12, modulatable: true },

    // Amygdala → Hippocampus (emotional memories consolidate more strongly)
    { from: 'amygdala', to: 'hippocampus',             weight: 0.8, delay: 5,  modulatable: false },
  ],

  // --- SERVER CONFIGURATION ---
  serverPort: 3000,
  wsPort: 3001,
  tickRate: 100,     // 100 Hz = 10ms per tick
  persistencePath: './brain_state.bin',
};

/**
 * Creates a custom configuration by merging partial values with the defaults.
 */
export function createBrainConfig(overrides: Partial<BrainConfiguration> = {}): BrainConfiguration {
  return {
    ...DEFAULT_BRAIN_CONFIG,
    ...overrides,
    regions: {
      ...DEFAULT_BRAIN_CONFIG.regions,
      ...(overrides.regions || {}),
    },
    snn: {
      ...DEFAULT_BRAIN_CONFIG.snn,
      ...(overrides.snn || {}),
      stdp: {
        ...DEFAULT_BRAIN_CONFIG.snn.stdp,
        ...(overrides.snn?.stdp || {}),
      },
    },
    memory: {
      ...DEFAULT_BRAIN_CONFIG.memory,
      ...(overrides.memory || {}),
    },
    modulators: {
      ...DEFAULT_BRAIN_CONFIG.modulators,
      ...(overrides.modulators || {}),
    },
  };
}

/**
 * Computes the total number of neurons in the configuration.
 */
export function getTotalNeurons(config: BrainConfiguration): number {
  return Object.values(config.regions).reduce((sum, r) => sum + r.neurons, 0);
}

/**
 * Estimates the memory in MB for the synaptic weights.
 */
export function estimateMemoryUsage(config: BrainConfiguration): number {
  let totalBytes = 0;
  for (const region of Object.values(config.regions)) {
    // Float32 (4 bytes) per weight, neurons × inputs weights
    totalBytes += region.neurons * region.inputs * 4;
  }
  return totalBytes / (1024 * 1024); // MB
}
