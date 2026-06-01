/**
 * CONFIGURACIÓN DEL CEREBRO DIGITAL
 * ==================================
 * Parámetros globales completamente parametrizables para escalar
 * el cerebro según el hardware disponible.
 * 
 * Inspirado en la neuroanatomía humana real:
 * - Proporciones de neuronas basadas en densidades corticales reales
 * - Retardos axonales basados en velocidades de conducción nerviosa
 * - Neuromoduladores con dinámicas biológicamente plausibles
 */

// ==================================================================
// TIPOS DE CONFIGURACIÓN
// ==================================================================

export interface RegionConfig {
  /** Número de neuronas en esta región */
  neurons: number;
  /** Número de inputs que recibe esta región */
  inputs: number;
  /** Fracción de neuronas que disparan simultáneamente (sparsity) */
  sparsity: number;
  /** Tipo de neurona predominante */
  neuronType: 'RegularSpiking' | 'FastSpiking' | 'Chattering' | 'IntrinsicBursting';
}

export interface SNNConfig {
  /** Modelo de neurona a utilizar */
  model: 'izhikevich' | 'lif';
  /** Paso temporal de simulación (ms) */
  dt: number;
  /** Tasa de aprendizaje base para STDP */
  baseLearningRate: number;
  /** Constantes de tiempo para STDP (ms) */
  stdp: {
    tauPlus: number;
    tauMinus: number;
    aPlus: number;
    aMinus: number;
  };
}

export interface MemoryConfig {
  /** Duración del buffer sensorial (ms) */
  sensoryBufferMs: number;
  /** Slots de memoria de trabajo (7±2) */
  workingMemorySlots: number;
  /** Capacidad máxima de memorias episódicas en hipocampo */
  hippocampalCapacity: number;
  /** Intervalo entre consolidaciones automáticas — "sueño" (ms) */
  consolidationIntervalMs: number;
  /** Número de replays por consolidación */
  consolidationReplays: number;
}

export interface ModulatorConfig {
  /** Nivel base (0-1) — hacia donde decae naturalmente */
  baseline: number;
  /** Velocidad de decaimiento hacia el baseline (por segundo) */
  decayRate: number;
  /** Cantidad máxima que puede alcanzar */
  max: number;
}

export interface ConnectionConfig {
  /** Región de origen */
  from: string;
  /** Región de destino */
  to: string;
  /** Peso de la conexión (0-1) — escala la amplitud de spikes transmitidos */
  weight: number;
  /** Retardo axonal (ms) — tiempo de conducción entre regiones */
  delay: number;
  /** Si el peso puede ser modulado por neuromoduladores */
  modulatable: boolean;
}

export interface BrainConfiguration {
  regions: Record<string, RegionConfig>;
  snn: SNNConfig;
  memory: MemoryConfig;
  modulators: Record<string, ModulatorConfig>;
  connectome: ConnectionConfig[];
  /** Puerto del servidor HTTP */
  serverPort: number;
  /** Puerto del WebSocket */
  wsPort: number;
  /** Frecuencia del tick principal del cerebro (Hz) */
  tickRate: number;
  /** Ruta para persistencia del estado cerebral */
  persistencePath: string;
}

// ==================================================================
// CONFIGURACIÓN POR DEFECTO (~50K neuronas)
// ==================================================================

export const DEFAULT_BRAIN_CONFIG: BrainConfiguration = {
  
  // --- REGIONES DEL CEREBRO ---
  regions: {
    thalamus: {
      neurons: 3000,
      inputs: 50000,
      sparsity: 0.05,
      neuronType: 'FastSpiking',  // El tálamo responde rápido
    },
    visualCortex: {
      neurons: 10000,
      inputs: 5000,
      sparsity: 0.1,
      neuronType: 'RegularSpiking',
    },
    auditoryCortex: {
      neurons: 5000,
      inputs: 800,   // 40 bandas × 20 timesteps
      sparsity: 0.1,
      neuronType: 'RegularSpiking',
    },
    hippocampus: {
      neurons: 5000,
      inputs: 5000,
      sparsity: 0.05,  // Hipocampo usa codificación muy sparse
      neuronType: 'RegularSpiking',
    },
    amygdala: {
      neurons: 2000,
      inputs: 3000,
      sparsity: 0.15,  // Amígdala es más reactiva
      neuronType: 'IntrinsicBursting',  // Respuestas en ráfaga
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
      neuronType: 'Chattering',  // Procesamiento lingüístico rítmico
    },
  },

  // --- CONFIGURACIÓN SNN ---
  snn: {
    model: 'izhikevich',
    dt: 1.0,
    baseLearningRate: 0.01,
    stdp: {
      tauPlus: 20.0,   // Ventana temporal LTP (ms)
      tauMinus: 20.0,  // Ventana temporal LTD (ms)
      aPlus: 0.01,     // Amplitud LTP
      aMinus: 0.012,   // Amplitud LTD (ligeramente mayor para estabilidad)
    },
  },

  // --- CONFIGURACIÓN DE MEMORIA ---
  memory: {
    sensoryBufferMs: 250,          // Buffer sensorial icónico/ecoico
    workingMemorySlots: 7,          // 7±2 de Miller
    hippocampalCapacity: 10000,     // Memorias episódicas máximas
    consolidationIntervalMs: 300_000,  // "Dormir" cada 5 minutos
    consolidationReplays: 10,       // Replays por memoria durante consolidación
  },

  // --- NEUROMODULADORES ---
  modulators: {
    dopamine: {
      baseline: 0.3,
      decayRate: 0.1,  // Decae lentamente
      max: 1.0,
    },
    serotonin: {
      baseline: 0.5,   // Normalmente alto (estabilidad)
      decayRate: 0.05,  // Muy lento
      max: 1.0,
    },
    norepinephrine: {
      baseline: 0.2,
      decayRate: 0.15,  // Decae moderadamente rápido
      max: 1.0,
    },
    cortisol: {
      baseline: 0.15,   // Normalmente bajo
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
      decayRate: 0.05,  // Muy lento
      max: 1.0,
    },
  },

  // --- CONECTOMA (Conexiones inter-regionales) ---
  // Inspirado en tractos de materia blanca reales
  connectome: [
    // === FEEDFORWARD (Sensorial → Asociativo → Ejecutivo) ===
    
    // Tálamo → Cortezas sensoriales (relay talámico)
    { from: 'thalamus', to: 'visualCortex',   weight: 1.0, delay: 5,  modulatable: true },
    { from: 'thalamus', to: 'auditoryCortex',  weight: 1.0, delay: 5,  modulatable: true },
    { from: 'thalamus', to: 'brocaWernicke',   weight: 0.8, delay: 8,  modulatable: true },
    
    // Cortezas sensoriales → Hipocampo (formación de memoria)
    { from: 'visualCortex',  to: 'hippocampus', weight: 0.9, delay: 10, modulatable: true },
    { from: 'auditoryCortex', to: 'hippocampus', weight: 0.9, delay: 10, modulatable: true },
    
    // Cortezas sensoriales → Amígdala (evaluación emocional rápida)
    { from: 'visualCortex',  to: 'amygdala',    weight: 0.7, delay: 8,  modulatable: true },
    { from: 'auditoryCortex', to: 'amygdala',    weight: 0.7, delay: 8,  modulatable: true },
    
    // Ruta tálamo-amígdala directa (respuesta emocional ultra-rápida, "low road")
    { from: 'thalamus', to: 'amygdala',         weight: 0.5, delay: 3,  modulatable: false },
    
    // Hipocampo + Amígdala → Corteza Prefrontal
    { from: 'hippocampus', to: 'prefrontalCortex', weight: 0.8, delay: 15, modulatable: true },
    { from: 'amygdala',    to: 'prefrontalCortex', weight: 0.9, delay: 10, modulatable: true },
    
    // Corteza Prefrontal → Broca/Wernicke (producción del lenguaje)
    { from: 'prefrontalCortex', to: 'brocaWernicke', weight: 1.0, delay: 12, modulatable: true },
    
    // Broca/Wernicke → Hipocampo (memorizar lo que se dice/comprende)
    { from: 'brocaWernicke', to: 'hippocampus', weight: 0.6, delay: 10, modulatable: true },

    // === FEEDBACK (Top-Down — Control ejecutivo) ===
    
    // Corteza Prefrontal → Tálamo (control atencional top-down)
    { from: 'prefrontalCortex', to: 'thalamus',      weight: 0.5, delay: 15, modulatable: true },
    
    // Corteza Prefrontal → Corteza Visual (imaginación, expectativas)
    { from: 'prefrontalCortex', to: 'visualCortex',   weight: 0.4, delay: 12, modulatable: true },
    
    // Corteza Prefrontal → Corteza Auditiva (atención selectiva auditiva)
    { from: 'prefrontalCortex', to: 'auditoryCortex',  weight: 0.4, delay: 12, modulatable: true },
    
    // Amígdala → Tálamo (modulación emocional de atención)
    { from: 'amygdala', to: 'thalamus',               weight: 0.6, delay: 8,  modulatable: true },
    
    // Hipocampo → Cortezas (reactivación de memorias)
    { from: 'hippocampus', to: 'visualCortex',         weight: 0.3, delay: 12, modulatable: true },
    { from: 'hippocampus', to: 'auditoryCortex',        weight: 0.3, delay: 12, modulatable: true },
    
    // Amígdala → Hipocampo (memorias emocionales se consolidan más fuerte)
    { from: 'amygdala', to: 'hippocampus',             weight: 0.8, delay: 5,  modulatable: false },
  ],

  // --- CONFIGURACIÓN DEL SERVIDOR ---
  serverPort: 3000,
  wsPort: 3001,
  tickRate: 100,     // 100 Hz = 10ms por tick
  persistencePath: './brain_state.bin',
};

/**
 * Crea una configuración personalizada mezclando valores parciales con los defaults.
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
 * Calcula el número total de neuronas en la configuración.
 */
export function getTotalNeurons(config: BrainConfiguration): number {
  return Object.values(config.regions).reduce((sum, r) => sum + r.neurons, 0);
}

/**
 * Calcula la memoria estimada en MB para los pesos sinápticos.
 */
export function estimateMemoryUsage(config: BrainConfiguration): number {
  let totalBytes = 0;
  for (const region of Object.values(config.regions)) {
    // Float32 (4 bytes) per weight, neurons × inputs weights
    totalBytes += region.neurons * region.inputs * 4;
  }
  return totalBytes / (1024 * 1024); // MB
}
