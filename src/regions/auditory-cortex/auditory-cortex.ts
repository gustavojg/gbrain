/**
 * CORTEZA AUDITIVA — Procesamiento de patrones sonoros
 * =====================================================
 * Refactorizado de 07_audio_cortex/server.ts
 *
 * Base biológica:
 *   La corteza auditiva primaria (A1) está organizada tonotópicamente:
 *   las neuronas se disponen según la frecuencia a la que responden
 *   preferentemente, creando un mapa de frecuencias sobre la superficie
 *   cortical (análogo al mapa retinotópico de V1).
 *
 *   La organización tonotópica se hereda de la membrana basilar coclear,
 *   donde las frecuencias altas activan la base y las bajas el ápex.
 *   Las conexiones tálamo→A1 preservan esta organización.
 *
 *   Esta implementación incluye:
 *   - Organización tonotópica: neuronas prefieren frecuencias específicas
 *   - Detección de voz: modula el procesamiento según presencia de voz
 *   - Aprendizaje contrastivo: lateral inhibition sharpens representations
 *   - Auto-aprendizaje: detecta patrones repetidos y les asigna etiquetas
 *   - Buffer de corto plazo: memoria ecoica para detección de repetición
 *
 * Escalabilidad:
 *   5,000 neuronas × 800 inputs = 4M pesos (16MB Float32).
 *   Ligero comparado con corteza visual.
 */

// ====================================================================
// Importaciones del core (existentes en el proyecto)
// ====================================================================

import type { SpikePacket } from '../../core/bus/spike-bus.js';
import { SpikeBus } from '../../core/bus/spike-bus.js';
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { SpikingNeuron, createNeuronPopulation } from '../../core/snn/neuron.js';
import type { NeuronTypeName } from '../../core/snn/neuron.js';

// ====================================================================
// Tipos de la Corteza Auditiva
// ====================================================================

/**
 * Características de voz extraídas del espectrograma.
 *
 * Base biológica:
 *   El sistema auditivo detecta la voz humana mediante la combinación
 *   de energía en el rango de frecuencias fundamentales de la voz
 *   (~85-300 Hz para adultos) y sus armónicos. Las neuronas de A1
 *   responden selectivamente a estos patrones espectro-temporales.
 */
export interface VoiceFeatures {
  /** Si se detecta voz humana en la señal */
  hasVoice: boolean;
  /** Centroide espectral (centro de masa frecuencial) */
  centroid: number;
  /** Energía total de la señal */
  energy: number;
}

/**
 * Memoria auditiva: patrón de neuronas asociado a un sonido.
 */
export interface AuditoryMemory {
  /** Índices de neuronas activas para este patrón */
  pattern: Int32Array;
  /** Etiqueta del sonido */
  label: string;
  /** Fuerza de la memoria (correlaciona con energía vocal) */
  strength: number;
  /** Timestamp de creación */
  createdAt: number;
}

/**
 * Resultado del procesamiento auditivo.
 */
export interface AuditoryProcessingResult {
  /** Tipo de resultado */
  type: 'KNOWN' | 'UNKNOWN' | 'NEW_LEARNING' | 'NO_VOICE';
  /** Neuronas ganadoras */
  winners: Int32Array;
  /** Predicción (label reconocido o '?') */
  prediction: string;
  /** Características de voz detectadas */
  voiceFeatures: VoiceFeatures;
  /** Puntuación de repetición (para auto-aprendizaje) */
  repetitionScore: number;
}

/**
 * Configuración de la corteza auditiva.
 */
export interface AuditoryCortexConfig {
  /** Número de neuronas corticales */
  neuronCount: number;
  /** Tamaño del input (numBands × numFrames del espectrograma) */
  inputCount: number;
  /** Número de bandas de frecuencia en el espectrograma */
  numBands: number;
  /** Número de frames temporales */
  numFrames: number;
  /** Sparsity de la codificación */
  sparsity: number;
  /** Número de neuronas ganadoras en k-WTA */
  kWinners: number;
  /** Factor de fatiga homeostática */
  fatigueFactor: number;
  /** Tasa de aprendizaje base */
  learningRate: number;
  /** Boost de aprendizaje para frecuencias vocales */
  voiceFreqBoost: number;
  /** Presupuesto máximo de peso por neurona */
  maxWeightBudget: number;
  /** Factor de inhibición contrastiva */
  contrastiveInhibitionFactor: number;
  /** Umbral de energía mínima para procesar */
  minEnergyThreshold: number;
  /** Umbral de repetición para auto-aprendizaje */
  autoLearnThreshold: number;
  /** Tamaño del buffer de corto plazo */
  shortTermBufferSize: number;
  /** Rango de bandas vocales [inicio, fin] */
  voiceBandRange: [number, number];
  /** Rango de bandas de alta frecuencia [inicio, fin] */
  highFreqRange: [number, number];
  /** Umbral de overlap para reconocimiento */
  recognitionOverlap: number;
  /** Tipo de neurona */
  neuronType: NeuronTypeName;
}

const DEFAULT_AUDITORY_CONFIG: AuditoryCortexConfig = {
  neuronCount: 5000,
  inputCount: 800,           // 40 bandas × 20 frames
  numBands: 40,
  numFrames: 20,
  sparsity: 0.1,
  kWinners: 3,
  fatigueFactor: 0.3,
  learningRate: 0.3,
  voiceFreqBoost: 1.5,
  maxWeightBudget: 15.0,
  contrastiveInhibitionFactor: 0.05,
  minEnergyThreshold: 0.5,
  autoLearnThreshold: 25,
  shortTermBufferSize: 60,
  voiceBandRange: [4, 24],
  highFreqRange: [30, 39],
  recognitionOverlap: 2,
  neuronType: 'RegularSpiking',
};

// ====================================================================
// Clase AuditoryCortex
// ====================================================================

/**
 * Corteza Auditiva — Procesamiento de patrones sonoros con especialización vocal.
 *
 * Base biológica:
 *   La corteza auditiva procesa sonidos con organización tonotópica
 *   (cada neurona "prefiere" un rango de frecuencias). Se especializa
 *   en detectar patrones temporospectrales como la voz humana.
 *
 *   Mecanismos implementados:
 *
 *   1. **Organización tonotópica**: Neuronas organizadas por frecuencia
 *      preferida. Las neuronas del 10% inferior responden a graves,
 *      las del 10% superior a agudos.
 *      Biología: A1 tiene un mapa tonotópico heredado de la cóclea.
 *
 *   2. **Detección de voz**: Las frecuencias vocales (bandas 4-24 de 40)
 *      reciben un boost de aprendizaje ×1.5. La voz se detecta cuando
 *      la energía low-mid domina sobre el ruido de alta frecuencia.
 *      Biología: El surco temporal superior tiene neuronas selectivas a voz.
 *
 *   3. **Aprendizaje contrastivo**: Los no-ganadores con activación
 *      residual reciben una inhibición que reduce sus pesos para el
 *      input actual. Esto sharpens la separación entre representaciones
 *      de sonidos distintos (ej: "A" vs "O").
 *      Biología: Inhibición lateral via interneuronas GABAérgicas.
 *
 *   4. **Auto-aprendizaje**: Cuando un patrón desconocido se repite
 *      consistentemente (score ≥ 25), se aprende automáticamente con
 *      una etiqueta generada ("Voice-N"). Modela el aprendizaje
 *      implícito de fonemas en bebés.
 *      Biología: Aprendizaje estadístico en la corteza auditiva.
 */
export class AuditoryCortex extends BrainRegion {
  /**
   * Contador de victorias por neurona (homeostasis).
   *
   * Base biológica:
   *   Previene el "winner dominance" donde pocas neuronas acaparan
   *   todas las representaciones. Análogo al escalado sináptico
   *   homeostático observado en cultivos de neuronas corticales.
   */
  private winCounts: Int32Array;

  /** Memorias auditivas almacenadas */
  private memories: AuditoryMemory[] = [];

  /** Configuración de la corteza */
  private config: AuditoryCortexConfig;

  /** Referencia al spike bus */
  private spikeBus: SpikeBus | null = null;

  /**
   * Buffer de corto plazo (memoria ecoica).
   *
   * Base biológica:
   *   La memoria ecoica es un buffer sensorial auditivo que retiene
   *   los últimos ~2-4 segundos de audio para comparación.
   *   Aquí almacenamos los últimos ~60 patrones de ganadores para
   *   detectar repeticiones que indican un patrón consistente.
   */
  private shortTermBuffer: Int32Array[] = [];

  /** Contador de auto-aprendizajes realizados */
  private autoLearnCounter: number = 0;

  /** Últimos ganadores calculados */
  private lastWinners: Int32Array = new Int32Array(0);

  /** Últimos potenciales calculados (para contrastive learning) */
  private lastPotentials: Float32Array = new Float32Array(0);

  /** Neuronas spiking locales de la corteza auditiva (modelo Izhikevich) */
  private localNeurons: SpikingNeuron[];

  /**
   * Crea una nueva corteza auditiva.
   *
   * @param config - Configuración parcial (se mezcla con defaults)
   */
  constructor(config: Partial<AuditoryCortexConfig> = {}) {
    const cfg = { ...DEFAULT_AUDITORY_CONFIG, ...config };
    super('auditoryCortex', 'Corteza Auditiva', cfg.neuronCount, cfg.inputCount);

    this.config = cfg;
    this.winCounts = new Int32Array(cfg.neuronCount);
    this.localNeurons = createNeuronPopulation(cfg.neuronCount, cfg.neuronType);
    this.initializeAuditoryWeights();
  }

  /**
   * Inicializa pesos con valores pequeños.
   *
   * Base biológica:
   *   Inicialización débil simula sinapsis inmaduras pre-experiencia.
   *   Valores más bajos (0.05) que corteza visual para evitar saturación
   *   con el rango dinámico más amplio del audio.
   */
  private initializeAuditoryWeights(): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = Math.random() * 0.05;
    }
  }

  /**
   * Conecta la corteza auditiva al spike bus.
   */
  connectBus(bus: SpikeBus): void {
    this.spikeBus = bus;
    bus.register(this.id);
  }

  /**
   * Calcula características de voz a partir del espectrograma.
   *
   * Base biológica:
   *   Las neuronas del surco temporal superior (STS) son selectivas
   *   a la voz humana. Detectan la combinación de:
   *   - Energía suficiente (señal > ruido de fondo)
   *   - Concentración en frecuencias low-mid (voz humana: 85-3000 Hz)
   *   - Baja energía relativa en altas frecuencias (no es ruido blanco)
   *
   * @param inputPattern - Espectrograma flat [time × freq]
   * @returns Características vocales detectadas
   */
  calculateVoiceFeatures(inputPattern: Float32Array): VoiceFeatures {
    const { numBands, voiceBandRange, highFreqRange } = this.config;

    // Último frame del espectrograma (más reciente)
    const lastFrameStart = inputPattern.length - numBands;
    const lastFrame = inputPattern.subarray(
      Math.max(0, lastFrameStart),
      inputPattern.length,
    );

    // 1. Energía total
    let energy = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      energy += lastFrame[i];
    }

    // 2. Centroide espectral (centro de masa frecuencial)
    let weightedSum = 0;
    let sum = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      weightedSum += i * lastFrame[i];
      sum += lastFrame[i];
    }
    const centroid = sum > 0 ? weightedSum / sum : 0;

    // 3. Energía en bandas vocales (low-mid)
    let lowMidEnergy = 0;
    for (let i = voiceBandRange[0]; i <= voiceBandRange[1] && i < lastFrame.length; i++) {
      lowMidEnergy += lastFrame[i];
    }

    // 4. Energía en altas frecuencias (ruido)
    let highFreqEnergy = 0;
    for (let i = highFreqRange[0]; i <= highFreqRange[1] && i < lastFrame.length; i++) {
      highFreqEnergy += lastFrame[i];
    }

    // Criterios de voz
    const isNotJustNoise = highFreqEnergy < (energy * 0.6);
    const hasVoice = energy > this.config.minEnergyThreshold &&
      lowMidEnergy > (energy * 0.25) &&
      isNotJustNoise;

    return { hasVoice, centroid, energy };
  }

  /**
   * Procesa un input auditivo recibido del tálamo.
   *
   * Pipeline:
   * 1. Detección de voz (gate: si no hay voz, no procesar)
   * 2. Feed-forward con ponderación por voz y homeostasis
   * 3. k-WTA (winner-take-all)
   *
   * @param input - Espectrograma como vector de spikes (Float32Array)
   * @param dt - Paso temporal (ms)
   * @param timestamp - Tiempo de simulación (ms)
   * @returns Vector de activación cortical
   */
  processInput(spikes: Float32Array, _modulationEffects: ModulationEffects): Float32Array {
    // Gate: solo procesar si hay señal significativa
    const voiceFeatures = this.calculateVoiceFeatures(spikes);

    const localPotentials = new Float32Array(this.neuronCount);
    const activeSpikes = new Float32Array(this.neuronCount);

    if (!voiceFeatures.hasVoice) {
      // Sin voz → corteza silente (todas las neuronas en reposo)
      for (let i = 0; i < this.localNeurons.length; i++) {
        this.localNeurons[i].fired = false;
      }
      this.lastWinners = new Int32Array(0);
      this.lastPotentials = localPotentials;
      return activeSpikes;
    }

    // --- Feed-forward con umbral de activación ---
    for (let n = 0; n < this.neuronCount; n++) {
      let sum = 0;
      const offset = n * this.inputCount;

      // Producto punto sparse: solo valores significativos (> 0.1)
      for (let i = 0; i < this.inputCount; i++) {
        if (spikes[i] > 0.1) {
          sum += spikes[i] * this.weights[offset + i];
        }
      }

      // Homeostasis: fatiga más suave que corteza visual
      const fatigue = this.winCounts[n] * this.config.fatigueFactor;
      localPotentials[n] = Math.max(0, sum - fatigue);
    }

    // --- Winner-Take-All ---
    const indices = new Int32Array(this.neuronCount);
    for (let i = 0; i < this.neuronCount; i++) indices[i] = i;
    indices.sort((a, b) => localPotentials[b] - localPotentials[a]);

    const winners = new Int32Array(this.config.kWinners);
    for (let i = 0; i < this.config.kWinners; i++) {
      winners[i] = indices[i];
    }

    // Actualizar estado neuronal
    for (let i = 0; i < this.localNeurons.length; i++) {
      this.localNeurons[i].fired = false;
    }
    for (let i = 0; i < this.config.kWinners; i++) {
      const winnerIdx = winners[i];
      this.localNeurons[winnerIdx].fired = true;
      activeSpikes[winnerIdx] = 1.0;
    }

    this.lastWinners = winners;
    this.lastPotentials = localPotentials;
    return activeSpikes;
  }

  /**
   * Aprende un patrón auditivo con etiqueta.
   *
   * Base biológica:
   *   Aprendizaje supervisado de patrones auditivos con:
   *
   *   1. **Boost vocal**: Las frecuencias en el rango de voz humana
   *      (bandas 4-24) reciben un aprendizaje ×1.5 más fuerte,
   *      modelando la especialización del STS para voz.
   *
   *   2. **Modulación por dopamina**: La recompensa (dopamina alta)
   *      amplifica la tasa de aprendizaje. Biológicamente, la dopamina
   *      facilita LTP en vías corticostriatales.
   *
   *   3. **Inhibición contrastiva (lateral inhibition)**:
   *      Los no-ganadores que tenían activación residual reciben
   *      una supresión de sus pesos. Esto sharpens la distinción
   *      entre fonemas similares (ej: /b/ vs /d/).
   *      Biología: Inhibición lateral via interneuronas PV+.
   *
   *   4. **Decaimiento global**: Todos los pesos decaen lentamente
   *      (×0.995), modelando la degradación sináptica natural.
   *
   * @param input - Espectrograma como vector
   * @param label - Etiqueta del sonido
   * @param dt - Paso temporal
   * @param timestamp - Tiempo de simulación
   * @param modulationEffects - Efectos de neuromodulación
   * @returns Resultado del procesamiento
   */
  learn(
    input: Float32Array,
    label: string,
    dt: number,
    timestamp: number,
    modulationEffects?: ModulationEffects,
  ): AuditoryProcessingResult {
    // Verificar tamaño de input
    if (input.length !== this.inputCount) {
      return {
        type: 'NO_VOICE',
        winners: new Int32Array(0),
        prediction: 'ERROR',
        voiceFeatures: { hasVoice: false, centroid: 0, energy: 0 },
        repetitionScore: 0,
      };
    }

    const voiceFeatures = this.calculateVoiceFeatures(input);
    if (!voiceFeatures.hasVoice) {
      return {
        type: 'NO_VOICE',
        winners: new Int32Array(0),
        prediction: '?',
        voiceFeatures,
        repetitionScore: 0,
      };
    }

    // 1. Procesar input
    const defaultMod: ModulationEffects = {
      learningRateMultiplier: 1.0,
      thresholdMultiplier: 1.0,
      attentionGain: 1.0,
      spikeGainMultiplier: 1.0,
      consolidationRate: 1.0,
      socialWeightBoost: 1.0,
    };
    this.processInput(input, defaultMod);
    const winners = this.lastWinners;
    const potentials = this.lastPotentials;

    // 2. Tasa de aprendizaje modulada por dopamina
    const lrMultiplier = modulationEffects?.learningRateMultiplier ?? 1.0;
    const effectiveLR = this.config.learningRate + (lrMultiplier - 1.0) * 0.4;

    // 3. Aprendizaje selectivo con boost vocal
    const { numBands, voiceBandRange } = this.config;

    for (let w = 0; w < winners.length; w++) {
      const winnerIdx = winners[w];
      this.winCounts[winnerIdx]++;

      const offset = winnerIdx * this.inputCount;
      let totalWeight = 0;

      for (let i = 0; i < this.inputCount; i++) {
        if (input[i] > 0.15) {
          // Determinar si este bin está en el rango vocal
          const freqIndex = i % numBands;
          const isVoiceFreq = freqIndex >= voiceBandRange[0] && freqIndex <= voiceBandRange[1];
          const boost = isVoiceFreq ? this.config.voiceFreqBoost : 0.5;

          // LTP con boost vocal
          this.weights[offset + i] += input[i] * effectiveLR * boost;
        }

        // Decaimiento global (degradación sináptica natural)
        this.weights[offset + i] *= 0.995;

        // Clamp a no-negativo (la corteza auditiva no usa pesos negativos)
        if (this.weights[offset + i] < 0) this.weights[offset + i] = 0;
        totalWeight += this.weights[offset + i];
      }

      // Normalización sináptica
      if (totalWeight > this.config.maxWeightBudget) {
        const factor = this.config.maxWeightBudget / totalWeight;
        for (let i = 0; i < this.inputCount; i++) {
          this.weights[offset + i] *= factor;
        }
      }
    }

    // 4. Almacenar memoria auditiva
    this.memories.push({
      pattern: new Int32Array(winners),
      label,
      strength: voiceFeatures.energy,
      createdAt: timestamp,
    });

    // 5. Aprendizaje contrastivo (inhibición lateral)
    this.applyContrastiveInhibition(input, winners, potentials, effectiveLR);

    // 6. Reconocimiento
    const predictions = this.recognizePattern(winners);

    return {
      type: predictions.length > 0 ? 'KNOWN' : 'UNKNOWN',
      winners,
      prediction: predictions.length > 0 ? predictions.join(', ') : '?',
      voiceFeatures,
      repetitionScore: 0,
    };
  }

  /**
   * Aplica inhibición contrastiva a los no-ganadores.
   *
   * Base biológica:
   *   Las interneuronas inhibitorias PV+ (parvalbumin-positivas) en
   *   la corteza auditiva proporcionan inhibición feedforward y lateral
   *   que sharpens la selectividad frecuencial. Los no-ganadores que
   *   tenían activación residual son suprimidos, reforzando la
   *   separación entre representaciones de fonemas.
   *
   * @param input - Input del espectrograma
   * @param winners - Índices de neuronas ganadoras
   * @param potentials - Potenciales de todas las neuronas
   * @param learningRate - Tasa de aprendizaje efectiva
   */
  private applyContrastiveInhibition(
    input: Float32Array,
    winners: Int32Array,
    potentials: Float32Array,
    learningRate: number,
  ): void {
    const inhibitionFactor = this.config.contrastiveInhibitionFactor * learningRate;
    const winnerSet = new Set<number>();
    for (let w = 0; w < winners.length; w++) {
      winnerSet.add(winners[w]);
    }

    for (let n = 0; n < this.neuronCount; n++) {
      // Solo afectar neuronas que tenían activación pero no ganaron
      if (!winnerSet.has(n) && potentials[n] > 0) {
        const offset = n * this.inputCount;

        // Reducir levemente los pesos para inputs activos
        for (let i = 0; i < this.inputCount; i++) {
          if (input[i] > 0.1) {
            this.weights[offset + i] -= input[i] * inhibitionFactor;
            if (this.weights[offset + i] < 0) {
              this.weights[offset + i] = 0;
            }
          }
        }
      }
    }
  }

  /**
   * Reconoce un patrón comparando con memorias almacenadas.
   *
   * Base biológica:
   *   El reconocimiento auditivo usa overlap de engramas con umbral
   *   dinámico basado en la fuerza de la memoria. Memorias más fuertes
   *   (aprendidas con más energía vocal) requieren menos overlap.
   *
   * @param currentWinners - Neuronas activas actualmente
   * @returns Lista de etiquetas reconocidas
   */
  private recognizePattern(currentWinners: Int32Array): string[] {
    const matches: string[] = [];
    const seenLabels = new Set<string>();

    for (const mem of this.memories) {
      let overlap = 0;

      for (let w = 0; w < currentWinners.length; w++) {
        for (let m = 0; m < mem.pattern.length; m++) {
          if (currentWinners[w] === mem.pattern[m]) {
            overlap++;
            break;
          }
        }
      }

      // Umbral dinámico basado en fuerza de la memoria
      const threshold = Math.max(
        this.config.recognitionOverlap,
        3 - (mem.strength / 10),
      );

      if (overlap >= threshold && !seenLabels.has(mem.label)) {
        matches.push(mem.label);
        seenLabels.add(mem.label);
      }
    }

    return matches;
  }

  /**
   * Procesamiento automático sin supervisión (auto-aprendizaje).
   *
   * Base biológica:
   *   Los bebés aprenden a distinguir fonemas de su lengua materna
   *   sin supervisión explícita, mediante exposición repetida.
   *   El cerebro detecta regularidades estadísticas en el input
   *   auditivo y forma categorías perceptuales automáticamente.
   *
   *   Implementamos esto detectando patrones que se repiten
   *   consistentemente en un buffer de corto plazo (memoria ecoica).
   *   Cuando la puntuación de repetición supera un umbral, se
   *   auto-aprende con una etiqueta generada.
   *
   * @param input - Espectrograma como vector
   * @param dt - Paso temporal
   * @param timestamp - Tiempo de simulación
   * @returns Resultado del procesamiento con detección de repetición
   */
  autoProcess(input: Float32Array, dt: number, timestamp: number): AuditoryProcessingResult {
    const voiceFeatures = this.calculateVoiceFeatures(input);

    // Sin voz o muy poca energía → ignorar
    if (!voiceFeatures.hasVoice || voiceFeatures.energy < this.config.minEnergyThreshold * 1.6) {
      return {
        type: 'NO_VOICE',
        winners: new Int32Array(0),
        prediction: '?',
        voiceFeatures,
        repetitionScore: 0,
      };
    }

    // Procesar input
    const defaultMod2: ModulationEffects = {
      learningRateMultiplier: 1.0,
      thresholdMultiplier: 1.0,
      attentionGain: 1.0,
      spikeGainMultiplier: 1.0,
      consolidationRate: 1.0,
      socialWeightBoost: 1.0,
    };
    this.processInput(input, defaultMod2);
    const winners = this.lastWinners;

    // Verificar si ya es conocido
    const predictions = this.recognizePattern(winners);
    if (predictions.length > 0) {
      return {
        type: 'KNOWN',
        winners,
        prediction: predictions.join(', '),
        voiceFeatures,
        repetitionScore: 0,
      };
    }

    // --- Detección de repetición para auto-aprendizaje ---
    this.shortTermBuffer.push(new Int32Array(winners));
    if (this.shortTermBuffer.length > this.config.shortTermBufferSize) {
      this.shortTermBuffer.shift();
    }

    let repetitionScore = 0;
    // Examinar historial reciente (~1.5 segundos)
    const recentCount = Math.min(30, this.shortTermBuffer.length);
    const recentStart = this.shortTermBuffer.length - recentCount;

    for (let h = recentStart; h < this.shortTermBuffer.length; h++) {
      const pastWinners = this.shortTermBuffer[h];
      let overlap = 0;

      for (let w = 0; w < winners.length; w++) {
        for (let p = 0; p < pastWinners.length; p++) {
          if (winners[w] === pastWinners[p]) {
            overlap++;
            break;
          }
        }
      }

      // Sistema de puntuación granular
      if (overlap >= 2) {
        repetitionScore += 3;     // Coincidencia fuerte
      } else if (overlap >= 1) {
        repetitionScore += 1.5;   // Coincidencia parcial
      }
    }

    // Auto-aprender si se repite consistentemente
    if (repetitionScore >= this.config.autoLearnThreshold) {
      this.autoLearnCounter++;
      const newLabel = `Voice-${this.autoLearnCounter}`;

      // Aprender con etiqueta automática
      this.learn(input, newLabel, dt, timestamp);

      // Limpiar buffer parcialmente para no reaprender inmediatamente
      this.shortTermBuffer = this.shortTermBuffer.slice(
        -Math.floor(this.config.shortTermBufferSize / 4),
      );

      return {
        type: 'NEW_LEARNING',
        winners,
        prediction: newLabel,
        voiceFeatures,
        repetitionScore,
      };
    }

    return {
      type: 'UNKNOWN',
      winners,
      prediction: '?',
      voiceFeatures,
      repetitionScore,
    };
  }

  /**
   * Envía la representación cortical actual al spike bus.
   *
   * @param timestamp - Tiempo de simulación
   * @param targets - Regiones destino (default: hippocampus + amygdala)
   */
  emitToDownstream(timestamp: number, targets: string[] = ['hippocampus', 'amygdala']): void {
    if (!this.spikeBus) return;

    const spikes = new Float32Array(this.neuronCount);
    for (let i = 0; i < this.lastWinners.length; i++) {
      spikes[this.lastWinners[i]] = 1.0;
    }

    this.spikeBus.send({
      source: this.id,
      targets,
      spikes,
      timestamp,
      metadata: {
        winnerCount: this.lastWinners.length,
        activity: this.getLocalActivity(),
        autoLearnCount: this.autoLearnCounter,
      },
    });
  }

  /**
   * Obtiene estadísticas de la corteza auditiva.
   */
  /**
   * Obtiene la fracción de actividad local (0-1) basada en las neuronas spiking.
   */
  getLocalActivity(): number {
    let active = 0;
    for (let i = 0; i < this.localNeurons.length; i++) {
      if (this.localNeurons[i].fired) active++;
    }
    return active / this.localNeurons.length;
  }

  getStats(): {
    memoryCount: number;
    averageWeight: number;
    maxFatigue: number;
    autoLearnCount: number;
    shortTermBufferSize: number;
    activity: number;
  } {
    let weightSum = 0;
    for (let i = 0; i < this.weights.length; i++) {
      weightSum += this.weights[i];
    }

    let maxFatigue = 0;
    for (let i = 0; i < this.winCounts.length; i++) {
      if (this.winCounts[i] > maxFatigue) maxFatigue = this.winCounts[i];
    }

    return {
      memoryCount: this.memories.length,
      averageWeight: weightSum / this.weights.length,
      maxFatigue,
      autoLearnCount: this.autoLearnCounter,
      shortTermBufferSize: this.shortTermBuffer.length,
      activity: this.getLocalActivity(),
    };
  }

  /** Número de memorias almacenadas */
  get memoryCount(): number {
    return this.memories.length;
  }

  /**
   * Resetea la fatiga homeostática.
   *
   * Base biológica:
   *   Equivalente al efecto restaurador del sueño sobre la
   *   homeostasis sináptica de la corteza auditiva.
   */
  resetFatigue(): void {
    this.winCounts.fill(0);
  }

  /**
   * Limpia el buffer de corto plazo (memoria ecoica).
   */
  clearShortTermBuffer(): void {
    this.shortTermBuffer = [];
  }
}
