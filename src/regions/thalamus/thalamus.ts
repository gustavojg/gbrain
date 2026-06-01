/**
 * TÁLAMO — Filtro Atencional y Relé Sensorial
 * =============================================
 * Refactorizado de 05_attention_scale/attention_brain_system.ts
 *
 * Base biológica:
 *   El tálamo es la "estación de relé" central del cerebro. Casi toda la
 *   información sensorial (excepto el olfato) pasa por el tálamo antes de
 *   llegar a la corteza. No es un relé pasivo: filtra activamente las señales,
 *   seleccionando las más salientes según el estado atencional.
 *
 *   El núcleo reticular talámico (NRT) actúa como compuerta inhibitoria,
 *   permitiendo pasar solo las señales más fuertes o relevantes. La
 *   acetilcolina del núcleo basal de Meynert modula esta compuerta:
 *   más ACh → compuerta más permisiva → más inputs pasan al córtex.
 *
 *   Implementamos esto como un filtro top-K con umbral de ruido,
 *   donde K (el bottleneck) se modula por los niveles de acetilcolina.
 *
 * Escalabilidad:
 *   Diseñado para manejar 50,000+ inputs sensoriales y filtrarlos
 *   a ~200 señales salientes usando un pase de umbral + sort parcial,
 *   evitando el costo de O(N log N) en el caso promedio.
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
// Tipos del Tálamo
// ====================================================================

/**
 * Tipo de señal sensorial para enrutamiento.
 * El tálamo enruta señales a diferentes cortezas según la modalidad.
 */
export type SensoryModality = 'visual' | 'auditory' | 'linguistic';

/**
 * Resultado del procesamiento talámico.
 * Contiene los índices salientes y el vector filtrado.
 */
export interface ThalamusOutput {
  /** Índices de los inputs más salientes seleccionados */
  salientIndices: Int32Array;
  /** Vector filtrado con solo los valores salientes (resto en 0) */
  filteredInput: Float32Array;
  /** Número de inputs que superaron el umbral de ruido */
  candidateCount: number;
  /** Tamaño del bottleneck efectivo (modulado por ACh) */
  effectiveBottleneck: number;
}

/**
 * Configuración específica del tálamo.
 */
export interface ThalamusConfig {
  /** Número de neuronas talámicas */
  neuronCount: number;
  /** Tamaño total del vector de entrada sensorial */
  totalInputSize: number;
  /** Número máximo de señales que pasan el filtro atencional */
  bottleneckSize: number;
  /** Umbral de ruido de fondo (señales por debajo se ignoran) */
  noiseFloor: number;
  /** Tipo de neurona (default: FastSpiking para respuesta rápida) */
  neuronType: NeuronTypeName;
  /** Sparsity de la codificación (fracción de neuronas activas) */
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
// Clase Thalamus
// ====================================================================

/**
 * Tálamo — Filtro atencional y relé sensorial del cerebro digital.
 *
 * Base biológica:
 *   El tálamo recibe ~50,000 inputs sensoriales simultáneos y los filtra
 *   a ~200 señales salientes que se envían a las cortezas especializadas.
 *   Este filtrado es modulado por:
 *
 *   1. **Acetilcolina** (ACh): Aumenta el bottleneck → más información pasa.
 *      Biológicamente, ACh del núcleo basal facilita la transmisión talámica
 *      al despolarizar neuronas de relé y inhibir al NRT.
 *
 *   2. **Ganancia atencional** (NE + ACh): Amplifica las señales salientes.
 *      Norepinefrina del locus coeruleus aumenta la relación señal/ruido.
 *
 *   3. **Umbral de ruido**: Filtra señales subliminales (< 0.1).
 *      Modela el umbral de disparo del NRT.
 *
 * Rendimiento:
 *   Para 50K inputs, usa un pase de umbral O(N) seguido de sort O(M log M)
 *   donde M << N (solo candidatos supraumbral), evitando sort completo.
 */
export class Thalamus extends BrainRegion {
  /** Número máximo base de señales que pasan el filtro */
  private attentionBottleneck: number;

  /** Umbral de ruido — señales por debajo se consideran ruido de fondo */
  private noiseFloor: number;

  /** Sparsity de la codificación interna */
  private thalamusSparsity: number;

  /** Referencia al spike bus para enrutar señales a cortezas */
  private spikeBus: SpikeBus | null = null;

  /** Último output calculado (para consultas externas) */
  private lastOutput: ThalamusOutput | null = null;

  /** Neuronas spiking locales del tálamo (modelo Izhikevich) */
  private localNeurons: SpikingNeuron[];

  /**
   * Crea una nueva instancia del tálamo.
   *
   * @param config - Configuración parcial (se mezcla con defaults)
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
   * Inicializa los pesos sinápticos con conectividad sparse.
   *
   * Base biológica:
   *   Las neuronas talámicas de relé no están conectadas a TODOS los inputs,
   *   sino que cada neurona recibe aferencias de un subconjunto de fibras
   *   sensoriales. Inicializamos ~1000 conexiones aleatorias por neurona.
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
   * Conecta el tálamo al spike bus para enrutamiento inter-regional.
   *
   * @param bus - Instancia del SpikeBus central
   */
  connectBus(bus: SpikeBus): void {
    this.spikeBus = bus;
    bus.register(this.id);
  }

  /**
   * Filtra un vector masivo de entrada sensorial a las top-K señales salientes.
   *
   * Base biológica:
   *   El núcleo reticular talámico (NRT) inhibe las neuronas de relé
   *   cuyos inputs son débiles, permitiendo pasar solo las señales que
   *   superan un umbral. La acetilcolina modula este umbral:
   *   - ACh alto → umbral más bajo → más señales pasan (atención amplia)
   *   - ACh bajo → umbral más alto → menos señales (atención focalizada)
   *
   *   Este proceso es análogo a la inhibición lateral en el NRT,
   *   implementado aquí como thresholding + selección top-K.
   *
   * @param rawInputs - Vector de entrada sensorial completo (50K+ valores, 0.0-1.0)
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Resultado del filtro atencional
   */
  processAttention(rawInputs: Float32Array, modulationEffects?: ModulationEffects): ThalamusOutput {
    // --- Modulación del bottleneck por acetilcolina ---
    // ACh alto → attentionGain alto → K más grande → más inputs pasan
    const achGain = modulationEffects?.attentionGain ?? 1.0;
    const effectiveK = Math.floor(this.attentionBottleneck * achGain);

    // --- Pase 1: Umbral de ruido (O(N), filtra ~90% de los inputs) ---
    // Estructura para candidatos: índice + valor para sort
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

    // --- Pase 2: Selección Top-K (sort parcial de M candidatos) ---
    // Crear array de índices para sorting (solo los M candidatos)
    const sortIndices = new Int32Array(candidateCount);
    for (let i = 0; i < candidateCount; i++) sortIndices[i] = i;

    // Sort descendente por valor
    sortIndices.sort((a, b) => candidateValues[b] - candidateValues[a]);

    // --- Construir output filtrado ---
    const k = Math.min(effectiveK, candidateCount);
    const salientIndices = new Int32Array(k);
    const filteredInput = new Float32Array(rawInputs.length);

    for (let i = 0; i < k; i++) {
      const candidateIdx = sortIndices[i];
      const originalIdx = candidateIndices[candidateIdx];
      salientIndices[i] = originalIdx;

      // Escalar por ganancia atencional (NE + ACh amplifican señales salientes)
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
   * Determina a qué región cortical enrutar la señal según su modalidad.
   *
   * Base biológica:
   *   El tálamo tiene núcleos especializados para cada modalidad:
   *   - Núcleo geniculado lateral (NGL) → Corteza visual V1
   *   - Núcleo geniculado medial (NGM) → Corteza auditiva A1
   *   - Núcleo ventral posterior → Corteza somatosensorial
   *   - Pulvinar → Áreas de asociación (aquí: Broca-Wernicke)
   *
   * @param inputType - Modalidad de la señal sensorial
   * @returns Identificador de la región destino
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
   * Procesa un input sensorial completo: filtra y enruta.
   *
   * Base biológica:
   *   Pipeline completo del relé talámico:
   *   1. Recibe input sensorial crudo (retina, cóclea, etc.)
   *   2. Aplica filtro atencional (NRT + modulación ACh)
   *   3. Procesa a través de neuronas de relé (SNN interna)
   *   4. Envía spikes resultantes a cortezas via tractos axonales
   *
   * @param input - Vector de entrada sensorial crudo
   * @param dt - Paso temporal (ms)
   * @param timestamp - Tiempo actual de simulación (ms)
   * @returns Vector filtrado por la atención
   */
  processInput(spikes: Float32Array, _modulationEffects: ModulationEffects): Float32Array {
    // 1. Aplicar filtro atencional (sin modulación por ahora)
    const attentionResult = this.processAttention(spikes);

    // 2. Procesar a través de neuronas talámicas (computación sparse)
    const localPotentials = new Float32Array(this.neuronCount);

    for (let n = 0; n < this.neuronCount; n++) {
      let excitation = 0;
      const offset = n * this.inputCount;

      // Producto punto sparse: solo calcular para índices salientes
      for (let i = 0; i < attentionResult.salientIndices.length; i++) {
        const idx = attentionResult.salientIndices[i];
        excitation += this.weights[offset + idx] * attentionResult.filteredInput[idx];
      }

      // Ruido neuronal (variabilidad intrínseca)
      excitation += Math.random() * 0.01;
      localPotentials[n] = excitation;
    }

    // 3. Winner-Take-All (inhibición lateral via NRT)
    const k = Math.max(1, Math.floor(this.neuronCount * this.thalamusSparsity));
    const indices = new Int32Array(this.neuronCount);
    for (let i = 0; i < this.neuronCount; i++) indices[i] = i;
    indices.sort((a, b) => localPotentials[b] - localPotentials[a]);

    // Activar solo top-k neuronas
    const activeSpikes = new Float32Array(this.neuronCount);
    for (let i = 0; i < k; i++) {
      const winnerIdx = indices[i];
      this.localNeurons[winnerIdx].fired = true;
      activeSpikes[winnerIdx] = 1.0;
    }

    // Resetear no-ganadores
    for (let i = k; i < this.neuronCount; i++) {
      this.localNeurons[indices[i]].fired = false;
    }

    return activeSpikes;
  }

  /**
   * Procesa y envía señal filtrada a una corteza específica via spike bus.
   *
   * @param rawInput - Input sensorial crudo
   * @param modality - Tipo de señal (visual, auditory, linguistic)
   * @param dt - Paso temporal
   * @param timestamp - Tiempo de simulación
   * @param modulationEffects - Efectos neuromoduladores opcionales
   */
  processAndRoute(
    rawInput: Float32Array,
    modality: SensoryModality,
    dt: number,
    timestamp: number,
    modulationEffects?: ModulationEffects,
  ): ThalamusOutput {
    // 1. Filtro atencional con modulación
    const attentionResult = this.processAttention(rawInput, modulationEffects);

    // 2. Procesar neuronas talámicas
    const defaultMod: ModulationEffects = {
      learningRateMultiplier: 1.0,
      thresholdMultiplier: 1.0,
      attentionGain: 1.0,
      spikeGainMultiplier: 1.0,
      consolidationRate: 1.0,
      socialWeightBoost: 1.0,
    };
    const activeSpikes = this.processInput(rawInput, modulationEffects ?? defaultMod);

    // 3. Enrutar via spike bus
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
   * Envía un broadcast a múltiples cortezas simultáneamente.
   *
   * Base biológica:
   *   Algunos estímulos (ej: un sonido fuerte repentino) activan
   *   múltiples vías talámicas simultáneamente, incluyendo la vía
   *   rápida tálamo→amígdala para respuesta emocional inmediata.
   *
   * @param rawInput - Input sensorial
   * @param targets - Lista de regiones destino
   * @param timestamp - Tiempo de simulación
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
   * Obtiene el último resultado del filtro atencional.
   */
  getLastOutput(): ThalamusOutput | null {
    return this.lastOutput;
  }

  /**
   * Obtiene estadísticas del tálamo para monitoreo.
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
