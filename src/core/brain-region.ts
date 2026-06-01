/**
 * Clase Base Abstracta para Regiones Cerebrales
 * ================================================
 * Define la interfaz común para todas las regiones funcionales del cerebro digital.
 *
 * Biología: Cada región cerebral (corteza visual, hipocampo, amígdala, etc.)
 * contiene una red de neuronas especializadas con patrones de conectividad y
 * dinámicas particulares. Sin embargo, todas comparten una arquitectura común:
 * reciben señales aferentes, las procesan mediante una SNN local, y emiten
 * señales eferentes. Esta clase abstracta captura esa estructura compartida.
 *
 * Las subclases concretas (VisualCortex, Hippocampus, etc.) implementan el
 * método processInput() con la lógica específica de cada región.
 *
 * Nota: Esta clase referencia SNNNetwork como tipo genérico. Las implementaciones
 * concretas pueden usar BinaryBiologicalNetwork, LargeScaleNetwork, o cualquier
 * variante compatible.
 */

import { SensoryBuffer, type SensoryEntry } from './memory/sensory-buffer.js';
import type { ModulationEffects } from './neuromodulators/modulator-system.js';

/**
 * Actividad instantánea de una región cerebral.
 * Resultado del procesamiento de un paso de simulación.
 */
export interface RegionActivity {
  /** Identificador de la región */
  id: string;
  /** Índices de neuronas activas (que dispararon spike) en este paso */
  activeNeurons: number[];
  /** Tasa de disparo promedio (Hz) de la región */
  firingRate: number;
  /**
   * Nivel de activación percibido (EMA, 0..1). A diferencia de `firingRate`
   * —que con codificación dispersa k-WTA es constante (= sparsity) y por tanto
   * no informa— este valor refleja cuánta señal está recibiendo realmente la
   * región (energía de drive de entrada) suavizada en el tiempo. Es lo que el
   * dashboard usa para las barras "ACTIVIDAD POR REGIÓN".
   */
  drive: number;
  /**
   * Novedad del patrón de disparo (EMA, 0..1). ~0 cuando la región repite el
   * mismo patrón (reposo / atractor estable) y sube cuando un estímulo cambia
   * QUÉ neuronas disparan. Es la señal reactiva del panel de actividad: a
   * diferencia de `firingRate` (constante con k-WTA), responde a la interacción.
   */
  novelty: number;
  /** Marca temporal del paso de simulación */
  timestamp: number;
  /** Spikes de salida para transmitir a otras regiones */
  outputSpikes: Float32Array;
}

/**
 * Configuración de red neural para una región cerebral.
 * Interfaz mínima que debe cumplir cualquier SNN usada como red local.
 */
export interface SNNNetworkConfig {
  /** Número total de neuronas en la red */
  neuronCount: number;
  /** Número de entradas a la red */
  inputCount: number;
  /** Pesos sinápticos (puede ser Float32Array plano o estructura compleja) */
  weights: Float32Array;
}

/**
 * Datos serializados de una región cerebral.
 */
export interface SerializedRegion {
  /** Identificador de la región */
  id: string;
  /** Nombre descriptivo */
  name: string;
  /** Configuración de la red neural */
  network: SNNNetworkConfig;
  /** Tiempo actual de simulación */
  currentTime: number;
}

/**
 * Clase base abstracta para regiones cerebrales del cerebro digital.
 *
 * Cada región contiene:
 * 1. Una red de neuronas de spikes (SNN) con pesos sinápticos propios
 * 2. Un buffer sensorial circular para almacenar entradas recientes
 * 3. Estado de activación y temporización
 *
 * Las subclases concretas deben implementar processInput() con la
 * lógica de procesamiento específica de la región.
 */
export abstract class BrainRegion {
  /** Identificador único de la región (ej: 'visualCortex') */
  public readonly id: string;

  /** Nombre descriptivo legible (ej: 'Corteza Visual Primaria') */
  public readonly name: string;

  /** Número de neuronas en esta región */
  protected readonly neuronCount: number;

  /** Número de entradas a la red neural de esta región */
  protected readonly inputCount: number;

  /** Pesos sinápticos de la red neural local (Float32Array plano) */
  protected weights: Float32Array;

  /** Buffer circular de memoria sensorial para entradas recientes */
  protected sensoryBuffer: SensoryBuffer;

  /** Potenciales de membrana de todas las neuronas */
  protected potentials: Float32Array;

  /** Estado de spike de cada neurona (1.0 = spike, 0.0 = silencio) */
  protected spikes: Float32Array;

  /** Si la región está activa y procesando señales */
  public isActive: boolean = true;

  /** Tiempo actual de simulación (ms) */
  public currentTime: number = 0;

  /** Tasa de aprendizaje base (modificable por neuromoduladores) */
  protected baseLearningRate: number = 0.1;

  /** Umbral de disparo base en mV (modificable por neuromoduladores) */
  protected baseThreshold: number = -55;

  /** Fracción de neuronas activas en el último paso (k-WTA sparsity) */
  protected sparsity: number = 0.1;

  /**
   * Nivel de activación percibido (EMA del drive de entrada, 0..1).
   * Refleja cuánta señal está recibiendo realmente la región, suavizado en el
   * tiempo. Sirve para el panel de actividad: a diferencia de `firingRate`,
   * varía de forma continua con la interacción (texto/webcam/micrófono).
   */
  protected driveEMA: number = 0;

  /**
   * Media móvil del patrón de disparo (qué neuronas suelen ganar). Sirve para
   * medir la NOVEDAD: cuánto se desvía el patrón actual de su régimen habitual.
   * Se asigna perezosamente en el primer `step` (necesita neuronCount).
   */
  protected spikeAvg: Float32Array | null = null;

  /**
   * Novedad del patrón (EMA, 0..1). ~0 cuando la región repite el mismo patrón
   * (reposo / atractor estable) y sube cuando un input cambia QUÉ neuronas
   * disparan. A diferencia de `firingRate` (constante con k-WTA) y de `drive`
   * (dominado por el fondo recurrente), esta señal SÍ reacciona a la
   * interacción: es lo que enciende las barras del panel de actividad.
   */
  protected noveltyEMA: number = 0;

  /**
   * Crea una nueva región cerebral.
   *
   * @param id - Identificador único (ej: 'hippocampus')
   * @param name - Nombre descriptivo (ej: 'Hipocampo - Formación de Memorias')
   * @param neuronCount - Número de neuronas en la región
   * @param inputCount - Número de entradas (dimensión del vector de entrada)
   * @param sensoryCapacity - Capacidad del buffer sensorial (default: 100 entradas)
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

    // Inicializar pesos sinápticos (neuronCount * inputCount pesos)
    this.weights = new Float32Array(neuronCount * inputCount);
    this.initializeWeights();

    // Inicializar vectores de estado
    this.potentials = new Float32Array(neuronCount);
    this.potentials.fill(-70); // Potencial de reposo
    this.spikes = new Float32Array(neuronCount);

    // Buffer sensorial para entradas recientes
    this.sensoryBuffer = new SensoryBuffer(sensoryCapacity, inputCount);
  }

  /**
   * Inicializa los pesos sinápticos con valores aleatorios pequeños.
   *
   * Biología: Las conexiones sinápticas iniciales son débiles y
   * parcialmente aleatorias, refinándose mediante experiencia y
   * plasticidad sináptica (STDP, Hebbian learning).
   */
  protected initializeWeights(): void {
    const initialConnections = Math.min(
      1000,
      this.inputCount
    );

    // Inicialización esparsa: solo conectar un subconjunto
    for (let n = 0; n < this.neuronCount; n++) {
      const baseOffset = n * this.inputCount;
      for (let k = 0; k < initialConnections; k++) {
        const inputIdx = Math.floor(Math.random() * this.inputCount);
        this.weights[baseOffset + inputIdx] = Math.random() * 0.1;
      }
    }
  }

  /**
   * Procesa un vector de spikes de entrada y produce spikes de salida.
   *
   * Método abstracto que cada región concreta debe implementar con
   * su lógica de procesamiento específica.
   *
   * @param spikes - Vector de spikes de entrada (Float32Array)
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Vector de spikes de salida (Float32Array)
   */
  abstract processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects
  ): Float32Array;

  /**
   * Ejecuta un paso de simulación de la región.
   *
   * 1. Recupera entradas recientes del buffer sensorial
   * 2. Aplica modulación a los parámetros de la red
   * 3. Procesa la entrada mediante la SNN local
   * 4. Retorna la actividad resultante
   *
   * @param dt - Paso de tiempo (ms)
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Actividad de la región en este paso
   */
  step(dt: number, modulationEffects: ModulationEffects): RegionActivity {
    this.currentTime += dt;

    if (!this.isActive) {
      // Sin actividad: drive y novedad decaen suavemente hacia 0.
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

    // Recuperar entradas recientes del buffer sensorial
    const recentEntries = this.sensoryBuffer.getRecent(dt * 2);
    let inputSpikes: Float32Array;

    if (recentEntries.length > 0) {
      // Promediar entradas recientes
      inputSpikes = this.averageEntries(recentEntries);
    } else {
      inputSpikes = new Float32Array(this.inputCount);
    }

    // Actualizar el nivel de drive percibido (EMA). Medimos la fracción de
    // canales de entrada con señal: 0 = región en reposo, sube al interactuar.
    let driven = 0;
    for (let i = 0; i < inputSpikes.length; i++) {
      if (inputSpikes[i] > 0) driven++;
    }
    const instDrive =
      inputSpikes.length > 0 ? driven / inputSpikes.length : 0;
    this.driveEMA = this.driveEMA * 0.85 + instDrive * 0.15;

    // Aplicar modulación a parámetros
    this.modulateBy(modulationEffects);

    // Procesar mediante la implementación específica de la región
    const outputSpikes = this.processInput(inputSpikes, modulationEffects);

    // Calcular actividad
    const activeNeurons: number[] = [];
    for (let i = 0; i < outputSpikes.length; i++) {
      if (outputSpikes[i] > 0) {
        activeNeurons.push(i);
      }
    }

    const firingRate =
      this.neuronCount > 0 ? activeNeurons.length / this.neuronCount : 0;

    // Persistir la salida en this.spikes para que getActivity() sea consistente
    // en TODAS las regiones (algunas no actualizaban this.spikes por su cuenta,
    // y el panel las mostraba congeladas a 0%).
    if (outputSpikes.length === this.spikes.length) {
      this.spikes.set(outputSpikes);
    }

    // Novedad del patrón: distancia L1 entre el disparo actual y su media móvil,
    // normalizada por el nº de neuronas activas. Mide si el input cambió QUÉ
    // neuronas ganan (≠ cuántas). Sube al interactuar, ~0 en régimen estable.
    if (!this.spikeAvg || this.spikeAvg.length !== outputSpikes.length) {
      this.spikeAvg = new Float32Array(outputSpikes.length);
    }
    const avg = this.spikeAvg;
    let l1 = 0;
    for (let i = 0; i < outputSpikes.length; i++) {
      const cur = outputSpikes[i] > 0 ? 1 : 0;
      l1 += Math.abs(cur - avg[i]);
      avg[i] = avg[i] * 0.9 + cur * 0.1; // EMA del patrón
    }
    // Normalizar: con k activas, una renovación total del patrón da L1≈2k.
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
   * Promedia múltiples entradas sensoriales recientes.
   *
   * @param entries - Entradas del buffer sensorial
   * @returns Vector promediado
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
   * Retorna la actividad actual de la región.
   *
   * @returns Snapshot de actividad con neuronas activas y tasa de disparo
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
   * Aplica efectos de neuromodulación a los parámetros de la red.
   *
   * Biología: Los neuromoduladores difusos (dopamina, serotonina, etc.)
   * modifican globalmente la excitabilidad neuronal, la plasticidad
   * sináptica y la ganancia de señales en la región.
   *
   * @param effects - Efectos de modulación a aplicar
   */
  modulateBy(effects: ModulationEffects): void {
    // Ajustar umbral de disparo: serotonina y cortisol ↑ → umbral ↑
    const modulatedThreshold = this.baseThreshold * effects.thresholdMultiplier;

    // Ajustar sparsity: más atención → más neuronas pueden activarse
    this.sparsity = Math.min(
      0.3,
      Math.max(0.02, 0.1 * effects.attentionGain)
    );

    // El umbral modulado se usa internamente en processInput
    // (almacenado para que las subclases lo accedan)
    this._modulatedThreshold = modulatedThreshold;
    this._modulatedLearningRate =
      this.baseLearningRate * effects.learningRateMultiplier;
  }

  /** Umbral de disparo después de modulación */
  protected _modulatedThreshold: number = -55;
  /** Tasa de aprendizaje después de modulación */
  protected _modulatedLearningRate: number = 0.1;

  /**
   * Alimenta el buffer sensorial con un nuevo vector de entrada.
   *
   * @param data - Vector de datos sensoriales
   * @param timestamp - Marca temporal de la entrada
   */
  feedInput(data: Float32Array, timestamp?: number): void {
    // Las regiones pueden recibir spikes de tamaño diferente a su inputCount
    // (e.g., el tálamo tiene 3000 neuronas pero la corteza visual espera 5000 entradas).
    // Adaptamos el tamaño: truncar si es mayor, pad con ceros si es menor.
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
   * Serializa el estado de la región para persistencia.
   *
   * @returns Buffer binario con el estado completo de la región
   */
  serialize(): Buffer {
    // Formato: id(string,len-prefixed) + neuronCount(4) + inputCount(4) + weights(Float32Array)
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
   * Restaura el estado de la región desde un buffer serializado.
   *
   * @param data - Buffer con datos previamente serializados
   */
  deserialize(data: Buffer): void {
    let offset = 0;

    // Leer id (length-prefixed)
    const idLen = data.readUInt32LE(offset);
    offset += 4;
    // Saltar el id (ya lo tenemos)
    offset += idLen;

    // Leer neuronCount e inputCount
    const neuronCount = data.readUInt32LE(offset);
    offset += 4;
    const inputCount = data.readUInt32LE(offset);
    offset += 4;

    // Verificar compatibilidad
    if (neuronCount !== this.neuronCount || inputCount !== this.inputCount) {
      throw new Error(
        `[BrainRegion] Incompatibilidad de dimensiones al deserializar '${this.id}': ` +
          `esperado ${this.neuronCount}x${this.inputCount}, ` +
          `recibido ${neuronCount}x${inputCount}`
      );
    }

    // Restaurar pesos
    const weightsByteLen = neuronCount * inputCount * 4;
    const weightsSlice = data.subarray(offset, offset + weightsByteLen);
    this.weights = new Float32Array(
      weightsSlice.buffer,
      weightsSlice.byteOffset,
      neuronCount * inputCount
    );
  }

  /**
   * Retorna la configuración de red neural de la región.
   */
  getNetworkConfig(): SNNNetworkConfig {
    return {
      neuronCount: this.neuronCount,
      inputCount: this.inputCount,
      weights: this.weights,
    };
  }

  /**
   * Reemplaza los pesos sinápticos (restauración desde persistencia).
   * Rechaza tamaños incompatibles para no corromper la red si las dimensiones
   * de la región cambiaron entre versiones guardadas.
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
   * Retorna el número de neuronas en la región.
   */
  get neurons(): number {
    return this.neuronCount;
  }

  /**
   * Retorna el número de entradas a la red.
   */
  get inputs(): number {
    return this.inputCount;
  }
}
