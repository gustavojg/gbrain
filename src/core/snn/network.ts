/**
 * RED NEURONAL PULSANTE (SNN) CON INHIBICIÓN LATERAL k-WTA
 * ==========================================================
 * Red de neuronas Izhikevich con almacenamiento plano de pesos en Float32Array,
 * inhibición lateral tipo k-Winners-Take-All, y aprendizaje STDP.
 *
 * Base biológica:
 *   La corteza cerebral organiza sus neuronas en columnas corticales donde
 *   la inhibición lateral (mediada por interneuronas GABAérgicas) asegura
 *   que solo un subconjunto "ganador" de neuronas (~5-15%) esté activo
 *   simultáneamente. Este mecanismo:
 *
 *   - Produce representaciones dispersas (sparse coding), maximizando
 *     la capacidad de almacenamiento de la red.
 *   - Implementa competencia entre neuronas, forzando la especialización.
 *   - Reduce el consumo energético (el cerebro real gasta ~20W con
 *     solo ~5% de neuronas activas en un instante dado).
 *
 *   El almacenamiento en Float32Array imita la organización densa de
 *   la matriz sináptica cortical, donde cada neurona excitadora recibe
 *   ~10,000 sinapsis de neuronas aferentes.
 *
 * Rendimiento:
 *   - Pesos en Float32Array plano: cache-friendly, ~4 bytes/sinapsis.
 *   - k-WTA con quickselect parcial: O(n) en promedio vs O(n log n) de sort.
 *   - Para 10,000 neuronas × 1,000 inputs = 40 MB de pesos.
 *
 * Referencia de arquitectura:
 *   - Patrón Float32Array: /04_binary_optimization/binary_brain_system.ts
 *   - k-WTA: /02_distributed_network/distributed_network_complete.ts
 */

import { SpikingNeuron, createNeuronPopulation, type NeuronTypeName } from './neuron.js';
import { computeSTDP, DEFAULT_STDP_PARAMS, type STDPParams } from './synapse.js';

// ====================================================================
// Interfaces
// ====================================================================

/**
 * Configuración para crear una red SNN.
 */
export interface SNNNetworkConfig {
  /** Número de neuronas en la red */
  readonly neuronCount: number;
  /** Número de entradas externas a la red */
  readonly inputCount: number;
  /** Fracción de neuronas que pueden ganar la competición k-WTA (0.0-1.0) */
  readonly sparsity: number;
  /** Tipo de neurona Izhikevich predominante */
  readonly neuronType: NeuronTypeName;
  /** Parámetros STDP opcionales */
  readonly stdp?: STDPParams;
  /** Peso máximo de sinapsis */
  readonly maxWeight?: number;
  /** Peso mínimo de sinapsis */
  readonly minWeight?: number;
  /** Presupuesto sináptico total para normalización */
  readonly synapticBudget?: number;
}

/**
 * Formato de datos serializado de la red para persistencia.
 *
 * Base biológica:
 *   Análogo a una "instantánea" del estado sináptico completo,
 *   similar a cómo las consolidación durante el sueño preserva
 *   las conexiones fortalecidas durante la vigilia.
 */
export interface SerializedNetwork {
  /** Número de neuronas */
  readonly neuronCount: number;
  /** Número de inputs */
  readonly inputCount: number;
  /** Esparcidad k-WTA */
  readonly sparsity: number;
  /** Tipo de neurona */
  readonly neuronType: NeuronTypeName;
  /** Pesos como ArrayBuffer (para serialización binaria) */
  readonly weightsBuffer: ArrayBuffer;
  /** Potenciales de membrana de las neuronas */
  readonly voltages: ArrayBuffer;
  /** Variables de recuperación de las neuronas */
  readonly recovery: ArrayBuffer;
}

// ====================================================================
// Clase SNNNetwork
// ====================================================================

/**
 * Red neuronal pulsante con k-WTA y STDP.
 *
 * Base biológica:
 *   Modela una población cortical homogénea (ej: una capa de corteza
 *   visual, una región del hipocampo) como un conjunto de neuronas
 *   Izhikevich densamente conectadas a un array de entradas, con
 *   inhibición lateral que asegura representaciones dispersas.
 *
 *   La combinación de:
 *   1. Neuronas Izhikevich (dinámica de membrana realista)
 *   2. k-WTA (inhibición lateral cortical)
 *   3. STDP (plasticidad Hebbiana con temporalidad)
 *   4. Normalización sináptica (homeostasis)
 *
 *   produce un sistema que auto-organiza representaciones dispersas
 *   estables para patrones de entrada, análogo a las columnas de
 *   orientación en V1 o las place cells en hipocampo.
 */
export class SNNNetwork {
  /** Neuronas Izhikevich de la red */
  public readonly neurons: SpikingNeuron[];

  /**
   * Matriz de pesos sinápticos en formato plano.
   *
   * Organización: weights[neuronIdx * inputCount + inputIdx]
   * Esto es equivalente a una matriz (neuronCount × inputCount) almacenada
   * por filas (row-major), optimizada para acceso cache-friendly.
   */
  public readonly weights: Float32Array;

  /** Número de neuronas en la red */
  public readonly neuronCount: number;

  /** Número de entradas externas */
  public readonly inputCount: number;

  /** Fracción de neuronas activas (k-WTA) */
  public readonly sparsity: number;

  /** Parámetros STDP */
  private readonly stdp: STDPParams;

  /** Peso máximo permitido */
  private readonly maxWeight: number;

  /** Peso mínimo permitido */
  private readonly minWeight: number;

  /** Presupuesto sináptico total para normalización homeostática */
  private readonly synapticBudget: number;

  // Buffers pre-alocados para evitar GC durante la simulación
  /** Potenciales crudos calculados en cada timestep */
  private readonly rawPotentials: Float32Array;
  /** Índices de neuronas para el sorting k-WTA */
  private readonly sortIndices: Uint32Array;

  /**
   * Crea una nueva red SNN.
   *
   * @param config - Configuración de la red, o parámetros posicionales
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

    // Crear población neuronal
    this.neurons = createNeuronPopulation(this.neuronCount, config.neuronType);

    // Alocar matriz de pesos plana (row-major: neuronCount × inputCount)
    const totalWeights = this.neuronCount * this.inputCount;
    this.weights = new Float32Array(totalWeights);

    // Pre-alocar buffers de trabajo (reutilizados en cada timestep)
    this.rawPotentials = new Float32Array(this.neuronCount);
    this.sortIndices = new Uint32Array(this.neuronCount);

    // Inicializar pesos aleatorios y normalizar
    this.initializeWeights();
  }

  /**
   * Inicializa los pesos sinápticos con valores aleatorios normalizados.
   *
   * Base biológica:
   *   Las sinapsis inmaduras tienen pesos distribuidos aleatoriamente
   *   (ruido sinaptogénico). La normalización impone un presupuesto
   *   total por neurona, modelando la homeostasis sináptica que
   *   mantiene la actividad total en un rango fisiológico.
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
   * Procesa un paso temporal completo de la red.
   *
   * Base biológica:
   *   Simula un ciclo completo de procesamiento cortical:
   *   1. Integración sináptica: cada neurona suma las corrientes de entrada
   *      ponderadas por sus pesos sinápticos (dot product parcial).
   *   2. Dinámica de membrana: Izhikevich step para cada neurona.
   *   3. Inhibición lateral (k-WTA): solo las k neuronas más excitadas
   *      mantienen su spike; las demás son silenciadas por interneuronas
   *      inhibitorias (modeladas implícitamente).
   *
   * @param inputs - Vector de entrada (Float32Array de 0s y 1s, o valores continuos)
   * @param dt - Paso temporal en milisegundos
   * @param currentTime - Tiempo actual de la simulación (ms), para STDP
   * @returns Índices de las neuronas que dispararon (ganadores k-WTA)
   */
  step(inputs: Float32Array, dt: number, currentTime: number = 0): Uint32Array {
    // --- 1. Calcular corriente de entrada para cada neurona ---
    for (let n = 0; n < this.neuronCount; n++) {
      const baseIdx = n * this.inputCount;
      let excitation = 0;

      // Producto punto: sum(weight_i * input_i)
      // Acceso secuencial en memoria → cache-friendly
      for (let i = 0; i < this.inputCount; i++) {
        excitation += this.weights[baseIdx + i] * inputs[i];
      }

      // Ruido sináptico leve (modela fluctuaciones de neurotransmisores)
      excitation += Math.random() * 0.05;

      this.rawPotentials[n] = excitation;
    }

    // --- 2. Simular dinámica de membrana de cada neurona ---
    for (let n = 0; n < this.neuronCount; n++) {
      this.neurons[n].step(this.rawPotentials[n], dt, currentTime);
    }

    // --- 3. Aplicar inhibición lateral k-WTA ---
    // Inicializar índices
    for (let n = 0; n < this.neuronCount; n++) {
      this.sortIndices[n] = n;
    }

    // Ordenar índices por potencial descendente
    // Nota: Uint32Array.sort usa una comparación que accede a rawPotentials
    const potentials = this.rawPotentials;
    this.sortIndices.sort((a, b) => potentials[b] - potentials[a]);

    // Solo los top-k pueden disparar
    const k = Math.max(1, Math.floor(this.neuronCount * this.sparsity));

    // Crear set de ganadores para lookup O(1)
    const winnerSet = new Uint8Array(this.neuronCount); // 0 = no ganador
    for (let i = 0; i < k; i++) {
      winnerSet[this.sortIndices[i]] = 1;
    }

    // Construir array de índices ganadores que realmente dispararon
    let firingCount = 0;
    // Buffer temporal para los ganadores (reutilizable)
    const firingBuffer = new Uint32Array(k);

    for (let n = 0; n < this.neuronCount; n++) {
      if (winnerSet[n] === 1 && this.neurons[n].fired) {
        firingBuffer[firingCount++] = n;
      } else {
        // Silenciar neuronas que no ganaron la competición
        this.neurons[n].fired = false;
      }
    }

    // Si ninguno de los k ganadores disparó naturalmente,
    // forzar el disparo de los top-k (activación por corriente subumbral)
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
   * Aplica la regla STDP a las sinapsis de las neuronas que dispararon.
   *
   * Base biológica:
   *   Implementa plasticidad Hebbiana temporalizada: las sinapsis que
   *   contribuyeron a hacer disparar la neurona postsináptica se
   *   fortalecen (LTP), mientras que las que no contribuyeron se
   *   debilitan (LTD). El factor de modulación permite control
   *   neuromodulador (dopamina para recompensa, etc.).
   *
   *   Después de STDP, se aplica normalización sináptica (synaptic scaling)
   *   para mantener la homeostasis: el presupuesto sináptico total por
   *   neurona se mantiene constante, modelando la regulación homeostática
   *   observada en cultivos corticales.
   *
   * @param firingNeurons - Índices de neuronas que dispararon (del step())
   * @param inputSpikeTimes - Tiempos de spike de cada entrada (Float32Array)
   * @param modulationFactor - Factor de modulación neuromoduladora (1.0 = normal)
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

      // Aplicar STDP para cada entrada
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
          // Aplicar límites homeostáticos
          newWeight = Math.max(this.minWeight, Math.min(this.maxWeight, newWeight));
          this.weights[baseIdx + i] = newWeight;
        }
      }

      // Normalización sináptica post-aprendizaje
      this.normalizeWeights(neuronIdx);
    }
  }

  /**
   * Retorna los índices de las neuronas actualmente activas (que dispararon).
   *
   * Base biológica:
   *   La representación activa es el "código poblacional" — el patrón
   *   de neuronas activas que codifica un concepto, objeto o recuerdo.
   *   Es la base de la representación distribuida en corteza.
   *
   * @returns Array de índices de neuronas activas
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
   * Normaliza los pesos sinápticos de una neurona para mantener
   * un presupuesto sináptico total fijo.
   *
   * Base biológica:
   *   "Synaptic scaling" (Turrigiano, 1998): mecanismo homeostático
   *   donde la neurona escala multiplicativamente todos sus pesos
   *   para mantener su actividad dentro de un rango fisiológico.
   *   Esto implementa heterosynaptic LTD implícita: cuando una sinapsis
   *   se fortalece (LTP), el rescalado debilita proporcionalmente
   *   las demás sinapsis.
   *
   * @param neuronIdx - Índice de la neurona a normalizar
   */
  normalizeWeights(neuronIdx: number): void {
    const baseIdx = neuronIdx * this.inputCount;

    // Calcular suma total de pesos
    let total = 0;
    for (let i = 0; i < this.inputCount; i++) {
      total += this.weights[baseIdx + i];
    }

    if (total === 0) return;

    // Escalar multiplicativamente para alcanzar el presupuesto
    const factor = this.synapticBudget / total;
    for (let i = 0; i < this.inputCount; i++) {
      this.weights[baseIdx + i] *= factor;
    }
  }

  /**
   * Reinicia todas las neuronas a su estado de reposo.
   * Los pesos sinápticos se mantienen (la memoria persiste).
   *
   * Base biológica:
   *   Equivale a un período de silencio cortical, como el observado
   *   entre estados DOWN (silencio) y UP (actividad) durante
   *   el sueño de ondas lentas.
   */
  resetNeurons(): void {
    for (let n = 0; n < this.neuronCount; n++) {
      this.neurons[n].reset();
    }
  }

  /**
   * Serializa el estado completo de la red para persistencia binaria.
   *
   * Base biológica:
   *   Análogo a la consolidación de memoria durante el sueño:
   *   se preserva la estructura sináptica (pesos) y el estado
   *   instantáneo de las neuronas para poder reanudar la simulación.
   *
   * @returns Objeto serializable con todos los datos de la red
   */
  serialize(): SerializedNetwork {
    // Copiar pesos (no compartir referencia)
    const weightsBuffer = this.weights.buffer.slice(0);

    // Extraer estados neuronales
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
   * Restaura el estado de la red desde datos serializados.
   *
   * @param data - Datos serializados previamente con serialize()
   * @returns Nueva instancia de SNNNetwork con el estado restaurado
   */
  static deserialize(data: SerializedNetwork): SNNNetwork {
    const network = new SNNNetwork({
      neuronCount: data.neuronCount,
      inputCount: data.inputCount,
      sparsity: data.sparsity,
      neuronType: data.neuronType,
    });

    // Restaurar pesos
    const loadedWeights = new Float32Array(data.weightsBuffer);
    network.weights.set(loadedWeights);

    // Restaurar estados neuronales
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
   * Infiere el nombre del tipo de neurona a partir de los parámetros.
   * Usado internamente para serialización.
   */
  private inferNeuronType(): NeuronTypeName {
    const p = this.neurons[0].params;
    if (p.a === 0.1 && p.c === -65 && p.d === 2) return 'FastSpiking';
    if (p.a === 0.02 && p.c === -50 && p.d === 2) return 'Chattering';
    if (p.a === 0.02 && p.c === -55 && p.d === 4) return 'IntrinsicBursting';
    return 'RegularSpiking';
  }

  /**
   * Retorna estadísticas de la red para monitoreo.
   *
   * @returns Objeto con métricas de actividad y pesos
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
