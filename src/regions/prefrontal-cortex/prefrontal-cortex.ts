/**
 * CORTEZA PREFRONTAL — Centro Ejecutivo del Cerebro Digital
 * ==========================================================
 * Modela la corteza prefrontal dorsolateral (dlPFC) y ventromedial (vmPFC),
 * centro de las funciones ejecutivas superiores del cerebro humano.
 *
 * Base biológica:
 *   La corteza prefrontal ocupa ~30% de la neocorteza humana y es la última
 *   región en madurar (~25 años). Cumple funciones ejecutivas críticas:
 *
 *   1. **Memoria de trabajo** (dlPFC): Mantiene información "online" para
 *      manipulación activa. Limitada a ~7±2 ítems (Miller, 1956) por la
 *      capacidad de mantener patrones de disparo sostenido en circuitos
 *      recurrentes. Cada "slot" es una población neural que mantiene un
 *      patrón de activación persistente mediante excitación recurrente.
 *
 *   2. **Integración multimodal**: Recibe aferencias de TODAS las áreas
 *      corticales (visual, auditiva, somatosensorial, límbica) y del
 *      tálamo. Es la única región con conectividad tan amplia, lo que le
 *      permite crear representaciones unificadas del estado global.
 *
 *   3. **Control top-down**: Envía señales descendentes que modulan el
 *      procesamiento en cortezas sensoriales y tálamo, implementando
 *      atención selectiva y supresión de distractores (Desimone & Duncan, 1995).
 *
 *   4. **Toma de decisiones**: Evaluación de opciones ponderada por
 *      información emocional (vmPFC, marcadores somáticos de Damasio) y
 *      recompensa esperada (señales dopaminérgicas del VTA).
 *
 * Implementación:
 *   - Memoria de trabajo como array de slots con patrón, etiqueta, edad y prioridad
 *   - Integración por superposición ponderada de inputs de múltiples regiones
 *   - Señales top-down como vectores de modulación atencional por región
 *   - Red SNN interna con k-WTA para selección competitiva
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';

/**
 * Piso de activación para el k-WTA. Los potenciales son sumas de pesos
 * adimensionales (~0 en reposo), no mV. En reposo (sin drive cortical) la
 * región queda a 0%; sólo dispara cuando hay señal real por encima del piso.
 */
const ACTIVATION_FLOOR = 1e-3;

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Slot de memoria de trabajo.
 *
 * Base biológica:
 *   Cada slot corresponde a un ensamble neural en la dlPFC que mantiene
 *   un patrón de disparo sostenido mediante excitación recurrente.
 *   La "edad" modela el decaimiento natural de la activación persistente
 *   en ausencia de reactivación (decay de la memoria de trabajo).
 */
export interface WorkingMemorySlot {
  /** Patrón de activación neural mantenido activamente */
  pattern: Float32Array;
  /** Etiqueta semántica del contenido (para debugging/introspección) */
  label: string;
  /** Edad del slot en pasos de simulación (decae con el tiempo) */
  age: number;
  /** Prioridad atencional (0.0–1.0). Determina resistencia a la evicción */
  priority: number;
}

/**
 * Señal de control top-down generada por la corteza prefrontal.
 *
 * Base biológica:
 *   Las proyecciones prefrontales descendentes modulan la ganancia de
 *   neuronas en cortezas sensoriales (atención basada en características)
 *   y en el tálamo (filtro atencional, modelo del foco atencional).
 */
export interface TopDownSignal {
  /** Identificador de la región objetivo */
  targetRegion: string;
  /** Vector de modulación atencional para la región objetivo */
  attentionMask: Float32Array;
  /** Intensidad global de la señal (0.0–1.0) */
  gain: number;
}

/**
 * Resultado del procesamiento ejecutivo de un paso de simulación.
 */
export interface ExecutiveOutput {
  /** Spikes de salida de la red SNN prefrontal */
  outputSpikes: Float32Array;
  /** Señales top-down generadas para otras regiones */
  topDownSignals: TopDownSignal[];
  /** Contenido actual de la memoria de trabajo */
  workingMemorySnapshot: ReadonlyArray<Readonly<{ label: string; priority: number; age: number }>>;
}

// ==================================================================
// Clase PrefrontalCortex
// ==================================================================

/**
 * Corteza Prefrontal — Centro ejecutivo del cerebro digital.
 *
 * Integra información de todas las regiones, mantiene representaciones
 * en memoria de trabajo (7±2 slots), y genera señales de control top-down
 * para modular el procesamiento en cortezas sensoriales y tálamo.
 *
 * La red SNN interna usa competencia k-WTA (k-Winners-Take-All) para
 * seleccionar las representaciones más relevantes y suprimir distractores,
 * modelando la inhibición lateral mediada por interneuronas GABAérgicas
 * de la capa III prefrontal.
 */
export class PrefrontalCortex extends BrainRegion {
  /**
   * Slots de memoria de trabajo.
   *
   * Biología: Circuitos recurrentes en la dlPFC mantienen patrones
   * de disparo sostenido. La capacidad está limitada por la competencia
   * entre ensambles neurales (~7±2 ítems, Miller 1956).
   */
  private workingMemory: WorkingMemorySlot[] = [];

  /**
   * Capacidad máxima de la memoria de trabajo.
   * Basada en la ley de Miller (7±2): el número mágico de ítems
   * que la memoria de trabajo humana puede mantener simultáneamente.
   */
  private readonly maxSlots: number = 7;

  /**
   * Tasa de decaimiento de la memoria de trabajo (por paso de simulación).
   *
   * Biología: Sin reactivación, la actividad persistente en la dlPFC
   * decae en ~15-30 segundos (Peterson & Peterson, 1959).
   */
  private readonly decayRate: number = 0.02;

  /**
   * Última representación integrada de todos los inputs.
   * Se mantiene entre pasos para modelar la persistencia de la
   * representación ejecutiva.
   */
  private integratedRepresentation: Float32Array;

  /**
   * Señales top-down generadas en el último paso.
   */
  private lastTopDownSignals: TopDownSignal[] = [];

  /**
   * Crea la Corteza Prefrontal del cerebro digital.
   *
   * Biología: La corteza prefrontal humana contiene ~300 millones de neuronas.
   * Modelamos 15.000 neuronas con 10.000 entradas, capturando la alta
   * convergencia de inputs que caracteriza a esta región.
   */
  constructor(neuronCount: number = 15000, inputCount: number = 10000) {
    super(
      'prefrontalCortex',
      'Corteza Prefrontal — Centro Ejecutivo',
      neuronCount,
      inputCount
    );
    this.integratedRepresentation = new Float32Array(this.neuronCount);
  }

  // ----------------------------------------------------------------
  // Memoria de Trabajo
  // ----------------------------------------------------------------

  /**
   * Atiende a un patrón, ingresándolo en la memoria de trabajo.
   *
   * Base biológica:
   *   Al atender a un estímulo, las neuronas prefrontales forman un
   *   ensamble de disparo persistente que mantiene la representación
   *   "en línea". Si la memoria está llena, se evicta el ítem con
   *   menor prioridad (competencia entre ensambles por recursos neurales
   *   limitados, modelo de capacidad de Cowan, 2001).
   *
   * @param pattern - Patrón neural a mantener en memoria de trabajo
   * @param label - Etiqueta semántica del patrón
   * @param priority - Prioridad atencional (0.0–1.0)
   */
  attendTo(pattern: Float32Array, label: string, priority: number): void {
    const clampedPriority = Math.max(0, Math.min(1, priority));

    // Si ya existe un slot con la misma etiqueta, actualizar en lugar de duplicar
    const existingIdx = this.workingMemory.findIndex(s => s.label === label);
    if (existingIdx >= 0) {
      const slot = this.workingMemory[existingIdx];
      slot.pattern = new Float32Array(pattern);
      slot.priority = clampedPriority;
      slot.age = 0; // Reactivar refresca el slot
      return;
    }

    // Si la memoria está llena, evictar el slot con menor prioridad
    if (this.workingMemory.length >= this.maxSlots) {
      this.evictLowestPriority();
    }

    this.workingMemory.push({
      pattern: new Float32Array(pattern),
      label,
      age: 0,
      priority: clampedPriority,
    });
  }

  /**
   * Evicta el slot con menor prioridad de la memoria de trabajo.
   *
   * Biología: Cuando los recursos atencionales se saturan, los ítems
   * menos relevantes pierden su activación sostenida y son desplazados
   * por nueva información (interferencia proactiva/retroactiva).
   */
  private evictLowestPriority(): void {
    if (this.workingMemory.length === 0) return;

    let minIdx = 0;
    let minScore = Infinity;

    for (let i = 0; i < this.workingMemory.length; i++) {
      // Puntuación combinada: prioridad baja + edad alta → más probable evicción
      const slot = this.workingMemory[i];
      const score = slot.priority - slot.age * this.decayRate;
      if (score < minScore) {
        minScore = score;
        minIdx = i;
      }
    }

    this.workingMemory.splice(minIdx, 1);
  }

  /**
   * Envejece todos los slots de la memoria de trabajo y elimina los decaídos.
   *
   * Biología: Sin rehearsal (repaso activo), la actividad persistente
   * decae exponencialmente. Los ítems cuya activación cae por debajo
   * de un umbral dejan de ser accesibles conscientemente.
   */
  private ageWorkingMemory(): void {
    for (const slot of this.workingMemory) {
      slot.age++;
      // Reducir prioridad gradualmente con la edad
      slot.priority *= (1 - this.decayRate);
    }

    // Eliminar slots cuya prioridad ha decaído por debajo de un umbral mínimo
    this.workingMemory = this.workingMemory.filter(s => s.priority > 0.01);
  }

  /**
   * Retorna el contenido actual de la memoria de trabajo.
   *
   * @returns Copia de solo lectura de los slots actuales
   */
  getWorkingMemory(): ReadonlyArray<Readonly<WorkingMemorySlot>> {
    return this.workingMemory;
  }

  // ----------------------------------------------------------------
  // Integración Multimodal
  // ----------------------------------------------------------------

  /**
   * Integra señales de múltiples regiones cerebrales en una representación unificada.
   *
   * Base biológica:
   *   La corteza prefrontal recibe proyecciones convergentes de todas las
   *   áreas corticales sensoriales, motoras, límbicas y del tálamo.
   *   La integración ocurre mediante superposición ponderada de patrones
   *   de activación, donde el peso refleja la relevancia atencional actual
   *   de cada fuente (atención biased competition, Desimone & Duncan 1995).
   *
   *   El resultado es una representación "global workspace" (Baars, 1988)
   *   que captura el estado integrado de la cognición del sistema.
   *
   * @param inputs - Mapa de regionId → spikes de salida de cada región
   * @returns Representación integrada como Float32Array
   */
  integrate(inputs: Map<string, Float32Array>): Float32Array {
    const integrated = new Float32Array(this.neuronCount);

    if (inputs.size === 0) return integrated;

    const invCount = 1.0 / inputs.size;

    for (const [, regionSpikes] of inputs) {
      // Proyectar los spikes de cada región al espacio prefrontal
      // usando los pesos sinápticos como matriz de proyección
      const projectionLen = Math.min(regionSpikes.length, this.inputCount);

      for (let n = 0; n < this.neuronCount; n++) {
        const baseOffset = n * this.inputCount;
        let activation = 0;

        for (let j = 0; j < projectionLen; j++) {
          activation += this.weights[baseOffset + j] * regionSpikes[j];
        }

        integrated[n] += activation * invCount;
      }
    }

    return integrated;
  }

  // ----------------------------------------------------------------
  // Señales Top-Down
  // ----------------------------------------------------------------

  /**
   * Genera una señal de control top-down para una región objetivo.
   *
   * Base biológica:
   *   Las proyecciones descendentes de la PFC modulan la ganancia de
   *   neuronas en las cortezas sensoriales y el tálamo. Esta modulación
   *   implementa la atención selectiva: amplifica señales relevantes y
   *   suprime distractores (modelo de sesgo atencional de Desimone & Duncan).
   *
   *   El mecanismo biológico involucra:
   *   - Proyecciones glutamatérgicas de capas II/III prefrontales
   *   - A neuronas piramidales e interneuronas en corteza objetivo
   *   - Resultando en aumento de ganancia (multiplicativa) para
   *     estímulos atendidos
   *
   * @param targetRegion - Identificador de la región a modular
   * @returns Señal top-down con máscara atencional y ganancia
   */
  generateTopDownSignal(targetRegion: string): TopDownSignal {
    // Generar máscara atencional basada en el estado integrado actual
    // y el contenido de la memoria de trabajo
    const maskSize = this.inputCount;
    const attentionMask = new Float32Array(maskSize);

    // La máscara se construye a partir de los patrones en memoria de trabajo
    // Biología: la atención amplifica las características que coinciden
    // con las representaciones mantenidas en la memoria de trabajo
    let totalPriority = 0;
    for (const slot of this.workingMemory) {
      totalPriority += slot.priority;
    }

    if (totalPriority > 0) {
      const invTotal = 1.0 / totalPriority;
      for (const slot of this.workingMemory) {
        const weight = slot.priority * invTotal;
        const len = Math.min(slot.pattern.length, maskSize);
        for (let i = 0; i < len; i++) {
          attentionMask[i] += slot.pattern[i] * weight;
        }
      }
    }

    // Normalizar la máscara al rango [0, 1]
    let maxVal = 0;
    for (let i = 0; i < maskSize; i++) {
      if (attentionMask[i] > maxVal) maxVal = attentionMask[i];
    }
    if (maxVal > 0) {
      const invMax = 1.0 / maxVal;
      for (let i = 0; i < maskSize; i++) {
        attentionMask[i] *= invMax;
      }
    }

    // La ganancia depende de la actividad prefrontal actual
    // (más actividad → señal top-down más fuerte)
    let activeCount = 0;
    for (let i = 0; i < this.spikes.length; i++) {
      if (this.spikes[i] > 0) activeCount++;
    }
    const gain = Math.min(1.0, activeCount / (this.neuronCount * this.sparsity + 1));

    return {
      targetRegion,
      attentionMask,
      gain,
    };
  }

  /**
   * Retorna las señales top-down generadas en el último paso.
   */
  getLastTopDownSignals(): ReadonlyArray<Readonly<TopDownSignal>> {
    return this.lastTopDownSignals;
  }

  // ----------------------------------------------------------------
  // k-WTA (k-Winners Take All) — Competencia neural
  // ----------------------------------------------------------------

  /**
   * Aplica selección competitiva k-WTA al vector de activaciones.
   *
   * Base biológica:
   *   La inhibición lateral mediada por interneuronas GABAérgicas en la
   *   capa III de la corteza prefrontal implementa una competencia
   *   "el ganador se lleva todo" (Winner-Take-All). Solo las k neuronas
   *   con mayor activación disparan, las demás son suprimidas.
   *   El valor de k está controlado por la sparsity (modulada por
   *   neuromoduladores como norepinefrina y acetilcolina).
   *
   * @param activations - Vector de activaciones pre-competencia
   * @returns Vector de spikes post-competencia (sparse)
   */
  private applyKWTA(activations: Float32Array): Float32Array {
    const output = new Float32Array(activations.length);
    const k = Math.max(1, Math.floor(activations.length * this.sparsity));

    // Encontrar el k-ésimo valor más alto (umbral de competencia)
    // Usamos selección parcial para eficiencia O(n)
    const sorted = new Float32Array(activations);
    sorted.sort();
    const threshold = sorted[sorted.length - k];

    // Solo las neuronas por encima del umbral disparan
    for (let i = 0; i < activations.length; i++) {
      if (activations[i] >= threshold && activations[i] > ACTIVATION_FLOOR) {
        output[i] = 1.0;
      }
    }

    return output;
  }

  // ----------------------------------------------------------------
  // processInput — Ciclo ejecutivo principal
  // ----------------------------------------------------------------

  /**
   * Procesa un paso de simulación de la corteza prefrontal.
   *
   * Base biológica:
   *   El ciclo ejecutivo prefrontal ocurre en ~100ms (ritmo theta, 4-8 Hz)
   *   y consiste en:
   *
   *   1. **Recepción**: Integrar aferencias de todas las regiones
   *   2. **Mantenimiento**: Actualizar la memoria de trabajo (envejecer,
   *      refrescar ítems relevantes, evictar ítems irrelevantes)
   *   3. **Competencia**: k-WTA sobre activaciones para seleccionar
   *      las representaciones más relevantes
   *   4. **Control**: Generar señales top-down para modular procesamiento
   *      en regiones sensoriales y tálamo
   *
   * @param spikes - Vector de spikes de entrada (integrado del buffer sensorial)
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Vector de spikes de salida (representación ejecutiva)
   */
  processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects
  ): Float32Array {
    // 1. Computar activaciones de la red SNN local
    //    Cada neurona prefrontal integra los inputs ponderados por sus pesos sinápticos
    const activations = new Float32Array(this.neuronCount);
    const inputLen = Math.min(spikes.length, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      const baseOffset = n * this.inputCount;
      let sum = 0;

      for (let j = 0; j < inputLen; j++) {
        sum += this.weights[baseOffset + j] * spikes[j];
      }

      // Aplicar ganancia de neuromodulación y umbral modulado
      sum *= modulationEffects.spikeGainMultiplier;

      // Modelo LIF simplificado: integrar en potencial de membrana
      this.potentials[n] += sum;
      this.potentials[n] *= 0.95; // Leak (fuga de membrana)

      // El potencial es una suma de pesos adimensionales (~0 en reposo), no mV.
      // Restar _modulatedThreshold (−55 mV) lo desplazaba +55 y el gate `>0` del
      // k-WTA siempre pasaba → actividad constante y falsa. Se usa el potencial
      // crudo y el k-WTA filtra contra un piso real.
      activations[n] = this.potentials[n];
    }

    // 2. Selección competitiva k-WTA
    const outputSpikes = this.applyKWTA(activations);

    // 3. Actualizar estado de spikes
    this.spikes.set(outputSpikes);

    // 4. Mantener la representación integrada (con decaimiento)
    for (let i = 0; i < this.neuronCount; i++) {
      this.integratedRepresentation[i] =
        this.integratedRepresentation[i] * 0.8 + outputSpikes[i] * 0.2;
    }

    // 5. Envejecer la memoria de trabajo
    this.ageWorkingMemory();

    // 6. Aprendizaje Hebbiano modulado por dopamina
    //    "Neurons that fire together wire together" (Hebb, 1949)
    if (modulationEffects.learningRateMultiplier > 0) {
      this.hebbianUpdate(spikes, outputSpikes, modulationEffects);
    }

    return outputSpikes;
  }

  /**
   * Actualización Hebbiana de pesos sinápticos.
   *
   * Base biológica:
   *   La plasticidad sináptica en la corteza prefrontal está fuertemente
   *   modulada por la dopamina (señal de recompensa/error de predicción).
   *   Las sinapsis entre neuronas que disparan juntas se fortalecen (LTP),
   *   mientras que las demás se debilitan (LTD). La dopamina actúa como
   *   una "puerta" que determina si el fortalecimiento se consolida
   *   (three-factor learning rule: pre × post × modulación).
   *
   * @param preSpikes - Spikes presinápticos (entradas)
   * @param postSpikes - Spikes postsinápticos (salida de k-WTA)
   * @param modulation - Efectos de neuromodulación (incluye learningRate)
   */
  private hebbianUpdate(
    preSpikes: Float32Array,
    postSpikes: Float32Array,
    modulation: ModulationEffects
  ): void {
    const lr = this._modulatedLearningRate * 0.01; // Escalar para estabilidad
    const inputLen = Math.min(preSpikes.length, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      if (postSpikes[n] === 0) continue; // Solo actualizar neuronas activas

      const baseOffset = n * this.inputCount;
      for (let j = 0; j < inputLen; j++) {
        if (preSpikes[j] > 0) {
          // LTP: reforzar conexión pre→post activa
          this.weights[baseOffset + j] += lr * preSpikes[j] * postSpikes[n];
        } else {
          // LTD suave: debilitar conexiones no usadas
          this.weights[baseOffset + j] *= (1 - lr * 0.1);
        }

        // Clamp de pesos para estabilidad
        if (this.weights[baseOffset + j] > 1.0) {
          this.weights[baseOffset + j] = 1.0;
        } else if (this.weights[baseOffset + j] < 0) {
          this.weights[baseOffset + j] = 0;
        }
      }
    }
  }
}
