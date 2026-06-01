/**
 * NEURONA PULSANTE — Modelo de Izhikevich
 * =========================================
 * Implementación de alto rendimiento del modelo de neurona de Izhikevich (2003).
 *
 * Base biológica:
 *   El modelo de Izhikevich captura la dinámica de membrana de neuronas reales
 *   usando solo dos ecuaciones diferenciales ordinarias:
 *
 *     dv/dt = 0.04v² + 5v + 140 - u + I   (dinámica de voltaje de membrana)
 *     du/dt = a(bv - u)                     (variable de recuperación lenta)
 *
 *   Con la regla de reset al disparar:
 *     si v ≥ 30 mV → v = c, u = u + d
 *
 *   Los parámetros (a, b, c, d) determinan el tipo de disparo:
 *   - Regular Spiking (RS): neuronas piramidales excitatorias corticales
 *   - Fast Spiking (FS): interneuronas inhibitorias (basket cells)
 *   - Chattering (CH): neuronas que disparan ráfagas rítmicas
 *   - Intrinsic Bursting (IB): neuronas con ráfagas intrínsecas
 *
 *   Este modelo es ~100× más eficiente que Hodgkin-Huxley y reproduce
 *   más de 20 patrones de disparo neocorticales observados experimentalmente.
 *
 * Referencia: Izhikevich, E.M. (2003). "Simple Model of Spiking Neurons."
 *             IEEE Trans. Neural Networks, 14(6):1569-1572.
 */

// ====================================================================
// Interfaces y tipos
// ====================================================================

/**
 * Parámetros del modelo de Izhikevich.
 *
 * Base biológica:
 *   - a: escala temporal de recuperación (más grande = recuperación más rápida)
 *   - b: sensibilidad de u al subumbral de v (acoplamiento u-v)
 *   - c: potencial de reset tras el spike (mV)
 *   - d: incremento de u tras el spike (corriente de adaptación)
 */
export interface IzhikevichParams {
  /** Escala temporal de la variable de recuperación u (típicamente 0.02-0.1) */
  readonly a: number;
  /** Sensibilidad de la variable de recuperación al potencial subumbral (típicamente 0.2-0.25) */
  readonly b: number;
  /** Potencial de reset post-spike (mV, típicamente -65 a -50) */
  readonly c: number;
  /** Incremento de recuperación post-spike (típicamente 2-8) */
  readonly d: number;
}

/**
 * Tipos de neurona predefinidos basados en observaciones electrofisiológicas.
 */
export type NeuronTypeName = 'RegularSpiking' | 'FastSpiking' | 'Chattering' | 'IntrinsicBursting';

// ====================================================================
// Parámetros predefinidos por tipo de neurona
// ====================================================================

/**
 * Parámetros predefinidos del modelo de Izhikevich para cada tipo celular.
 *
 * Base biológica:
 *   - RegularSpiking: la neurona excitadora más común en corteza (>80%).
 *     Adapta su frecuencia de disparo progresivamente.
 *   - FastSpiking: interneuronas GABAérgicas (basket/chandelier cells).
 *     Respuesta rápida sin adaptación, mediando la inhibición lateral.
 *   - Chattering: neuronas de capas superficiales que producen ráfagas
 *     rítmicas, posiblemente involucradas en sincronización gamma.
 *   - IntrinsicBursting: neuronas de capa V que producen una ráfaga
 *     inicial seguida de spikes regulares, importantes en señalización
 *     cortico-cortical.
 */
export const NEURON_PRESETS: Record<NeuronTypeName, IzhikevichParams> = {
  RegularSpiking:    { a: 0.02, b: 0.2, c: -65, d: 8 },
  FastSpiking:       { a: 0.1,  b: 0.2, c: -65, d: 2 },
  Chattering:        { a: 0.02, b: 0.2, c: -50, d: 2 },
  IntrinsicBursting: { a: 0.02, b: 0.2, c: -55, d: 4 },
} as const;

// ====================================================================
// Clase SpikingNeuron
// ====================================================================

/**
 * Neurona pulsante individual basada en el modelo de Izhikevich.
 *
 * Base biológica:
 *   Modela una neurona cortical individual con su potencial de membrana (v),
 *   variable de recuperación (u), y capacidad de generar potenciales de
 *   acción (spikes) cuando v cruza el umbral de ~30 mV.
 *
 *   La variable u modela las corrientes iónicas lentas (K+ de activación
 *   lenta y Na+ de inactivación) que producen la adaptación de frecuencia
 *   y los períodos refractarios.
 *
 * Rendimiento:
 *   Se usa Float32Array cuando es posible. Los campos escalares (v, u)
 *   se mantienen como number nativo por eficiencia en acceso individual.
 */
export class SpikingNeuron {
  /** Potencial de membrana actual (mV). Reposo típico: -65 mV */
  public v: number;

  /** Variable de recuperación. Modela corrientes iónicas lentas. */
  public u: number;

  /**
   * Tiempo del último spike en ms (relativo al inicio de la simulación).
   * -Infinity si nunca ha disparado. Usado por STDP para calcular Δt.
   */
  public lastSpikeTime: number;

  /** Indica si la neurona disparó en el último paso de simulación */
  public fired: boolean;

  /** Parámetros (a, b, c, d) que definen el tipo de disparo */
  public readonly params: IzhikevichParams;

  /**
   * Crea una nueva neurona pulsante.
   *
   * @param params - Parámetros de Izhikevich, o nombre de un preset
   */
  constructor(params: IzhikevichParams | NeuronTypeName = 'RegularSpiking') {
    if (typeof params === 'string') {
      this.params = NEURON_PRESETS[params];
    } else {
      this.params = params;
    }

    this.v = -65.0;           // Potencial de reposo (mV)
    this.u = this.params.b * this.v;  // Estado estable de u
    this.lastSpikeTime = -Infinity;
    this.fired = false;
  }

  /**
   * Ejecuta un paso de simulación del modelo de Izhikevich.
   *
   * Base biológica:
   *   Integra las ecuaciones diferenciales de membrana por el método de Euler
   *   con paso dt. La corriente de entrada I modela la suma de corrientes
   *   sinápticas (excitatorias + inhibitorias) + corrientes de fondo.
   *
   *   Se usan 2 sub-pasos de Euler de dt/2 cada uno para mejorar la
   *   estabilidad numérica (el término 0.04v² puede ser inestable con
   *   pasos grandes).
   *
   * @param I - Corriente de entrada total (sináptica + externa) en unidades arbitrarias
   * @param dt - Paso temporal en milisegundos (típicamente 0.5-1.0 ms)
   * @param currentTime - Tiempo actual de simulación en ms (para registrar spike times)
   * @returns true si la neurona disparó un potencial de acción en este paso
   */
  step(I: number, dt: number, currentTime: number): boolean {
    const { a, b, c, d } = this.params;
    this.fired = false;

    // --- Integración numérica con 2 sub-pasos de Euler (estabilidad) ---
    // Sub-paso 1: dt/2
    const halfDt = dt * 0.5;
    this.v += halfDt * (0.04 * this.v * this.v + 5.0 * this.v + 140.0 - this.u + I);
    // Sub-paso 2: dt/2
    this.v += halfDt * (0.04 * this.v * this.v + 5.0 * this.v + 140.0 - this.u + I);
    // Actualizar variable de recuperación
    this.u += dt * a * (b * this.v - this.u);

    // --- Detección de spike ---
    if (this.v >= 30.0) {
      this.v = c;         // Reset del potencial de membrana
      this.u += d;         // Incremento de recuperación post-spike
      this.fired = true;
      this.lastSpikeTime = currentTime;
    }

    return this.fired;
  }

  /**
   * Reinicia la neurona a su estado de reposo.
   *
   * Base biológica:
   *   Equivale a un período de silencio prolongado donde la neurona
   *   regresa a su estado electroquímico basal.
   */
  reset(): void {
    this.v = -65.0;
    this.u = this.params.b * this.v;
    this.lastSpikeTime = -Infinity;
    this.fired = false;
  }
}

// ====================================================================
// Utilidades para creación masiva de neuronas
// ====================================================================

/**
 * Crea un array de neuronas del mismo tipo.
 * Optimizado para instanciación masiva en regiones cerebrales.
 *
 * @param count - Número de neuronas a crear
 * @param neuronType - Tipo de neurona (preset o parámetros custom)
 * @returns Array de neuronas inicializadas
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
 * Estructura de estado compacto para serialización/transferencia masiva.
 * Almacena v, u, lastSpikeTime de N neuronas en arrays contiguos.
 *
 * Base biológica:
 *   Formato de almacenamiento eficiente análogo a cómo técnicas de
 *   imagen funcional (fMRI, EEG) almacenan estados de miles de
 *   vóxels/canales en arrays contiguos.
 */
export interface NeuronStateBuffer {
  /** Potenciales de membrana de todas las neuronas */
  readonly voltages: Float32Array;
  /** Variables de recuperación de todas las neuronas */
  readonly recovery: Float32Array;
  /** Tiempos de último spike (ms) */
  readonly lastSpikeTimes: Float64Array;
}

/**
 * Extrae el estado de un array de neuronas a buffers compactos.
 *
 * @param neurons - Array de neuronas
 * @returns Buffers compactos con el estado de todas las neuronas
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
 * Restaura el estado de neuronas desde buffers compactos.
 *
 * @param neurons - Array de neuronas a restaurar
 * @param state - Buffers compactos con el estado guardado
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
