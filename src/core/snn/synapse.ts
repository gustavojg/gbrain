/**
 * SINAPSIS CON STDP — Plasticidad Dependiente del Tiempo de Spikes
 * ==================================================================
 * Implementación de la regla de aprendizaje sináptico STDP
 * (Spike-Timing-Dependent Plasticity).
 *
 * Base biológica:
 *   STDP es el mecanismo de plasticidad sináptica más estudiado en
 *   neurociencia experimental. Descubierto por Markram (1997) y
 *   Bi & Poo (1998), establece que:
 *
 *   - Si la neurona presináptica dispara ANTES que la postsináptica (Δt > 0):
 *     → Potenciación a Largo Plazo (LTP): la sinapsis se fortalece.
 *     → Δw = A+ · exp(-Δt / τ+)
 *     → Interpretación: "la pre CAUSÓ la post" → reforzar la conexión.
 *
 *   - Si la neurona postsináptica dispara ANTES que la presináptica (Δt < 0):
 *     → Depresión a Largo Plazo (LTD): la sinapsis se debilita.
 *     → Δw = -A- · exp(Δt / τ-)
 *     → Interpretación: "la pre NO causó la post" → debilitar.
 *
 *   La ventana temporal asimétrica (τ ≈ 20ms) coincide con la duración
 *   de los potenciales postsinápticos excitatorios (EPSPs) en corteza.
 *
 *   El factor de modulación permite que neuromoduladores (dopamina, etc.)
 *   escalen la magnitud del cambio sináptico, implementando así
 *   "three-factor learning rules" (elegibilidad × recompensa × STDP).
 *
 * Referencias:
 *   - Bi, G. & Poo, M. (1998). J. Neuroscience, 18(24):10464-10472.
 *   - Markram, H. et al. (1997). Science, 275(5297):213-215.
 */

// ====================================================================
// Interfaces y tipos
// ====================================================================

/**
 * Configuración de una sinapsis individual.
 *
 * Base biológica:
 *   - weight: eficacia sináptica (amplitud del EPSP/IPSP).
 *   - maxWeight: límite superior por saturación de receptores AMPA/NMDA.
 *   - minWeight: límite inferior (sinapsis silenciosa, no eliminada).
 *   - delay: retardo de conducción axonal en ms.
 */
export interface SynapseConfig {
  /** Peso sináptico inicial (eficacia de transmisión) */
  weight: number;
  /** Peso máximo alcanzable (saturación de receptores) */
  maxWeight: number;
  /** Peso mínimo (sinapsis silenciosa, pero preservada estructuralmente) */
  minWeight: number;
  /** Retardo de conducción axonal en milisegundos */
  delay: number;
}

/**
 * Parámetros de la ventana temporal de STDP.
 *
 * Base biológica:
 *   Los valores por defecto están calibrados según datos experimentales
 *   de sinapsis corticales glutamatérgicas:
 *   - A+ < A- asegura que la competencia sináptica tienda hacia
 *     la estabilidad (más LTD que LTP en promedio).
 *   - τ ≈ 20ms corresponde a la duración típica de un EPSP.
 */
export interface STDPParams {
  /** Amplitud máxima de potenciación (LTP). Valor típico: 0.01 */
  readonly aPlus: number;
  /** Amplitud máxima de depresión (LTD). Valor típico: 0.012 */
  readonly aMinus: number;
  /** Constante de tiempo de la ventana LTP en ms. Valor típico: 20 */
  readonly tauPlus: number;
  /** Constante de tiempo de la ventana LTD en ms. Valor típico: 20 */
  readonly tauMinus: number;
}

/** Parámetros STDP por defecto basados en datos experimentales corticales */
export const DEFAULT_STDP_PARAMS: STDPParams = {
  aPlus: 0.01,
  aMinus: 0.012,
  tauPlus: 20.0,
  tauMinus: 20.0,
} as const;

// ====================================================================
// Clase STDPSynapse
// ====================================================================

/**
 * Sinapsis con plasticidad dependiente del tiempo de spikes (STDP).
 *
 * Base biológica:
 *   Modela una conexión sináptica individual entre dos neuronas, con
 *   capacidad de modificar su eficacia (peso) según la regla STDP.
 *
 *   La sinapsis incluye:
 *   - Un peso que determina la amplitud de corriente inyectada
 *   - Un retardo de conducción axonal
 *   - Límites homeostáticos para evitar pesos patológicos
 *   - Regla STDP modulable por neuromoduladores
 *
 * Rendimiento:
 *   Para redes grandes (50K+), se recomienda usar la clase SNNNetwork
 *   que almacena pesos en Float32Array plano. Esta clase es útil para
 *   conexiones inter-regionales individuales o para depuración.
 */
export class STDPSynapse {
  /** Peso sináptico actual (eficacia de transmisión) */
  private weight: number;
  /** Peso máximo permitido (homeostasis) */
  private readonly maxWeight: number;
  /** Peso mínimo permitido (homeostasis) */
  private readonly minWeight: number;
  /** Retardo de conducción axonal (ms) */
  public readonly delay: number;
  /** Parámetros de la ventana STDP */
  private readonly stdp: STDPParams;

  /**
   * Crea una nueva sinapsis STDP.
   *
   * @param config - Configuración de la sinapsis (peso, límites, retardo)
   * @param stdpParams - Parámetros opcionales de la ventana STDP
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
   * Aplica la regla STDP basada en los tiempos de spike pre y postsináptico.
   *
   * Base biológica:
   *   Calcula el cambio de peso Δw según la diferencia temporal:
   *     Δt = t_post - t_pre
   *
   *   - Δt > 0 (pre antes que post): LTP → la sinapsis se fortalece
   *     porque la actividad presináptica predijo/causó el disparo post.
   *     Δw = A+ · exp(-Δt / τ+)
   *
   *   - Δt < 0 (post antes que pre): LTD → la sinapsis se debilita
   *     porque la actividad presináptica NO predijo el disparo post.
   *     Δw = -A- · exp(Δt / τ-)   (nota: Δt < 0 aquí)
   *
   *   El modulationFactor escala Δw, permitiendo que neuromoduladores
   *   (dopamina = recompensa, norepinefrina = alerta) controlen
   *   cuánto se aprende. Esto implementa la "three-factor rule":
   *     Δw_final = STDP(Δt) × modulación
   *
   * @param preSpikeTime - Tiempo del spike presináptico (ms)
   * @param postSpikeTime - Tiempo del spike postsináptico (ms)
   * @param modulationFactor - Factor de modulación neuromoduladora (default: 1.0)
   * @returns El cambio de peso aplicado (Δw)
   */
  applySTDP(
    preSpikeTime: number,
    postSpikeTime: number,
    modulationFactor: number = 1.0,
  ): number {
    // No hay aprendizaje si alguna neurona nunca ha disparado
    if (!isFinite(preSpikeTime) || !isFinite(postSpikeTime)) {
      return 0;
    }

    const deltaT = postSpikeTime - preSpikeTime;

    let deltaW = 0;

    if (deltaT > 0) {
      // Pre dispara antes que post → LTP (potenciación)
      deltaW = this.stdp.aPlus * Math.exp(-deltaT / this.stdp.tauPlus);
    } else if (deltaT < 0) {
      // Post dispara antes que pre → LTD (depresión)
      deltaW = -this.stdp.aMinus * Math.exp(deltaT / this.stdp.tauMinus);
    }
    // Si deltaT === 0, no se aplica cambio (evento simultáneo, ambiguo)

    // Aplicar modulación neuromoduladora (three-factor rule)
    deltaW *= modulationFactor;

    // Actualizar peso con límites homeostáticos
    this.weight = Math.max(
      this.minWeight,
      Math.min(this.maxWeight, this.weight + deltaW),
    );

    return deltaW;
  }

  /**
   * Retorna el peso sináptico actual.
   *
   * @returns Peso sináptico (eficacia de transmisión)
   */
  getWeight(): number {
    return this.weight;
  }

  /**
   * Establece el peso sináptico directamente (respetando límites).
   * Útil para inicialización o restauración desde persistencia.
   *
   * @param newWeight - Nuevo peso sináptico
   */
  setWeight(newWeight: number): void {
    this.weight = Math.max(this.minWeight, Math.min(this.maxWeight, newWeight));
  }

  /**
   * Reinicia el peso sináptico al valor medio entre min y max.
   *
   * Base biológica:
   *   Simula una despotenciación sináptica masiva, como la observada
   *   durante el sueño de ondas lentas (slow-wave sleep), donde las
   *   sinapsis se reescalan globalmente a niveles basales.
   */
  reset(): void {
    this.weight = (this.maxWeight + this.minWeight) * 0.5;
  }

  /**
   * Calcula la corriente sináptica transmitida (peso × spike presináptico).
   * En la práctica: retorna el peso si hay spike, 0 si no.
   *
   * @param presynapticSpike - true si la neurona presináptica disparó
   * @returns Corriente inyectada en la neurona postsináptica
   */
  transmit(presynapticSpike: boolean): number {
    return presynapticSpike ? this.weight : 0;
  }
}

// ====================================================================
// Funciones utilitarias para STDP a nivel de red
// ====================================================================

/**
 * Calcula el cambio de peso STDP entre un par de neuronas sin aplicarlo.
 * Función pura para uso en redes que almacenan pesos en Float32Array.
 *
 * Base biológica:
 *   Versión funcional del cálculo STDP, diseñada para integración con
 *   la clase SNNNetwork que usa almacenamiento plano de pesos.
 *
 * @param preSpikeTime - Tiempo del spike presináptico (ms)
 * @param postSpikeTime - Tiempo del spike postsináptico (ms)
 * @param stdp - Parámetros de la ventana STDP
 * @param modulationFactor - Factor de modulación neuromoduladora
 * @returns Δw calculado (sin aplicar límites)
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
