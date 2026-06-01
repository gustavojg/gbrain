/**
 * Sistema de Neuromodulación
 * ===========================
 * Modela los 6 principales sistemas neuromoduladores del cerebro humano.
 *
 * Biología: Los neuromoduladores son sustancias químicas liberadas por núcleos
 * subcorticales que afectan globalmente el procesamiento neural. A diferencia
 * de la neurotransmisión sináptica punto-a-punto, la neuromodulación actúa
 * de forma difusa ("volumétrica") sobre regiones enteras, alterando parámetros
 * como la excitabilidad neuronal, la plasticidad sináptica y la ganancia
 * de señales.
 *
 * Sistemas modelados:
 * 1. Dopamina (VTA/SNc): Recompensa, motivación, aprendizaje por refuerzo
 * 2. Serotonina (Núcleos del rafe): Regulación emocional, estabilidad
 * 3. Norepinefrina (Locus coeruleus): Alerta, atención, respuesta al estrés
 * 4. Cortisol (Eje HPA): Estrés crónico, supresión de consolidación
 * 5. Acetilcolina (Núcleo basal): Atención focalizada, consolidación mnémica
 * 6. Oxitocina (Hipotálamo): Cognición social, confianza, vinculación
 */

/**
 * Tipos de neuromoduladores implementados en el sistema.
 */
export enum ModulatorType {
  /** Dopamina: señal de recompensa y error de predicción */
  Dopamine = 'dopamine',
  /** Serotonina: regulación del ánimo y estabilidad conductual */
  Serotonin = 'serotonin',
  /** Norepinefrina: alerta, vigilia y respuesta fight-or-flight */
  Norepinephrine = 'norepinephrine',
  /** Cortisol: respuesta al estrés sostenido */
  Cortisol = 'cortisol',
  /** Acetilcolina: atención focalizada y formación de memorias */
  Acetylcholine = 'acetylcholine',
  /** Oxitocina: cognición social y vínculos afectivos */
  Oxytocin = 'oxytocin',
}

/**
 * Estado instantáneo de un neuromodulador.
 */
export interface ModulatorState {
  /** Nivel actual del modulador (0.0 - 1.0) */
  level: number;
  /** Nivel basal al que tiende el modulador en reposo */
  baseline: number;
  /** Tasa de decaimiento hacia el baseline (fracción por ms) */
  decayRate: number;
  /** Marca temporal de la última actualización (ms) */
  lastUpdate: number;
}

/**
 * Efectos combinados de todos los neuromoduladores sobre los parámetros de la SNN.
 * Estos multiplicadores se aplican globalmente a todas las regiones cerebrales.
 */
export interface ModulationEffects {
  /** Multiplicador de la tasa de aprendizaje (STDP/Hebbian). >1 = más plasticidad */
  learningRateMultiplier: number;
  /** Multiplicador del umbral de disparo. >1 = neuronas más difíciles de activar */
  thresholdMultiplier: number;
  /** Ganancia atencional. >1 = señales sensoriales amplificadas */
  attentionGain: number;
  /** Tasa de consolidación mnémica. >1 = memorias se fortalecen más rápido */
  consolidationRate: number;
  /** Multiplicador de ganancia de spikes. >1 = actividad neural amplificada */
  spikeGainMultiplier: number;
  /** Boost de pesos relacionados con procesamiento social */
  socialWeightBoost: number;
}

/**
 * Estado serializable del sistema completo de neuromodulación.
 */
export interface NeuromodulatorSnapshot {
  /** Estado de cada modulador */
  modulators: Record<ModulatorType, ModulatorState>;
  /** Marca temporal de la captura */
  timestamp: number;
}

/**
 * Sistema central de neuromodulación del cerebro digital.
 *
 * Gestiona los niveles de 6 neuromoduladores y computa sus efectos
 * combinados sobre los parámetros globales de la red neuronal.
 * Cada modulador decae exponencialmente hacia su nivel basal,
 * modelando la recaptación y degradación enzimática biológica.
 */
export class NeuromodulatorSystem {
  /** Estado interno de todos los moduladores */
  private readonly modulators: Map<ModulatorType, ModulatorState>;

  /**
   * Crea el sistema de neuromodulación con niveles basales por defecto.
   *
   * Biología: Los niveles basales representan la actividad tónica
   * (constante) de cada sistema neuromodulador en estado de vigilia
   * tranquila. Los valores oscilan entre 0.3 y 0.5 reflejando
   * actividad moderada.
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
   * Libera una cantidad de neuromodulador (aumenta su nivel).
   *
   * Biología: Modela la liberación fásica (burst) de neuromodulador
   * desde los núcleos subcorticales en respuesta a eventos relevantes.
   * Ejemplo: Una recompensa inesperada causa un burst de dopamina.
   *
   * @param type - Tipo de neuromodulador a liberar
   * @param amount - Cantidad a liberar (0.0 - 1.0). Se suma al nivel actual.
   */
  release(type: ModulatorType, amount: number): void {
    const state = this.modulators.get(type);
    if (!state) return;

    // Clamp al rango [0, 1]
    state.level = Math.min(1.0, Math.max(0, state.level + amount));
    state.lastUpdate = Date.now();
  }

  /**
   * Aplica decaimiento temporal a todos los moduladores.
   *
   * Biología: Los neuromoduladores son recaptados por transportadores
   * presinápticos y degradados enzimáticamente (ej: MAO para dopamina,
   * AChE para acetilcolina). Este proceso hace que los niveles retornen
   * exponencialmente al baseline tónico.
   *
   * @param dt - Tiempo transcurrido desde el último tick (ms)
   */
  decay(dt: number): void {
    for (const [, state] of this.modulators) {
      // Decaimiento exponencial hacia el baseline
      const diff = state.level - state.baseline;
      state.level = state.baseline + diff * Math.exp(-state.decayRate * dt);
    }
  }

  /**
   * Computa los efectos combinados de todos los neuromoduladores.
   *
   * Biología: Los neuromoduladores interactúan de forma compleja. Por ejemplo,
   * el cortisol alto inhibe la consolidación hipocampal y contrarresta los
   * efectos potenciadores de la dopamina sobre el aprendizaje. La acetilcolina
   * y la norepinefrina actúan sinérgicamente para amplificar la atención.
   *
   * @returns Efectos de modulación combinados para aplicar a la SNN
   */
  getEffects(): ModulationEffects {
    const da = this.getLevel(ModulatorType.Dopamine);
    const ser = this.getLevel(ModulatorType.Serotonin);
    const ne = this.getLevel(ModulatorType.Norepinephrine);
    const cort = this.getLevel(ModulatorType.Cortisol);
    const ach = this.getLevel(ModulatorType.Acetylcholine);
    const oxy = this.getLevel(ModulatorType.Oxytocin);

    return {
      // Dopamina ↑ → aprendizaje ↑ | Cortisol ↑ → aprendizaje ↓
      // DA potencia LTP en sinapsis corticoestriatales; cortisol bloquea NMDA hipocampal
      learningRateMultiplier: (1.0 + da * 0.8) * (1.0 - cort * 0.4),

      // Serotonina ↑ → umbral ↑ (más estabilidad, menos impulsividad)
      // Cortisol ↑ → umbral ↑ (inhibición por estrés)
      // 5-HT eleva el umbral de disparo de neuronas piramidales vía receptores 5-HT1A
      thresholdMultiplier: 1.0 + ser * 0.3 + cort * 0.2,

      // Norepinefrina ↑ → atención ↑ | Acetilcolina ↑ → atención ↑
      // NE del locus coeruleus modula la relación señal/ruido cortical
      // ACh del núcleo basal amplifica selectivamente aferencias sensoriales
      attentionGain: 1.0 + ne * 0.6 + ach * 0.5,

      // Acetilcolina ↑ → consolidación ↑ | Cortisol ↑ → consolidación ↓
      // ACh facilita LTP hipocampal; cortisol crónico atrofia dendritas CA3
      consolidationRate: (1.0 + ach * 0.7) * (1.0 - cort * 0.3),

      // Norepinefrina ↑ → ganancia de spikes ↑ (más activación general)
      // NE aumenta la excitabilidad neuronal vía receptores β-adrenérgicos
      spikeGainMultiplier: 1.0 + ne * 0.5,

      // Oxitocina ↑ → boost social (procesamiento de caras, empatía, confianza)
      // OXT potencia neuronas en el surco temporal superior y amígdala
      socialWeightBoost: 1.0 + oxy * 0.6,
    };
  }

  /**
   * Obtiene el nivel actual de un neuromodulador específico.
   *
   * @param type - Tipo de neuromodulador
   * @returns Nivel actual (0.0 - 1.0)
   */
  getLevel(type: ModulatorType): number {
    return this.modulators.get(type)?.level ?? 0;
  }

  /**
   * Obtiene el estado completo de todos los moduladores.
   *
   * @returns Mapa con el estado de cada neuromodulador
   */
  getState(): Map<ModulatorType, ModulatorState> {
    // Retornar copia profunda para evitar mutación externa
    const copy = new Map<ModulatorType, ModulatorState>();
    for (const [type, state] of this.modulators) {
      copy.set(type, { ...state });
    }
    return copy;
  }

  /**
   * Serializa el estado completo del sistema de neuromodulación.
   *
   * @returns Snapshot serializable del estado actual
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
   * Restaura el estado del sistema desde un snapshot serializado.
   *
   * @param snapshot - Estado previamente serializado
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
   * Resetea todos los moduladores a sus niveles basales.
   *
   * Biología: Equivale al retorno al estado homeostático después de
   * un período prolongado de reposo.
   */
  reset(): void {
    for (const [, state] of this.modulators) {
      state.level = state.baseline;
    }
  }
}
