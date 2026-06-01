/**
 * Motor de Consolidación Mnémica (Replay durante el Sueño)
 * ==========================================================
 * Implementa el proceso de consolidación de memorias del hipocampo a la corteza.
 *
 * Biología: Durante el sueño de ondas lentas (NREM), el hipocampo "reproduce"
 * memorias recientes (sharp-wave ripples a ~200Hz) hacia las regiones corticales
 * correspondientes. Este replay comprimido (~20x más rápido que el tiempo real)
 * fortalece las conexiones corticales que almacenarán la memoria a largo plazo.
 *
 * Proceso:
 * 1. Seleccionar memorias hipocampales recientes ordenadas por relevancia
 * 2. Para cada memoria, identificar la región cortical asociada
 * 3. Reproducir el patrón de activación como entrada a la región con aprendizaje activo
 * 4. Repetir múltiples veces (reactivación cíclica durante el sueño)
 *
 * El resultado neto es la transferencia gradual del almacenamiento mnémico
 * desde el hipocampo (rápido pero temporal) a la corteza (lento pero duradero).
 */

import type { BrainRegion } from '../brain-region.js';
import type { ModulationEffects } from '../neuromodulators/modulator-system.js';

/**
 * Entrada de memoria a corto plazo del hipocampo.
 * Representa un episodio memorizado que puede ser consolidado.
 */
export interface ShortTermEntry {
  /** Patrón de activación neural almacenado */
  pattern: Float32Array;
  /** Etiqueta semántica del recuerdo */
  label: string;
  /** Marca temporal de adquisición (ms) */
  timestamp: number;
  /** Fuerza de la traza mnémica (0.0 - 1.0) */
  strength: number;
  /** Identificador de la región cortical asociada */
  associatedRegion: string;
  /** Número de veces que esta memoria ha sido replayada */
  replayCount: number;
}

/**
 * Estadísticas del proceso de consolidación.
 */
export interface ConsolidationStats {
  /** Número total de memorias reproducidas */
  memoriesReplayed: number;
  /** Número de sinapsis fortalecidas durante el replay */
  synapsesStrengthened: number;
  /** Duración total del proceso en ms de simulación */
  duration: number;
  /** Memorias que fueron exitosamente consolidadas */
  consolidatedLabels: string[];
  /** Memorias que fueron descartadas por debilidad */
  prunedLabels: string[];
}

/**
 * Motor de consolidación que implementa el replay hipocampal.
 *
 * Simula el proceso de consolidación de memorias durante el sueño,
 * reproduciendo patrones hipocampales hacia las cortezas asociadas
 * para fortalecer el almacenamiento a largo plazo.
 */
export class ConsolidationEngine {
  /** Número de ciclos de replay por memoria (modela las ondas lentas NREM) */
  private readonly replayCycles: number;

  /** Umbral mínimo de fuerza para consolidar (memorias más débiles se pierden) */
  private readonly minimumStrength: number;

  /** Efectos de modulación durante el sueño (acetilcolina alta, cortisol bajo) */
  private readonly sleepModulationEffects: ModulationEffects;

  /**
   * Crea un nuevo motor de consolidación.
   *
   * @param replayCycles - Número de veces que cada memoria se reproduce (default: 5)
   * @param minimumStrength - Fuerza mínima para que una memoria sea consolidada (default: 0.2)
   */
  constructor(replayCycles: number = 5, minimumStrength: number = 0.2) {
    this.replayCycles = replayCycles;
    this.minimumStrength = minimumStrength;

    // Durante el sueño NREM:
    // - Acetilcolina BAJA (permite replay hipocampo → corteza)
    // - Cortisol BAJO (no hay estrés)
    // - Aprendizaje moderado (consolidación, no adquisición)
    this.sleepModulationEffects = {
      learningRateMultiplier: 1.2,
      thresholdMultiplier: 0.8, // Umbral más bajo para facilitar reactivación
      attentionGain: 0.5,       // Atención reducida (dormido)
      consolidationRate: 1.5,   // Consolidación amplificada
      spikeGainMultiplier: 0.7, // Actividad global reducida
      socialWeightBoost: 1.0,   // Neutral durante sueño
    };
  }

  /**
   * Ejecuta el proceso de consolidación mnémica.
   *
   * Biología: Simula múltiples ciclos de sharp-wave ripples durante
   * el sueño NREM. Cada ripple reproduce un episodio hipocampal
   * comprimido (~20x) hacia la región cortical correspondiente,
   * permitiendo que los pesos sinápticos corticales se ajusten
   * gradualmente para almacenar la memoria a largo plazo.
   *
   * @param hippocampalMemories - Memorias del hipocampo a consolidar
   * @param regions - Mapa de regiones cerebrales disponibles
   * @param modulationOverride - Efectos de modulación opcionales (override del default de sueño)
   * @returns Estadísticas del proceso de consolidación
   */
  consolidate(
    hippocampalMemories: ShortTermEntry[],
    regions: Map<string, BrainRegion>,
    modulationOverride?: ModulationEffects
  ): ConsolidationStats {
    const startTime = performance.now();
    const effects = modulationOverride ?? this.sleepModulationEffects;

    let synapsesStrengthened = 0;
    const consolidatedLabels: string[] = [];
    const prunedLabels: string[] = [];

    // Ordenar memorias por fuerza descendente (las más fuertes primero)
    const sortedMemories = [...hippocampalMemories].sort(
      (a, b) => b.strength - a.strength
    );

    let memoriesReplayed = 0;

    for (const memory of sortedMemories) {
      // Podar memorias demasiado débiles
      if (memory.strength < this.minimumStrength) {
        prunedLabels.push(memory.label);
        continue;
      }

      // Encontrar la región cortical asociada
      const targetRegion = regions.get(memory.associatedRegion);
      if (!targetRegion) {
        // Región no encontrada, saltar
        continue;
      }

      // Verificar que el patrón sea compatible con la región
      if (memory.pattern.length !== targetRegion.inputs) {
        continue;
      }

      // Replay cíclico: reproducir el patrón múltiples veces
      for (let cycle = 0; cycle < this.replayCycles; cycle++) {
        // Alimentar el patrón al buffer sensorial de la región
        targetRegion.feedInput(memory.pattern);

        // Ejecutar un paso de procesamiento con modulación de sueño
        const activity = targetRegion.step(1, effects);

        // Contar sinapsis fortalecidas (neuronas que dispararon = synapses activas)
        synapsesStrengthened += activity.activeNeurons.length;
        memoriesReplayed++;
      }

      // Incrementar contador de replay de la memoria
      memory.replayCount += this.replayCycles;
      consolidatedLabels.push(memory.label);
    }

    const duration = performance.now() - startTime;

    return {
      memoriesReplayed,
      synapsesStrengthened,
      duration,
      consolidatedLabels,
      prunedLabels,
    };
  }

  /**
   * Ejecuta un ciclo de consolidación selectiva.
   *
   * A diferencia de consolidate(), este método consolida solo
   * memorias asociadas a una emoción fuerte (strength > umbral alto),
   * modelando cómo las memorias emocionales se consolidan preferentemente.
   *
   * @param hippocampalMemories - Memorias del hipocampo
   * @param regions - Regiones cerebrales
   * @param emotionalThreshold - Umbral de fuerza para consolidación selectiva (default: 0.7)
   * @returns Estadísticas del proceso
   */
  consolidateEmotional(
    hippocampalMemories: ShortTermEntry[],
    regions: Map<string, BrainRegion>,
    emotionalThreshold: number = 0.7
  ): ConsolidationStats {
    // Filtrar solo memorias con fuerza emocional alta
    const emotionalMemories = hippocampalMemories.filter(
      m => m.strength >= emotionalThreshold
    );

    // Usar más ciclos de replay para memorias emocionales (la amígdala potencia)
    const emotionalEffects: ModulationEffects = {
      ...this.sleepModulationEffects,
      learningRateMultiplier: 1.8, // La amígdala amplifica la plasticidad
      consolidationRate: 2.0,       // Consolidación aún más fuerte
    };

    return this.consolidate(emotionalMemories, regions, emotionalEffects);
  }
}
