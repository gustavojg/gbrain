/**
 * DECODIFICADOR EMOCIONAL — Neuromoduladores a expresiones
 * ========================================================
 * Convierte los niveles de neuromoduladores del sistema límbico
 * en estados emocionales legibles y expresivos.
 * 
 * Basado en el modelo dimensional de emociones (Russell, 1980):
 * - Valencia: placer ↔ displacer (eje horizontal)
 * - Arousal: activación ↔ calma (eje vertical)
 * 
 * Los neuromoduladores se mapean a este espacio 2D, y de ahí
 * a categorías emocionales discretas.
 */

/** Estado emocional del cerebro */
export interface EmotionalState {
  /** Valencia: -1 (displacer) a 1 (placer) */
  valence: number;
  /** Arousal: 0 (calma) a 1 (máxima activación) */
  arousal: number;
  /** Emoción primaria identificada */
  primaryEmotion: string;
  /** Emoji representativo */
  emoji: string;
  /** Intensidad de la emoción (0-1) */
  intensity: number;
  /** Descripción textual */
  description: string;
  /** Color asociado (para visualización) */
  color: string;
}

/** Niveles de neuromoduladores */
export interface ModulatorLevels {
  dopamine: number;
  serotonin: number;
  norepinephrine: number;
  cortisol: number;
  acetylcholine: number;
  oxytocin: number;
}

/**
 * Mapa de emociones en el espacio Valencia × Arousal.
 * Cada emoción ocupa una zona del espacio circumplex.
 */
const EMOTION_MAP: Array<{
  name: string;
  emoji: string;
  valence: [number, number]; // [min, max]
  arousal: [number, number]; // [min, max]
  color: string;
  description: string;
}> = [
  // Alto arousal, alta valencia positiva
  { name: 'Eufórico', emoji: '🤩', valence: [0.6, 1.0], arousal: [0.7, 1.0], color: '#FFD700', description: 'Extremadamente positivo y activado' },
  { name: 'Entusiasmado', emoji: '😄', valence: [0.3, 0.7], arousal: [0.6, 0.9], color: '#FFA500', description: 'Muy motivado y positivo' },
  { name: 'Curioso', emoji: '🤔', valence: [0.1, 0.5], arousal: [0.5, 0.8], color: '#7B68EE', description: 'Interesado en explorar algo nuevo' },
  
  // Bajo arousal, alta valencia positiva
  { name: 'Contento', emoji: '😊', valence: [0.3, 0.7], arousal: [0.2, 0.5], color: '#90EE90', description: 'Satisfecho y tranquilo' },
  { name: 'Sereno', emoji: '😌', valence: [0.1, 0.5], arousal: [0.0, 0.3], color: '#87CEEB', description: 'En paz, relajado' },
  { name: 'Amoroso', emoji: '🥰', valence: [0.5, 1.0], arousal: [0.2, 0.6], color: '#FF69B4', description: 'Conexión y cariño' },
  
  // Alto arousal, valencia negativa
  { name: 'Asustado', emoji: '😨', valence: [-1.0, -0.3], arousal: [0.7, 1.0], color: '#FF4500', description: 'Amenaza detectada, alerta máxima' },
  { name: 'Estresado', emoji: '😰', valence: [-0.6, -0.1], arousal: [0.5, 0.8], color: '#FF6347', description: 'Bajo presión' },
  { name: 'Frustrado', emoji: '😤', valence: [-0.7, -0.2], arousal: [0.6, 0.9], color: '#DC143C', description: 'Impedido, bloqueado' },
  
  // Bajo arousal, valencia negativa
  { name: 'Triste', emoji: '😢', valence: [-0.8, -0.2], arousal: [0.0, 0.3], color: '#4682B4', description: 'Bajo y desanimado' },
  { name: 'Aburrido', emoji: '😑', valence: [-0.3, 0.1], arousal: [0.0, 0.2], color: '#A9A9A9', description: 'Sin estímulos interesantes' },
  
  // Neutro
  { name: 'Neutral', emoji: '😐', valence: [-0.15, 0.15], arousal: [0.1, 0.4], color: '#C0C0C0', description: 'Estado base, sin emoción dominante' },
  { name: 'Alerta', emoji: '👀', valence: [-0.1, 0.2], arousal: [0.4, 0.7], color: '#FFD700', description: 'Atento pero sin emoción fuerte' },
  { name: 'Concentrado', emoji: '🧐', valence: [0.0, 0.3], arousal: [0.4, 0.7], color: '#4169E1', description: 'Enfocado en una tarea' },
];

/**
 * Decodificador Emocional — Traduce bioquímica a emociones.
 */
export class EmotionDecoder {
  
  /**
   * Decodifica niveles de neuromoduladores a estado emocional.
   * 
   * @param levels - Niveles actuales de cada neuromodulador (0-1)
   * @returns Estado emocional completo
   */
  decode(levels: ModulatorLevels): EmotionalState {
    // 1. Calcular valencia y arousal desde neuromoduladores
    const valence = this.calculateValence(levels);
    const arousal = this.calculateArousal(levels);
    
    // 2. Encontrar la emoción más cercana en el circumplex
    const emotion = this.findClosestEmotion(valence, arousal);
    
    // 3. Calcular intensidad
    const intensity = Math.sqrt(valence * valence + arousal * arousal) / Math.sqrt(2);
    
    return {
      valence,
      arousal,
      primaryEmotion: emotion.name,
      emoji: emotion.emoji,
      intensity: Math.min(1, intensity),
      description: emotion.description,
      color: emotion.color,
    };
  }

  /**
   * Calcula la valencia (placer/displacer) desde los neuromoduladores.
   * 
   * Dopamina y serotonina → positivo
   * Cortisol → negativo
   * Oxitocina → ligeramente positivo
   */
  private calculateValence(levels: ModulatorLevels): number {
    const positive = (levels.dopamine * 0.4) + (levels.serotonin * 0.3) + (levels.oxytocin * 0.2);
    const negative = (levels.cortisol * 0.5);
    
    // Rango: -1 a 1
    return Math.max(-1, Math.min(1, (positive - negative) * 2 - 0.3));
  }

  /**
   * Calcula el arousal (activación/calma) desde los neuromoduladores.
   * 
   * Norepinefrina y cortisol → alta activación
   * Serotonina → calma
   * Acetilcolina → moderadamente activante
   */
  private calculateArousal(levels: ModulatorLevels): number {
    const activating = (levels.norepinephrine * 0.4) + (levels.cortisol * 0.3) + (levels.acetylcholine * 0.2);
    const calming = (levels.serotonin * 0.3);
    
    // Rango: 0 a 1
    return Math.max(0, Math.min(1, activating - calming + 0.2));
  }

  /**
   * Encuentra la emoción más cercana en el mapa circumplex.
   */
  private findClosestEmotion(valence: number, arousal: number): typeof EMOTION_MAP[0] {
    let bestMatch = EMOTION_MAP[EMOTION_MAP.length - 1]; // Neutral por defecto
    let bestDist = Infinity;

    for (const emotion of EMOTION_MAP) {
      // Verificar si estamos dentro del rango de esta emoción
      const inValence = valence >= emotion.valence[0] && valence <= emotion.valence[1];
      const inArousal = arousal >= emotion.arousal[0] && arousal <= emotion.arousal[1];

      if (inValence && inArousal) {
        // Calcular distancia al centro de la emoción
        const centerV = (emotion.valence[0] + emotion.valence[1]) / 2;
        const centerA = (emotion.arousal[0] + emotion.arousal[1]) / 2;
        const dist = Math.sqrt((valence - centerV) ** 2 + (arousal - centerA) ** 2);

        if (dist < bestDist) {
          bestDist = dist;
          bestMatch = emotion;
        }
      }
    }

    // Si no encontramos match exacto, buscar el más cercano
    if (bestDist === Infinity) {
      for (const emotion of EMOTION_MAP) {
        const centerV = (emotion.valence[0] + emotion.valence[1]) / 2;
        const centerA = (emotion.arousal[0] + emotion.arousal[1]) / 2;
        const dist = Math.sqrt((valence - centerV) ** 2 + (arousal - centerA) ** 2);

        if (dist < bestDist) {
          bestDist = dist;
          bestMatch = emotion;
        }
      }
    }

    return bestMatch;
  }

  /**
   * Genera una descripción narrativa del estado emocional.
   */
  narrate(state: EmotionalState): string {
    const intensityWord = state.intensity > 0.7 ? 'intensamente' :
                          state.intensity > 0.4 ? 'moderadamente' : 'ligeramente';
    
    return `${state.emoji} Me siento ${intensityWord} ${state.primaryEmotion.toLowerCase()}. ${state.description}.`;
  }
}
