/**
 * EMOTION DECODER — Neuromodulators to expressions
 * ========================================================
 * Converts the neuromodulator levels of the limbic system
 * into readable, expressive emotional states.
 *
 * Based on the dimensional model of emotions (Russell, 1980):
 * - Valence: pleasure ↔ displeasure (horizontal axis)
 * - Arousal: activation ↔ calm (vertical axis)
 *
 * The neuromodulators are mapped to this 2D space, and from there
 * to discrete emotional categories.
 */

/** Emotional state of the brain */
export interface EmotionalState {
  /** Valence: -1 (displeasure) to 1 (pleasure) */
  valence: number;
  /** Arousal: 0 (calm) to 1 (maximum activation) */
  arousal: number;
  /** Identified primary emotion */
  primaryEmotion: string;
  /** Representative emoji */
  emoji: string;
  /** Emotion intensity (0-1) */
  intensity: number;
  /** Textual description */
  description: string;
  /** Associated color (for visualization) */
  color: string;
}

/** Neuromodulator levels */
export interface ModulatorLevels {
  dopamine: number;
  serotonin: number;
  norepinephrine: number;
  cortisol: number;
  acetylcholine: number;
  oxytocin: number;
}

/**
 * Map of emotions in the Valence × Arousal space.
 * Each emotion occupies a zone of the circumplex space.
 */
const EMOTION_MAP: Array<{
  name: string;
  emoji: string;
  valence: [number, number]; // [min, max]
  arousal: [number, number]; // [min, max]
  color: string;
  description: string;
}> = [
  // High arousal, high positive valence
  { name: 'Euphoric', emoji: '🤩', valence: [0.6, 1.0], arousal: [0.7, 1.0], color: '#FFD700', description: 'Extremely positive and activated' },
  { name: 'Excited', emoji: '😄', valence: [0.3, 0.7], arousal: [0.6, 0.9], color: '#FFA500', description: 'Highly motivated and positive' },
  { name: 'Curious', emoji: '🤔', valence: [0.1, 0.5], arousal: [0.5, 0.8], color: '#7B68EE', description: 'Interested in exploring something new' },

  // Low arousal, high positive valence
  { name: 'Content', emoji: '😊', valence: [0.3, 0.7], arousal: [0.2, 0.5], color: '#90EE90', description: 'Satisfied and calm' },
  { name: 'Serene', emoji: '😌', valence: [0.1, 0.5], arousal: [0.0, 0.3], color: '#87CEEB', description: 'At peace, relaxed' },
  { name: 'Loving', emoji: '🥰', valence: [0.5, 1.0], arousal: [0.2, 0.6], color: '#FF69B4', description: 'Connection and affection' },

  // High arousal, negative valence
  { name: 'Afraid', emoji: '😨', valence: [-1.0, -0.3], arousal: [0.7, 1.0], color: '#FF4500', description: 'Threat detected, maximum alertness' },
  { name: 'Stressed', emoji: '😰', valence: [-0.6, -0.1], arousal: [0.5, 0.8], color: '#FF6347', description: 'Under pressure' },
  { name: 'Frustrated', emoji: '😤', valence: [-0.7, -0.2], arousal: [0.6, 0.9], color: '#DC143C', description: 'Blocked, impeded' },

  // Low arousal, negative valence
  { name: 'Sad', emoji: '😢', valence: [-0.8, -0.2], arousal: [0.0, 0.3], color: '#4682B4', description: 'Low and discouraged' },
  { name: 'Bored', emoji: '😑', valence: [-0.3, 0.1], arousal: [0.0, 0.2], color: '#A9A9A9', description: 'No interesting stimuli' },

  // Neutral
  { name: 'Neutral', emoji: '😐', valence: [-0.15, 0.15], arousal: [0.1, 0.4], color: '#C0C0C0', description: 'Baseline state, no dominant emotion' },
  { name: 'Alert', emoji: '👀', valence: [-0.1, 0.2], arousal: [0.4, 0.7], color: '#FFD700', description: 'Attentive but no strong emotion' },
  { name: 'Focused', emoji: '🧐', valence: [0.0, 0.3], arousal: [0.4, 0.7], color: '#4169E1', description: 'Concentrated on a task' },
];

/**
 * Emotion Decoder — Translates biochemistry into emotions.
 */
export class EmotionDecoder {

  /**
   * Decodes neuromodulator levels into an emotional state.
   *
   * @param levels - Current levels of each neuromodulator (0-1)
   * @returns Complete emotional state
   */
  decode(levels: ModulatorLevels): EmotionalState {
    // 1. Compute valence and arousal from neuromodulators
    const valence = this.calculateValence(levels);
    const arousal = this.calculateArousal(levels);

    // 2. Find the closest emotion in the circumplex
    const emotion = this.findClosestEmotion(valence, arousal);

    // 3. Compute intensity
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
   * Computes the valence (pleasure/displeasure) from the neuromodulators.
   *
   * Dopamine and serotonin → positive
   * Cortisol → negative
   * Oxytocin → slightly positive
   */
  private calculateValence(levels: ModulatorLevels): number {
    const positive = (levels.dopamine * 0.4) + (levels.serotonin * 0.3) + (levels.oxytocin * 0.2);
    const negative = (levels.cortisol * 0.5);

    // Range: -1 to 1
    return Math.max(-1, Math.min(1, (positive - negative) * 2 - 0.3));
  }

  /**
   * Computes the arousal (activation/calm) from the neuromodulators.
   *
   * Norepinephrine and cortisol → high activation
   * Serotonin → calm
   * Acetylcholine → moderately activating
   */
  private calculateArousal(levels: ModulatorLevels): number {
    const activating = (levels.norepinephrine * 0.4) + (levels.cortisol * 0.3) + (levels.acetylcholine * 0.2);
    const calming = (levels.serotonin * 0.3);

    // Range: 0 to 1
    return Math.max(0, Math.min(1, activating - calming + 0.2));
  }

  /**
   * Finds the closest emotion in the circumplex map.
   */
  private findClosestEmotion(valence: number, arousal: number): typeof EMOTION_MAP[0] {
    let bestMatch = EMOTION_MAP[EMOTION_MAP.length - 1]; // Neutral by default
    let bestDist = Infinity;

    for (const emotion of EMOTION_MAP) {
      // Check whether we are within this emotion's range
      const inValence = valence >= emotion.valence[0] && valence <= emotion.valence[1];
      const inArousal = arousal >= emotion.arousal[0] && arousal <= emotion.arousal[1];

      if (inValence && inArousal) {
        // Compute the distance to the emotion's center
        const centerV = (emotion.valence[0] + emotion.valence[1]) / 2;
        const centerA = (emotion.arousal[0] + emotion.arousal[1]) / 2;
        const dist = Math.sqrt((valence - centerV) ** 2 + (arousal - centerA) ** 2);

        if (dist < bestDist) {
          bestDist = dist;
          bestMatch = emotion;
        }
      }
    }

    // If we don't find an exact match, look for the closest one
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
   * Generates a narrative description of the emotional state.
   */
  narrate(state: EmotionalState): string {
    const intensityWord = state.intensity > 0.7 ? 'intensely' :
                          state.intensity > 0.4 ? 'moderately' : 'slightly';

    return `${state.emoji} I feel ${intensityWord} ${state.primaryEmotion.toLowerCase()}. ${state.description}.`;
  }
}
