/**
 * AMÍGDALA — Centro de Procesamiento Emocional
 * ===============================================
 * Modela el complejo amigdalino del lóbulo temporal, responsable de
 * la evaluación emocional de estímulos y la generación de respuestas
 * afectivas que modulan todo el procesamiento cerebral.
 *
 * Base biológica:
 *   La amígdala humana contiene ~12 millones de neuronas organizadas en
 *   varios núcleos con funciones específicas:
 *
 *   - Núcleo lateral (LA): Puerta de entrada. Recibe aferencias sensoriales
 *     del tálamo (vía rápida subcortical, "low road") y de cortezas
 *     sensoriales (vía lenta cortical, "high road"). Evalúa la relevancia
 *     emocional del estímulo.
 *
 *   - Núcleo basolateral (BLA): Integra información sensorial con memoria
 *     emocional. Almacena asociaciones estímulo-emoción aprendidas
 *     (condicionamiento clásico, LeDoux 2000).
 *
 *   - Núcleo central (CeA): Salida eferente. Proyecta a:
 *     * Hipotálamo → respuestas autonómicas (frecuencia cardíaca, cortisol)
 *     * Tronco encefálico → respuestas motoras (congelamiento, lucha/huida)
 *     * Núcleos neuromoduladores → modulación global (DA, NE, 5-HT, cortisol)
 *
 *   La amígdala es el "centro de alarma" emocional del cerebro: puede
 *   secuestrar la atención y los recursos cognitivos cuando detecta
 *   estímulos potencialmente amenazantes o muy salientes.
 *
 * Referencia: LeDoux, J. (2000). "Emotion circuits in the brain."
 *             Annual Review of Neuroscience, 23, 155-184.
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Estado emocional bidimensional basado en el modelo circumplejo de Russell.
 *
 * Base biológica:
 *   Las emociones se representan como puntos en un espacio bidimensional:
 *   - Valencia (valence): dimensión placer/displacer, codificada por el
 *     balance de actividad entre núcleos BLA positivos y negativos
 *   - Arousal: nivel de activación fisiológica, codificado por la tasa
 *     de disparo del CeA y sus proyecciones autonómicas
 *
 *   Russell, J.A. (1980). "A circumplex model of affect."
 */
export interface EmotionalState {
  /**
   * Valencia emocional: -1 (muy negativo/aversivo) a +1 (muy positivo/placentero).
   * Neutral = 0.
   */
  valence: number;
  /**
   * Nivel de activación/arousal: 0 (calma profunda) a 1 (activación máxima).
   * Modula la intensidad de la respuesta emocional.
   */
  arousal: number;
}

/**
 * Memoria emocional condicionada almacenada en la amígdala.
 *
 * Base biológica:
 *   El condicionamiento del miedo (y otras emociones) ocurre por potenciación
 *   a largo plazo (LTP) en las sinapsis del núcleo lateral de la amígdala.
 *   Estas memorias son muy resistentes a la extinción y pueden reactivarse
 *   incluso años después (ej: PTSD).
 */
export interface EmotionalMemory {
  /** Patrón de estímulo condicionado (CS) */
  pattern: Float32Array;
  /** Respuesta emocional condicionada (CR) */
  emotion: EmotionalState;
  /**
   * Fuerza de la asociación estímulo-emoción (0–1).
   * Aumenta con repetición, disminuye con extinción.
   */
  strength: number;
}

/**
 * Liberaciones neuromoduladoras producidas por la amígdala.
 *
 * Base biológica:
 *   La amígdala modula la actividad de los principales sistemas
 *   neuromoduladores del cerebro a través de sus proyecciones
 *   eferentes al hipotálamo, VTA, locus coeruleus y núcleos del rafe.
 */
export interface NeuromodulatorRelease {
  /** Dopamina: recompensa, motivación. VTA/SNc. */
  dopamine: number;
  /** Serotonina: regulación emocional, bienestar. Núcleos del rafe. */
  serotonin: number;
  /** Norepinefrina: alerta, atención. Locus coeruleus. */
  norepinephrine: number;
  /** Cortisol: respuesta al estrés. Eje HPA (hipotálamo). */
  cortisol: number;
  /** Acetilcolina: atención focalizada. Núcleo basal de Meynert. */
  acetylcholine: number;
  /** Oxitocina: vinculación social, confianza. Hipotálamo. */
  oxytocin: number;
}

// ==================================================================
// Clase Amygdala
// ==================================================================

/**
 * Amígdala — Centro de evaluación emocional y modulación afectiva.
 *
 * Base biológica:
 *   Implementa las tres funciones principales del complejo amigdalino:
 *   1. Evaluación rápida de valencia emocional de estímulos (LA/BLA)
 *   2. Almacenamiento de memorias emocionales condicionadas (BLA)
 *   3. Producción de señales neuromoduladoras que afectan todo el cerebro (CeA)
 *
 *   La amígdala opera en dos modos:
 *   - Reactivo: evaluación rápida de estímulos novedosos o previamente
 *     condicionados (latencia ~12ms vía tálamo-amígdala directa)
 *   - Modulatorio: ajuste continuo del tono emocional del procesamiento
 *     cerebral vía neuromoduladores
 */
export class Amygdala extends BrainRegion {
  /** Estado emocional actual del sistema */
  private emotionalState: EmotionalState = { valence: 0, arousal: 0.2 };

  /** Memorias emocionales condicionadas (asociaciones estímulo-emoción) */
  private emotionalMemories: EmotionalMemory[] = [];

  /** Última liberación de neuromoduladores producida */
  private lastRelease: NeuromodulatorRelease = {
    dopamine: 0, serotonin: 0, norepinephrine: 0,
    cortisol: 0, acetylcholine: 0, oxytocin: 0,
  };

  /**
   * Capacidad máxima de memorias emocionales.
   * Biología: la amígdala tiene capacidad prácticamente ilimitada para
   * condicionamiento del miedo, pero modelamos un límite práctico.
   */
  private readonly maxEmotionalMemories: number = 5000;

  /**
   * Umbral de novedad: si ninguna memoria emocional tiene similitud
   * superior a este valor, el estímulo se considera "nuevo".
   */
  private readonly noveltyThreshold: number = 0.3;

  /**
   * Inercia emocional: cuánto del estado anterior se preserva en cada tick.
   * Biología: las emociones no cambian instantáneamente; hay inercia
   * debida a la liberación lenta de neuropéptidos y hormonas.
   */
  private readonly emotionalInertia: number = 0.7;

  /**
   * Crea la amígdala.
   *
   * @param neuronCount - Número de neuronas (default: 2000)
   * @param inputCount - Dimensionalidad del input (default: 3000)
   */
  constructor(neuronCount: number = 2000, inputCount: number = 3000) {
    super(
      'amygdala',
      'Amígdala — Procesamiento Emocional',
      neuronCount,
      inputCount,
    );
  }

  // ----------------------------------------------------------------
  // Evaluación Emocional
  // ----------------------------------------------------------------

  /**
   * Evalúa la valencia y arousal emocional de un estímulo.
   *
   * Base biológica:
   *   El núcleo lateral (LA) de la amígdala recibe inputs convergentes del
   *   tálamo sensorial y las cortezas de asociación. Las neuronas del LA
   *   responden a estímulos que:
   *   1. Coinciden con memorias emocionales previas (respuesta condicionada)
   *   2. Son novedosos o inesperados (respuesta de orientación)
   *   3. Tienen propiedades intrínsecamente aversivas (ej: dolor, sonidos fuertes)
   *
   *   Luego el BLA integra esta evaluación con el contexto y el estado
   *   motivacional actual para producir una respuesta emocional modulada.
   *
   * @param input - Vector de spikes del estímulo a evaluar
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Estado emocional evaluado para este estímulo
   */
  evaluateStimulus(
    input: Float32Array,
    modulationEffects: ModulationEffects,
  ): EmotionalState {
    let newValence = 0;
    let newArousal = 0.1; // Arousal base mínimo (estado de vigilia)

    // --- Búsqueda en memorias emocionales (respuesta condicionada) ---
    let bestMatchSimilarity = 0;
    let bestMatchEmotion: EmotionalState | null = null;

    for (const memory of this.emotionalMemories) {
      const similarity = this.cosineSimilarity(input, memory.pattern);
      const adjustedSim = similarity * memory.strength;

      if (adjustedSim > bestMatchSimilarity) {
        bestMatchSimilarity = adjustedSim;
        bestMatchEmotion = memory.emotion;
      }
    }

    if (bestMatchEmotion && bestMatchSimilarity > this.noveltyThreshold) {
      // --- Estímulo familiar con asociación emocional ---
      newValence = bestMatchEmotion.valence * bestMatchSimilarity;
      newArousal = bestMatchEmotion.arousal * bestMatchSimilarity;

      // Amplificar arousal para patrones negativos familiares
      // Biología: la amígdala es especialmente sensible a amenazas conocidas
      if (bestMatchEmotion.valence < -0.3) {
        newArousal = Math.min(1.0, newArousal * 1.5);
      }
    } else {
      // --- Estímulo novedoso ---
      // Biología: la novedad genera una leve respuesta positiva (curiosidad)
      // mediada por el sistema dopaminérgico mesolímbico
      newValence = 0.1; // Leve sesgo positivo (curiosidad)
      newArousal = 0.3 + (1 - bestMatchSimilarity) * 0.2; // Más novedad = más arousal

      // La atención modulada aumenta el arousal ante novedad
      newArousal *= modulationEffects.attentionGain;
    }

    // Aplicar inercia emocional (suavizado temporal)
    const finalValence =
      this.emotionalState.valence * this.emotionalInertia +
      newValence * (1 - this.emotionalInertia);
    const finalArousal =
      this.emotionalState.arousal * this.emotionalInertia +
      newArousal * (1 - this.emotionalInertia);

    // Clamp a rangos válidos
    this.emotionalState = {
      valence: Math.max(-1, Math.min(1, finalValence)),
      arousal: Math.max(0, Math.min(1, finalArousal)),
    };

    return { ...this.emotionalState };
  }

  // ----------------------------------------------------------------
  // Producción de Neuromoduladores
  // ----------------------------------------------------------------

  /**
   * Produce señales neuromoduladoras basadas en el estado emocional actual.
   *
   * Base biológica:
   *   El núcleo central (CeA) de la amígdala es el principal efector
   *   emocional del cerebro. Sus proyecciones eferentes activan/inhiben
   *   los núcleos neuromoduladores subcorticales:
   *
   *   - Valencia positiva alta → VTA (dopamina ↑)
   *     Recompensa activa neuronas dopaminérgicas del área tegmental ventral
   *
   *   - Arousal alto + valencia negativa → LC (norepinefrina ↑) + HPA (cortisol ↑)
   *     Estrés/amenaza activa el locus coeruleus y el eje hipotálamo-hipófisis-adrenal
   *
   *   - Estímulo novedoso → LC (norepinefrina ↑ moderada)
   *     La novedad genera una respuesta de orientación mediada por NE
   *
   *   - Valencia positiva + familiaridad → Rafe (serotonina ↑) + PVN (oxitocina ↑)
   *     Seguridad y confort activan los sistemas de bienestar y vinculación
   *
   *   - Atención requerida → NB (acetilcolina ↑)
   *     Estímulos salientes activan el núcleo basal de Meynert
   *
   * @returns Cantidades de liberación para cada neuromodulador (0–1)
   */
  produceNeuromodulators(): NeuromodulatorRelease {
    const { valence, arousal } = this.emotionalState;

    // --- Dopamina: valencia positiva ---
    // VTA dopamine neurons fire in response to reward prediction errors
    const dopamine = Math.max(0, valence) * 0.5 + (valence > 0.5 ? 0.2 : 0);

    // --- Serotonina: estabilidad emocional, sesgo positivo familiar ---
    // Los núcleos del rafe mantienen tono serotoninérgico modulado por seguridad
    const serotonin =
      (valence > 0 ? valence * 0.3 : 0) + (1 - arousal) * 0.2;

    // --- Norepinefrina: arousal, novedad, alerta ---
    // LC tonic and phasic firing modulated by arousal and novelty
    const norepinephrine =
      arousal * 0.4 + (valence < -0.3 ? Math.abs(valence) * 0.3 : 0);

    // --- Cortisol: estrés (arousal alto + valencia negativa) ---
    // HPA axis activation during sustained negative affect
    const cortisol =
      valence < 0 ? Math.abs(valence) * arousal * 0.5 : 0;

    // --- Acetilcolina: atención focalizada (estímulos salientes) ---
    // NBM activation proportional to stimulus salience
    const acetylcholine = arousal * 0.3 + Math.abs(valence) * 0.2;

    // --- Oxitocina: confort social, valencia positiva baja arousal ---
    // PVN oxytocin release during safe social contexts
    const oxytocin =
      valence > 0.2 && arousal < 0.5 ? valence * 0.4 : 0;

    this.lastRelease = {
      dopamine: Math.min(1, Math.max(0, dopamine)),
      serotonin: Math.min(1, Math.max(0, serotonin)),
      norepinephrine: Math.min(1, Math.max(0, norepinephrine)),
      cortisol: Math.min(1, Math.max(0, cortisol)),
      acetylcholine: Math.min(1, Math.max(0, acetylcholine)),
      oxytocin: Math.min(1, Math.max(0, oxytocin)),
    };

    return { ...this.lastRelease };
  }

  // ----------------------------------------------------------------
  // Condicionamiento Emocional
  // ----------------------------------------------------------------

  /**
   * Condiciona una respuesta emocional a un estímulo.
   *
   * Base biológica:
   *   El condicionamiento ocurre por LTP en las sinapsis LA → BLA.
   *   Un estímulo neutro (CS) que co-ocurre con un estímulo emocional
   *   (US) adquiere la capacidad de evocar la respuesta emocional por
   *   sí solo. Este aprendizaje es:
   *   - Rápido (puede ocurrir en una sola exposición para estímulos aversivos)
   *   - Resistente a extinción (las asociaciones se debilitan pero no se borran)
   *   - Modulado por NE y cortisol (los estímulos estresantes se condicionan más fuerte)
   *
   * @param stimulus - Patrón del estímulo condicionado (CS)
   * @param emotion - Respuesta emocional a asociar (CR)
   */
  conditionResponse(stimulus: Float32Array, emotion: EmotionalState): void {
    // Verificar si ya existe una asociación similar
    for (const memory of this.emotionalMemories) {
      const similarity = this.cosineSimilarity(stimulus, memory.pattern);
      if (similarity > 0.8) {
        // Actualizar asociación existente (reconsolidación)
        memory.emotion.valence =
          memory.emotion.valence * 0.5 + emotion.valence * 0.5;
        memory.emotion.arousal =
          memory.emotion.arousal * 0.5 + emotion.arousal * 0.5;
        memory.strength = Math.min(1.0, memory.strength + 0.1);
        return;
      }
    }

    // Nueva asociación
    if (this.emotionalMemories.length >= this.maxEmotionalMemories) {
      // Eliminar la asociación más débil
      let weakestIdx = 0;
      let weakestStrength = this.emotionalMemories[0].strength;
      for (let i = 1; i < this.emotionalMemories.length; i++) {
        if (this.emotionalMemories[i].strength < weakestStrength) {
          weakestStrength = this.emotionalMemories[i].strength;
          weakestIdx = i;
        }
      }
      this.emotionalMemories.splice(weakestIdx, 1);
    }

    this.emotionalMemories.push({
      pattern: new Float32Array(stimulus),
      emotion: { ...emotion },
      strength: 1.0,
    });
  }

  // ----------------------------------------------------------------
  // Procesamiento principal
  // ----------------------------------------------------------------

  /**
   * Procesa spikes de entrada y produce spikes modulados emocionalmente.
   *
   * Base biológica:
   *   Flujo de procesamiento amigdalino:
   *   1. Input llega por vía rápida (tálamo→LA) y lenta (corteza→LA)
   *   2. LA evalúa valencia y arousal comparando con memorias
   *   3. BLA integra con contexto y produce estado emocional
   *   4. CeA traduce estado emocional a señales neuromoduladoras
   *   5. Output: spikes de salida + señales neuromoduladoras
   *
   * @param spikes - Vector de spikes de entrada
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Vector de spikes de salida teñidos emocionalmente
   */
  processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects,
  ): Float32Array {
    // Adaptar dimensionalidad
    const adaptedInput = this.adaptInput(spikes);

    // 1. Evaluar contenido emocional del estímulo
    this.evaluateStimulus(adaptedInput, modulationEffects);

    // 2. Producir señales neuromoduladoras
    this.produceNeuromodulators();

    // 3. Generar spikes de salida modulados por emoción
    const output = new Float32Array(this.neuronCount);
    const { valence, arousal } = this.emotionalState;

    // La magnitud de la respuesta escala con arousal
    // Biología: el CeA modula la ganancia de sus outputs por arousal
    const responseGain = 0.5 + arousal * 0.5;

    // Computar corrientes y generar spikes
    const copyLen = Math.min(adaptedInput.length, this.neuronCount);
    for (let i = 0; i < copyLen; i++) {
      // Los spikes de salida llevan la "marca" emocional
      output[i] = adaptedInput[i] * responseGain;
    }

    // Añadir señal de valencia como modulación de los spikes
    // Biología: neuronas del CeA codifican valencia en su tasa de disparo
    const valenceSignalStart = Math.min(copyLen, this.neuronCount - 100);
    for (let i = valenceSignalStart; i < this.neuronCount; i++) {
      // Últimas neuronas codifican valencia y arousal directamente
      const t = (i - valenceSignalStart) / Math.max(1, this.neuronCount - valenceSignalStart);
      output[i] = t < 0.5 ? (valence + 1) / 2 * arousal : arousal;
    }

    // Aplicar ganancia de neuromodulación
    for (let i = 0; i < output.length; i++) {
      output[i] *= modulationEffects.spikeGainMultiplier;
    }

    return output;
  }

  // ----------------------------------------------------------------
  // Utilidades
  // ----------------------------------------------------------------

  /**
   * Adapta un vector de entrada a la dimensionalidad esperada.
   */
  private adaptInput(input: Float32Array): Float32Array {
    if (input.length === this.inputCount) return input;

    const adapted = new Float32Array(this.inputCount);
    const copyLen = Math.min(input.length, this.inputCount);
    for (let i = 0; i < copyLen; i++) {
      adapted[i] = input[i];
    }
    return adapted;
  }

  /**
   * Similitud coseno entre dos vectores.
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < len; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  // ----------------------------------------------------------------
  // Accesores públicos
  // ----------------------------------------------------------------

  /** Retorna el estado emocional actual */
  getEmotionalState(): EmotionalState {
    return { ...this.emotionalState };
  }

  /** Retorna la última liberación de neuromoduladores */
  getLastRelease(): NeuromodulatorRelease {
    return { ...this.lastRelease };
  }

  /** Número de memorias emocionales condicionadas */
  get conditionedMemoryCount(): number {
    return this.emotionalMemories.length;
  }
}
