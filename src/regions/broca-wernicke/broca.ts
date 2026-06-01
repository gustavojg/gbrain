/**
 * ÁREA DE BROCA — Producción del Lenguaje
 * =========================================
 * Modela el área de Broca (pars opercularis y pars triangularis del giro
 * frontal inferior, áreas de Brodmann 44/45) del cerebro digital,
 * responsable de la producción y planificación del habla.
 *
 * Base biológica:
 *   El área de Broca fue descubierta por Paul Broca en 1861 al estudiar
 *   al paciente "Tan" (Louis Leborgne), quien solo podía pronunciar la
 *   sílaba "tan" tras una lesión en el giro frontal inferior izquierdo.
 *   La "afasia de Broca" (afasia expresiva/no fluente) se caracteriza
 *   por comprensión preservada pero producción verbal severamente reducida
 *   y agramatismo (habla telegráfica).
 *
 *   Funciones modeladas:
 *   1. **Planificación articulatoria**: Traduce representaciones semánticas
 *      (intención comunicativa de la corteza prefrontal) en secuencias de
 *      patrones motores del habla.
 *
 *   2. **Selección léxica para producción**: Busca en el léxico compartido
 *      las palabras que mejor expresan la intención semántica. A diferencia
 *      de Wernicke (que recibe input y busca significado), Broca parte
 *      del significado y busca la forma (acceso léxico invertido).
 *
 *   3. **Secuenciación**: Ordena los patrones léxicos seleccionados en
 *      una secuencia temporalmente correcta (sintaxis simplificada).
 *
 *   4. **Modulación emocional**: El estado emocional (valence/arousal de
 *      la amígdala) modifica la prosodia y la selección léxica, modelando
 *      cómo las emociones afectan el habla (ej: vocabulario más reducido
 *      bajo estrés alto).
 *
 *   Conexiones:
 *   - Recibe: intención semántica de la corteza prefrontal (vía fibras
 *     fronto-frontales) y estado emocional de la amígdala
 *   - Envía: patrones motores del habla a la corteza motora primaria
 *   - Comparte: léxico con el área de Wernicke (vía fascículo arqueado)
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { Lexicon, type LexiconMatch } from './lexicon.js';

/**
 * Piso de activación para el k-WTA. Los potenciales son sumas de pesos
 * adimensionales (~0 en reposo), no mV. Sin este piso el gate compara contra
 * `_modulatedThreshold` (−55 mV), siempre pasa, y el k-WTA enciende k neuronas
 * por tick → actividad constante y falsa. Con el piso, en reposo queda a 0%.
 */
const ACTIVATION_FLOOR = 1e-3;

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Estado emocional que modula la producción del lenguaje.
 *
 * Base biológica:
 *   La amígdala proyecta al área de Broca modulando la prosodia
 *   (entonación emocional) y la selección léxica. Un arousal alto
 *   reduce la fluidez verbal; una valencia positiva sesga hacia
 *   vocabulario positivo (mood-congruent lexical access).
 */
export interface EmotionalState {
  /**
   * Valencia emocional (-1.0 = muy negativo, +1.0 = muy positivo).
   * Afecta la selección de palabras y el tono de la respuesta.
   */
  valence: number;
  /**
   * Nivel de activación emocional (0.0 = calma, 1.0 = alta excitación).
   * Afecta la longitud y complejidad de la respuesta: arousal alto
   * produce respuestas más cortas y menos elaboradas.
   */
  arousal: number;
}

/**
 * Palabra seleccionada para producción con su puntuación de relevancia.
 */
export interface SelectedWord {
  /** Palabra seleccionada del léxico */
  word: string;
  /** Puntuación de relevancia semántica (0.0–1.0) */
  relevance: number;
  /** Patrón neural asociado en el léxico */
  pattern: Float32Array;
}

/**
 * Respuesta lingüística generada por el área de Broca.
 */
export interface LanguageResponse {
  /** Secuencia de patrones motores del habla (uno por palabra) */
  wordPatterns: Float32Array[];
  /** Palabras seleccionadas en orden */
  words: string[];
  /** Confianza global en la respuesta (0.0–1.0) */
  confidence: number;
}

// ==================================================================
// Clase BrocaArea
// ==================================================================

/**
 * Área de Broca — Producción lingüística del cerebro digital.
 *
 * Recibe intenciones semánticas de la corteza prefrontal, selecciona
 * palabras del léxico compartido por similitud coseno, y genera una
 * secuencia de patrones motores del habla modulada por el estado emocional.
 *
 * Base biológica:
 *   Las columnas corticales en el giro frontal inferior se especializan
 *   en la secuenciación de patrones motores complejos. La red SNN
 *   interna modela la competencia entre candidatos léxicos durante
 *   la producción (modelo de selección léxica por competencia,
 *   Levelt et al., 1999). La inhibición lateral asegura que solo
 *   una palabra se seleccione por slot temporal.
 */
export class BrocaArea extends BrainRegion {
  /**
   * Léxico compartido con el área de Wernicke.
   *
   * Biología: Broca accede al mismo almacén léxico que Wernicke pero
   * por la vía dorsal (fascículo arqueado): busca la forma léxica
   * a partir del significado (dirección inversa a la comprensión).
   */
  private readonly lexicon: Lexicon;

  /**
   * Última respuesta lingüística generada.
   */
  private lastResponse: LanguageResponse | null = null;

  /**
   * Número máximo de palabras por respuesta.
   *
   * Biología: La longitud de las emisiones verbales está limitada por
   * la capacidad de planificación articulatoria y la memoria de trabajo
   * fonológica (bucle fonológico de Baddeley, ~2 segundos de habla).
   */
  private readonly maxResponseLength: number = 10;

  /**
   * Crea el Área de Broca del cerebro digital.
   *
   * Biología: El área de Broca contiene ~100 millones de neuronas en
   * las áreas 44/45. Modelamos 5.000 neuronas con 5.000 entradas,
   * reflejando la estructura columnar del giro frontal inferior.
   *
   * @param lexicon - Léxico compartido (misma instancia que Wernicke)
   */
  constructor(lexicon: Lexicon, neuronCount: number = 5000, inputCount: number = 5000) {
    super(
      'broca',
      'Área de Broca — Producción del Lenguaje',
      neuronCount,
      inputCount
    );
    this.lexicon = lexicon;
  }

  // ----------------------------------------------------------------
  // Selección Léxica
  // ----------------------------------------------------------------

  /**
   * Selecciona las K palabras más relevantes para expresar un patrón semántico.
   *
   * Base biológica:
   *   Modela el acceso léxico para producción (lematización): dado un
   *   concepto semántico, se busca la forma léxica que mejor lo expresa.
   *   La selección ocurre por competencia: múltiples lemas se activan
   *   parcialmente y compiten por selección (modelo WEAVER++, Levelt
   *   et al., 1999). Las palabras con mayor similitud coseno al
   *   patrón de intención ganan la competencia.
   *
   * @param semanticPattern - Patrón semántico de la intención comunicativa
   * @param topK - Número de palabras a seleccionar
   * @returns Palabras seleccionadas ordenadas por relevancia descendente
   */
  selectWords(semanticPattern: Float32Array, topK: number = 5): SelectedWord[] {
    if (this.lexicon.size === 0) return [];

    const matches: LexiconMatch[] = this.lexicon.findClosest(semanticPattern, topK);

    return matches.map(m => {
      const pattern = this.lexicon.lookup(m.word);
      return {
        word: m.word,
        relevance: m.similarity,
        pattern: pattern ? new Float32Array(pattern) : new Float32Array(this.lexicon.dimensions),
      };
    });
  }

  // ----------------------------------------------------------------
  // Generación de Respuesta
  // ----------------------------------------------------------------

  /**
   * Genera una respuesta lingüística a partir de una intención semántica
   * y un estado emocional.
   *
   * Base biológica:
   *   La producción del habla en Broca sigue un pipeline de 3 etapas
   *   (modelo de Levelt, 1989):
   *
   *   1. **Conceptualización** → recibida como intención de la PFC
   *   2. **Formulación** → selección léxica + codificación gramatical
   *   3. **Articulación** → generación de patrones motores
   *
   *   El estado emocional modula la formulación:
   *   - Arousal alto → respuestas más cortas (la ansiedad reduce la
   *     fluidez verbal, efecto Yerkes-Dodson en producción)
   *   - Valencia negativa → sesgo hacia palabras de alta frecuencia
   *     (vocabulario "seguro" bajo estrés)
   *   - Valencia positiva → mayor diversidad léxica
   *
   * @param intention - Patrón semántico de la intención comunicativa (de la PFC)
   * @param emotionalState - Estado emocional actual (de la amígdala)
   * @returns Respuesta lingüística con secuencia de patrones y palabras
   */
  generateResponse(
    intention: Float32Array,
    emotionalState: EmotionalState
  ): LanguageResponse {
    // 1. Determinar longitud de la respuesta según arousal
    //    Alto arousal → respuesta más corta (efecto del estrés sobre fluidez)
    const arousalFactor = 1.0 - emotionalState.arousal * 0.6;
    const responseLength = Math.max(
      1,
      Math.floor(this.maxResponseLength * arousalFactor)
    );

    // 2. Determinar cantidad de candidatos a evaluar según valencia
    //    Valencia positiva → más candidatos → mayor diversidad
    const diversityFactor = 1.0 + Math.max(0, emotionalState.valence) * 0.5;
    const candidateCount = Math.ceil(responseLength * diversityFactor * 2);

    // 3. Seleccionar candidatos léxicos
    const candidates = this.selectWords(intention, candidateCount);

    if (candidates.length === 0) {
      return { wordPatterns: [], words: [], confidence: 0 };
    }

    // 4. Construir secuencia de respuesta
    const wordPatterns: Float32Array[] = [];
    const words: string[] = [];
    let totalRelevance = 0;

    // Seleccionar las mejores palabras sin repetir
    const used = new Set<string>();

    for (const candidate of candidates) {
      if (words.length >= responseLength) break;
      if (used.has(candidate.word)) continue;

      // Bajo valencia negativa, favorecer palabras de alta relevancia
      // (vocabulario más conservador bajo estrés)
      const valencePenalty = emotionalState.valence < 0
        ? (1.0 - candidate.relevance) * Math.abs(emotionalState.valence) * 0.3
        : 0;

      if (candidate.relevance - valencePenalty > 0.05) {
        words.push(candidate.word);
        wordPatterns.push(candidate.pattern);
        totalRelevance += candidate.relevance;
        used.add(candidate.word);
      }
    }

    const confidence = words.length > 0 ? totalRelevance / words.length : 0;

    const response: LanguageResponse = {
      wordPatterns,
      words,
      confidence,
    };

    this.lastResponse = response;
    return response;
  }

  /**
   * Retorna la última respuesta lingüística generada.
   */
  getLastResponse(): LanguageResponse | null {
    return this.lastResponse;
  }

  /**
   * Retorna una referencia al léxico compartido.
   */
  getLexicon(): Lexicon {
    return this.lexicon;
  }

  // ----------------------------------------------------------------
  // processInput — Ciclo de producción lingüística
  // ----------------------------------------------------------------

  /**
   * Procesa un paso de simulación del área de Broca.
   *
   * Base biológica:
   *   El ciclo de producción de Broca opera en ~150ms y consiste en:
   *
   *   1. **Recepción de intención**: Los spikes de entrada representan
   *      la intención comunicativa de la corteza prefrontal, transmitida
   *      vía fibras fronto-frontales.
   *
   *   2. **Procesamiento SNN**: La red neural local procesa la intención
   *      mediante competencia k-WTA, seleccionando las representaciones
   *      motoras más relevantes.
   *
   *   3. **Formulación**: Se seleccionan las palabras del léxico que
   *      mejor capturan la intención (acceso léxico para producción).
   *
   *   4. **Generación**: Se produce una secuencia de patrones motores
   *      del habla que se envían a la corteza motora primaria para
   *      la ejecución articulatoria.
   *
   *   El estado emocional se estima a partir de la actividad de spikes
   *   en la ausencia de un input directo de la amígdala.
   *
   * @param spikes - Vector de spikes de entrada (intención de la PFC)
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Vector de spikes de salida (patrones motores del habla)
   */
  processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects
  ): Float32Array {
    // 1. Computar activaciones de la SNN local
    const activations = new Float32Array(this.neuronCount);
    const inputLen = Math.min(spikes.length, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      const baseOffset = n * this.inputCount;
      let sum = 0;

      for (let j = 0; j < inputLen; j++) {
        sum += this.weights[baseOffset + j] * spikes[j];
      }

      // Modulación: ganancia de spikes y atención
      sum *= modulationEffects.spikeGainMultiplier;

      // Integración LIF simplificada
      this.potentials[n] += sum;
      this.potentials[n] *= 0.92; // Leak temporal (ligeramente más rápido que Wernicke)

      activations[n] = this.potentials[n];
    }

    // 2. Competencia k-WTA
    const k = Math.max(1, Math.floor(this.neuronCount * this.sparsity));
    const outputSpikes = new Float32Array(this.neuronCount);

    const sortedActivations = new Float32Array(activations);
    sortedActivations.sort();
    const kwtaThreshold = sortedActivations[this.neuronCount - k];

    for (let i = 0; i < this.neuronCount; i++) {
      if (activations[i] >= kwtaThreshold && activations[i] > ACTIVATION_FLOOR) {
        outputSpikes[i] = 1.0;
      }
    }

    // 3. Actualizar estado de spikes
    this.spikes.set(outputSpikes);

    // 4. Generar respuesta lingüística a partir del spike pattern
    //    Estimar estado emocional desde la neuromodulación:
    //    - Serotonina alta → valencia positiva; cortisol alto → arousal alto
    const estimatedEmotionalState: EmotionalState = {
      valence: (modulationEffects.spikeGainMultiplier - 1.0) * 0.5,
      arousal: Math.max(0, Math.min(1, (modulationEffects.attentionGain - 1.0) * 0.8)),
    };

    this.generateResponse(spikes, estimatedEmotionalState);

    // 5. Aprendizaje Hebbiano modulado
    if (modulationEffects.learningRateMultiplier > 0) {
      this.hebbianUpdate(spikes, outputSpikes);
    }

    return outputSpikes;
  }

  /**
   * Actualización Hebbiana de pesos sinápticos para producción.
   *
   * Base biológica:
   *   Las sinapsis en el giro frontal inferior se fortalecen cuando
   *   un patrón de intención (pre) activa exitosamente un patrón
   *   motor (post). Esto refina la asociación entre significado y
   *   forma léxica, mejorando la fluidez de producción con la práctica.
   *
   * @param preSpikes - Spikes de entrada (intención prefrontal)
   * @param postSpikes - Spikes de salida (patrones motores)
   */
  private hebbianUpdate(preSpikes: Float32Array, postSpikes: Float32Array): void {
    const lr = this._modulatedLearningRate * 0.005;
    const inputLen = Math.min(preSpikes.length, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      if (postSpikes[n] === 0) continue;

      const baseOffset = n * this.inputCount;
      for (let j = 0; j < inputLen; j++) {
        if (preSpikes[j] > 0) {
          this.weights[baseOffset + j] += lr * preSpikes[j];
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
