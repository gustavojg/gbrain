/**
 * ÁREA DE WERNICKE — Comprensión del Lenguaje
 * =============================================
 * Modela el área de Wernicke (parte posterior del giro temporal superior,
 * área de Brodmann 22) del cerebro digital, responsable de la comprensión
 * del lenguaje hablado y escrito.
 *
 * Base biológica:
 *   El área de Wernicke fue identificada por Carl Wernicke en 1874 al
 *   observar que lesiones en la parte posterior del giro temporal superior
 *   causaban "afasia receptiva": los pacientes podían hablar fluidamente
 *   pero no comprendían el lenguaje. Sus emisiones verbales eran
 *   gramaticalmente correctas pero semánticamente vacías ("ensalada de
 *   palabras").
 *
 *   Funciones modeladas:
 *   1. **Decodificación auditiva/visual → léxica**: Mapea patrones de
 *      activación (procedentes del tálamo) a entradas en el léxico mental.
 *      Modelo de acceso léxico por cohorte (Marslen-Wilson, 1987).
 *
 *   2. **Comprensión semántica**: Extrae significado de secuencias de
 *      patrones léxicos activados. Las neuronas de Wernicke responden
 *      selectivamente a combinaciones semánticas (no solo palabras
 *      individuales).
 *
 *   3. **Aprendizaje léxico**: Nuevas palabras se incorporan al léxico
 *      compartido, creando nuevos engramas neurales en el lóbulo temporal.
 *
 *   Conexiones:
 *   - Recibe: spikes del tálamo (señales auditivas/visuales procesadas)
 *   - Envía: representaciones semánticas a la corteza prefrontal y al
 *     área de Broca (vía fascículo arqueado)
 *   - Comparte: léxico con el área de Broca
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { Lexicon, type LexiconMatch } from './lexicon.js';

/**
 * Piso de activación para el k-WTA. Los "potenciales" aquí son sumas de
 * pesos adimensionales (~0 en reposo), no mV de membrana. Sin este piso el
 * gate compararía contra `_modulatedThreshold` (−55 mV), siempre pasaría y
 * el k-WTA encendería exactamente k neuronas por tick → actividad constante
 * y falsa. Con el piso, en reposo (sin drive) la región queda a 0%.
 */
const ACTIVATION_FLOOR = 1e-3;

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Resultado de la comprensión de un patrón de spikes.
 *
 * Base biológica:
 *   Cada resultado representa una activación léxica candidata,
 *   análoga a la cohorte de candidatos léxicos que se activan
 *   parcialmente al procesar una señal de habla (modelo de cohorte).
 */
export interface ComprehensionResult {
  /** Palabra identificada en el léxico */
  word: string;
  /** Grado de similitud con el patrón de entrada (0.0–1.0) */
  similarity: number;
}

/**
 * Representación semántica producida por el procesamiento de Wernicke.
 */
export interface SemanticRepresentation {
  /** Patrón neural de la representación semántica */
  pattern: Float32Array;
  /** Palabras comprendidas y sus similitudes */
  comprehendedWords: ComprehensionResult[];
  /** Confianza global de la comprensión (0.0–1.0) */
  confidence: number;
}

// ==================================================================
// Clase WernickeArea
// ==================================================================

/**
 * Área de Wernicke — Comprensión lingüística del cerebro digital.
 *
 * Procesa patrones de spikes recibidos del tálamo (señales auditivas/
 * visuales preprocesadas) y los mapea a representaciones semánticas
 * mediante acceso léxico por similitud. Comparte un léxico neural
 * con el área de Broca.
 *
 * Base biológica:
 *   Las neuronas de Wernicke se organizan tonotópicamente y responden
 *   selectivamente a categorías fonéticas y combinaciones semánticas.
 *   La comprensión ocurre en ~200ms (componente N400 del ERP para
 *   violaciones semánticas). La red SNN interna modela las columnas
 *   corticales del giro temporal superior con competencia lateral.
 */
export class WernickeArea extends BrainRegion {
  /**
   * Léxico compartido con el área de Broca.
   *
   * Biología: El léxico mental es un almacén compartido en el lóbulo
   * temporal, accedido por la vía ventral (Wernicke → comprensión) y
   * la vía dorsal (Broca → producción).
   */
  private readonly lexicon: Lexicon;

  /**
   * Última representación semántica producida.
   * Se mantiene para que otras regiones puedan consultarla.
   */
  private lastSemanticOutput: SemanticRepresentation | null = null;

  /**
   * Umbral mínimo de similitud para considerar una coincidencia léxica válida.
   *
   * Biología: Modela el umbral de activación léxica. Patrones que no
   * superan este umbral no activan ningún engrama léxico (palabra
   * desconocida o ruido).
   */
  private readonly comprehensionThreshold: number = 0.15;

  /**
   * Crea el Área de Wernicke del cerebro digital.
   *
   * Biología: El área de Wernicke contiene ~150 millones de neuronas.
   * Modelamos 5.000 neuronas con 5.000 entradas, reflejando la estructura
   * columnar del giro temporal superior.
   *
   * @param lexicon - Léxico compartido (inyectado para compartir con Broca)
   */
  constructor(lexicon: Lexicon, neuronCount: number = 5000, inputCount: number = 5000) {
    super(
      'wernicke',
      'Área de Wernicke — Comprensión Lingüística',
      neuronCount,
      inputCount
    );
    this.lexicon = lexicon;
  }

  // ----------------------------------------------------------------
  // Comprensión Lingüística
  // ----------------------------------------------------------------

  /**
   * Comprende un patrón de spikes mapeándolo a las entradas léxicas más cercanas.
   *
   * Base biológica:
   *   Modela el acceso léxico auditivo/visual. Al recibir un patrón de
   *   activación (ej: representación acústica procesada por la corteza
   *   auditiva), Wernicke lo compara con todos los engramas léxicos
   *   almacenados. Los más similares se activan formando una "cohorte"
   *   de candidatos (modelo de cohorte, Marslen-Wilson 1987). La
   *   competencia lateral resuelve la ambigüedad seleccionando la
   *   coincidencia más fuerte.
   *
   *   El efecto de frecuencia léxica está integrado en el método
   *   findClosest() del léxico (palabras frecuentes tienen ventaja).
   *
   * @param inputSpikes - Patrón de spikes a comprender
   * @returns Array de palabras candidatas con su similitud, ordenadas descendentemente
   */
  comprehend(inputSpikes: Float32Array): ComprehensionResult[] {
    if (this.lexicon.size === 0) return [];

    // Buscar las coincidencias más cercanas en el léxico
    const matches: LexiconMatch[] = this.lexicon.findClosest(inputSpikes, 5);

    // Filtrar por umbral de comprensión
    return matches
      .filter(m => m.similarity >= this.comprehensionThreshold)
      .map(m => ({
        word: m.word,
        similarity: m.similarity,
      }));
  }

  /**
   * Aprende una nueva palabra agregándola al léxico compartido.
   *
   * Base biológica:
   *   El aprendizaje léxico ocurre cuando un patrón de activación nuevo
   *   (sin coincidencia suficiente en el léxico existente) se presenta
   *   repetidamente y se asocia con un significado. En el cerebro, esto
   *   involucra la formación de un nuevo engrama neural en el lóbulo
   *   temporal, inicialmente dependiente del hipocampo y gradualmente
   *   consolidado en neocorteza (complementary learning systems theory,
   *   McClelland et al., 1995).
   *
   * @param word - Palabra a aprender
   * @param pattern - Patrón de activación neural asociado
   */
  learnWord(word: string, pattern: Float32Array): void {
    this.lexicon.add(word, pattern);
  }

  /**
   * Retorna la última representación semántica producida.
   *
   * @returns Representación semántica o null si no se ha procesado nada
   */
  getLastSemanticOutput(): SemanticRepresentation | null {
    return this.lastSemanticOutput;
  }

  /**
   * Retorna una referencia de solo lectura al léxico compartido.
   */
  getLexicon(): Lexicon {
    return this.lexicon;
  }

  // ----------------------------------------------------------------
  // processInput — Ciclo de comprensión lingüística
  // ----------------------------------------------------------------

  /**
   * Procesa un paso de simulación del área de Wernicke.
   *
   * Base biológica:
   *   El ciclo de comprensión de Wernicke opera en ~200ms y consiste en:
   *
   *   1. **Recepción**: Recibir spikes del tálamo (señales auditivas/
   *      visuales preprocesadas por cortezas sensoriales primarias)
   *
   *   2. **Acceso léxico**: Mapear el patrón de entrada a candidatos
   *      léxicos mediante comparación por similitud (cohorte)
   *
   *   3. **Procesamiento SNN**: Pasar los spikes por la red neural
   *      local para producir una representación semántica distribuida.
   *      La competencia k-WTA en la SNN modela la selección del
   *      candidato léxico ganador.
   *
   *   4. **Integración semántica**: Combinar la entrada con las
   *      coincidencias léxicas para producir una representación semántica
   *      unificada que se envía a la corteza prefrontal y al área de Broca.
   *
   * @param spikes - Vector de spikes de entrada del tálamo
   * @param modulationEffects - Efectos de neuromodulación actuales
   * @returns Vector de spikes de salida (representación semántica)
   */
  processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects
  ): Float32Array {
    // 1. Acceso léxico: comprender el patrón de entrada
    const comprehended = this.comprehend(spikes);

    // 2. Computar activaciones de la SNN local
    const activations = new Float32Array(this.neuronCount);
    const inputLen = Math.min(spikes.length, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      const baseOffset = n * this.inputCount;
      let sum = 0;

      for (let j = 0; j < inputLen; j++) {
        sum += this.weights[baseOffset + j] * spikes[j];
      }

      // Modulación atencional: la acetilcolina amplifica señales lingüísticas
      sum *= modulationEffects.attentionGain;

      // Integración LIF simplificada
      this.potentials[n] += sum;
      this.potentials[n] *= 0.93; // Leak temporal

      activations[n] = this.potentials[n];
    }

    // 3. Competencia k-WTA: selección de representación ganadora
    const k = Math.max(1, Math.floor(this.neuronCount * this.sparsity));
    const outputSpikes = new Float32Array(this.neuronCount);

    // Encontrar el k-ésimo umbral
    const sortedActivations = new Float32Array(activations);
    sortedActivations.sort();
    const kwtaThreshold = sortedActivations[this.neuronCount - k];

    for (let i = 0; i < this.neuronCount; i++) {
      if (activations[i] >= kwtaThreshold && activations[i] > ACTIVATION_FLOOR) {
        outputSpikes[i] = 1.0;
      }
    }

    // 4. Actualizar estado de spikes
    this.spikes.set(outputSpikes);

    // 5. Construir representación semántica
    let confidence = 0;
    if (comprehended.length > 0) {
      // Confianza basada en la mejor coincidencia léxica
      confidence = comprehended[0].similarity;
    }

    this.lastSemanticOutput = {
      pattern: new Float32Array(outputSpikes),
      comprehendedWords: comprehended,
      confidence,
    };

    // 6. Aprendizaje Hebbiano: reforzar las conexiones que contribuyeron
    //    a la comprensión exitosa
    if (confidence > 0.3 && modulationEffects.learningRateMultiplier > 0) {
      this.hebbianUpdate(spikes, outputSpikes);
    }

    return outputSpikes;
  }

  /**
   * Actualización Hebbiana de pesos sinápticos para aprendizaje léxico.
   *
   * Base biológica:
   *   Las sinapsis en el giro temporal superior se fortalecen cuando
   *   hay correlación entre el patrón de entrada (señal acústica/visual)
   *   y la activación de neuronas que representan la palabra reconocida.
   *   Esto refina progresivamente la representación léxica (tuning).
   *
   * @param preSpikes - Spikes de entrada (presinápticos)
   * @param postSpikes - Spikes de salida (postsinápticos)
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

        // Clamp de pesos
        if (this.weights[baseOffset + j] > 1.0) {
          this.weights[baseOffset + j] = 1.0;
        } else if (this.weights[baseOffset + j] < 0) {
          this.weights[baseOffset + j] = 0;
        }
      }
    }
  }
}
