/**
 * DECODIFICADOR DE TEXTO — Spikes a palabras
 * ============================================
 * Convierte los trenes de spikes del área de Broca
 * en texto legible (secuencia de palabras).
 * 
 * Biología: El área de Broca produce patrones de activación motora
 * que controlan la articulación del habla. Aquí, "articulación"
 * se traduce en selección de tokens del lexicón.
 */

/**
 * Decodifica un patrón de spikes cortical a texto.
 */
export class TextDecoder {
  /** Umbral mínimo de similitud para considerar un match */
  private matchThreshold: number;

  constructor(matchThreshold: number = 0.3) {
    this.matchThreshold = matchThreshold;
  }

  /**
   * Decodifica spikes de Broca a secuencia de palabras.
   * 
   * @param outputSpikes - Patrón de activación de Broca
   * @param lexicon - Vocabulario disponible (palabra → patrón)
   * @param topK - Número máximo de tokens a producir
   * @returns Texto generado
   */
  decode(
    outputSpikes: Float32Array,
    lexicon: Map<string, Float32Array>,
    topK: number = 10
  ): string {
    const matches: { word: string; similarity: number }[] = [];

    for (const [word, pattern] of lexicon) {
      const similarity = this.cosineSimilarity(outputSpikes, pattern);
      if (similarity > this.matchThreshold) {
        matches.push({ word, similarity });
      }
    }

    // Ordenar por similitud descendente
    matches.sort((a, b) => b.similarity - a.similarity);

    // Tomar los top K
    const selectedWords = matches.slice(0, topK).map(m => m.word);

    return selectedWords.join(' ');
  }

  /**
   * Decodifica múltiples patrones secuenciales a texto.
   */
  decodeSequence(
    spikeSequence: Float32Array[],
    lexicon: Map<string, Float32Array>
  ): string {
    const words: string[] = [];

    for (const spikes of spikeSequence) {
      const word = this.decodeToken(spikes, lexicon);
      if (word) words.push(word);
    }

    return words.join(' ');
  }

  /**
   * Decodifica un solo patrón de spikes al token más probable.
   */
  private decodeToken(
    spikes: Float32Array,
    lexicon: Map<string, Float32Array>
  ): string | null {
    let bestWord: string | null = null;
    let bestSim = this.matchThreshold;

    for (const [word, pattern] of lexicon) {
      const sim = this.cosineSimilarity(spikes, pattern);
      if (sim > bestSim) {
        bestSim = sim;
        bestWord = word;
      }
    }

    return bestWord;
  }

  /**
   * Similitud coseno entre dos vectores.
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < len; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator > 0 ? dotProduct / denominator : 0;
  }
}
