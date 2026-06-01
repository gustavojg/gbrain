/**
 * TEXT DECODER — Spikes to words
 * ============================================
 * Converts the spike trains from Broca's area
 * into readable text (sequence of words).
 *
 * Biology: Broca's area produces motor activation patterns
 * that control speech articulation. Here, "articulation"
 * translates into selecting tokens from the lexicon.
 */

/**
 * Decodes a cortical spike pattern into text.
 */
export class TextDecoder {
  /** Minimum similarity threshold to consider a match */
  private matchThreshold: number;

  constructor(matchThreshold: number = 0.3) {
    this.matchThreshold = matchThreshold;
  }

  /**
   * Decodes Broca spikes into a sequence of words.
   *
   * @param outputSpikes - Activation pattern of Broca
   * @param lexicon - Available vocabulary (word → pattern)
   * @param topK - Maximum number of tokens to produce
   * @returns Generated text
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

    // Sort by descending similarity
    matches.sort((a, b) => b.similarity - a.similarity);

    // Take the top K
    const selectedWords = matches.slice(0, topK).map(m => m.word);

    return selectedWords.join(' ');
  }

  /**
   * Decodes multiple sequential patterns into text.
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
   * Decodes a single spike pattern into the most probable token.
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
   * Cosine similarity between two vectors.
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
