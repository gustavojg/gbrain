/**
 * QUESTION MEMORY — what a question asks for
 * ===========================================================================
 * "¿De qué color es?" is not understood by its grammar. A child learns it as
 * a signal that goes with a kind of answer: the adult asks it and answers
 * it — "¿de qué color es? azul" — many times, and the question comes to
 * point at a DIMENSION (colour, shape, sound). Once it does, hearing it turns
 * attention to that dimension of what is in front (feature-based attention;
 * Treue & Martínez-Trujillo 1999), and the answer is what that dimension of
 * the thing brings to mind.
 *
 * This memory counts, for each question (a lexical percept), which modality
 * the answer that followed it belonged to.
 */

export interface QuestionTarget {
  modality: string;
  probability: number;
  observations: number;
}

/** Fewest question–answer pairs before a question points anywhere. */
const MIN_OBSERVATIONS = 2;
const MIN_PROBABILITY = 0.6;

export class QuestionMemory {
  /** question key → modality → count */
  private counts = new Map<string, Map<string, number>>();

  /** A question was followed by an answer word bound to `modality`. */
  observe(question: string, modality: string): void {
    let byModality = this.counts.get(question);
    if (!byModality) {
      byModality = new Map();
      this.counts.set(question, byModality);
    }
    byModality.set(modality, (byModality.get(modality) ?? 0) + 1);
  }

  /** The dimension a question points at, if it has been answered often enough in one way. */
  refersTo(question: string): QuestionTarget | null {
    const byModality = this.counts.get(question);
    if (!byModality) return null;
    let total = 0;
    let best: [string, number] | null = null;
    for (const entry of byModality) {
      total += entry[1];
      if (!best || entry[1] > best[1]) best = entry;
    }
    if (!best || total < MIN_OBSERVATIONS) return null;
    const probability = best[1] / total;
    return probability >= MIN_PROBABILITY ? { modality: best[0], probability, observations: total } : null;
  }

  get size(): number {
    return this.counts.size;
  }

  serialize(): unknown {
    return Array.from(this.counts.entries()).map(([q, byModality]) => [q, Array.from(byModality.entries())]);
  }

  deserialize(data: unknown): void {
    if (!Array.isArray(data)) return;
    this.counts = new Map();
    for (const entry of data.slice(0, 2000)) {
      if (!Array.isArray(entry) || entry.length !== 2 || typeof entry[0] !== 'string' || !Array.isArray(entry[1])) continue;
      const byModality = new Map<string, number>();
      for (const pair of entry[1].slice(0, 16)) {
        if (!Array.isArray(pair) || typeof pair[0] !== 'string' || !Number.isInteger(pair[1]) || pair[1] <= 0) continue;
        byModality.set(String(pair[0]).slice(0, 32), Math.min(1e6, pair[1] as number));
      }
      if (byModality.size > 0) this.counts.set(entry[0].slice(0, 80), byModality);
    }
  }
}
