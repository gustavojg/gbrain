/**
 * SHARED LEXICON — Distributed neural vocabulary
 * ===================================================
 * Vocabulary store shared between the Wernicke (comprehension)
 * and Broca (production) areas of the digital brain.
 *
 * Biological basis:
 *   The human mental lexicon stores ~60,000 words as neural activation
 *   patterns distributed in the temporal lobe. Each word is represented as
 *   a sparse activation vector over a neuronal population (distributed
 *   coding, Hebb 1949). Lexical retrieval occurs by pattern
 *   similarity: upon hearing a partial word, the closest pattern is activated
 *   (phonological/semantic priming).
 *
 *   The lexicon is shared between the comprehension and production areas, although
 *   in real biology these areas access the lexicon via partially
 *   distinct pathways (ventral pathway for Wernicke, dorsal pathway for Broca).
 *
 * Implementation:
 *   Each entry contains:
 *   - The neural activation pattern (sparse Float32Array)
 *   - Usage frequency (for retrieval biases — lexical frequency effect)
 *   - Timestamp of last use (for recency effect)
 *
 *   The similarity search uses cosine, optimized with precomputed norms.
 */

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Lexical entry: associates a word with its distributed neural pattern.
 */
export interface LexiconEntry {
  /** Distributed neural activation pattern (sparse Float32Array) */
  pattern: Float32Array;
  /**
   * Cumulative usage frequency. More frequent words are retrieved faster
   * (lexical frequency effect, Oldfield & Wingfield 1965).
   */
  frequency: number;
  /** Timestamp of last access (ms). Biases retrieval by recency. */
  lastUsed: number;
}

/**
 * Result of a similarity search in the lexicon.
 */
export interface LexiconMatch {
  /** Word found */
  word: string;
  /** Cosine similarity with the query pattern (0–1) */
  similarity: number;
}

/**
 * Serialization format of the complete lexicon.
 */
export interface SerializedLexicon {
  /** Serialized entries: word → { pattern as number[], frequency, lastUsed } */
  entries: Array<{
    word: string;
    pattern: number[];
    frequency: number;
    lastUsed: number;
  }>;
  /** Dimensionality of the patterns */
  patternSize: number;
}

// ==================================================================
// Lexicon class
// ==================================================================

/**
 * Neural lexicon shared between the language areas.
 *
 * Biological basis:
 *   Models the lexical store of the temporal lobe where each concept/word
 *   is represented as an activation pattern distributed over a neuronal
 *   population. Retrieval occurs by competition: the stored pattern
 *   most similar to the input wins (interactive activation model of
 *   lexical access, McClelland & Rumelhart 1981).
 *
 *   Implemented features:
 *   - Lexical frequency effect: frequent words have a lower activation
 *     threshold (they are retrieved faster)
 *   - Recency effect: recently used words have residual priming
 *   - Lexical competition: when searching, the K best matches are returned
 */
export class Lexicon {
  /** Main store: word → lexical entry */
  private readonly entries: Map<string, LexiconEntry> = new Map();

  /**
   * Cache of L2 norms to accelerate cosine similarity computations.
   * Invalidated when an entry is modified.
   */
  private readonly normCache: Map<string, number> = new Map();

  /** Dimensionality of the neural patterns */
  private readonly patternSize: number;

  /**
   * Creates a new neural lexicon.
   *
   * @param patternSize - Dimensionality of the neural patterns (default: 5000)
   */
  constructor(patternSize: number = 5000) {
    this.patternSize = patternSize;
  }

  // ----------------------------------------------------------------
  // CRUD operations
  // ----------------------------------------------------------------

  /**
   * Adds or updates a word in the lexicon.
   *
   * Biological basis:
   *   Models lexical learning: the first exposure to a word
   *   creates a new neural engram. Repeated exposures strengthen
   *   the frequency (repetition effect) and update the pattern if
   *   the context generates a slightly different representation
   *   (engram refinement).
   *
   * @param word - Word to add (normalized to lowercase)
   * @param pattern - Associated neural activation pattern
   */
  add(word: string, pattern: Float32Array): void {
    const key = word.toLowerCase();
    const existing = this.entries.get(key);

    if (existing) {
      // Update: blend existing pattern with new one (weighted average)
      // Biology: engram refinement through repeated exposure
      const alpha = 0.3; // Update rate
      for (let i = 0; i < this.patternSize; i++) {
        existing.pattern[i] = existing.pattern[i] * (1 - alpha) + pattern[i] * alpha;
      }
      existing.frequency++;
      existing.lastUsed = Date.now();
    } else {
      // New lexical engram
      this.entries.set(key, {
        pattern: new Float32Array(pattern),
        frequency: 1,
        lastUsed: Date.now(),
      });
    }

    // Invalidate norm cache
    this.normCache.delete(key);
  }

  /**
   * Looks up a word in the lexicon and returns its neural pattern.
   *
   * Biological basis:
   *   Direct lexical access (by known orthographic/phonological form).
   *   Updates recency to model priming.
   *
   * @param word - Word to look up
   * @returns Neural pattern if it exists, null if it is not in the lexicon
   */
  lookup(word: string): Float32Array | null {
    const entry = this.entries.get(word.toLowerCase());
    if (!entry) return null;

    entry.lastUsed = Date.now();
    return entry.pattern;
  }

  /**
   * Finds the K words most similar to a given neural pattern.
   *
   * Biological basis:
   *   Models lexical access by activation: the neural input (e.g.: partial
   *   auditory pattern) is compared against all the stored lexical
   *   engrams. The most similar engrams are activated first
   *   (cohort model, Marslen-Wilson 1987). Lexical frequency
   *   biases the competition in favor of more common words.
   *
   * @param pattern - Query neural pattern
   * @param topK - Number of matches to return (default: 5)
   * @returns Array of matches ordered by descending similarity
   */
  findClosest(pattern: Float32Array, topK: number = 5): LexiconMatch[] {
    const queryNorm = this.computeNorm(pattern);
    if (queryNorm === 0) return [];

    const matches: LexiconMatch[] = [];

    for (const [word, entry] of this.entries) {
      const entryNorm = this.getCachedNorm(word, entry.pattern);
      if (entryNorm === 0) continue;

      // Base cosine similarity
      let similarity = this.dotProduct(pattern, entry.pattern) / (queryNorm * entryNorm);

      // Lexical frequency bias (subtle effect, ~5% max boost)
      // Biology: frequent words have lower thresholds
      const freqBoost = Math.min(0.05, Math.log1p(entry.frequency) * 0.01);
      similarity = Math.min(1.0, similarity + freqBoost);

      matches.push({ word, similarity });
    }

    // Sort by descending similarity and return the top K
    matches.sort((a, b) => b.similarity - a.similarity);
    return matches.slice(0, topK);
  }

  // ----------------------------------------------------------------
  // Math utilities
  // ----------------------------------------------------------------

  /**
   * Computes the cosine similarity between two neural patterns.
   *
   * Biological basis:
   *   Cosine similarity captures the similarity of the activation
   *   "direction" of two neural populations, independent of magnitude.
   *   This models how the brain compares distributed activation
   *   patterns (analogous to neural population correlation).
   *
   * @param a - First neural pattern
   * @param b - Second neural pattern
   * @returns Cosine similarity in range [-1, 1]
   */
  cosineSimilarity(a: Float32Array, b: Float32Array): number {
    const normA = this.computeNorm(a);
    const normB = this.computeNorm(b);

    if (normA === 0 || normB === 0) return 0;

    return this.dotProduct(a, b) / (normA * normB);
  }

  /**
   * Dot product optimized for Float32Array.
   *
   * @param a - First vector
   * @param b - Second vector
   * @returns Scalar product
   */
  private dotProduct(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let sum = 0;
    // Manual loop unrolling for better performance (4-way unroll)
    const limit = len - (len % 4);
    let i = 0;

    for (; i < limit; i += 4) {
      sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
    }
    for (; i < len; i++) {
      sum += a[i] * b[i];
    }

    return sum;
  }

  /**
   * Computes the L2 norm of a vector.
   *
   * @param v - Vector
   * @returns ||v||₂
   */
  private computeNorm(v: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
      sum += v[i] * v[i];
    }
    return Math.sqrt(sum);
  }

  /**
   * Gets the cached L2 norm of a lexical entry.
   * Avoids recomputing the norm on every search.
   */
  private getCachedNorm(word: string, pattern: Float32Array): number {
    let norm = this.normCache.get(word);
    if (norm === undefined) {
      norm = this.computeNorm(pattern);
      this.normCache.set(word, norm);
    }
    return norm;
  }

  // ----------------------------------------------------------------
  // Properties and utilities
  // ----------------------------------------------------------------

  /**
   * Total number of words in the lexicon.
   */
  get size(): number {
    return this.entries.size;
  }

  /**
   * Dimensionality of the neural patterns.
   */
  get dimensions(): number {
    return this.patternSize;
  }

  /**
   * Returns all the words in the lexicon.
   */
  getWords(): string[] {
    return Array.from(this.entries.keys());
  }

  /**
   * Checks whether a word exists in the lexicon.
   */
  has(word: string): boolean {
    return this.entries.has(word.toLowerCase());
  }

  // ----------------------------------------------------------------
  // Serialization / Deserialization
  // ----------------------------------------------------------------

  /**
   * Serializes the complete lexicon for persistence.
   *
   * @returns Serializable object with all the lexical entries
   */
  serialize(): SerializedLexicon {
    const serializedEntries: SerializedLexicon['entries'] = [];

    for (const [word, entry] of this.entries) {
      serializedEntries.push({
        word,
        pattern: Array.from(entry.pattern),
        frequency: entry.frequency,
        lastUsed: entry.lastUsed,
      });
    }

    return {
      entries: serializedEntries,
      patternSize: this.patternSize,
    };
  }

  /**
   * Restores the lexicon from serialized data.
   *
   * @param data - Data previously serialized with serialize()
   */
  deserialize(data: SerializedLexicon): void {
    this.entries.clear();
    this.normCache.clear();

    for (const entry of data.entries) {
      this.entries.set(entry.word, {
        pattern: new Float32Array(entry.pattern),
        frequency: entry.frequency,
        lastUsed: entry.lastUsed,
      });
    }
  }
}
