/**
 * TEXT ENCODER — Linguistic encoding to spikes
 * =====================================================
 * Transforms text (user input) into spike trains
 * for Wernicke's area (linguistic comprehension).
 *
 * Pipeline:
 * 1. Tokenization by characters/words
 * 2. Positional encoding (position in the sequence)
 * 3. Sparse random projection (hash → distributed pattern)
 * 4. Rate coding to spikes
 *
 * The "random projection" approach is well grounded in computational
 * neuroscience: the insect olfactory cortex uses exactly this
 * scheme to encode odors (Caron et al., Nature 2013).
 */

import { encodeSpikeVector } from '../core/snn/spike-train.js';

/** Text encoder configuration */
export interface TextEncoderConfig {
  /** Output vector size (dimensionality of the representation) */
  vectorSize: number;
  /** Pattern sparsity per token (~5% of the vector active per token) */
  tokenSparsity: number;
  /** Maximum number of tokens to process per input */
  maxTokens: number;
  /** Whether to use positional encoding */
  positionalEncoding: boolean;
  /** Tokenization method */
  tokenization: 'character' | 'word' | 'subword';
}

const DEFAULT_TEXT_CONFIG: TextEncoderConfig = {
  vectorSize: 5000,
  tokenSparsity: 0.05,
  maxTokens: 128,
  positionalEncoding: true,
  tokenization: 'word',
};

/**
 * Text Encoder — Simulates early linguistic processing.
 * Converts text into distributed spike patterns.
 */
export class TextEncoder {
  private config: TextEncoderConfig;
  /** Output size */
  public outputSize: number;
  /** Cache of patterns per token (learned vocabulary) */
  private tokenCache: Map<string, Float32Array> = new Map();
  /** Base seed for deterministic hashing */
  private hashSeed: number;

  constructor(config: Partial<TextEncoderConfig> = {}) {
    this.config = { ...DEFAULT_TEXT_CONFIG, ...config };
    this.outputSize = this.config.vectorSize;
    this.hashSeed = 42; // Reproducible
  }

  /**
   * Encodes a full text as a spike vector.
   *
   * @param text - Text to encode
   * @param dt - Time step
   * @returns Spike vector
   */
  encode(text: string, dt: number = 1.0): Float32Array {
    // 1. Tokenize
    const tokens = this.tokenize(text);

    // 2. Generate distributed representation
    const representation = new Float32Array(this.config.vectorSize);

    for (let i = 0; i < Math.min(tokens.length, this.config.maxTokens); i++) {
      const token = tokens[i];

      // Get or generate a pattern for this token
      const tokenPattern = this.getTokenPattern(token);

      // Add positional encoding if enabled
      const positionWeight = this.config.positionalEncoding
        ? this.getPositionWeight(i, tokens.length)
        : 1.0;

      // Accumulate into the representation (superposition of patterns)
      for (let j = 0; j < this.config.vectorSize; j++) {
        representation[j] += tokenPattern[j] * positionWeight;
      }
    }

    // 3. Normalize to range 0-1
    this.normalize(representation);

    // 4. Convert to spikes
    return encodeSpikeVector(representation, dt, 120);
  }

  /**
   * Tokenizes the text according to the configured method.
   */
  private tokenize(text: string): string[] {
    const cleaned = text.toLowerCase().trim();
    
    switch (this.config.tokenization) {
      case 'character':
        return cleaned.split('');
      
      case 'word':
        // Word tokenization with punctuation preservation
        return cleaned.split(/\s+/).filter(t => t.length > 0);

      case 'subword':
        // Simple tokenization by syllables/bigrams
        return this.subwordTokenize(cleaned);
      
      default:
        return cleaned.split(/\s+/);
    }
  }

  /**
   * Subword tokenization (character bigrams).
   * A simple approximation to BPE without needing a pretrained vocabulary.
   */
  private subwordTokenize(text: string): string[] {
    const tokens: string[] = [];
    const words = text.split(/\s+/);
    
    for (const word of words) {
      if (word.length <= 3) {
        tokens.push(word);
      } else {
        // Overlapping bigrams
        for (let i = 0; i < word.length - 1; i++) {
          tokens.push(word.substring(i, i + 2));
        }
        // Also add the full word (for direct recognition)
        tokens.push(word);
      }
    }
    
    return tokens;
  }

  /**
   * Generates or retrieves from the cache the activation pattern for a token.
   * Uses deterministic hashing to generate a reproducible sparse pattern.
   *
   * Inspired by the sparse distributed representation (SDR) of the neocortex:
   * each concept is represented by a unique subset of active neurons.
   */
  private getTokenPattern(token: string): Float32Array {
    // Check cache
    if (this.tokenCache.has(token)) {
      return this.tokenCache.get(token)!;
    }

    const pattern = new Float32Array(this.config.vectorSize);
    const numActive = Math.floor(this.config.vectorSize * this.config.tokenSparsity);

    // Deterministic hash of the token → active positions
    let seed = this.hashString(token);

    for (let a = 0; a < numActive; a++) {
      seed = this.nextRandom(seed);
      const idx = Math.abs(seed) % this.config.vectorSize;

      // Activation with variation (not all positions equally active)
      seed = this.nextRandom(seed);
      const strength = 0.5 + 0.5 * (Math.abs(seed) % 100) / 100;

      pattern[idx] = Math.min(1.0, pattern[idx] + strength);
    }

    // Cache the pattern
    this.tokenCache.set(token, pattern);

    return pattern;
  }

  /**
   * Computes a positional weight for the token.
   * Tokens at the end of the input are slightly more relevant
   * (recency effect in working memory).
   */
  private getPositionWeight(position: number, totalTokens: number): number {
    // Base weight + recency (recent tokens carry more weight)
    const recency = 0.7 + 0.3 * (position / Math.max(1, totalTokens - 1));

    // Sinusoidal modulation (like in Transformers, but simplified)
    const freq = position / 10;
    const sinMod = 0.9 + 0.1 * Math.sin(freq);

    return recency * sinMod;
  }

  /**
   * Deterministic hash of a string to a number.
   */
  private hashString(str: string): number {
    let hash = this.hashSeed;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return hash;
  }

  /**
   * Deterministic pseudo-random generator (LCG).
   */
  private nextRandom(seed: number): number {
    return (seed * 1103515245 + 12345) & 0x7fffffff;
  }

  /**
   * Normalizes a vector to the range 0-1.
   */
  private normalize(vec: Float32Array): void {
    let max = 0;
    for (let i = 0; i < vec.length; i++) {
      if (vec[i] > max) max = vec[i];
    }
    if (max > 0) {
      for (let i = 0; i < vec.length; i++) {
        vec[i] /= max;
      }
    }
  }

  /** Size of the cached vocabulary */
  get vocabularySize(): number {
    return this.tokenCache.size;
  }

  /** Clears the token cache */
  reset(): void {
    this.tokenCache.clear();
  }
}
