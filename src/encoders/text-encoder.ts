/**
 * ENCODER DE TEXTO — Codificación lingüística a spikes
 * =====================================================
 * Transforma texto (input del usuario) en trenes de spikes
 * para el área de Wernicke (comprensión lingüística).
 * 
 * Pipeline:
 * 1. Tokenización por caracteres/palabras
 * 2. Codificación posicional (posición en la secuencia)
 * 3. Proyección aleatoria sparse (hash → patrón distribuido)
 * 4. Rate coding a spikes
 * 
 * El enfoque de "random projection" está bien fundamentado en neurociencia
 * computacional: la corteza olfatoria del insecto usa exactamente este
 * esquema para codificar olores (Caron et al., Nature 2013).
 */

import { encodeSpikeVector } from '../core/snn/spike-train.js';

/** Configuración del encoder de texto */
export interface TextEncoderConfig {
  /** Tamaño del vector de salida (dimensionalidad de la representación) */
  vectorSize: number;
  /** Sparsity del patrón por token (~5% del vector activo por token) */
  tokenSparsity: number;
  /** Número máximo de tokens a procesar por input */
  maxTokens: number;
  /** Si usar codificación posicional */
  positionalEncoding: boolean;
  /** Método de tokenización */
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
 * Encoder de Texto — Simula el procesamiento lingüístico temprano.
 * Convierte texto en patrones de spikes distribuidos.
 */
export class TextEncoder {
  private config: TextEncoderConfig;
  /** Tamaño del output */
  public outputSize: number;
  /** Cache de patrones por token (vocabulario aprendido) */
  private tokenCache: Map<string, Float32Array> = new Map();
  /** Seed base para hashing determinista */
  private hashSeed: number;

  constructor(config: Partial<TextEncoderConfig> = {}) {
    this.config = { ...DEFAULT_TEXT_CONFIG, ...config };
    this.outputSize = this.config.vectorSize;
    this.hashSeed = 42; // Reproducible
  }

  /**
   * Codifica un texto completo como vector de spikes.
   * 
   * @param text - Texto a codificar
   * @param dt - Paso temporal
   * @returns Vector de spikes
   */
  encode(text: string, dt: number = 1.0): Float32Array {
    // 1. Tokenizar
    const tokens = this.tokenize(text);
    
    // 2. Generar representación distribuida
    const representation = new Float32Array(this.config.vectorSize);
    
    for (let i = 0; i < Math.min(tokens.length, this.config.maxTokens); i++) {
      const token = tokens[i];
      
      // Obtener o generar patrón para este token
      const tokenPattern = this.getTokenPattern(token);
      
      // Añadir codificación posicional si está habilitada
      const positionWeight = this.config.positionalEncoding 
        ? this.getPositionWeight(i, tokens.length)
        : 1.0;
      
      // Acumular en la representación (superposición de patrones)
      for (let j = 0; j < this.config.vectorSize; j++) {
        representation[j] += tokenPattern[j] * positionWeight;
      }
    }
    
    // 3. Normalizar a rango 0-1
    this.normalize(representation);
    
    // 4. Convertir a spikes
    return encodeSpikeVector(representation, dt, 120);
  }

  /**
   * Tokeniza el texto según el método configurado.
   */
  private tokenize(text: string): string[] {
    const cleaned = text.toLowerCase().trim();
    
    switch (this.config.tokenization) {
      case 'character':
        return cleaned.split('');
      
      case 'word':
        // Tokenización por palabras con preservación de puntuación
        return cleaned.split(/\s+/).filter(t => t.length > 0);
      
      case 'subword':
        // Tokenización simple por sílabas/bigramas
        return this.subwordTokenize(cleaned);
      
      default:
        return cleaned.split(/\s+/);
    }
  }

  /**
   * Tokenización por subpalabras (bigramas de caracteres).
   * Aproximación simple a BPE sin necesidad de vocabulario preentrenado.
   */
  private subwordTokenize(text: string): string[] {
    const tokens: string[] = [];
    const words = text.split(/\s+/);
    
    for (const word of words) {
      if (word.length <= 3) {
        tokens.push(word);
      } else {
        // Bigramas solapados
        for (let i = 0; i < word.length - 1; i++) {
          tokens.push(word.substring(i, i + 2));
        }
        // También añadir la palabra completa (para reconocimiento directo)
        tokens.push(word);
      }
    }
    
    return tokens;
  }

  /**
   * Genera o recupera del cache el patrón de activación para un token.
   * Usa hashing determinista para generar un patrón sparse reproducible.
   * 
   * Inspirado en la representación distribuida sparse (SDR) del neocórtex:
   * cada concepto se representa por un subconjunto único de neuronas activas.
   */
  private getTokenPattern(token: string): Float32Array {
    // Verificar cache
    if (this.tokenCache.has(token)) {
      return this.tokenCache.get(token)!;
    }
    
    const pattern = new Float32Array(this.config.vectorSize);
    const numActive = Math.floor(this.config.vectorSize * this.config.tokenSparsity);
    
    // Hash determinista del token → posiciones activas
    let seed = this.hashString(token);
    
    for (let a = 0; a < numActive; a++) {
      seed = this.nextRandom(seed);
      const idx = Math.abs(seed) % this.config.vectorSize;
      
      // Activación con variación (no todas las posiciones igualmente activas)
      seed = this.nextRandom(seed);
      const strength = 0.5 + 0.5 * (Math.abs(seed) % 100) / 100;
      
      pattern[idx] = Math.min(1.0, pattern[idx] + strength);
    }
    
    // Cachear el patrón
    this.tokenCache.set(token, pattern);
    
    return pattern;
  }

  /**
   * Calcula un peso posicional para el token.
   * Tokens al final del input son ligeramente más relevantes
   * (efecto de recencia en la memoria de trabajo).
   */
  private getPositionWeight(position: number, totalTokens: number): number {
    // Peso base + recencia (tokens recientes tienen más peso)
    const recency = 0.7 + 0.3 * (position / Math.max(1, totalTokens - 1));
    
    // Modulación sinusoidal (como en Transformers, pero simplificada)
    const freq = position / 10;
    const sinMod = 0.9 + 0.1 * Math.sin(freq);
    
    return recency * sinMod;
  }

  /**
   * Hash determinista de string a número.
   */
  private hashString(str: string): number {
    let hash = this.hashSeed;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return hash;
  }

  /**
   * Generador pseudo-aleatorio determinista (LCG).
   */
  private nextRandom(seed: number): number {
    return (seed * 1103515245 + 12345) & 0x7fffffff;
  }

  /**
   * Normaliza un vector al rango 0-1.
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

  /** Tamaño del vocabulario en cache */
  get vocabularySize(): number {
    return this.tokenCache.size;
  }

  /** Limpia el cache de tokens */
  reset(): void {
    this.tokenCache.clear();
  }
}
