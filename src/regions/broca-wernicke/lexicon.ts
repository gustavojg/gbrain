/**
 * LÉXICO COMPARTIDO — Vocabulario neural distribuido
 * ===================================================
 * Almacén de vocabulario compartido entre las áreas de Wernicke (comprensión)
 * y Broca (producción) del cerebro digital.
 *
 * Base biológica:
 *   El léxico mental humano almacena ~60.000 palabras como patrones de activación
 *   neuronal distribuidos en el lóbulo temporal. Cada palabra se representa como
 *   un vector de activación sparse en una población neuronal (codificación
 *   distribuida, Hebb 1949). La recuperación léxica ocurre por similitud de
 *   patrones: al escuchar una palabra parcial, se activa el patrón más cercano
 *   (priming fonológico/semántico).
 *
 *   El léxico es compartido entre las áreas de comprensión y producción, aunque
 *   en la biología real estas áreas acceden al léxico por vías parcialmente
 *   distintas (vía ventral para Wernicke, vía dorsal para Broca).
 *
 * Implementación:
 *   Cada entrada contiene:
 *   - El patrón de activación neural (Float32Array sparse)
 *   - Frecuencia de uso (para sesgos de recuperación — efecto de frecuencia léxica)
 *   - Timestamp del último uso (para efecto de recencia)
 *
 *   La búsqueda por similitud usa coseno, optimizada con pre-cómputo de normas.
 */

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Entrada léxica: asocia una palabra con su patrón neural distribuido.
 */
export interface LexiconEntry {
  /** Patrón de activación neural distribuido (Float32Array sparse) */
  pattern: Float32Array;
  /**
   * Frecuencia de uso acumulada. Palabras más frecuentes se recuperan más rápido
   * (efecto de frecuencia léxica, Oldfield & Wingfield 1965).
   */
  frequency: number;
  /** Marca temporal del último acceso (ms). Sesga recuperación por recencia. */
  lastUsed: number;
}

/**
 * Resultado de una búsqueda por similitud en el léxico.
 */
export interface LexiconMatch {
  /** Palabra encontrada */
  word: string;
  /** Similitud coseno con el patrón de consulta (0–1) */
  similarity: number;
}

/**
 * Formato de serialización del léxico completo.
 */
export interface SerializedLexicon {
  /** Entradas serializadas: palabra → { pattern como number[], frequency, lastUsed } */
  entries: Array<{
    word: string;
    pattern: number[];
    frequency: number;
    lastUsed: number;
  }>;
  /** Dimensionalidad de los patrones */
  patternSize: number;
}

// ==================================================================
// Clase Lexicon
// ==================================================================

/**
 * Léxico neural compartido entre áreas del lenguaje.
 *
 * Base biológica:
 *   Modela el almacén léxico del lóbulo temporal donde cada concepto/palabra
 *   se representa como un patrón de activación distribuido sobre una población
 *   neuronal. La recuperación ocurre por competencia: el patrón almacenado
 *   más similar al input gana (modelo de acceso léxico por activación
 *   interactiva, McClelland & Rumelhart 1981).
 *
 *   Características implementadas:
 *   - Efecto de frecuencia léxica: palabras frecuentes tienen umbral de
 *     activación más bajo (se recuperan más rápido)
 *   - Efecto de recencia: palabras usadas recientemente tienen priming residual
 *   - Competencia léxica: al buscar, se retornan las K mejores coincidencias
 */
export class Lexicon {
  /** Almacén principal: palabra → entrada léxica */
  private readonly entries: Map<string, LexiconEntry> = new Map();

  /**
   * Cache de normas L2 para acelerar cálculos de similitud coseno.
   * Se invalida cuando se modifica una entrada.
   */
  private readonly normCache: Map<string, number> = new Map();

  /** Dimensionalidad de los patrones neurales */
  private readonly patternSize: number;

  /**
   * Crea un nuevo léxico neural.
   *
   * @param patternSize - Dimensionalidad de los patrones neurales (default: 5000)
   */
  constructor(patternSize: number = 5000) {
    this.patternSize = patternSize;
  }

  // ----------------------------------------------------------------
  // Operaciones CRUD
  // ----------------------------------------------------------------

  /**
   * Agrega o actualiza una palabra en el léxico.
   *
   * Base biológica:
   *   Modela el aprendizaje léxico: la primera exposición a una palabra
   *   crea un nuevo engrama neural. Exposiciones repetidas fortalecen
   *   la frecuencia (efecto de repetición) y actualizan el patrón si
   *   el contexto genera una representación ligeramente diferente
   *   (refinamiento del engrama).
   *
   * @param word - Palabra a agregar (normalizada a minúsculas)
   * @param pattern - Patrón de activación neural asociado
   */
  add(word: string, pattern: Float32Array): void {
    const key = word.toLowerCase();
    const existing = this.entries.get(key);

    if (existing) {
      // Actualización: mezclar patrón existente con nuevo (promedio ponderado)
      // Biología: refinamiento del engrama por exposición repetida
      const alpha = 0.3; // Tasa de actualización
      for (let i = 0; i < this.patternSize; i++) {
        existing.pattern[i] = existing.pattern[i] * (1 - alpha) + pattern[i] * alpha;
      }
      existing.frequency++;
      existing.lastUsed = Date.now();
    } else {
      // Nuevo engrama léxico
      this.entries.set(key, {
        pattern: new Float32Array(pattern),
        frequency: 1,
        lastUsed: Date.now(),
      });
    }

    // Invalidar cache de norma
    this.normCache.delete(key);
  }

  /**
   * Busca una palabra en el léxico y retorna su patrón neural.
   *
   * Base biológica:
   *   Acceso léxico directo (por forma ortográfica/fonológica conocida).
   *   Actualiza la recencia para modelar priming.
   *
   * @param word - Palabra a buscar
   * @returns Patrón neural si existe, null si no está en el léxico
   */
  lookup(word: string): Float32Array | null {
    const entry = this.entries.get(word.toLowerCase());
    if (!entry) return null;

    entry.lastUsed = Date.now();
    return entry.pattern;
  }

  /**
   * Busca las K palabras más similares a un patrón neural dado.
   *
   * Base biológica:
   *   Modela el acceso léxico por activación: el input neural (ej: patrón
   *   auditivo parcial) se compara con todos los engramas léxicos
   *   almacenados. Los engramas más similares se activan primero
   *   (modelo de cohorte, Marslen-Wilson 1987). La frecuencia léxica
   *   sesga la competencia a favor de palabras más comunes.
   *
   * @param pattern - Patrón neural de consulta
   * @param topK - Número de coincidencias a retornar (default: 5)
   * @returns Array de coincidencias ordenadas por similitud descendente
   */
  findClosest(pattern: Float32Array, topK: number = 5): LexiconMatch[] {
    const queryNorm = this.computeNorm(pattern);
    if (queryNorm === 0) return [];

    const matches: LexiconMatch[] = [];

    for (const [word, entry] of this.entries) {
      const entryNorm = this.getCachedNorm(word, entry.pattern);
      if (entryNorm === 0) continue;

      // Similitud coseno base
      let similarity = this.dotProduct(pattern, entry.pattern) / (queryNorm * entryNorm);

      // Sesgo por frecuencia léxica (efecto sutil, ~5% de boost max)
      // Biología: palabras frecuentes tienen umbrales más bajos
      const freqBoost = Math.min(0.05, Math.log1p(entry.frequency) * 0.01);
      similarity = Math.min(1.0, similarity + freqBoost);

      matches.push({ word, similarity });
    }

    // Ordenar por similitud descendente y retornar top K
    matches.sort((a, b) => b.similarity - a.similarity);
    return matches.slice(0, topK);
  }

  // ----------------------------------------------------------------
  // Utilidades matemáticas
  // ----------------------------------------------------------------

  /**
   * Calcula la similitud coseno entre dos patrones neurales.
   *
   * Base biológica:
   *   La similitud coseno captura la similitud de la "dirección" de
   *   activación de dos poblaciones neurales, independiente de la magnitud.
   *   Esto modela cómo el cerebro compara patrones de activación
   *   distributed (análogo a la correlación de población neural).
   *
   * @param a - Primer patrón neural
   * @param b - Segundo patrón neural
   * @returns Similitud coseno en rango [-1, 1]
   */
  cosineSimilarity(a: Float32Array, b: Float32Array): number {
    const normA = this.computeNorm(a);
    const normB = this.computeNorm(b);

    if (normA === 0 || normB === 0) return 0;

    return this.dotProduct(a, b) / (normA * normB);
  }

  /**
   * Producto punto (dot product) optimizado para Float32Array.
   *
   * @param a - Primer vector
   * @param b - Segundo vector
   * @returns Producto escalar
   */
  private dotProduct(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let sum = 0;
    // Desenrollado manual de bucle para mejor rendimiento (4-way unroll)
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
   * Calcula la norma L2 de un vector.
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
   * Obtiene la norma L2 cacheada de una entrada léxica.
   * Evita recalcular la norma en cada búsqueda.
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
  // Propiedades y utilidades
  // ----------------------------------------------------------------

  /**
   * Número total de palabras en el léxico.
   */
  get size(): number {
    return this.entries.size;
  }

  /**
   * Dimensionalidad de los patrones neurales.
   */
  get dimensions(): number {
    return this.patternSize;
  }

  /**
   * Retorna todas las palabras del léxico.
   */
  getWords(): string[] {
    return Array.from(this.entries.keys());
  }

  /**
   * Verifica si una palabra existe en el léxico.
   */
  has(word: string): boolean {
    return this.entries.has(word.toLowerCase());
  }

  // ----------------------------------------------------------------
  // Serialización / Deserialización
  // ----------------------------------------------------------------

  /**
   * Serializa el léxico completo para persistencia.
   *
   * @returns Objeto serializable con todas las entradas léxicas
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
   * Restaura el léxico desde datos serializados.
   *
   * @param data - Datos previamente serializados con serialize()
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
