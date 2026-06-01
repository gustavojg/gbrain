/**
 * SPANISH LEXICON — Initial vocabulary seed
 * =================================================
 * Populates the neural lexicon with ~190 fundamental Spanish words,
 * organized by semantic category.
 *
 * Biological basis:
 *   The native speaker's mental lexicon contains ~60,000 words, but
 *   a core of ~2,000 words covers ~80% of everyday use (Zipf's
 *   law). This module seeds the ~190 most essential ones as a starting
 *   point, analogous to how a child acquires its early vocabulary:
 *   first the pronouns, greetings, basic emotions, and concrete
 *   high-frequency words.
 *
 *   The patterns are generated with character n-gram hashing.
 *   This produces distributed representations where words that
 *   share substrings (e.g.: "triste"/"tristeza", "pensar"/"pensamiento")
 *   have partially overlapping patterns — modeling the
 *   phonological/orthographic priming observed in human lexical access
 *   (Marslen-Wilson, 1987; Coltheart et al., 2001).
 */

import { Lexicon } from './lexicon.js';

// ==================================================================
// Deterministic n-gram pattern generator
// ==================================================================

/**
 * djb2 hash function (Daniel J. Bernstein).
 * Generates a deterministic integer hash for a string.
 *
 * Biological basis:
 *   Emulates the nonlinear transformation that converts an
 *   acoustic/visual signal into a cortical activation pattern. Different
 *   inputs produce different patterns, but similar inputs
 *   (shared n-grams) activate overlapping positions.
 *
 * @param str - String to hash
 * @returns 32-bit integer (may be negative)
 */
function hashCode(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    // hash * 33 + charCode — classic djb2 variant
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return hash;
}

/**
 * Converts a word into a distributed neural activation pattern
 * using character n-gram hashing (uni-, bi-, and tri-grams).
 *
 * Biological basis:
 *   The human reading system decomposes words into sub-lexical
 *   units (graphemes, open bigrams, etc.) that are processed
 *   in parallel in the visual word form area (VWFA,
 *   Cohen et al., 2000). Each sub-unit activates a subset of
 *   neurons. Words that share sub-units (e.g.: "casa"/"casado")
 *   activate overlapping neurons → partially similar patterns.
 *
 *   Activation weights:
 *   - Unigrams: 0.3 (weak activation, low specificity)
 *   - Bigrams:  0.5 (medium activation, captures local sequence)
 *   - Trigrams: 0.7 (strong activation, high specificity)
 *
 * @param word - Word to encode
 * @param size - Dimensionality of the pattern (default: 5000)
 * @returns Normalized activation pattern [0, 1]
 */
export function wordToPattern(word: string, size: number = 5000): Float32Array {
  const pattern = new Float32Array(size);

  for (let i = 0; i < word.length; i++) {
    // Unigram: weak activation for each individual character
    const h1 = hashCode(word[i]) % size;
    pattern[Math.abs(h1)] += 0.3;

    // Bigram: captures transitions between adjacent characters
    if (i < word.length - 1) {
      const h2 = hashCode(word.substring(i, i + 2)) % size;
      pattern[Math.abs(h2)] += 0.5;
    }

    // Trigram: captures wider local context
    if (i < word.length - 2) {
      const h3 = hashCode(word.substring(i, i + 3)) % size;
      pattern[Math.abs(h3)] += 0.7;
    }
  }

  // Normalization: scale to the range [0, 1]
  // Biology: synaptic homeostasis — neurons regulate their activation
  // to stay within a functional range (Turrigiano, 2008)
  let max = 0;
  for (let j = 0; j < size; j++) {
    if (pattern[j] > max) max = pattern[j];
  }
  if (max > 0) {
    for (let j = 0; j < size; j++) {
      pattern[j] /= max;
    }
  }

  return pattern;
}

// ==================================================================
// Seed vocabulary organized by semantic category
// ==================================================================

/**
 * Greetings and social expressions (15 entries).
 *
 * Biological basis:
 *   Social formulas are stored as holistic units in
 *   procedural memory (basal ganglia); they are not composed word
 *   by word. That is why Broca's aphasics often retain
 *   automatic greetings.
 */
const SALUDOS: readonly string[] = [
  'hola', 'adiós', 'buenos', 'días', 'noches',
  'tardes', 'gracias', 'perdón', 'por favor', 'de nada',
  'bienvenido', 'hasta luego', 'chao', 'hey', 'oye',
];

/**
 * Personal pronouns and clitics (12 entries).
 *
 * Biological basis:
 *   Pronouns are closed-class words, processed
 *   mainly in the left hemisphere (Broca's area).
 *   Their extremely high frequency makes them the first
 *   lexical items acquired (along with determiners).
 */
const PRONOMBRES: readonly string[] = [
  'yo', 'tú', 'él', 'ella', 'nosotros', 'ellos',
  'me', 'te', 'se', 'nos', 'mi', 'tu',
];

/**
 * Emotional vocabulary (25 entries).
 *
 * Biological basis:
 *   Emotional words have privileged access to the lexicon:
 *   they are recognized faster and remembered better (emotionality effect,
 *   Kensinger & Corkin, 2003). This is due to the direct connection
 *   between the amygdala and the temporal lexical processing areas.
 */
const EMOCIONES: readonly string[] = [
  'feliz', 'triste', 'miedo', 'amor', 'odio',
  'alegría', 'tristeza', 'sorpresa', 'asco', 'curiosidad',
  'calma', 'paz', 'ansiedad', 'estrés', 'emoción',
  'enojo', 'vergüenza', 'orgullo', 'esperanza', 'soledad',
  'cariño', 'ternura', 'nostalgia', 'frustración', 'entusiasmo',
];

/**
 * Action and cognition verbs (25 entries).
 *
 * Biological basis:
 *   Action verbs activate the motor cortex somatotopically
 *   (Hauk et al., 2004): "caminar" activates the leg area,
 *   "escribir" the hand area. Cognitive verbs ("pensar",
 *   "recordar") activate the prefrontal cortex more.
 */
const ACCIONES: readonly string[] = [
  'ver', 'mirar', 'escuchar', 'oír', 'hablar',
  'decir', 'pensar', 'recordar', 'olvidar', 'sentir',
  'tocar', 'comer', 'dormir', 'soñar', 'caminar',
  'correr', 'leer', 'escribir', 'aprender', 'enseñar',
  'crear', 'imaginar', 'entender', 'saber', 'querer',
];

/**
 * Nervous system concepts (15 entries).
 *
 * Biological basis:
 *   Meta-cognitive vocabulary: words that the brain system itself
 *   can use to describe itself. In a digital brain,
 *   these words are especially relevant for introspection
 *   and self-reporting of internal states.
 */
const CONCEPTOS_CEREBRALES: readonly string[] = [
  'cerebro', 'neurona', 'sinapsis', 'memoria', 'recuerdo',
  'sueño', 'consciencia', 'mente', 'pensamiento', 'idea',
  'atención', 'concentración', 'percepción', 'emoción', 'aprendizaje',
];

/**
 * Descriptive adjectives (20 entries).
 *
 * Biological basis:
 *   Adjectives are represented in the anterior temporal lobe as
 *   dimensions of a continuous semantic space (Patterson et al., 2007).
 *   Antonym pairs (grande/pequeño, rápido/lento) share nearby
 *   cortical regions, differing in the polarity of the semantic axis.
 */
const DESCRIPTIVOS: readonly string[] = [
  'grande', 'pequeño', 'rápido', 'lento', 'nuevo',
  'viejo', 'bueno', 'malo', 'bonito', 'feo',
  'fuerte', 'débil', 'alto', 'bajo', 'largo',
  'corto', 'claro', 'oscuro', 'frío', 'caliente',
];

/**
 * Connectors and prepositions (15 entries).
 *
 * Biological basis:
 *   Function words (connectors, prepositions) are processed
 *   predominantly in Broca's area (BA44/45) and the left
 *   inferior prefrontal cortex. Broca's aphasics have
 *   specific difficulties with these words (agrammatism).
 */
const CONECTORES: readonly string[] = [
  'y', 'o', 'pero', 'porque', 'cuando',
  'si', 'que', 'como', 'donde', 'para',
  'por', 'con', 'sin', 'entre', 'sobre',
];

/**
 * Interrogative words (8 entries).
 *
 * Biological basis:
 *   Questions involve a special prosody (rising intonational
 *   contour) processed in the right hemisphere. Interrogative
 *   words in Spanish carry a diacritical accent that distinguishes them
 *   from their relative homophones (qué/que, cómo/como).
 */
const PREGUNTAS: readonly string[] = [
  'qué', 'cómo', 'cuándo', 'dónde', 'quién',
  'cuál', 'cuánto', 'por qué',
];

/**
 * High-frequency concrete nouns (25 entries).
 *
 * Biological basis:
 *   Concrete nouns ("casa", "mano", "sol") have a
 *   concreteness advantage: they are processed faster and remembered
 *   better than abstract ones (Paivio, 1971), because they activate
 *   both verbal and imagistic representations (dual
 *   coding theory). They are distributed across the ventral and
 *   lateral temporal cortex.
 */
const SUSTANTIVOS: readonly string[] = [
  'persona', 'casa', 'mundo', 'tiempo', 'vida',
  'día', 'noche', 'agua', 'sol', 'luna',
  'cielo', 'tierra', 'mar', 'luz', 'color',
  'música', 'sonido', 'voz', 'palabra', 'historia',
  'camino', 'nombre', 'mano', 'ojo', 'corazón',
];

/**
 * Short responses and discourse fillers (10 entries).
 *
 * Biological basis:
 *   Fillers ("mmm", "ah") and automatic responses ("sí", "no")
 *   are processed as overlearned motor units in the basal
 *   ganglia. They survive even severe aphasias, which indicates
 *   redundant subcortical storage.
 */
const RESPUESTAS: readonly string[] = [
  'sí', 'no', 'tal vez', 'claro', 'bien',
  'mal', 'vale', 'ok', 'mmm', 'ah',
];

/**
 * Adverbs and temporal expressions (10 entries).
 *
 * Biological basis:
 *   Temporal processing involves the cerebellum and the
 *   dorsolateral prefrontal cortex. Temporal words anchor
 *   events on the subjective timeline, essential for
 *   episodic memory and planning.
 */
const TIEMPO: readonly string[] = [
  'hoy', 'ayer', 'mañana', 'ahora', 'antes',
  'después', 'siempre', 'nunca', 'pronto', 'tarde',
];

/**
 * Basic cardinal numbers (10 entries).
 *
 * Biological basis:
 *   Numbers are processed in the intraparietal sulcus (Dehaene, 1997)
 *   with a logarithmic representation of magnitude (the
 *   representational distance between 1 and 2 is greater than between 8 and 9).
 *   Number words are the interface between the linguistic system
 *   and the numerical magnitude system.
 */
const NUMEROS: readonly string[] = [
  'uno', 'dos', 'tres', 'cuatro', 'cinco',
  'seis', 'siete', 'ocho', 'nueve', 'diez',
];

// ==================================================================
// Main seeding function
// ==================================================================

/**
 * Populates the lexicon with a seed vocabulary of ~190 Spanish words.
 *
 * Biological basis:
 *   Analogous to the critical period of early lexical acquisition (0–6 years),
 *   where the child's brain rapidly incorporates its core vocabulary.
 *   The words are encoded as distributed activation patterns
 *   using n-gram hashing, which produces natural overlap
 *   between morphologically related words (e.g.: "triste"/"tristeza")
 *   — modeling orthographic/phonological priming.
 *
 *   The insertion order (greetings → pronouns → emotions → actions → ...)
 *   roughly reflects the acquisition sequence of the first
 *   childhood lexicon (Clark, 1993).
 *
 * @param lexicon - Lexicon instance where the words will be inserted
 */
export function seedSpanishLexicon(lexicon: Lexicon): void {
  /** Neural pattern size, taken from the lexicon instance */
  const size = lexicon.dimensions;

  /**
   * All the semantic categories to seed.
   * Each category is an array of Spanish words.
   */
  const allCategories: readonly (readonly string[])[] = [
    SALUDOS,
    PRONOMBRES,
    EMOCIONES,
    ACCIONES,
    CONCEPTOS_CEREBRALES,
    DESCRIPTIVOS,
    CONECTORES,
    PREGUNTAS,
    SUSTANTIVOS,
    RESPUESTAS,
    TIEMPO,
    NUMEROS,
  ];

  for (const category of allCategories) {
    for (const word of category) {
      const pattern = wordToPattern(word, size);
      lexicon.add(word, pattern);
    }
  }
}

/**
 * Encodes a complete sentence into the SAME representational space as the
 * lexicon (superimposing the n-gram patterns of each word).
 *
 * Why it exists:
 *   The generic `TextEncoder` uses its own hash and produces vectors
 *   ORTHOGONAL to those of the lexicon (cos ≈ 0), so that Wernicke does not
 *   recognize any word. This function guarantees that the input text falls in
 *   the lexicon space, so that `Wernicke.comprehend()` retrieves the
 *   real words and Broca can produce a reproducible and discriminative
 *   associative response.
 *
 * Biological basis:
 *   Language comprehension projects the acoustic/orthographic signal onto
 *   the stored lexical engrams (lexical access by pattern similarity,
 *   McClelland & Rumelhart 1981). We use the same n-gram encoding that
 *   created the engrams → natural overlap between related words.
 *
 * @param text - Input sentence
 * @param size - Dimensionality (must match lexicon.dimensions)
 * @returns Normalized distributed pattern [0, 1] in the lexicon space
 */
export function encodeSentenceToLexiconSpace(text: string, size: number = 5000): Float32Array {
  const pattern = new Float32Array(size);

  // Typographic normalization: lowercase, no accents, words only.
  const words = text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\w\sáéíóúüñ]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 0);

  if (words.length === 0) return pattern;

  // Superimpose the n-gram pattern of each word (known or not): they all fall
  // in the same space as the lexicon engrams.
  for (const word of words) {
    const wp = wordToPattern(word, size);
    for (let i = 0; i < size; i++) pattern[i] += wp[i];
  }

  // Normalize to the range [0, 1] to keep a homeostatic scale.
  let max = 0;
  for (let i = 0; i < size; i++) if (pattern[i] > max) max = pattern[i];
  if (max > 0) {
    for (let i = 0; i < size; i++) pattern[i] /= max;
  }

  return pattern;
}
