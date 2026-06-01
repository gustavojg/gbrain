/**
 * LÉXICO ESPAÑOL — Semilla de vocabulario inicial
 * =================================================
 * Puebla el léxico neural con ~190 palabras fundamentales del español,
 * organizadas por categoría semántica.
 *
 * Base biológica:
 *   El léxico mental del hablante nativo contiene ~60.000 palabras, pero
 *   un núcleo de ~2.000 palabras cubre el ~80% del uso cotidiano (ley de
 *   Zipf). Este módulo siembra las ~190 más esenciales como punto de
 *   partida, análogo a cómo un niño adquiere su vocabulario temprano:
 *   primero los pronombres, saludos, emociones básicas y palabras
 *   concretas de alta frecuencia.
 *
 *   Los patrones se generan con hashing de n-gramas de caracteres.
 *   Esto produce representaciones distribuidas donde palabras que
 *   comparten sub-cadenas (ej: "triste"/"tristeza", "pensar"/"pensamiento")
 *   tienen patrones parcialmente superpuestos — modelando el priming
 *   fonológico/ortográfico observado en el acceso léxico humano
 *   (Marslen-Wilson, 1987; Coltheart et al., 2001).
 */

import { Lexicon } from './lexicon.js';

// ==================================================================
// Generador de patrones deterministas por n-gramas
// ==================================================================

/**
 * Función hash djb2 (Daniel J. Bernstein).
 * Genera un hash entero determinista para una cadena.
 *
 * Base biológica:
 *   Emula la transformación no lineal que convierte una señal
 *   acústica/visual en un patrón de activación cortical. Distintas
 *   entradas producen patrones distintos, pero entradas similares
 *   (n-gramas compartidos) activan posiciones superpuestas.
 *
 * @param str - Cadena a hashear
 * @returns Entero de 32 bits (puede ser negativo)
 */
function hashCode(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    // hash * 33 + charCode — variante clásica djb2
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return hash;
}

/**
 * Convierte una palabra en un patrón de activación neural distribuido
 * usando hashing de n-gramas de caracteres (uni-, bi- y tri-gramas).
 *
 * Base biológica:
 *   El sistema de lectura humano descompone las palabras en unidades
 *   sub-léxicas (grafemas, bigramas abiertos, etc.) que se procesan
 *   en paralelo en el área visual de la forma de las palabras (VWFA,
 *   Cohen et al., 2000). Cada sub-unidad activa un subconjunto de
 *   neuronas. Palabras que comparten sub-unidades (ej: "casa"/"casado")
 *   activan neuronas solapadas → patrones parcialmente similares.
 *
 *   Pesos de activación:
 *   - Unigramas: 0.3 (activación débil, poca especificidad)
 *   - Bigramas:  0.5 (activación media, captura secuencia local)
 *   - Trigramas: 0.7 (activación fuerte, alta especificidad)
 *
 * @param word - Palabra a codificar
 * @param size - Dimensionalidad del patrón (default: 5000)
 * @returns Patrón de activación normalizado [0, 1]
 */
export function wordToPattern(word: string, size: number = 5000): Float32Array {
  const pattern = new Float32Array(size);

  for (let i = 0; i < word.length; i++) {
    // Unigrama: activación débil por cada carácter individual
    const h1 = hashCode(word[i]) % size;
    pattern[Math.abs(h1)] += 0.3;

    // Bigrama: captura transiciones entre caracteres adyacentes
    if (i < word.length - 1) {
      const h2 = hashCode(word.substring(i, i + 2)) % size;
      pattern[Math.abs(h2)] += 0.5;
    }

    // Trigrama: captura contexto local más amplio
    if (i < word.length - 2) {
      const h3 = hashCode(word.substring(i, i + 3)) % size;
      pattern[Math.abs(h3)] += 0.7;
    }
  }

  // Normalización: escalar al rango [0, 1]
  // Biología: homeostasis sináptica — las neuronas regulan su activación
  // para mantenerse en un rango funcional (Turrigiano, 2008)
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
// Vocabulario semilla organizado por categoría semántica
// ==================================================================

/**
 * Saludos y expresiones sociales (15 entradas).
 *
 * Base biológica:
 *   Las fórmulas sociales se almacenan como unidades holísticas en la
 *   memoria procedimental (ganglios basales), no se componen palabra
 *   por palabra. Por eso los afásicos de Broca a menudo retienen
 *   saludos automáticos.
 */
const SALUDOS: readonly string[] = [
  'hola', 'adiós', 'buenos', 'días', 'noches',
  'tardes', 'gracias', 'perdón', 'por favor', 'de nada',
  'bienvenido', 'hasta luego', 'chao', 'hey', 'oye',
];

/**
 * Pronombres personales y clíticos (12 entradas).
 *
 * Base biológica:
 *   Los pronombres son palabras de clase cerrada, procesados
 *   principalmente en el hemisferio izquierdo (área de Broca).
 *   Su frecuencia extremadamente alta los convierte en los primeros
 *   ítems léxicos adquiridos (junto con determinantes).
 */
const PRONOMBRES: readonly string[] = [
  'yo', 'tú', 'él', 'ella', 'nosotros', 'ellos',
  'me', 'te', 'se', 'nos', 'mi', 'tu',
];

/**
 * Vocabulario emocional (25 entradas).
 *
 * Base biológica:
 *   Las palabras emocionales tienen acceso privilegiado al léxico:
 *   se reconocen más rápido y se recuerdan mejor (efecto de emocionalidad,
 *   Kensinger & Corkin, 2003). Esto se debe a la conexión directa
 *   entre la amígdala y las áreas de procesamiento léxico temporal.
 */
const EMOCIONES: readonly string[] = [
  'feliz', 'triste', 'miedo', 'amor', 'odio',
  'alegría', 'tristeza', 'sorpresa', 'asco', 'curiosidad',
  'calma', 'paz', 'ansiedad', 'estrés', 'emoción',
  'enojo', 'vergüenza', 'orgullo', 'esperanza', 'soledad',
  'cariño', 'ternura', 'nostalgia', 'frustración', 'entusiasmo',
];

/**
 * Verbos de acción y cognición (25 entradas).
 *
 * Base biológica:
 *   Los verbos de acción activan la corteza motora de forma somatotópica
 *   (Hauk et al., 2004): "caminar" activa la zona de las piernas,
 *   "escribir" la zona de las manos. Los verbos cognitivos ("pensar",
 *   "recordar") activan más la corteza prefrontal.
 */
const ACCIONES: readonly string[] = [
  'ver', 'mirar', 'escuchar', 'oír', 'hablar',
  'decir', 'pensar', 'recordar', 'olvidar', 'sentir',
  'tocar', 'comer', 'dormir', 'soñar', 'caminar',
  'correr', 'leer', 'escribir', 'aprender', 'enseñar',
  'crear', 'imaginar', 'entender', 'saber', 'querer',
];

/**
 * Conceptos del sistema nervioso (15 entradas).
 *
 * Base biológica:
 *   Vocabulario meta-cognitivo: palabras que el propio sistema cerebral
 *   puede usar para describirse a sí mismo. En un cerebro digital,
 *   estas palabras son especialmente relevantes para la introspección
 *   y el auto-reporte de estados internos.
 */
const CONCEPTOS_CEREBRALES: readonly string[] = [
  'cerebro', 'neurona', 'sinapsis', 'memoria', 'recuerdo',
  'sueño', 'consciencia', 'mente', 'pensamiento', 'idea',
  'atención', 'concentración', 'percepción', 'emoción', 'aprendizaje',
];

/**
 * Adjetivos descriptivos (20 entradas).
 *
 * Base biológica:
 *   Los adjetivos se representan en el lóbulo temporal anterior como
 *   dimensiones de un espacio semántico continuo (Patterson et al., 2007).
 *   Los pares antónimos (grande/pequeño, rápido/lento) comparten regiones
 *   corticales cercanas, diferenciándose en la polaridad del eje semántico.
 */
const DESCRIPTIVOS: readonly string[] = [
  'grande', 'pequeño', 'rápido', 'lento', 'nuevo',
  'viejo', 'bueno', 'malo', 'bonito', 'feo',
  'fuerte', 'débil', 'alto', 'bajo', 'largo',
  'corto', 'claro', 'oscuro', 'frío', 'caliente',
];

/**
 * Conectores y preposiciones (15 entradas).
 *
 * Base biológica:
 *   Las palabras funcionales (conectores, preposiciones) se procesan
 *   predominantemente en el área de Broca (BA44/45) y la corteza
 *   prefrontal inferior izquierda. Los afásicos de Broca tienen
 *   dificultades específicas con estas palabras (agramatismo).
 */
const CONECTORES: readonly string[] = [
  'y', 'o', 'pero', 'porque', 'cuando',
  'si', 'que', 'como', 'donde', 'para',
  'por', 'con', 'sin', 'entre', 'sobre',
];

/**
 * Palabras interrogativas (8 entradas).
 *
 * Base biológica:
 *   Las preguntas involucran una prosodia especial (curva entonacional
 *   ascendente) procesada en el hemisferio derecho. Las palabras
 *   interrogativas en español llevan tilde diacrítica que las distingue
 *   de sus homófonos relativos (qué/que, cómo/como).
 */
const PREGUNTAS: readonly string[] = [
  'qué', 'cómo', 'cuándo', 'dónde', 'quién',
  'cuál', 'cuánto', 'por qué',
];

/**
 * Sustantivos concretos de alta frecuencia (25 entradas).
 *
 * Base biológica:
 *   Los sustantivos concretos ("casa", "mano", "sol") tienen una
 *   ventaja de concretud: se procesan más rápido y se recuerdan
 *   mejor que los abstractos (Paivio, 1971), porque activan
 *   representaciones tanto verbales como imagénicas (teoría de
 *   doble codificación). Se distribuyen en la corteza temporal
 *   ventral y lateral.
 */
const SUSTANTIVOS: readonly string[] = [
  'persona', 'casa', 'mundo', 'tiempo', 'vida',
  'día', 'noche', 'agua', 'sol', 'luna',
  'cielo', 'tierra', 'mar', 'luz', 'color',
  'música', 'sonido', 'voz', 'palabra', 'historia',
  'camino', 'nombre', 'mano', 'ojo', 'corazón',
];

/**
 * Respuestas cortas y fillers discursivos (10 entradas).
 *
 * Base biológica:
 *   Los fillers ("mmm", "ah") y respuestas automáticas ("sí", "no")
 *   se procesan como unidades motoras sobreaprendidas en los ganglios
 *   basales. Sobreviven incluso a afasias graves, lo que indica
 *   almacenamiento subcortical redundante.
 */
const RESPUESTAS: readonly string[] = [
  'sí', 'no', 'tal vez', 'claro', 'bien',
  'mal', 'vale', 'ok', 'mmm', 'ah',
];

/**
 * Adverbios y expresiones temporales (10 entradas).
 *
 * Base biológica:
 *   El procesamiento temporal involucra el cerebelo y la corteza
 *   prefrontal dorsolateral. Las palabras temporales anclan los
 *   eventos en la línea temporal subjetiva, esencial para la
 *   memoria episódica y la planificación.
 */
const TIEMPO: readonly string[] = [
  'hoy', 'ayer', 'mañana', 'ahora', 'antes',
  'después', 'siempre', 'nunca', 'pronto', 'tarde',
];

/**
 * Números cardinales básicos (10 entradas).
 *
 * Base biológica:
 *   Los números se procesan en el surco intraparietal (Dehaene, 1997)
 *   con una representación logarítmica de magnitud (la distancia
 *   representacional entre 1 y 2 es mayor que entre 8 y 9).
 *   Las palabras numéricas son la interfaz entre el sistema lingüístico
 *   y el sistema de magnitud numérica.
 */
const NUMEROS: readonly string[] = [
  'uno', 'dos', 'tres', 'cuatro', 'cinco',
  'seis', 'siete', 'ocho', 'nueve', 'diez',
];

// ==================================================================
// Función principal de siembra
// ==================================================================

/**
 * Puebla el léxico con un vocabulario semilla de ~190 palabras en español.
 *
 * Base biológica:
 *   Análogo al período crítico de adquisición léxica temprana (0–6 años),
 *   donde el cerebro del niño incorpora rápidamente su vocabulario nuclear.
 *   Las palabras se codifican como patrones de activación distribuidos
 *   usando hashing de n-gramas, lo que produce superposición natural
 *   entre palabras morfológicamente relacionadas (ej: "triste"/"tristeza")
 *   — modelando el priming ortográfico/fonológico.
 *
 *   El orden de inserción (saludos → pronombres → emociones → acciones → ...)
 *   refleja aproximadamente la secuencia de adquisición del primer léxico
 *   infantil (Clark, 1993).
 *
 * @param lexicon - Instancia de Lexicon donde se insertarán las palabras
 */
export function seedSpanishLexicon(lexicon: Lexicon): void {
  /** Tamaño del patrón neural, tomado de la instancia del léxico */
  const size = lexicon.dimensions;

  /**
   * Todas las categorías semánticas a sembrar.
   * Cada categoría es un arreglo de palabras en español.
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
 * Codifica una frase completa en el MISMO espacio representacional que el
 * léxico (superponiendo los patrones n-grama de cada palabra).
 *
 * Por qué existe:
 *   El `TextEncoder` genérico usa su propio hash y produce vectores
 *   ORTOGONALES a los del léxico (cos ≈ 0), de modo que Wernicke no reconoce
 *   ninguna palabra. Esta función garantiza que el texto de entrada caiga en
 *   el espacio del léxico, para que `Wernicke.comprehend()` recupere las
 *   palabras reales y Broca pueda producir una respuesta asociativa
 *   reproducible y discriminativa.
 *
 * Base biológica:
 *   La comprensión del lenguaje proyecta la señal acústica/ortográfica sobre
 *   los engramas léxicos almacenados (acceso léxico por similitud de patrón,
 *   McClelland & Rumelhart 1981). Usamos la misma codificación n-grama que
 *   creó los engramas → solapamiento natural entre palabras relacionadas.
 *
 * @param text - Frase de entrada
 * @param size - Dimensionalidad (debe coincidir con lexicon.dimensions)
 * @returns Patrón distribuido normalizado [0, 1] en el espacio del léxico
 */
export function encodeSentenceToLexiconSpace(text: string, size: number = 5000): Float32Array {
  const pattern = new Float32Array(size);

  // Normalización tipográfica: minúsculas, sin acentos, solo palabras.
  const words = text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\w\sáéíóúüñ]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 0);

  if (words.length === 0) return pattern;

  // Superponer el patrón n-grama de cada palabra (conocida o no): todas caen
  // en el mismo espacio que los engramas del léxico.
  for (const word of words) {
    const wp = wordToPattern(word, size);
    for (let i = 0; i < size; i++) pattern[i] += wp[i];
  }

  // Normalizar al rango [0, 1] para mantener escala homeostática.
  let max = 0;
  for (let i = 0; i < size; i++) if (pattern[i] > max) max = pattern[i];
  if (max > 0) {
    for (let i = 0; i < size; i++) pattern[i] /= max;
  }

  return pattern;
}
