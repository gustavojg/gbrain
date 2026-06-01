/**
 * ENGLISH LEXICON — Initial vocabulary seed
 * =================================================
 * Populates the neural lexicon with ~190 fundamental English words,
 * mirroring the categories of the Spanish lexicon so the brain becomes
 * bilingual. Both vocabularies share the SAME representational space
 * because they use the same character n-gram encoder (`wordToPattern`),
 * which is language-agnostic.
 *
 * Biological basis:
 *   Bilingual brains store both lexicons in largely overlapping
 *   left-hemisphere language areas (Broca/Wernicke), with words from
 *   each language competing for activation during lexical access
 *   (Kroll & Stewart, 1994; Abutalebi & Green, 2007). Here that is
 *   modeled naturally: English and Spanish engrams coexist in one
 *   distributed pattern space, and similar spellings overlap regardless
 *   of language.
 */

import { Lexicon } from './lexicon.js';
import { wordToPattern } from './spanish-lexicon.js';

// ==================================================================
// Seed vocabulary organized by semantic category (English mirror)
// ==================================================================

/**
 * Greetings and social expressions (15 entries).
 *
 * Biological basis:
 *   Social formulas are stored as holistic units in procedural memory
 *   (basal ganglia); Broca's aphasics often retain automatic greetings.
 */
const GREETINGS: readonly string[] = [
  'hello', 'goodbye', 'good', 'morning', 'night',
  'afternoon', 'thanks', 'sorry', 'please', 'welcome',
  'bye', 'later', 'hi', 'hey', 'cheers',
];

/**
 * Personal pronouns and clitics (12 entries).
 *
 * Biological basis:
 *   Pronouns are closed-class words processed mainly in the left
 *   hemisphere (Broca's area). Their extremely high frequency makes
 *   them among the first lexical items acquired.
 */
const PRONOUNS: readonly string[] = [
  'i', 'you', 'he', 'she', 'we', 'they',
  'me', 'him', 'her', 'us', 'my', 'your',
];

/**
 * Emotional vocabulary (25 entries).
 *
 * Biological basis:
 *   Emotional words have privileged lexical access: recognized faster
 *   and remembered better (Kensinger & Corkin, 2003), via the direct
 *   amygdala–temporal lexicon connection.
 */
const EMOTIONS: readonly string[] = [
  'happy', 'sad', 'fear', 'love', 'hate',
  'joy', 'sadness', 'surprise', 'disgust', 'curiosity',
  'calm', 'peace', 'anxiety', 'stress', 'emotion',
  'anger', 'shame', 'pride', 'hope', 'loneliness',
  'affection', 'tenderness', 'nostalgia', 'frustration', 'enthusiasm',
];

/**
 * Action and cognition verbs (25 entries).
 *
 * Biological basis:
 *   Action verbs activate the motor cortex somatotopically (Hauk et al.,
 *   2004): "walk" activates the leg area, "write" the hand area.
 *   Cognitive verbs ("think", "remember") engage the prefrontal cortex.
 */
const ACTIONS: readonly string[] = [
  'see', 'look', 'listen', 'hear', 'talk',
  'say', 'think', 'remember', 'forget', 'feel',
  'touch', 'eat', 'sleep', 'dream', 'walk',
  'run', 'read', 'write', 'learn', 'teach',
  'create', 'imagine', 'understand', 'know', 'want',
];

/**
 * Nervous system concepts (15 entries).
 *
 * Biological basis:
 *   Meta-cognitive vocabulary: words the brain system can use to
 *   describe itself, especially relevant for introspection and
 *   self-reporting of internal states in a digital brain.
 */
const BRAIN_CONCEPTS: readonly string[] = [
  'brain', 'neuron', 'synapse', 'memory', 'recall',
  'sleep', 'consciousness', 'mind', 'thought', 'idea',
  'attention', 'concentration', 'perception', 'feeling', 'learning',
];

/**
 * Descriptive adjectives (20 entries).
 *
 * Biological basis:
 *   Adjectives are represented in the anterior temporal lobe as
 *   dimensions of a continuous semantic space (Patterson et al., 2007).
 *   Antonym pairs (big/small, fast/slow) share nearby cortical regions.
 */
const DESCRIPTIVE: readonly string[] = [
  'big', 'small', 'fast', 'slow', 'new',
  'old', 'good', 'bad', 'pretty', 'ugly',
  'strong', 'weak', 'high', 'low', 'long',
  'short', 'light', 'dark', 'cold', 'hot',
];

/**
 * Connectors and prepositions (15 entries).
 *
 * Biological basis:
 *   Function words are processed predominantly in Broca's area
 *   (BA44/45); Broca's aphasics have specific difficulties with them
 *   (agrammatism).
 */
const CONNECTORS: readonly string[] = [
  'and', 'or', 'but', 'because', 'when',
  'if', 'that', 'as', 'where', 'for',
  'by', 'with', 'without', 'between', 'about',
];

/**
 * Interrogative words (8 entries).
 *
 * Biological basis:
 *   Questions involve a special prosody (rising intonational contour)
 *   processed in the right hemisphere.
 */
const QUESTIONS: readonly string[] = [
  'what', 'how', 'when', 'where', 'who',
  'which', 'much', 'why',
];

/**
 * High-frequency concrete nouns (25 entries).
 *
 * Biological basis:
 *   Concrete nouns ("house", "hand", "sun") have a concreteness
 *   advantage (Paivio, 1971): processed faster and remembered better,
 *   activating both verbal and imagistic representations.
 */
const NOUNS: readonly string[] = [
  'person', 'house', 'world', 'time', 'life',
  'day', 'night', 'water', 'sun', 'moon',
  'sky', 'earth', 'sea', 'light', 'color',
  'music', 'sound', 'voice', 'word', 'story',
  'path', 'name', 'hand', 'eye', 'heart',
];

/**
 * Short responses and discourse fillers (10 entries).
 *
 * Biological basis:
 *   Fillers ("hmm", "uh") and automatic responses ("yes", "no") are
 *   overlearned motor units in the basal ganglia, surviving even severe
 *   aphasias.
 */
const RESPONSES: readonly string[] = [
  'yes', 'no', 'maybe', 'sure', 'fine',
  'wrong', 'okay', 'ok', 'hmm', 'ah',
];

/**
 * Adverbs and temporal expressions (10 entries).
 *
 * Biological basis:
 *   Temporal processing involves the cerebellum and dorsolateral
 *   prefrontal cortex. Temporal words anchor events on the subjective
 *   timeline, essential for episodic memory and planning.
 */
const TIME: readonly string[] = [
  'today', 'yesterday', 'tomorrow', 'now', 'before',
  'after', 'always', 'never', 'soon', 'late',
];

/**
 * Basic cardinal numbers (10 entries).
 *
 * Biological basis:
 *   Numbers are processed in the intraparietal sulcus (Dehaene, 1997)
 *   with a logarithmic representation of magnitude. Number words are the
 *   interface between language and the numerical magnitude system.
 */
const NUMBERS: readonly string[] = [
  'one', 'two', 'three', 'four', 'five',
  'six', 'seven', 'eight', 'nine', 'ten',
];

// ==================================================================
// Main seeding function
// ==================================================================

/**
 * Populates the lexicon with a seed vocabulary of ~190 English words,
 * mirroring the Spanish categories so the brain is bilingual.
 *
 * Biological basis:
 *   English and Spanish engrams share one distributed pattern space
 *   (same n-gram encoder), modeling how a bilingual brain stores both
 *   lexicons in overlapping left-hemisphere language areas.
 *
 * Note on collisions:
 *   A few tokens are identical across languages (e.g. "no", "ok", "hey").
 *   `lexicon.add` is idempotent for identical patterns, so re-adding the
 *   same spelling simply reaffirms the existing engram.
 *
 * @param lexicon - Lexicon instance where the words will be inserted
 */
export function seedEnglishLexicon(lexicon: Lexicon): void {
  /** Neural pattern size, taken from the lexicon instance */
  const size = lexicon.dimensions;

  /** All the semantic categories to seed (English mirror). */
  const allCategories: readonly (readonly string[])[] = [
    GREETINGS,
    PRONOUNS,
    EMOTIONS,
    ACTIONS,
    BRAIN_CONCEPTS,
    DESCRIPTIVE,
    CONNECTORS,
    QUESTIONS,
    NOUNS,
    RESPONSES,
    TIME,
    NUMBERS,
  ];

  for (const category of allCategories) {
    for (const word of category) {
      const pattern = wordToPattern(word, size);
      lexicon.add(word, pattern);
    }
  }
}
