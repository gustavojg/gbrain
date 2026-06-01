/**
 * WERNICKE'S AREA — Language Comprehension
 * =============================================
 * Models Wernicke's area (posterior part of the superior temporal gyrus,
 * Brodmann area 22) of the digital brain, responsible for the comprehension
 * of spoken and written language.
 *
 * Biological basis:
 *   Wernicke's area was identified by Carl Wernicke in 1874 upon
 *   observing that lesions in the posterior part of the superior temporal gyrus
 *   caused "receptive aphasia": patients could speak fluently
 *   but did not comprehend language. Their verbal utterances were
 *   grammatically correct but semantically empty ("word
 *   salad").
 *
 *   Modeled functions:
 *   1. **Auditory/visual → lexical decoding**: Maps activation
 *      patterns (coming from the thalamus) to entries in the mental lexicon.
 *      Cohort model of lexical access (Marslen-Wilson, 1987).
 *
 *   2. **Semantic comprehension**: Extracts meaning from sequences of
 *      activated lexical patterns. Wernicke's neurons respond
 *      selectively to semantic combinations (not just individual
 *      words).
 *
 *   3. **Lexical learning**: New words are incorporated into the shared
 *      lexicon, creating new neural engrams in the temporal lobe.
 *
 *   Connections:
 *   - Receives: spikes from the thalamus (processed auditory/visual signals)
 *   - Sends: semantic representations to the prefrontal cortex and to
 *     Broca's area (via the arcuate fasciculus)
 *   - Shares: lexicon with Broca's area
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { Lexicon, type LexiconMatch } from './lexicon.js';

/**
 * Activation floor for the k-WTA. The "potentials" here are sums of
 * dimensionless weights (~0 at rest), not membrane mV. Without this floor the
 * gate would compare against `_modulatedThreshold` (−55 mV), always pass, and
 * the k-WTA would fire exactly k neurons per tick → constant and false
 * activity. With the floor, at rest (no drive) the region stays at 0%.
 */
const ACTIVATION_FLOOR = 1e-3;

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Result of comprehending a spike pattern.
 *
 * Biological basis:
 *   Each result represents a candidate lexical activation,
 *   analogous to the cohort of lexical candidates that are
 *   partially activated when processing a speech signal (cohort model).
 */
export interface ComprehensionResult {
  /** Word identified in the lexicon */
  word: string;
  /** Degree of similarity with the input pattern (0.0–1.0) */
  similarity: number;
}

/**
 * Semantic representation produced by Wernicke's processing.
 */
export interface SemanticRepresentation {
  /** Neural pattern of the semantic representation */
  pattern: Float32Array;
  /** Comprehended words and their similarities */
  comprehendedWords: ComprehensionResult[];
  /** Overall comprehension confidence (0.0–1.0) */
  confidence: number;
}

// ==================================================================
// WernickeArea class
// ==================================================================

/**
 * Wernicke's area — Linguistic comprehension of the digital brain.
 *
 * Processes spike patterns received from the thalamus (preprocessed
 * auditory/visual signals) and maps them to semantic representations
 * through lexical access by similarity. Shares a neural lexicon
 * with Broca's area.
 *
 * Biological basis:
 *   Wernicke's neurons are organized tonotopically and respond
 *   selectively to phonetic categories and semantic combinations.
 *   Comprehension occurs in ~200ms (N400 ERP component for
 *   semantic violations). The internal SNN models the cortical
 *   columns of the superior temporal gyrus with lateral competition.
 */
export class WernickeArea extends BrainRegion {
  /**
   * Lexicon shared with Broca's area.
   *
   * Biology: The mental lexicon is a shared store in the temporal
   * lobe, accessed via the ventral pathway (Wernicke → comprehension) and
   * the dorsal pathway (Broca → production).
   */
  private readonly lexicon: Lexicon;

  /**
   * Last semantic representation produced.
   * Kept so that other regions can query it.
   */
  private lastSemanticOutput: SemanticRepresentation | null = null;

  /**
   * Minimum similarity threshold to consider a lexical match valid.
   *
   * Biology: Models the lexical activation threshold. Patterns that do not
   * exceed this threshold do not activate any lexical engram (unknown
   * word or noise).
   */
  private readonly comprehensionThreshold: number = 0.15;

  /**
   * Creates Wernicke's area of the digital brain.
   *
   * Biology: Wernicke's area contains ~150 million neurons.
   * We model 5,000 neurons with 5,000 inputs, reflecting the columnar
   * structure of the superior temporal gyrus.
   *
   * @param lexicon - Shared lexicon (injected to share with Broca)
   */
  constructor(lexicon: Lexicon, neuronCount: number = 5000, inputCount: number = 5000) {
    super(
      'wernicke',
      'Área de Wernicke — Comprensión Lingüística',
      neuronCount,
      inputCount
    );
    this.lexicon = lexicon;
  }

  // ----------------------------------------------------------------
  // Linguistic Comprehension
  // ----------------------------------------------------------------

  /**
   * Comprehends a spike pattern by mapping it to the closest lexical entries.
   *
   * Biological basis:
   *   Models auditory/visual lexical access. Upon receiving an
   *   activation pattern (e.g.: acoustic representation processed by the
   *   auditory cortex), Wernicke compares it against all the stored lexical
   *   engrams. The most similar ones are activated, forming a "cohort"
   *   of candidates (cohort model, Marslen-Wilson 1987). Lateral
   *   competition resolves the ambiguity by selecting the
   *   strongest match.
   *
   *   The lexical frequency effect is built into the lexicon's
   *   findClosest() method (frequent words have an advantage).
   *
   * @param inputSpikes - Spike pattern to comprehend
   * @returns Array of candidate words with their similarity, ordered descending
   */
  comprehend(inputSpikes: Float32Array): ComprehensionResult[] {
    if (this.lexicon.size === 0) return [];

    // Find the closest matches in the lexicon
    const matches: LexiconMatch[] = this.lexicon.findClosest(inputSpikes, 5);

    // Filter by comprehension threshold
    return matches
      .filter(m => m.similarity >= this.comprehensionThreshold)
      .map(m => ({
        word: m.word,
        similarity: m.similarity,
      }));
  }

  /**
   * Learns a new word by adding it to the shared lexicon.
   *
   * Biological basis:
   *   Lexical learning occurs when a new activation pattern
   *   (without a sufficient match in the existing lexicon) is presented
   *   repeatedly and associated with a meaning. In the brain, this
   *   involves the formation of a new neural engram in the temporal
   *   lobe, initially dependent on the hippocampus and gradually
   *   consolidated in the neocortex (complementary learning systems theory,
   *   McClelland et al., 1995).
   *
   * @param word - Word to learn
   * @param pattern - Associated neural activation pattern
   */
  learnWord(word: string, pattern: Float32Array): void {
    this.lexicon.add(word, pattern);
  }

  /**
   * Returns the last semantic representation produced.
   *
   * @returns Semantic representation or null if nothing has been processed
   */
  getLastSemanticOutput(): SemanticRepresentation | null {
    return this.lastSemanticOutput;
  }

  /**
   * Returns a read-only reference to the shared lexicon.
   */
  getLexicon(): Lexicon {
    return this.lexicon;
  }

  // ----------------------------------------------------------------
  // processInput — Linguistic comprehension cycle
  // ----------------------------------------------------------------

  /**
   * Processes one simulation step of Wernicke's area.
   *
   * Biological basis:
   *   Wernicke's comprehension cycle operates in ~200ms and consists of:
   *
   *   1. **Reception**: Receive spikes from the thalamus (auditory/
   *      visual signals preprocessed by primary sensory cortices)
   *
   *   2. **Lexical access**: Map the input pattern to lexical
   *      candidates through similarity comparison (cohort)
   *
   *   3. **SNN processing**: Pass the spikes through the local neural
   *      network to produce a distributed semantic representation.
   *      The k-WTA competition in the SNN models the selection of the
   *      winning lexical candidate.
   *
   *   4. **Semantic integration**: Combine the input with the
   *      lexical matches to produce a unified semantic representation
   *      that is sent to the prefrontal cortex and Broca's area.
   *
   * @param spikes - Input spike vector from the thalamus
   * @param modulationEffects - Current neuromodulation effects
   * @returns Output spike vector (semantic representation)
   */
  processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects
  ): Float32Array {
    // 1. Lexical access: comprehend the input pattern
    const comprehended = this.comprehend(spikes);

    // 2. Compute activations of the local SNN
    const activations = new Float32Array(this.neuronCount);
    const inputLen = Math.min(spikes.length, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      const baseOffset = n * this.inputCount;
      let sum = 0;

      for (let j = 0; j < inputLen; j++) {
        sum += this.weights[baseOffset + j] * spikes[j];
      }

      // Attentional modulation: acetylcholine amplifies linguistic signals
      sum *= modulationEffects.attentionGain;

      // Simplified LIF integration
      this.potentials[n] += sum;
      this.potentials[n] *= 0.93; // Temporal leak

      activations[n] = this.potentials[n];
    }

    // 3. k-WTA competition: selection of the winning representation
    const k = Math.max(1, Math.floor(this.neuronCount * this.sparsity));
    const outputSpikes = new Float32Array(this.neuronCount);

    // Find the k-th threshold
    const sortedActivations = new Float32Array(activations);
    sortedActivations.sort();
    const kwtaThreshold = sortedActivations[this.neuronCount - k];

    for (let i = 0; i < this.neuronCount; i++) {
      if (activations[i] >= kwtaThreshold && activations[i] > ACTIVATION_FLOOR) {
        outputSpikes[i] = 1.0;
      }
    }

    // 4. Update spike state
    this.spikes.set(outputSpikes);

    // 5. Build the semantic representation
    let confidence = 0;
    if (comprehended.length > 0) {
      // Confidence based on the best lexical match
      confidence = comprehended[0].similarity;
    }

    this.lastSemanticOutput = {
      pattern: new Float32Array(outputSpikes),
      comprehendedWords: comprehended,
      confidence,
    };

    // 6. Hebbian learning: strengthen the connections that contributed
    //    to successful comprehension
    if (confidence > 0.3 && modulationEffects.learningRateMultiplier > 0) {
      this.hebbianUpdate(spikes, outputSpikes);
    }

    return outputSpikes;
  }

  /**
   * Hebbian update of synaptic weights for lexical learning.
   *
   * Biological basis:
   *   The synapses in the superior temporal gyrus are strengthened when
   *   there is correlation between the input pattern (acoustic/visual signal)
   *   and the activation of neurons that represent the recognized word.
   *   This progressively refines the lexical representation (tuning).
   *
   * @param preSpikes - Input spikes (presynaptic)
   * @param postSpikes - Output spikes (postsynaptic)
   */
  private hebbianUpdate(preSpikes: Float32Array, postSpikes: Float32Array): void {
    const lr = this._modulatedLearningRate * 0.005;
    const inputLen = Math.min(preSpikes.length, this.inputCount);

    for (let n = 0; n < this.neuronCount; n++) {
      if (postSpikes[n] === 0) continue;

      const baseOffset = n * this.inputCount;
      for (let j = 0; j < inputLen; j++) {
        if (preSpikes[j] > 0) {
          this.weights[baseOffset + j] += lr * preSpikes[j];
        }

        // Weight clamp
        if (this.weights[baseOffset + j] > 1.0) {
          this.weights[baseOffset + j] = 1.0;
        } else if (this.weights[baseOffset + j] < 0) {
          this.weights[baseOffset + j] = 0;
        }
      }
    }
  }
}
