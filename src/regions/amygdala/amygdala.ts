/**
 * AMYGDALA — Emotional Processing Center
 * ===============================================
 * Models the amygdaloid complex of the temporal lobe, responsible for
 * the emotional evaluation of stimuli and the generation of affective
 * responses that modulate all brain processing.
 *
 * Biological basis:
 *   The human amygdala contains ~12 million neurons organized into
 *   several nuclei with specific functions:
 *
 *   - Lateral nucleus (LA): Entry gate. Receives sensory afferents
 *     from the thalamus (fast subcortical pathway, "low road") and from
 *     sensory cortices (slow cortical pathway, "high road"). Evaluates the
 *     emotional relevance of the stimulus.
 *
 *   - Basolateral nucleus (BLA): Integrates sensory information with
 *     emotional memory. Stores learned stimulus-emotion associations
 *     (classical conditioning, LeDoux 2000).
 *
 *   - Central nucleus (CeA): Efferent output. Projects to:
 *     * Hypothalamus → autonomic responses (heart rate, cortisol)
 *     * Brainstem → motor responses (freezing, fight/flight)
 *     * Neuromodulatory nuclei → global modulation (DA, NE, 5-HT, cortisol)
 *
 *   The amygdala is the brain's emotional "alarm center": it can
 *   hijack attention and cognitive resources when it detects
 *   potentially threatening or highly salient stimuli.
 *
 * Reference: LeDoux, J. (2000). "Emotion circuits in the brain."
 *             Annual Review of Neuroscience, 23, 155-184.
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';

// ==================================================================
// Interfaces
// ==================================================================

/**
 * Two-dimensional emotional state based on Russell's circumplex model.
 *
 * Biological basis:
 *   Emotions are represented as points in a two-dimensional space:
 *   - Valence (valence): pleasure/displeasure dimension, encoded by the
 *     balance of activity between positive and negative BLA nuclei
 *   - Arousal: level of physiological activation, encoded by the firing
 *     rate of the CeA and its autonomic projections
 *
 *   Russell, J.A. (1980). "A circumplex model of affect."
 */
export interface EmotionalState {
  /**
   * Emotional valence: -1 (very negative/aversive) to +1 (very positive/pleasant).
   * Neutral = 0.
   */
  valence: number;
  /**
   * Activation/arousal level: 0 (deep calm) to 1 (maximum activation).
   * Modulates the intensity of the emotional response.
   */
  arousal: number;
}

/**
 * Conditioned emotional memory stored in the amygdala.
 *
 * Biological basis:
 *   Fear conditioning (and that of other emotions) occurs through long-term
 *   potentiation (LTP) at the synapses of the lateral nucleus of the amygdala.
 *   These memories are highly resistant to extinction and can be reactivated
 *   even years later (e.g.: PTSD).
 */
export interface EmotionalMemory {
  /** Conditioned stimulus pattern (CS) */
  pattern: Float32Array;
  /** Conditioned emotional response (CR) */
  emotion: EmotionalState;
  /**
   * Strength of the stimulus-emotion association (0–1).
   * Increases with repetition, decreases with extinction.
   */
  strength: number;
}

/**
 * Neuromodulatory releases produced by the amygdala.
 *
 * Biological basis:
 *   The amygdala modulates the activity of the brain's main
 *   neuromodulatory systems through its efferent projections
 *   to the hypothalamus, VTA, locus coeruleus and raphe nuclei.
 */
export interface NeuromodulatorRelease {
  /** Dopamine: reward, motivation. VTA/SNc. */
  dopamine: number;
  /** Serotonin: emotional regulation, well-being. Raphe nuclei. */
  serotonin: number;
  /** Norepinephrine: alertness, attention. Locus coeruleus. */
  norepinephrine: number;
  /** Cortisol: stress response. HPA axis (hypothalamus). */
  cortisol: number;
  /** Acetylcholine: focused attention. Nucleus basalis of Meynert. */
  acetylcholine: number;
  /** Oxytocin: social bonding, trust. Hypothalamus. */
  oxytocin: number;
}

// ==================================================================
// Amygdala class
// ==================================================================

/**
 * Amygdala — Center of emotional evaluation and affective modulation.
 *
 * Biological basis:
 *   Implements the three main functions of the amygdaloid complex:
 *   1. Fast evaluation of the emotional valence of stimuli (LA/BLA)
 *   2. Storage of conditioned emotional memories (BLA)
 *   3. Production of neuromodulatory signals that affect the whole brain (CeA)
 *
 *   The amygdala operates in two modes:
 *   - Reactive: fast evaluation of novel or previously conditioned
 *     stimuli (latency ~12ms via the direct thalamo-amygdala pathway)
 *   - Modulatory: continuous adjustment of the emotional tone of brain
 *     processing via neuromodulators
 */
export class Amygdala extends BrainRegion {
  /** Current emotional state of the system */
  private emotionalState: EmotionalState = { valence: 0, arousal: 0.2 };

  /** Conditioned emotional memories (stimulus-emotion associations) */
  private emotionalMemories: EmotionalMemory[] = [];

  /** Last neuromodulator release produced */
  private lastRelease: NeuromodulatorRelease = {
    dopamine: 0, serotonin: 0, norepinephrine: 0,
    cortisol: 0, acetylcholine: 0, oxytocin: 0,
  };

  /**
   * Maximum capacity of emotional memories.
   * Biology: the amygdala has a practically unlimited capacity for
   * fear conditioning, but we model a practical limit.
   */
  private readonly maxEmotionalMemories: number = 5000;

  /**
   * Novelty threshold: if no emotional memory has a similarity
   * above this value, the stimulus is considered "new".
   */
  private readonly noveltyThreshold: number = 0.3;

  /**
   * Emotional inertia: how much of the previous state is preserved on each tick.
   * Biology: emotions do not change instantaneously; there is inertia
   * due to the slow release of neuropeptides and hormones.
   */
  private readonly emotionalInertia: number = 0.7;

  /**
   * Creates the amygdala.
   *
   * @param neuronCount - Number of neurons (default: 2000)
   * @param inputCount - Dimensionality of the input (default: 3000)
   */
  constructor(neuronCount: number = 2000, inputCount: number = 3000) {
    super(
      'amygdala',
      'Amígdala — Procesamiento Emocional',
      neuronCount,
      inputCount,
    );
  }

  // ----------------------------------------------------------------
  // Emotional Evaluation
  // ----------------------------------------------------------------

  /**
   * Evaluates the emotional valence and arousal of a stimulus.
   *
   * Biological basis:
   *   The lateral nucleus (LA) of the amygdala receives convergent inputs from
   *   the sensory thalamus and the association cortices. LA neurons
   *   respond to stimuli that:
   *   1. Match previous emotional memories (conditioned response)
   *   2. Are novel or unexpected (orienting response)
   *   3. Have intrinsically aversive properties (e.g.: pain, loud sounds)
   *
   *   The BLA then integrates this evaluation with the context and the current
   *   motivational state to produce a modulated emotional response.
   *
   * @param input - Spike vector of the stimulus to evaluate
   * @param modulationEffects - Current neuromodulation effects
   * @returns Emotional state evaluated for this stimulus
   */
  evaluateStimulus(
    input: Float32Array,
    modulationEffects: ModulationEffects,
  ): EmotionalState {
    let newValence = 0;
    let newArousal = 0.1; // Minimum baseline arousal (wakeful state)

    // --- Search in emotional memories (conditioned response) ---
    let bestMatchSimilarity = 0;
    let bestMatchEmotion: EmotionalState | null = null;

    for (const memory of this.emotionalMemories) {
      const similarity = this.cosineSimilarity(input, memory.pattern);
      const adjustedSim = similarity * memory.strength;

      if (adjustedSim > bestMatchSimilarity) {
        bestMatchSimilarity = adjustedSim;
        bestMatchEmotion = memory.emotion;
      }
    }

    if (bestMatchEmotion && bestMatchSimilarity > this.noveltyThreshold) {
      // --- Familiar stimulus with emotional association ---
      newValence = bestMatchEmotion.valence * bestMatchSimilarity;
      newArousal = bestMatchEmotion.arousal * bestMatchSimilarity;

      // Amplify arousal for familiar negative patterns
      // Biology: the amygdala is especially sensitive to known threats
      if (bestMatchEmotion.valence < -0.3) {
        newArousal = Math.min(1.0, newArousal * 1.5);
      }
    } else {
      // --- Novel stimulus ---
      // Biology: novelty generates a slight positive response (curiosity)
      // mediated by the mesolimbic dopaminergic system
      newValence = 0.1; // Slight positive bias (curiosity)
      newArousal = 0.3 + (1 - bestMatchSimilarity) * 0.2; // More novelty = more arousal

      // Modulated attention increases arousal in response to novelty
      newArousal *= modulationEffects.attentionGain;
    }

    // Apply emotional inertia (temporal smoothing)
    const finalValence =
      this.emotionalState.valence * this.emotionalInertia +
      newValence * (1 - this.emotionalInertia);
    const finalArousal =
      this.emotionalState.arousal * this.emotionalInertia +
      newArousal * (1 - this.emotionalInertia);

    // Clamp to valid ranges
    this.emotionalState = {
      valence: Math.max(-1, Math.min(1, finalValence)),
      arousal: Math.max(0, Math.min(1, finalArousal)),
    };

    return { ...this.emotionalState };
  }

  // ----------------------------------------------------------------
  // Neuromodulator Production
  // ----------------------------------------------------------------

  /**
   * Produces neuromodulatory signals based on the current emotional state.
   *
   * Biological basis:
   *   The central nucleus (CeA) of the amygdala is the brain's main
   *   emotional effector. Its efferent projections activate/inhibit
   *   the subcortical neuromodulatory nuclei:
   *
   *   - High positive valence → VTA (dopamine ↑)
   *     Reward activates dopaminergic neurons of the ventral tegmental area
   *
   *   - High arousal + negative valence → LC (norepinephrine ↑) + HPA (cortisol ↑)
   *     Stress/threat activates the locus coeruleus and the hypothalamic-pituitary-adrenal axis
   *
   *   - Novel stimulus → LC (moderate norepinephrine ↑)
   *     Novelty generates an orienting response mediated by NE
   *
   *   - Positive valence + familiarity → Raphe (serotonin ↑) + PVN (oxytocin ↑)
   *     Safety and comfort activate the well-being and bonding systems
   *
   *   - Attention required → NB (acetylcholine ↑)
   *     Salient stimuli activate the nucleus basalis of Meynert
   *
   * @returns Release amounts for each neuromodulator (0–1)
   */
  produceNeuromodulators(): NeuromodulatorRelease {
    const { valence, arousal } = this.emotionalState;

    // --- Dopamine: positive valence ---
    // VTA dopamine neurons fire in response to reward prediction errors
    const dopamine = Math.max(0, valence) * 0.5 + (valence > 0.5 ? 0.2 : 0);

    // --- Serotonin: emotional stability, familiar positive bias ---
    // The raphe nuclei maintain serotonergic tone modulated by safety
    const serotonin =
      (valence > 0 ? valence * 0.3 : 0) + (1 - arousal) * 0.2;

    // --- Norepinephrine: arousal, novelty, alertness ---
    // LC tonic and phasic firing modulated by arousal and novelty
    const norepinephrine =
      arousal * 0.4 + (valence < -0.3 ? Math.abs(valence) * 0.3 : 0);

    // --- Cortisol: stress (high arousal + negative valence) ---
    // HPA axis activation during sustained negative affect
    const cortisol =
      valence < 0 ? Math.abs(valence) * arousal * 0.5 : 0;

    // --- Acetylcholine: focused attention (salient stimuli) ---
    // NBM activation proportional to stimulus salience
    const acetylcholine = arousal * 0.3 + Math.abs(valence) * 0.2;

    // --- Oxytocin: social comfort, positive valence low arousal ---
    // PVN oxytocin release during safe social contexts
    const oxytocin =
      valence > 0.2 && arousal < 0.5 ? valence * 0.4 : 0;

    this.lastRelease = {
      dopamine: Math.min(1, Math.max(0, dopamine)),
      serotonin: Math.min(1, Math.max(0, serotonin)),
      norepinephrine: Math.min(1, Math.max(0, norepinephrine)),
      cortisol: Math.min(1, Math.max(0, cortisol)),
      acetylcholine: Math.min(1, Math.max(0, acetylcholine)),
      oxytocin: Math.min(1, Math.max(0, oxytocin)),
    };

    return { ...this.lastRelease };
  }

  // ----------------------------------------------------------------
  // Emotional Conditioning
  // ----------------------------------------------------------------

  /**
   * Conditions an emotional response to a stimulus.
   *
   * Biological basis:
   *   Conditioning occurs through LTP at the LA → BLA synapses.
   *   A neutral stimulus (CS) that co-occurs with an emotional stimulus
   *   (US) acquires the ability to evoke the emotional response by
   *   itself. This learning is:
   *   - Fast (it can occur in a single exposure for aversive stimuli)
   *   - Resistant to extinction (associations weaken but are not erased)
   *   - Modulated by NE and cortisol (stressful stimuli are conditioned more strongly)
   *
   * @param stimulus - Conditioned stimulus pattern (CS)
   * @param emotion - Emotional response to associate (CR)
   */
  conditionResponse(stimulus: Float32Array, emotion: EmotionalState): void {
    // Check whether a similar association already exists
    for (const memory of this.emotionalMemories) {
      const similarity = this.cosineSimilarity(stimulus, memory.pattern);
      if (similarity > 0.8) {
        // Update existing association (reconsolidation)
        memory.emotion.valence =
          memory.emotion.valence * 0.5 + emotion.valence * 0.5;
        memory.emotion.arousal =
          memory.emotion.arousal * 0.5 + emotion.arousal * 0.5;
        memory.strength = Math.min(1.0, memory.strength + 0.1);
        return;
      }
    }

    // New association
    if (this.emotionalMemories.length >= this.maxEmotionalMemories) {
      // Remove the weakest association
      let weakestIdx = 0;
      let weakestStrength = this.emotionalMemories[0].strength;
      for (let i = 1; i < this.emotionalMemories.length; i++) {
        if (this.emotionalMemories[i].strength < weakestStrength) {
          weakestStrength = this.emotionalMemories[i].strength;
          weakestIdx = i;
        }
      }
      this.emotionalMemories.splice(weakestIdx, 1);
    }

    this.emotionalMemories.push({
      pattern: new Float32Array(stimulus),
      emotion: { ...emotion },
      strength: 1.0,
    });
  }

  // ----------------------------------------------------------------
  // Main processing
  // ----------------------------------------------------------------

  /**
   * Processes input spikes and produces emotionally modulated spikes.
   *
   * Biological basis:
   *   Amygdaloid processing flow:
   *   1. Input arrives via the fast pathway (thalamus→LA) and slow pathway (cortex→LA)
   *   2. LA evaluates valence and arousal by comparing with memories
   *   3. BLA integrates with context and produces an emotional state
   *   4. CeA translates the emotional state into neuromodulatory signals
   *   5. Output: output spikes + neuromodulatory signals
   *
   * @param spikes - Input spike vector
   * @param modulationEffects - Current neuromodulation effects
   * @returns Emotionally tinted output spike vector
   */
  processInput(
    spikes: Float32Array,
    modulationEffects: ModulationEffects,
  ): Float32Array {
    // Adapt dimensionality
    const adaptedInput = this.adaptInput(spikes);

    // 1. Evaluate the emotional content of the stimulus
    this.evaluateStimulus(adaptedInput, modulationEffects);

    // 2. Produce neuromodulatory signals
    this.produceNeuromodulators();

    // 3. Generate output spikes modulated by emotion
    const output = new Float32Array(this.neuronCount);
    const { valence, arousal } = this.emotionalState;

    // The magnitude of the response scales with arousal
    // Biology: the CeA modulates the gain of its outputs by arousal
    const responseGain = 0.5 + arousal * 0.5;

    // Compute currents and generate spikes
    const copyLen = Math.min(adaptedInput.length, this.neuronCount);
    for (let i = 0; i < copyLen; i++) {
      // The output spikes carry the emotional "mark"
      output[i] = adaptedInput[i] * responseGain;
    }

    // Add a valence signal as modulation of the spikes
    // Biology: CeA neurons encode valence in their firing rate
    const valenceSignalStart = Math.min(copyLen, this.neuronCount - 100);
    for (let i = valenceSignalStart; i < this.neuronCount; i++) {
      // The last neurons encode valence and arousal directly
      const t = (i - valenceSignalStart) / Math.max(1, this.neuronCount - valenceSignalStart);
      output[i] = t < 0.5 ? (valence + 1) / 2 * arousal : arousal;
    }

    // Apply neuromodulation gain
    for (let i = 0; i < output.length; i++) {
      output[i] *= modulationEffects.spikeGainMultiplier;
    }

    return output;
  }

  // ----------------------------------------------------------------
  // Utilities
  // ----------------------------------------------------------------

  /**
   * Adapts an input vector to the expected dimensionality.
   */
  private adaptInput(input: Float32Array): Float32Array {
    if (input.length === this.inputCount) return input;

    const adapted = new Float32Array(this.inputCount);
    const copyLen = Math.min(input.length, this.inputCount);
    for (let i = 0; i < copyLen; i++) {
      adapted[i] = input[i];
    }
    return adapted;
  }

  /**
   * Cosine similarity between two vectors.
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < len; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  // ----------------------------------------------------------------
  // Public accessors
  // ----------------------------------------------------------------

  /** Returns the current emotional state */
  getEmotionalState(): EmotionalState {
    return { ...this.emotionalState };
  }

  /** Returns the last neuromodulator release */
  getLastRelease(): NeuromodulatorRelease {
    return { ...this.lastRelease };
  }

  /** Number of conditioned emotional memories */
  get conditionedMemoryCount(): number {
    return this.emotionalMemories.length;
  }
}
