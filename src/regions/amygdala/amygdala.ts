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
/**
 * A conditioned emotional response to a sensory CUE: the population code of a
 * category (its engram's units) that has been paired with an emotional event.
 */
export interface CueMemory {
  modality: string;
  units: number[];
  emotion: EmotionalState;
  /** 0..1: the association's strength; extinction lowers it, never to zero. */
  strength: number;
}

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
   * Conditioned SEMANTIC associations (word pattern → emotion), in the
   * lexicon's representational space. Kept apart from `emotionalMemories`,
   * which live in the space of the sensory afferents.
   */
  private semanticAssociations: EmotionalMemory[] = [];

  /** Minimum similarity for a semantic stimulus to evoke its conditioned response. */
  private static readonly SEMANTIC_MATCH = 0.8;

  /**
   * Inertia of a conditioned semantic appraisal. Lower than `emotionalInertia`:
   * conditioned responses are fast (LeDoux, 1996), so a clearly emotional word
   * moves the state more than a tick of diffuse sensory input does.
   */
  private static readonly APPRAISAL_INERTIA = 0.5;

  /**
   * Conditioned responses to sensory CUES (visual / auditory category codes).
   * Fear conditioning at the lateral nucleus: fast (one pairing for an aversive
   * event), extinguishable but never erased (extinction is new learning that
   * inhibits the memory, so it recovers under stress; Bouton 2004).
   */
  private cueMemories: CueMemory[] = [];
  /** Overlap (Jaccard) from which a cue's code evokes its conditioned response. */
  private static readonly CUE_MATCH = 0.5;
  /** Strength left after each safe exposure (extinction), and the floor it never crosses. */
  private static readonly EXTINCTION_KEEP = 0.85;
  private static readonly EXTINCTION_FLOOR = 0.1;
  /** How much stress (cortisol) restores an extinguished memory: effective = strength × (1 + stress × this). */
  private static readonly RECOVERY_GAIN = 1.0;
  /** Effective strength below which a cue evokes nothing. */
  private static readonly CUE_MIN_EFFECT = 0.2;

  /** Neurons that report the affective state (half valence, half arousal). */
  private static readonly AFFECT_POPULATION = 20;
  /** Amplitude of an affect spike relative to the amygdala's response gain. */
  private static readonly AFFECT_AMPLITUDE = 0.5;

  /** Arousal the affective state relaxes to when there is no stimulus (wakeful rest). */
  private static readonly RESTING_AROUSAL = 0.1;

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
    // The raphe nuclei maintain serotonergic tone modulated by safety. Negative
    // affect does not recruit it (low mood goes with LOW serotonergic tone), so
    // a calm-but-sad state must not read as well-being.
    const serotonin =
      valence >= 0 ? valence * 0.3 + (1 - arousal) * 0.2 : 0;

    // --- Norepinephrine: arousal, novelty, alertness ---
    // LC tonic and phasic firing modulated by arousal and novelty
    const norepinephrine =
      arousal * 0.4 + (valence < -0.3 ? Math.abs(valence) * 0.3 : 0);

    // --- Cortisol: stress (negative valence, amplified by arousal) ---
    // HPA axis activation during negative affect: strongest under threat
    // (high arousal), but sadness (low arousal) recruits it too.
    const cortisol =
      valence < 0 ? Math.abs(valence) * (0.3 + 0.7 * arousal) * 0.8 : 0;

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
  // Semantic appraisal (cortex → amygdala route)
  // ----------------------------------------------------------------

  /**
   * Conditions an emotional response to a SEMANTIC stimulus (a word's pattern
   * in lexicon space). Re-conditioning the same word updates the association.
   *
   * @param pattern - Pattern of the word in lexicon space
   * @param emotion - Emotion to associate
   */
  conditionSemantic(pattern: Float32Array, emotion: EmotionalState, strength: number = 1.0): void {
    for (const memory of this.semanticAssociations) {
      if (this.cosineSimilarity(pattern, memory.pattern) > 0.99) {
        memory.emotion = {
          valence: memory.emotion.valence * 0.5 + emotion.valence * 0.5,
          arousal: memory.emotion.arousal * 0.5 + emotion.arousal * 0.5,
        };
        memory.strength = Math.min(1.0, memory.strength + 0.15);
        return;
      }
    }
    if (this.semanticAssociations.length >= this.maxEmotionalMemories) return;
    this.semanticAssociations.push({
      pattern: new Float32Array(pattern),
      emotion: { ...emotion },
      strength: Math.max(0, Math.min(1, strength)),
    });
  }

  // ----------------------------------------------------------------
  // Innate appraisal and cue conditioning (thalamus → amygdala route)
  // ----------------------------------------------------------------

  /**
   * An innate appraisal (a tone of voice, a startle, something looming, a
   * face) pulls the affective state toward the evoked emotion. Fast route:
   * the same inertia as a conditioned response.
   */
  appraiseInnate(evoked: EmotionalState): EmotionalState {
    const inertia = Amygdala.APPRAISAL_INERTIA;
    this.emotionalState = {
      valence: Math.max(-1, Math.min(1, this.emotionalState.valence * inertia + evoked.valence * (1 - inertia))),
      arousal: Math.max(0, Math.min(1, this.emotionalState.arousal * inertia + evoked.arousal * (1 - inertia))),
    };
    return { ...this.emotionalState };
  }

  /**
   * Conditions a sensory cue (a category's population code) to an emotion.
   * Pairing the same cue again reconsolidates: the emotion is averaged and the
   * strength restored.
   */
  conditionCue(modality: string, units: ArrayLike<number>, emotion: EmotionalState, strength: number = 1.0): void {
    const code = Array.from(units);
    if (code.length === 0) return;
    const existing = this.findCue(modality, code);
    if (existing) {
      existing.memory.emotion = {
        valence: existing.memory.emotion.valence * 0.5 + emotion.valence * 0.5,
        arousal: existing.memory.emotion.arousal * 0.5 + emotion.arousal * 0.5,
      };
      existing.memory.strength = Math.min(1, Math.max(existing.memory.strength, strength));
      return;
    }
    if (this.cueMemories.length >= this.maxEmotionalMemories) {
      let weakest = 0;
      for (let i = 1; i < this.cueMemories.length; i++) {
        if (this.cueMemories[i].strength < this.cueMemories[weakest].strength) weakest = i;
      }
      this.cueMemories.splice(weakest, 1);
    }
    this.cueMemories.push({ modality, units: code, emotion: { ...emotion }, strength: Math.max(0, Math.min(1, strength)) });
  }

  /**
   * A cue has just been perceived: if it was conditioned, its emotion is
   * evoked (scaled by the memory's effective strength, which stress restores)
   * and pulls the affective state.
   *
   * @returns The evoked emotion, or `null` if the cue carries no learned affect
   */
  appraiseCue(modality: string, units: ArrayLike<number>, stress: number = 0): EmotionalState | null {
    const found = this.findCue(modality, Array.from(units));
    if (!found) return null;
    const effective = Math.min(1, found.memory.strength * (1 + Math.max(0, stress) * Amygdala.RECOVERY_GAIN)) * found.overlap;
    if (effective < Amygdala.CUE_MIN_EFFECT) return null;
    const evoked = { valence: found.memory.emotion.valence * effective, arousal: found.memory.emotion.arousal * effective };
    this.appraiseInnate(evoked);
    return evoked;
  }

  /**
   * The cue was perceived and nothing happened: extinction weakens the
   * association a little. Never to zero — extinction inhibits, it does not erase.
   */
  extinguishCue(modality: string, units: ArrayLike<number>): void {
    const found = this.findCue(modality, Array.from(units));
    if (!found) return;
    found.memory.strength = Math.max(Amygdala.EXTINCTION_FLOOR, found.memory.strength * Amygdala.EXTINCTION_KEEP);
  }

  private findCue(modality: string, code: number[]): { memory: CueMemory; overlap: number } | null {
    const set = new Set(code);
    let best: CueMemory | null = null;
    let bestOverlap = 0;
    for (const memory of this.cueMemories) {
      if (memory.modality !== modality) continue;
      let inter = 0;
      for (const u of memory.units) if (set.has(u)) inter++;
      const union = set.size + memory.units.length - inter;
      const overlap = union > 0 ? inter / union : 0;
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = memory;
      }
    }
    return best && bestOverlap >= Amygdala.CUE_MATCH ? { memory: best, overlap: bestOverlap } : null;
  }

  /** Number of conditioned sensory cues */
  get conditionedCueCount(): number {
    return this.cueMemories.length;
  }

  /** Strength of the conditioned response to a cue (0 if none): what tests and the dashboard read. */
  cueStrength(modality: string, units: ArrayLike<number>): number {
    const found = this.findCue(modality, Array.from(units));
    return found ? found.memory.strength : 0;
  }

  // ----------------------------------------------------------------
  // Persistence of what the amygdala has learned
  // ----------------------------------------------------------------

  serializeExtra(): unknown {
    return {
      semantic: this.semanticAssociations.map((m) => ({ pattern: Array.from(m.pattern), emotion: m.emotion, strength: m.strength })),
      cues: this.cueMemories.map((m) => ({ modality: m.modality, units: m.units, emotion: m.emotion, strength: m.strength })),
    };
  }

  deserializeExtra(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as { semantic?: unknown; cues?: unknown };
    const finite = (x: unknown, lo: number, hi: number): number | null =>
      typeof x === 'number' && Number.isFinite(x) ? Math.max(lo, Math.min(hi, x)) : null;
    const emotionOf = (e: unknown): EmotionalState | null => {
      if (typeof e !== 'object' || e === null) return null;
      const valence = finite((e as { valence?: unknown }).valence, -1, 1);
      const arousal = finite((e as { arousal?: unknown }).arousal, 0, 1);
      return valence === null || arousal === null ? null : { valence, arousal };
    };
    if (Array.isArray(d.semantic)) {
      this.semanticAssociations = [];
      for (const item of d.semantic.slice(0, this.maxEmotionalMemories)) {
        const m = item as { pattern?: unknown; emotion?: unknown; strength?: unknown };
        const emotion = emotionOf(m.emotion);
        const strength = finite(m.strength, 0, 1);
        if (!Array.isArray(m.pattern) || !emotion || strength === null) continue;
        const pattern = Float32Array.from(m.pattern, (v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0));
        this.semanticAssociations.push({ pattern, emotion, strength });
      }
    }
    if (Array.isArray(d.cues)) {
      this.cueMemories = [];
      for (const item of d.cues.slice(0, this.maxEmotionalMemories)) {
        const m = item as { modality?: unknown; units?: unknown; emotion?: unknown; strength?: unknown };
        const emotion = emotionOf(m.emotion);
        const strength = finite(m.strength, 0, 1);
        if (typeof m.modality !== 'string' || !Array.isArray(m.units) || !emotion || strength === null) continue;
        const units = m.units.filter((u): u is number => Number.isInteger(u) && u >= 0);
        if (units.length === 0) continue;
        this.cueMemories.push({ modality: m.modality.slice(0, 32), units, emotion, strength });
      }
    }
  }

  /**
   * Appraises a comprehended word: if its pattern matches a conditioned
   * association, the conditioned emotion is evoked (scaled by the match) and
   * pulls the affective state toward it.
   *
   * Biological basis:
   *   Besides the fast thalamic route, the amygdala receives highly processed
   *   input from the temporal association cortex; that is how a WORD — not
   *   just a loud noise — can evoke fear or joy. The response is a recall of
   *   a learned association, so it generalizes only to close variants of the
   *   conditioned stimulus ("miedo" → "miedos"), by pattern similarity.
   *
   * @param pattern - Pattern of the word in lexicon space
   * @returns The evoked emotion, or `null` if the word carries no learned affect
   */
  appraise(pattern: Float32Array): EmotionalState | null {
    let bestSimilarity = 0;
    let best: EmotionalMemory | null = null;
    for (const memory of this.semanticAssociations) {
      const similarity = this.cosineSimilarity(pattern, memory.pattern) * memory.strength;
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        best = memory;
      }
    }
    if (!best || bestSimilarity < Amygdala.SEMANTIC_MATCH) return null;

    const evoked: EmotionalState = {
      valence: best.emotion.valence * bestSimilarity,
      arousal: best.emotion.arousal * bestSimilarity,
    };
    const inertia = Amygdala.APPRAISAL_INERTIA;
    this.emotionalState = {
      valence: Math.max(-1, Math.min(1, this.emotionalState.valence * inertia + evoked.valence * (1 - inertia))),
      arousal: Math.max(0, Math.min(1, this.emotionalState.arousal * inertia + evoked.arousal * (1 - inertia))),
    };
    return evoked;
  }

  /** Number of conditioned semantic (word → emotion) associations */
  get semanticAssociationCount(): number {
    return this.semanticAssociations.length;
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

    // 0. No stimulus → nothing to evaluate. Silence is not a "novel stimulus":
    // the affective state relaxes toward its resting baseline and the region
    // stays silent (otherwise its tonic valence/arousal output re-excites the
    // thalamus every tick and the whole brain never comes to rest).
    let hasStimulus = false;
    for (let i = 0; i < adaptedInput.length; i++) {
      if (adaptedInput[i] > 0) {
        hasStimulus = true;
        break;
      }
    }
    if (!hasStimulus) {
      this.emotionalState = {
        valence: this.emotionalState.valence * this.emotionalInertia,
        arousal:
          this.emotionalState.arousal * this.emotionalInertia +
          Amygdala.RESTING_AROUSAL * (1 - this.emotionalInertia),
      };
      return new Float32Array(this.neuronCount);
    }

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

    // Affect population: the last neurons report the affective state as a
    // sparse "thermometer" code — the higher the arousal (or the valence), the
    // more neurons of its half fire. It is kept small on purpose: with 100
    // always-on graded neurons, the content-agnostic affect signal outweighed
    // the ~30 content-bearing spikes relayed to the prefrontal cortex, which
    // then responded almost identically to any stimulus.
    const affectStart = this.neuronCount - Amygdala.AFFECT_POPULATION;
    const half = Amygdala.AFFECT_POPULATION / 2;
    const valenceUnits = Math.round(((valence + 1) / 2) * half);
    const arousalUnits = Math.round(arousal * half);
    for (let i = 0; i < Amygdala.AFFECT_POPULATION; i++) {
      const fires = i < half ? i < valenceUnits : i - half < arousalUnits;
      // Same scale as the relayed content (which arrives through the weak
      // thalamic low road): affect colours the message, it does not shout over it.
      output[affectStart + i] = fires ? responseGain * Amygdala.AFFECT_AMPLITUDE : 0;
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
