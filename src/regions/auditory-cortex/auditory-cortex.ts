/**
 * AUDITORY CORTEX — Processing of sound patterns
 * =====================================================
 * Refactored from 07_audio_cortex/server.ts
 *
 * Biological basis:
 *   The primary auditory cortex (A1) is organized tonotopically:
 *   the neurons are arranged according to the frequency to which they
 *   preferentially respond, creating a frequency map over the cortical
 *   surface (analogous to the retinotopic map of V1).
 *
 *   The tonotopic organization is inherited from the cochlear basilar
 *   membrane, where high frequencies activate the base and low ones the apex.
 *   The thalamus→A1 connections preserve this organization.
 *
 *   This implementation includes:
 *   - Tonotopic organization: neurons prefer specific frequencies
 *   - Voice detection: modulates processing according to the presence of voice
 *   - Contrastive learning: lateral inhibition sharpens representations
 *   - Self-learning: detects repeated patterns and assigns them labels
 *   - Short-term buffer: echoic memory for repetition detection
 *
 * Scalability:
 *   5,000 neurons × 800 inputs = 4M weights (16MB Float32).
 *   Light compared to the visual cortex.
 */

// ====================================================================
// Core imports (existing in the project)
// ====================================================================

import type { SpikePacket } from '../../core/bus/spike-bus.js';
import { SpikeBus } from '../../core/bus/spike-bus.js';
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { SpikingNeuron, createNeuronPopulation } from '../../core/snn/neuron.js';
import type { NeuronTypeName } from '../../core/snn/neuron.js';

// ====================================================================
// Auditory Cortex Types
// ====================================================================

/**
 * Voice features extracted from the spectrogram.
 *
 * Biological basis:
 *   The auditory system detects the human voice through the combination
 *   of energy in the range of the voice's fundamental frequencies
 *   (~85-300 Hz for adults) and its harmonics. A1 neurons
 *   respond selectively to these spectro-temporal patterns.
 */
export interface VoiceFeatures {
  /** Whether a human voice is detected in the signal */
  hasVoice: boolean;
  /** Spectral centroid (frequency center of mass) */
  centroid: number;
  /** Total energy of the signal */
  energy: number;
}

/**
 * Auditory memory: neuron pattern associated with a sound.
 */
export interface AuditoryMemory {
  /** Indices of active neurons for this pattern */
  pattern: Int32Array;
  /** Label of the sound */
  label: string;
  /** Strength of the memory (correlates with vocal energy) */
  strength: number;
  /** Creation timestamp */
  createdAt: number;
}

/**
 * Result of auditory processing.
 */
export interface AuditoryProcessingResult {
  /** Result type */
  type: 'KNOWN' | 'UNKNOWN' | 'NEW_LEARNING' | 'NO_VOICE';
  /** Winning neurons */
  winners: Int32Array;
  /** Prediction (recognized label or '?') */
  prediction: string;
  /** Detected voice features */
  voiceFeatures: VoiceFeatures;
  /** Repetition score (for self-learning) */
  repetitionScore: number;
}

/**
 * Auditory cortex configuration.
 */
export interface AuditoryCortexConfig {
  /** Number of cortical neurons */
  neuronCount: number;
  /** Input size (numBands × numFrames of the spectrogram) */
  inputCount: number;
  /** Number of frequency bands in the spectrogram */
  numBands: number;
  /** Number of temporal frames */
  numFrames: number;
  /** Sparsity of the encoding */
  sparsity: number;
  /** Number of winning neurons in k-WTA */
  kWinners: number;
  /** Homeostatic fatigue factor */
  fatigueFactor: number;
  /** Base learning rate */
  learningRate: number;
  /** Learning boost for vocal frequencies */
  voiceFreqBoost: number;
  /** Maximum weight budget per neuron */
  maxWeightBudget: number;
  /** Contrastive inhibition factor */
  contrastiveInhibitionFactor: number;
  /** Minimum energy threshold for processing */
  minEnergyThreshold: number;
  /** Repetition threshold for self-learning */
  autoLearnThreshold: number;
  /** Size of the short-term buffer */
  shortTermBufferSize: number;
  /** Vocal band range [start, end] */
  voiceBandRange: [number, number];
  /** High-frequency band range [start, end] */
  highFreqRange: [number, number];
  /** Overlap threshold for recognition */
  recognitionOverlap: number;
  /** Neuron type */
  neuronType: NeuronTypeName;
}

const DEFAULT_AUDITORY_CONFIG: AuditoryCortexConfig = {
  neuronCount: 5000,
  inputCount: 800,           // 40 bands × 20 frames
  numBands: 40,
  numFrames: 20,
  sparsity: 0.1,
  kWinners: 3,
  fatigueFactor: 0.3,
  learningRate: 0.3,
  voiceFreqBoost: 1.5,
  maxWeightBudget: 15.0,
  contrastiveInhibitionFactor: 0.05,
  minEnergyThreshold: 0.5,
  autoLearnThreshold: 25,
  shortTermBufferSize: 60,
  voiceBandRange: [4, 24],
  highFreqRange: [30, 39],
  recognitionOverlap: 2,
  neuronType: 'RegularSpiking',
};

// ====================================================================
// AuditoryCortex class
// ====================================================================

/**
 * Auditory Cortex — Processing of sound patterns with vocal specialization.
 *
 * Biological basis:
 *   The auditory cortex processes sounds with tonotopic organization
 *   (each neuron "prefers" a range of frequencies). It specializes
 *   in detecting spectro-temporal patterns such as the human voice.
 *
 *   Implemented mechanisms:
 *
 *   1. **Tonotopic organization**: Neurons organized by preferred
 *      frequency. The neurons in the lower 10% respond to low pitches,
 *      those in the upper 10% to high pitches.
 *      Biology: A1 has a tonotopic map inherited from the cochlea.
 *
 *   2. **Voice detection**: The vocal frequencies (bands 4-24 of 40)
 *      receive a learning boost of ×1.5. Voice is detected when
 *      the low-mid energy dominates over the high-frequency noise.
 *      Biology: The superior temporal sulcus has voice-selective neurons.
 *
 *   3. **Contrastive learning**: Non-winners with residual
 *      activation receive an inhibition that reduces their weights for the
 *      current input. This sharpens the separation between representations
 *      of distinct sounds (e.g.: "A" vs "O").
 *      Biology: Lateral inhibition via GABAergic interneurons.
 *
 *   4. **Self-learning**: When an unknown pattern repeats
 *      consistently (score ≥ 25), it is learned automatically with
 *      a generated label ("Voice-N"). Models the implicit
 *      learning of phonemes in infants.
 *      Biology: Statistical learning in the auditory cortex.
 */
export class AuditoryCortex extends BrainRegion {
  /**
   * Win counter per neuron (homeostasis).
   *
   * Biological basis:
   *   Prevents "winner dominance" where a few neurons monopolize
   *   all the representations. Analogous to the homeostatic synaptic
   *   scaling observed in cultures of cortical neurons.
   */
  private winCounts: Int32Array;

  /** Stored auditory memories */
  private memories: AuditoryMemory[] = [];

  /** Cortex configuration */
  private config: AuditoryCortexConfig;

  /** Reference to the spike bus */
  private spikeBus: SpikeBus | null = null;

  /**
   * Short-term buffer (echoic memory).
   *
   * Biological basis:
   *   Echoic memory is an auditory sensory buffer that retains
   *   the last ~2-4 seconds of audio for comparison.
   *   Here we store the last ~60 winner patterns to
   *   detect repetitions that indicate a consistent pattern.
   */
  private shortTermBuffer: Int32Array[] = [];

  /** Counter of self-learning events performed */
  private autoLearnCounter: number = 0;

  /** Last computed winners */
  private lastWinners: Int32Array = new Int32Array(0);

  /** Last computed potentials (for contrastive learning) */
  private lastPotentials: Float32Array = new Float32Array(0);

  /** Local spiking neurons of the auditory cortex (Izhikevich model) */
  private localNeurons: SpikingNeuron[];

  /**
   * Creates a new auditory cortex.
   *
   * @param config - Partial configuration (merged with defaults)
   */
  constructor(config: Partial<AuditoryCortexConfig> = {}) {
    const cfg = { ...DEFAULT_AUDITORY_CONFIG, ...config };
    super('auditoryCortex', 'Corteza Auditiva', cfg.neuronCount, cfg.inputCount);

    this.config = cfg;
    this.winCounts = new Int32Array(cfg.neuronCount);
    this.localNeurons = createNeuronPopulation(cfg.neuronCount, cfg.neuronType);
    this.initializeAuditoryWeights();
  }

  /**
   * Initializes weights with small values.
   *
   * Biological basis:
   *   Weak initialization simulates immature pre-experience synapses.
   *   Lower values (0.05) than the visual cortex to avoid saturation
   *   with the wider dynamic range of audio.
   */
  private initializeAuditoryWeights(): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = Math.random() * 0.05;
    }
  }

  /**
   * Connects the auditory cortex to the spike bus.
   */
  connectBus(bus: SpikeBus): void {
    this.spikeBus = bus;
    bus.register(this.id);
  }

  /**
   * Computes voice features from the spectrogram.
   *
   * Biological basis:
   *   The neurons of the superior temporal sulcus (STS) are selective
   *   to the human voice. They detect the combination of:
   *   - Sufficient energy (signal > background noise)
   *   - Concentration in low-mid frequencies (human voice: 85-3000 Hz)
   *   - Low relative energy in high frequencies (it is not white noise)
   *
   * @param inputPattern - Flat spectrogram [time × freq]
   * @returns Detected vocal features
   */
  calculateVoiceFeatures(inputPattern: Float32Array): VoiceFeatures {
    const { numBands, voiceBandRange, highFreqRange } = this.config;

    // Last frame of the spectrogram (most recent)
    const lastFrameStart = inputPattern.length - numBands;
    const lastFrame = inputPattern.subarray(
      Math.max(0, lastFrameStart),
      inputPattern.length,
    );

    // 1. Total energy
    let energy = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      energy += lastFrame[i];
    }

    // 2. Spectral centroid (frequency center of mass)
    let weightedSum = 0;
    let sum = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      weightedSum += i * lastFrame[i];
      sum += lastFrame[i];
    }
    const centroid = sum > 0 ? weightedSum / sum : 0;

    // 3. Energy in vocal bands (low-mid)
    let lowMidEnergy = 0;
    for (let i = voiceBandRange[0]; i <= voiceBandRange[1] && i < lastFrame.length; i++) {
      lowMidEnergy += lastFrame[i];
    }

    // 4. Energy in high frequencies (noise)
    let highFreqEnergy = 0;
    for (let i = highFreqRange[0]; i <= highFreqRange[1] && i < lastFrame.length; i++) {
      highFreqEnergy += lastFrame[i];
    }

    // Voice criteria
    const isNotJustNoise = highFreqEnergy < (energy * 0.6);
    const hasVoice = energy > this.config.minEnergyThreshold &&
      lowMidEnergy > (energy * 0.25) &&
      isNotJustNoise;

    return { hasVoice, centroid, energy };
  }

  /**
   * Processes an auditory input received from the thalamus.
   *
   * Pipeline:
   * 1. Voice detection (gate: if there is no voice, do not process)
   * 2. Feed-forward with voice weighting and homeostasis
   * 3. k-WTA (winner-take-all)
   *
   * @param input - Spectrogram as a spike vector (Float32Array)
   * @param dt - Time step (ms)
   * @param timestamp - Simulation time (ms)
   * @returns Cortical activation vector
   */
  processInput(spikes: Float32Array, _modulationEffects: ModulationEffects): Float32Array {
    // Gate: only process if there is a significant signal
    const voiceFeatures = this.calculateVoiceFeatures(spikes);

    const localPotentials = new Float32Array(this.neuronCount);
    const activeSpikes = new Float32Array(this.neuronCount);

    if (!voiceFeatures.hasVoice) {
      // No voice → silent cortex (all neurons at rest)
      for (let i = 0; i < this.localNeurons.length; i++) {
        this.localNeurons[i].fired = false;
      }
      this.lastWinners = new Int32Array(0);
      this.lastPotentials = localPotentials;
      return activeSpikes;
    }

    // --- Feed-forward with activation threshold ---
    for (let n = 0; n < this.neuronCount; n++) {
      let sum = 0;
      const offset = n * this.inputCount;

      // Sparse dot product: only significant values (> 0.1)
      for (let i = 0; i < this.inputCount; i++) {
        if (spikes[i] > 0.1) {
          sum += spikes[i] * this.weights[offset + i];
        }
      }

      // Homeostasis: softer fatigue than the visual cortex
      const fatigue = this.winCounts[n] * this.config.fatigueFactor;
      localPotentials[n] = Math.max(0, sum - fatigue);
    }

    // --- Winner-Take-All ---
    const indices = new Int32Array(this.neuronCount);
    for (let i = 0; i < this.neuronCount; i++) indices[i] = i;
    indices.sort((a, b) => localPotentials[b] - localPotentials[a]);

    const winners = new Int32Array(this.config.kWinners);
    for (let i = 0; i < this.config.kWinners; i++) {
      winners[i] = indices[i];
    }

    // Update neuronal state
    for (let i = 0; i < this.localNeurons.length; i++) {
      this.localNeurons[i].fired = false;
    }
    for (let i = 0; i < this.config.kWinners; i++) {
      const winnerIdx = winners[i];
      this.localNeurons[winnerIdx].fired = true;
      activeSpikes[winnerIdx] = 1.0;
    }

    this.lastWinners = winners;
    this.lastPotentials = localPotentials;
    return activeSpikes;
  }

  /**
   * Learns an auditory pattern with a label.
   *
   * Biological basis:
   *   Supervised learning of auditory patterns with:
   *
   *   1. **Vocal boost**: The frequencies in the human voice range
   *      (bands 4-24) receive ×1.5 stronger learning,
   *      modeling the specialization of the STS for voice.
   *
   *   2. **Dopamine modulation**: Reward (high dopamine)
   *      amplifies the learning rate. Biologically, dopamine
   *      facilitates LTP in corticostriatal pathways.
   *
   *   3. **Contrastive inhibition (lateral inhibition)**:
   *      The non-winners that had residual activation receive
   *      a suppression of their weights. This sharpens the distinction
   *      between similar phonemes (e.g.: /b/ vs /d/).
   *      Biology: Lateral inhibition via PV+ interneurons.
   *
   *   4. **Global decay**: All weights decay slowly
   *      (×0.995), modeling natural synaptic degradation.
   *
   * @param input - Spectrogram as a vector
   * @param label - Label of the sound
   * @param dt - Time step
   * @param timestamp - Simulation time
   * @param modulationEffects - Neuromodulation effects
   * @returns Processing result
   */
  learn(
    input: Float32Array,
    label: string,
    dt: number,
    timestamp: number,
    modulationEffects?: ModulationEffects,
  ): AuditoryProcessingResult {
    // Check input size
    if (input.length !== this.inputCount) {
      return {
        type: 'NO_VOICE',
        winners: new Int32Array(0),
        prediction: 'ERROR',
        voiceFeatures: { hasVoice: false, centroid: 0, energy: 0 },
        repetitionScore: 0,
      };
    }

    const voiceFeatures = this.calculateVoiceFeatures(input);
    if (!voiceFeatures.hasVoice) {
      return {
        type: 'NO_VOICE',
        winners: new Int32Array(0),
        prediction: '?',
        voiceFeatures,
        repetitionScore: 0,
      };
    }

    // 1. Process input
    const defaultMod: ModulationEffects = {
      learningRateMultiplier: 1.0,
      thresholdMultiplier: 1.0,
      attentionGain: 1.0,
      spikeGainMultiplier: 1.0,
      consolidationRate: 1.0,
      socialWeightBoost: 1.0,
    };
    this.processInput(input, defaultMod);
    const winners = this.lastWinners;
    const potentials = this.lastPotentials;

    // 2. Learning rate modulated by dopamine
    const lrMultiplier = modulationEffects?.learningRateMultiplier ?? 1.0;
    const effectiveLR = this.config.learningRate + (lrMultiplier - 1.0) * 0.4;

    // 3. Selective learning with vocal boost
    const { numBands, voiceBandRange } = this.config;

    for (let w = 0; w < winners.length; w++) {
      const winnerIdx = winners[w];
      this.winCounts[winnerIdx]++;

      const offset = winnerIdx * this.inputCount;
      let totalWeight = 0;

      for (let i = 0; i < this.inputCount; i++) {
        if (input[i] > 0.15) {
          // Determine whether this bin is in the vocal range
          const freqIndex = i % numBands;
          const isVoiceFreq = freqIndex >= voiceBandRange[0] && freqIndex <= voiceBandRange[1];
          const boost = isVoiceFreq ? this.config.voiceFreqBoost : 0.5;

          // LTP with vocal boost
          this.weights[offset + i] += input[i] * effectiveLR * boost;
        }

        // Global decay (natural synaptic degradation)
        this.weights[offset + i] *= 0.995;

        // Clamp to non-negative (the auditory cortex does not use negative weights)
        if (this.weights[offset + i] < 0) this.weights[offset + i] = 0;
        totalWeight += this.weights[offset + i];
      }

      // Synaptic normalization
      if (totalWeight > this.config.maxWeightBudget) {
        const factor = this.config.maxWeightBudget / totalWeight;
        for (let i = 0; i < this.inputCount; i++) {
          this.weights[offset + i] *= factor;
        }
      }
    }

    // 4. Store auditory memory
    this.memories.push({
      pattern: new Int32Array(winners),
      label,
      strength: voiceFeatures.energy,
      createdAt: timestamp,
    });

    // 5. Contrastive learning (lateral inhibition)
    this.applyContrastiveInhibition(input, winners, potentials, effectiveLR);

    // 6. Recognition
    const predictions = this.recognizePattern(winners);

    return {
      type: predictions.length > 0 ? 'KNOWN' : 'UNKNOWN',
      winners,
      prediction: predictions.length > 0 ? predictions.join(', ') : '?',
      voiceFeatures,
      repetitionScore: 0,
    };
  }

  /**
   * Applies contrastive inhibition to the non-winners.
   *
   * Biological basis:
   *   The inhibitory PV+ (parvalbumin-positive) interneurons in
   *   the auditory cortex provide feedforward and lateral inhibition
   *   that sharpens frequency selectivity. The non-winners that
   *   had residual activation are suppressed, reinforcing the
   *   separation between phoneme representations.
   *
   * @param input - Spectrogram input
   * @param winners - Indices of winning neurons
   * @param potentials - Potentials of all neurons
   * @param learningRate - Effective learning rate
   */
  private applyContrastiveInhibition(
    input: Float32Array,
    winners: Int32Array,
    potentials: Float32Array,
    learningRate: number,
  ): void {
    const inhibitionFactor = this.config.contrastiveInhibitionFactor * learningRate;
    const winnerSet = new Set<number>();
    for (let w = 0; w < winners.length; w++) {
      winnerSet.add(winners[w]);
    }

    for (let n = 0; n < this.neuronCount; n++) {
      // Only affect neurons that had activation but did not win
      if (!winnerSet.has(n) && potentials[n] > 0) {
        const offset = n * this.inputCount;

        // Slightly reduce the weights for active inputs
        for (let i = 0; i < this.inputCount; i++) {
          if (input[i] > 0.1) {
            this.weights[offset + i] -= input[i] * inhibitionFactor;
            if (this.weights[offset + i] < 0) {
              this.weights[offset + i] = 0;
            }
          }
        }
      }
    }
  }

  /**
   * Recognizes a pattern by comparing it with stored memories.
   *
   * Biological basis:
   *   Auditory recognition uses engram overlap with a dynamic
   *   threshold based on the strength of the memory. Stronger memories
   *   (learned with more vocal energy) require less overlap.
   *
   * @param currentWinners - Currently active neurons
   * @returns List of recognized labels
   */
  private recognizePattern(currentWinners: Int32Array): string[] {
    const matches: string[] = [];
    const seenLabels = new Set<string>();

    for (const mem of this.memories) {
      let overlap = 0;

      for (let w = 0; w < currentWinners.length; w++) {
        for (let m = 0; m < mem.pattern.length; m++) {
          if (currentWinners[w] === mem.pattern[m]) {
            overlap++;
            break;
          }
        }
      }

      // Dynamic threshold based on the strength of the memory
      const threshold = Math.max(
        this.config.recognitionOverlap,
        3 - (mem.strength / 10),
      );

      if (overlap >= threshold && !seenLabels.has(mem.label)) {
        matches.push(mem.label);
        seenLabels.add(mem.label);
      }
    }

    return matches;
  }

  /**
   * Automatic unsupervised processing (self-learning).
   *
   * Biological basis:
   *   Infants learn to distinguish the phonemes of their native language
   *   without explicit supervision, through repeated exposure.
   *   The brain detects statistical regularities in the auditory
   *   input and forms perceptual categories automatically.
   *
   *   We implement this by detecting patterns that repeat
   *   consistently in a short-term buffer (echoic memory).
   *   When the repetition score exceeds a threshold, it is
   *   self-learned with a generated label.
   *
   * @param input - Spectrogram as a vector
   * @param dt - Time step
   * @param timestamp - Simulation time
   * @returns Processing result with repetition detection
   */
  autoProcess(input: Float32Array, dt: number, timestamp: number): AuditoryProcessingResult {
    const voiceFeatures = this.calculateVoiceFeatures(input);

    // No voice or very little energy → ignore
    if (!voiceFeatures.hasVoice || voiceFeatures.energy < this.config.minEnergyThreshold * 1.6) {
      return {
        type: 'NO_VOICE',
        winners: new Int32Array(0),
        prediction: '?',
        voiceFeatures,
        repetitionScore: 0,
      };
    }

    // Process input
    const defaultMod2: ModulationEffects = {
      learningRateMultiplier: 1.0,
      thresholdMultiplier: 1.0,
      attentionGain: 1.0,
      spikeGainMultiplier: 1.0,
      consolidationRate: 1.0,
      socialWeightBoost: 1.0,
    };
    this.processInput(input, defaultMod2);
    const winners = this.lastWinners;

    // Check whether it is already known
    const predictions = this.recognizePattern(winners);
    if (predictions.length > 0) {
      return {
        type: 'KNOWN',
        winners,
        prediction: predictions.join(', '),
        voiceFeatures,
        repetitionScore: 0,
      };
    }

    // --- Repetition detection for self-learning ---
    this.shortTermBuffer.push(new Int32Array(winners));
    if (this.shortTermBuffer.length > this.config.shortTermBufferSize) {
      this.shortTermBuffer.shift();
    }

    let repetitionScore = 0;
    // Examine recent history (~1.5 seconds)
    const recentCount = Math.min(30, this.shortTermBuffer.length);
    const recentStart = this.shortTermBuffer.length - recentCount;

    for (let h = recentStart; h < this.shortTermBuffer.length; h++) {
      const pastWinners = this.shortTermBuffer[h];
      let overlap = 0;

      for (let w = 0; w < winners.length; w++) {
        for (let p = 0; p < pastWinners.length; p++) {
          if (winners[w] === pastWinners[p]) {
            overlap++;
            break;
          }
        }
      }

      // Granular scoring system
      if (overlap >= 2) {
        repetitionScore += 3;     // Strong match
      } else if (overlap >= 1) {
        repetitionScore += 1.5;   // Partial match
      }
    }

    // Self-learn if it repeats consistently
    if (repetitionScore >= this.config.autoLearnThreshold) {
      this.autoLearnCounter++;
      const newLabel = `Voice-${this.autoLearnCounter}`;

      // Learn with an automatic label
      this.learn(input, newLabel, dt, timestamp);

      // Partially clear the buffer to avoid relearning immediately
      this.shortTermBuffer = this.shortTermBuffer.slice(
        -Math.floor(this.config.shortTermBufferSize / 4),
      );

      return {
        type: 'NEW_LEARNING',
        winners,
        prediction: newLabel,
        voiceFeatures,
        repetitionScore,
      };
    }

    return {
      type: 'UNKNOWN',
      winners,
      prediction: '?',
      voiceFeatures,
      repetitionScore,
    };
  }

  /**
   * Sends the current cortical representation to the spike bus.
   *
   * @param timestamp - Simulation time
   * @param targets - Destination regions (default: hippocampus + amygdala)
   */
  emitToDownstream(timestamp: number, targets: string[] = ['hippocampus', 'amygdala']): void {
    if (!this.spikeBus) return;

    const spikes = new Float32Array(this.neuronCount);
    for (let i = 0; i < this.lastWinners.length; i++) {
      spikes[this.lastWinners[i]] = 1.0;
    }

    this.spikeBus.send({
      source: this.id,
      targets,
      spikes,
      timestamp,
      metadata: {
        winnerCount: this.lastWinners.length,
        activity: this.getLocalActivity(),
        autoLearnCount: this.autoLearnCounter,
      },
    });
  }

  /**
   * Gets statistics of the auditory cortex.
   */
  /**
   * Gets the fraction of local activity (0-1) based on the spiking neurons.
   */
  getLocalActivity(): number {
    let active = 0;
    for (let i = 0; i < this.localNeurons.length; i++) {
      if (this.localNeurons[i].fired) active++;
    }
    return active / this.localNeurons.length;
  }

  getStats(): {
    memoryCount: number;
    averageWeight: number;
    maxFatigue: number;
    autoLearnCount: number;
    shortTermBufferSize: number;
    activity: number;
  } {
    let weightSum = 0;
    for (let i = 0; i < this.weights.length; i++) {
      weightSum += this.weights[i];
    }

    let maxFatigue = 0;
    for (let i = 0; i < this.winCounts.length; i++) {
      if (this.winCounts[i] > maxFatigue) maxFatigue = this.winCounts[i];
    }

    return {
      memoryCount: this.memories.length,
      averageWeight: weightSum / this.weights.length,
      maxFatigue,
      autoLearnCount: this.autoLearnCounter,
      shortTermBufferSize: this.shortTermBuffer.length,
      activity: this.getLocalActivity(),
    };
  }

  /** Number of stored memories */
  get memoryCount(): number {
    return this.memories.length;
  }

  /**
   * Resets the homeostatic fatigue.
   *
   * Biological basis:
   *   Equivalent to the restorative effect of sleep on the
   *   synaptic homeostasis of the auditory cortex.
   */
  resetFatigue(): void {
    this.winCounts.fill(0);
  }

  /**
   * Clears the short-term buffer (echoic memory).
   */
  clearShortTermBuffer(): void {
    this.shortTermBuffer = [];
  }
}
