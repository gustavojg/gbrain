/**
 * Memory Consolidation Engine (Sleep Replay)
 * ==========================================================
 * Implements the process of consolidating memories from the hippocampus to the cortex.
 *
 * Biology: During slow-wave sleep (NREM), the hippocampus "replays"
 * recent memories (sharp-wave ripples at ~200Hz) toward the corresponding
 * cortical regions. This compressed replay (~20x faster than real time)
 * strengthens the cortical connections that will store the memory long-term.
 *
 * Process:
 * 1. Select recent hippocampal memories ordered by relevance
 * 2. For each memory, identify the associated cortical region
 * 3. Replay the activation pattern as input to the region with active learning
 * 4. Repeat multiple times (cyclic reactivation during sleep)
 *
 * The net result is the gradual transfer of mnemonic storage
 * from the hippocampus (fast but temporary) to the cortex (slow but durable).
 */

import type { BrainRegion } from '../brain-region.js';
import type { ModulationEffects } from '../neuromodulators/modulator-system.js';

/**
 * Hippocampal short-term memory entry.
 * Represents a memorized episode that can be consolidated.
 */
export interface ShortTermEntry {
  /** Stored neural activation pattern */
  pattern: Float32Array;
  /** Semantic label of the memory */
  label: string;
  /** Acquisition timestamp (ms) */
  timestamp: number;
  /** Strength of the mnemonic trace (0.0 - 1.0) */
  strength: number;
  /** Identifier of the associated cortical region */
  associatedRegion: string;
  /** Number of times this memory has been replayed */
  replayCount: number;
}

/**
 * Statistics of the consolidation process.
 */
export interface ConsolidationStats {
  /** Total number of memories replayed */
  memoriesReplayed: number;
  /** Number of synapses strengthened during the replay */
  synapsesStrengthened: number;
  /** Total duration of the process in ms of simulation */
  duration: number;
  /** Memories that were successfully consolidated */
  consolidatedLabels: string[];
  /** Memories that were discarded due to weakness */
  prunedLabels: string[];
}

/**
 * Consolidation engine that implements hippocampal replay.
 *
 * Simulates the process of memory consolidation during sleep,
 * replaying hippocampal patterns toward the associated cortices
 * to strengthen long-term storage.
 */
export class ConsolidationEngine {
  /** Number of replay cycles per memory (models the NREM slow waves) */
  private readonly replayCycles: number;

  /** Minimum strength threshold to consolidate (weaker memories are lost) */
  private readonly minimumStrength: number;

  /** Modulation effects during sleep (high acetylcholine, low cortisol) */
  private readonly sleepModulationEffects: ModulationEffects;

  /**
   * Creates a new consolidation engine.
   *
   * @param replayCycles - Number of times each memory is replayed (default: 5)
   * @param minimumStrength - Minimum strength for a memory to be consolidated (default: 0.2)
   */
  constructor(replayCycles: number = 5, minimumStrength: number = 0.2) {
    this.replayCycles = replayCycles;
    this.minimumStrength = minimumStrength;

    // During NREM sleep:
    // - LOW acetylcholine (enables hippocampus → cortex replay)
    // - LOW cortisol (no stress)
    // - Moderate learning (consolidation, not acquisition)
    this.sleepModulationEffects = {
      learningRateMultiplier: 1.2,
      thresholdMultiplier: 0.8, // Lower threshold to facilitate reactivation
      attentionGain: 0.5,       // Reduced attention (asleep)
      consolidationRate: 1.5,   // Amplified consolidation
      spikeGainMultiplier: 0.7, // Reduced global activity
      socialWeightBoost: 1.0,   // Neutral during sleep
    };
  }

  /**
   * Runs the memory consolidation process.
   *
   * Biology: Simulates multiple cycles of sharp-wave ripples during
   * NREM sleep. Each ripple replays a compressed (~20x) hippocampal
   * episode toward the corresponding cortical region,
   * allowing the cortical synaptic weights to be adjusted
   * gradually to store the memory long-term.
   *
   * @param hippocampalMemories - Hippocampal memories to consolidate
   * @param regions - Map of available brain regions
   * @param modulationOverride - Optional modulation effects (override of the sleep default)
   * @returns Statistics of the consolidation process
   */
  consolidate(
    hippocampalMemories: ShortTermEntry[],
    regions: Map<string, BrainRegion>,
    modulationOverride?: ModulationEffects
  ): ConsolidationStats {
    const startTime = performance.now();
    const effects = modulationOverride ?? this.sleepModulationEffects;

    let synapsesStrengthened = 0;
    const consolidatedLabels: string[] = [];
    const prunedLabels: string[] = [];

    // Sort memories by descending strength (strongest first)
    const sortedMemories = [...hippocampalMemories].sort(
      (a, b) => b.strength - a.strength
    );

    let memoriesReplayed = 0;

    for (const memory of sortedMemories) {
      // Prune memories that are too weak
      if (memory.strength < this.minimumStrength) {
        prunedLabels.push(memory.label);
        continue;
      }

      // Find the associated cortical region
      const targetRegion = regions.get(memory.associatedRegion);
      if (!targetRegion) {
        // Region not found, skip
        continue;
      }

      // Verify that the pattern is compatible with the region
      if (memory.pattern.length !== targetRegion.inputs) {
        continue;
      }

      // Cyclic replay: replay the pattern multiple times
      for (let cycle = 0; cycle < this.replayCycles; cycle++) {
        // Feed the pattern into the region's sensory buffer
        targetRegion.feedInput(memory.pattern);

        // Run one processing step with sleep modulation
        const activity = targetRegion.step(1, effects);

        // Count strengthened synapses (neurons that fired = active synapses)
        synapsesStrengthened += activity.activeNeurons.length;
        memoriesReplayed++;
      }

      // Increment the memory's replay counter
      memory.replayCount += this.replayCycles;
      consolidatedLabels.push(memory.label);
    }

    const duration = performance.now() - startTime;

    return {
      memoriesReplayed,
      synapsesStrengthened,
      duration,
      consolidatedLabels,
      prunedLabels,
    };
  }

  /**
   * Runs a selective consolidation cycle.
   *
   * Unlike consolidate(), this method consolidates only
   * memories associated with a strong emotion (strength > high threshold),
   * modeling how emotional memories are preferentially consolidated.
   *
   * @param hippocampalMemories - Hippocampal memories
   * @param regions - Brain regions
   * @param emotionalThreshold - Strength threshold for selective consolidation (default: 0.7)
   * @returns Statistics of the process
   */
  consolidateEmotional(
    hippocampalMemories: ShortTermEntry[],
    regions: Map<string, BrainRegion>,
    emotionalThreshold: number = 0.7
  ): ConsolidationStats {
    // Filter only memories with high emotional strength
    const emotionalMemories = hippocampalMemories.filter(
      m => m.strength >= emotionalThreshold
    );

    // Use more replay cycles for emotional memories (the amygdala potentiates)
    const emotionalEffects: ModulationEffects = {
      ...this.sleepModulationEffects,
      learningRateMultiplier: 1.8, // The amygdala amplifies plasticity
      consolidationRate: 2.0,       // Even stronger consolidation
    };

    return this.consolidate(emotionalMemories, regions, emotionalEffects);
  }
}
