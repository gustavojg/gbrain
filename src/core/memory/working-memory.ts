/**
 * Working Memory (7±2 slots)
 * ================================
 * Models working memory according to the Baddeley & Hitch model.
 *
 * Biology: The prefrontal cortex maintains ~7±2 elements active simultaneously
 * (Miller, 1956). Each "slot" is a neural activation pattern sustained by
 * recurrent activity in prefrontal-parietal circuits. Without rehearsal
 * (attentional refresh), the patterns decay in ~15-30 seconds.
 *
 * Implementation: Each slot stores a pattern (Float32Array), a label,
 * its timestamp and a refresh counter. Unrefreshed patterns
 * lose strength and are evicted when their intensity falls below a threshold.
 */

/**
 * Individual working memory slot.
 * Represents an item actively maintained in awareness.
 */
export interface WorkingMemorySlot {
  /** Neural activation pattern representing the content */
  pattern: Float32Array;
  /** Semantic label of the content (e.g. "rostro_conocido", "palabra_escuchada") */
  label: string;
  /** Timestamp of the last access/refresh (ms) */
  timestamp: number;
  /** Number of times this pattern has been refreshed (rehearsal) */
  refreshCount: number;
  /** Current strength of the pattern (1.0 = maximum, decays without refresh) */
  strength: number;
}

/**
 * Working memory system with limited capacity.
 *
 * Implements the classic 7±2 slot model with temporal decay.
 * Patterns that are not attended to (refreshed) decay exponentially
 * and are eventually evicted, freeing space for new information.
 */
export class WorkingMemory {
  /** Active working memory slots */
  private slots: WorkingMemorySlot[] = [];
  /** Maximum number of available slots (7±2) */
  public readonly maxSlots: number;
  /** Minimum strength threshold; below this value the slot is evicted */
  private readonly evictionThreshold: number = 0.15;
  /** Temporal decay constant (ms). Half-life ~20s */
  private readonly decayHalfLife: number = 20_000;

  /**
   * Creates a new working memory system.
   *
   * @param maxSlots - Maximum capacity (default 7, per Miller 1956)
   */
  constructor(maxSlots: number = 7) {
    if (maxSlots <= 0) {
      throw new Error(`[WorkingMemory] maxSlots debe ser > 0, recibido: ${maxSlots}`);
    }
    this.maxSlots = maxSlots;
  }

  /**
   * Attends to (adds or refreshes) a pattern in working memory.
   *
   * Biology: Equivalent to the prefrontal rehearsal process. If the pattern
   * already exists (by cosine similarity with an existing slot), its
   * strength and timestamp are refreshed. If it is new and there is space, it is added.
   * If there is no space, the weakest slot is evicted.
   *
   * @param pattern - Neural activation vector of the pattern to maintain
   * @param label - Descriptive semantic label
   * @returns true if it was added/refreshed successfully
   */
  attend(pattern: Float32Array, label: string): boolean {
    const now = Date.now();

    // Look for an existing slot with the same label to refresh
    const existingIndex = this.slots.findIndex(s => s.label === label);

    if (existingIndex >= 0) {
      // Rehearsal: refresh the existing pattern
      const slot = this.slots[existingIndex];
      slot.pattern.set(pattern);
      slot.timestamp = now;
      slot.refreshCount++;
      slot.strength = Math.min(1.0, slot.strength + 0.3);
      return true;
    }

    // Create a new slot
    const newSlot: WorkingMemorySlot = {
      pattern: new Float32Array(pattern),
      label,
      timestamp: now,
      refreshCount: 0,
      strength: 1.0,
    };

    if (this.slots.length < this.maxSlots) {
      // There is available space
      this.slots.push(newSlot);
    } else {
      // Evict the weakest slot (lowest strength)
      let weakestIndex = 0;
      let weakestStrength = this.slots[0].strength;

      for (let i = 1; i < this.slots.length; i++) {
        if (this.slots[i].strength < weakestStrength) {
          weakestStrength = this.slots[i].strength;
          weakestIndex = i;
        }
      }

      this.slots[weakestIndex] = newSlot;
    }

    return true;
  }

  /**
   * Applies temporal decay to all slots.
   *
   * Biology: Without recurrent prefrontal activity to keep the patterns
   * active, the neural representation fades due to interference and
   * synaptic noise. The decay constant models this as an
   * exponential with a half-life of ~20 seconds.
   *
   * @param dt - Time elapsed since the last tick (ms)
   */
  decay(dt: number): void {
    // Exponential decay factor: strength *= e^(-dt * ln(2) / halfLife)
    const decayFactor = Math.exp((-dt * Math.LN2) / this.decayHalfLife);

    // Iterate in reverse to be able to remove elements safely
    for (let i = this.slots.length - 1; i >= 0; i--) {
      this.slots[i].strength *= decayFactor;

      // Evict if it fell below the threshold
      if (this.slots[i].strength < this.evictionThreshold) {
        this.slots.splice(i, 1);
      }
    }
  }

  /**
   * Returns the patterns currently maintained in working memory.
   *
   * @returns Copy of the active slots, ordered by descending strength
   */
  getActive(): ReadonlyArray<WorkingMemorySlot> {
    // Return a copy ordered by strength (strongest first)
    return [...this.slots].sort((a, b) => b.strength - a.strength);
  }

  /**
   * Looks up a slot by its label.
   *
   * @param label - Label of the slot to look up
   * @returns The slot if it exists, undefined otherwise
   */
  find(label: string): WorkingMemorySlot | undefined {
    return this.slots.find(s => s.label === label);
  }

  /**
   * Indicates whether working memory is full.
   * When full, new patterns evict the weakest one.
   */
  isFull(): boolean {
    return this.slots.length >= this.maxSlots;
  }

  /**
   * Completely clears working memory.
   *
   * Biology: Equivalent to an abrupt distraction or task switch
   * that frees all prefrontal resources.
   */
  clear(): void {
    this.slots = [];
  }

  /**
   * Returns the number of slots currently occupied.
   */
  get activeCount(): number {
    return this.slots.length;
  }
}
