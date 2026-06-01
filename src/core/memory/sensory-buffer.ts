/**
 * Sensory Memory Buffer (~250ms)
 * =====================================
 * Models the iconic (visual) and echoic (auditory) memory of the human brain.
 *
 * Biology: Sensory memory retains an almost exact copy of the stimulus
 * for ~250ms (iconic) or ~3-4s (echoic). It is implemented as a high-speed
 * ring buffer using Float32Array to store sensory
 * activation vectors with their timestamps.
 *
 * The ring buffer avoids memory allocations by reusing positions,
 * critical for real-time operations at the scale of 50K+ neurons.
 */

/**
 * Individual entry in the sensory buffer.
 * Contains the activation vector and its timestamp.
 */
export interface SensoryEntry {
  /** Sensory activation vector (spatial pattern of the stimulus) */
  data: Float32Array;
  /** Timestamp in milliseconds of the capture moment */
  timestamp: number;
}

/**
 * Sensory memory ring buffer.
 *
 * Implements a fixed-size ring buffer that stores sensory
 * vectors with timestamps. When the buffer fills up, the oldest
 * entries are automatically overwritten, mimicking the natural
 * decay of the sensory trace in the brain.
 */
export class SensoryBuffer {
  /** Contiguous storage for all sensory vectors */
  private readonly storage: Float32Array;
  /** Timestamps corresponding to each position of the ring */
  private readonly timestamps: Float64Array;
  /** Maximum capacity of the buffer (number of entries) */
  private readonly capacity: number;
  /** Dimension of each sensory vector */
  private readonly vectorSize: number;
  /** Current write index (head of the ring) */
  private head: number = 0;
  /** Number of valid entries stored */
  private count: number = 0;

  /**
   * Creates a new circular sensory buffer.
   *
   * @param capacity - Maximum number of entries the buffer can store
   * @param vectorSize - Dimension of each sensory vector (e.g. 128 for a spectrogram)
   */
  constructor(capacity: number, vectorSize: number) {
    if (capacity <= 0 || vectorSize <= 0) {
      throw new Error(
        `[SensoryBuffer] Capacidad (${capacity}) y tamaño de vector (${vectorSize}) deben ser > 0`
      );
    }
    this.capacity = capacity;
    this.vectorSize = vectorSize;
    // Flat storage: capacity * vectorSize floats contiguous in memory
    this.storage = new Float32Array(capacity * vectorSize);
    this.timestamps = new Float64Array(capacity);
  }

  /**
   * Inserts a new sensory vector into the buffer.
   *
   * Biology: Equivalent to the arrival of a new retinal frame or
   * cochlear segment at the sensory register.
   *
   * @param data - Sensory activation vector to store
   * @param timestamp - Timestamp in ms of the stimulus
   */
  push(data: Float32Array, timestamp: number): void {
    if (data.length !== this.vectorSize) {
      throw new Error(
        `[SensoryBuffer] Tamaño de datos (${data.length}) no coincide con vectorSize (${this.vectorSize})`
      );
    }

    // Compute offset in the flat storage
    const offset = this.head * this.vectorSize;
    // Copy data into the ring (O(vectorSize) operation, very fast with typed arrays)
    this.storage.set(data, offset);
    this.timestamps[this.head] = timestamp;

    // Advance the head of the ring
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  /**
   * Retrieves all entries within a recent time window.
   *
   * Biology: Simulates access to the sensory trace that still persists
   * within the iconic/echoic memory window.
   *
   * @param durationMs - Duration of the window backward from the most recent entry (ms)
   * @returns Array of sensory entries within the window, ordered chronologically
   */
  getRecent(durationMs: number): SensoryEntry[] {
    if (this.count === 0) return [];

    // Find the most recent timestamp
    const lastIndex = (this.head - 1 + this.capacity) % this.capacity;
    const latestTimestamp = this.timestamps[lastIndex];
    const cutoff = latestTimestamp - durationMs;

    const results: SensoryEntry[] = [];

    // Traverse the ring from the oldest entry to the most recent
    for (let i = 0; i < this.count; i++) {
      const idx = (this.head - this.count + i + this.capacity) % this.capacity;
      const ts = this.timestamps[idx];

      if (ts >= cutoff) {
        // Extract a copy of the vector from the flat storage
        const offset = idx * this.vectorSize;
        const data = new Float32Array(this.vectorSize);
        data.set(this.storage.subarray(offset, offset + this.vectorSize));
        results.push({ data, timestamp: ts });
      }
    }

    return results;
  }

  /**
   * Completely clears the sensory buffer.
   *
   * Biology: Equivalent to an abrupt attentional reset, like the effect
   * of a blink or a change of visual fixation (saccade).
   */
  clear(): void {
    this.storage.fill(0);
    this.timestamps.fill(0);
    this.head = 0;
    this.count = 0;
  }

  /**
   * Indicates whether the buffer has reached its maximum capacity.
   * Once full, new entries overwrite the oldest ones.
   */
  isFull(): boolean {
    return this.count >= this.capacity;
  }

  /**
   * Returns the number of valid entries currently stored.
   */
  get size(): number {
    return this.count;
  }

  /**
   * Returns the maximum capacity of the buffer.
   */
  get maxCapacity(): number {
    return this.capacity;
  }

  /**
   * Returns the dimension of each sensory vector.
   */
  get dimensions(): number {
    return this.vectorSize;
  }
}
