/**
 * PROTOTYPE MEMORY — Unsupervised perceptual categories, learned by exposure
 * =========================================================================
 * A sensory cortex answers every stimulus with an engram (the neurons that
 * fired most while the stimulus was present). This module turns the stream of
 * engrams into perceptual categories WITHOUT labels or a teacher:
 *
 *   - an engram close enough to a known prototype is a re-encounter: the
 *     prototype is refreshed (it tracks the engram as the cortex's weights keep
 *     maturing) and its exposure count grows;
 *   - otherwise the stimulus is novel and founds a new prototype.
 *
 * Biological basis:
 *   Infants form perceptual categories from mere exposure, by picking up the
 *   statistical regularities of their input (Saffran et al., 1996; Quinn &
 *   Eimas, 1996) — long before anything has a name. Familiarity ("I have seen
 *   this before") is the first thing a perceptual system learns to signal.
 *
 * Two helpers live here:
 *   - `PresentationTracker` segments the tick-by-tick activity of a region into
 *     presentations (a stimulus held for a while, then silence) and extracts
 *     the engram of each one.
 *   - `PrototypeMemory` matches engrams against the learned prototypes.
 */

/**
 * Perceptual narrowing (critical periods): how much a fully entrenched
 * category relaxes its vigilance (`gain`), and the exposures at which a
 * category is ~63% entrenched (`tau`). Early on every contrast founds its own
 * category; once a category is well worn, nearby inputs are assimilated to it
 * and a contrast never met before is harder to learn (Werker & Tees 1984).
 */
export const PERCEPTUAL_NARROWING = { gain: 0.35, tau: 10 } as const;

/** What is left of a released neuron's synapses (pruning frees it for recruitment). */
export const RELEASE_KEEP = 0.2;

/** Outcome of showing one stimulus to a `PrototypeMemory`. */
export interface Recognition {
  /** Stable identifier of the matched (or newly founded) category. */
  id: number;
  /** Auto-generated name, e.g. "Visual-3". */
  label: string;
  /** Overlap (Jaccard, 0–1) between the engram and the prototype BEFORE updating it. 0 when novel. */
  familiarity: number;
  /** How many times this category has been encountered, this one included. */
  exposures: number;
  /**
   * How badly the category predicted this input (0..1): 1 − cosine between
   * the input and what the category's neurons reconstruct (predictive coding:
   * the error between the cortex's expectation and what arrived). Set by the
   * cortex that owns the category; absent where not measured.
   */
  surprise?: number;
  /** Whether this stimulus founded the category. */
  isNew: boolean;
  /** Simulation time of the recognition (ms). */
  timestamp: number;
}

interface Prototype {
  id: number;
  units: number[];
  exposures: number;
  lastSeen: number;
  /** Sleeps since it was last seen (pruning). */
  unseenSleeps: number;
}

/** A learned category, read-only. */
export interface CategoryView {
  id: number;
  label: string;
  units: readonly number[];
  exposures: number;
  lastSeen: number;
}

/** Serialized form of a `PrototypeMemory`. */
export interface SerializedPrototypes {
  nextId: number;
  prototypes: Prototype[];
}

export interface PrototypeMemoryOptions {
  /** Prefix of the generated labels ("Visual" → "Visual-1"). */
  labelPrefix: string;
  /** Minimum Jaccard overlap with a prototype to count as a re-encounter. */
  matchThreshold: number;
  /** Maximum number of categories; the least recently seen one is evicted. */
  capacity?: number;
  /** Units are neuron indices in [0, unitCount). Used to validate restored data. */
  unitCount: number;
}

export class PrototypeMemory {
  private prototypes: Prototype[] = [];
  private nextId = 1;
  private readonly capacity: number;

  constructor(private readonly options: PrototypeMemoryOptions) {
    this.capacity = options.capacity ?? 200;
  }

  /** Number of categories learned so far. */
  get size(): number {
    return this.prototypes.length;
  }

  /** Jaccard overlap between two engrams given as index lists. */
  static overlap(a: ArrayLike<number>, b: ArrayLike<number>): number {
    if (a.length === 0 || b.length === 0) return 0;
    const setA = new Set<number>();
    for (let i = 0; i < a.length; i++) setA.add(a[i]);
    let inter = 0;
    for (let i = 0; i < b.length; i++) if (setA.has(b[i])) inter++;
    return inter / (a.length + b.length - inter);
  }

  /**
   * Shows an engram to the memory.
   *
   * @param engram - Indices of the neurons that represent the stimulus
   * @param timestamp - Current simulation time (ms)
   * @returns The recognition, or `null` for an empty engram (nothing perceived)
   */
  observe(engram: ArrayLike<number>, timestamp: number): Recognition | null {
    if (engram.length === 0) return null;

    let best: Prototype | null = null;
    let bestOverlap = 0;
    for (const prototype of this.prototypes) {
      const overlap = PrototypeMemory.overlap(engram, prototype.units);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = prototype;
      }
    }

    if (best && bestOverlap >= this.options.matchThreshold) {
      // Re-encounter: the prototype follows the engram, which still drifts
      // while the cortex's weights are maturing.
      best.units = Array.from(engram);
      best.exposures++;
      best.lastSeen = timestamp;
      best.unseenSleeps = 0;
      return this.describe(best, bestOverlap, false);
    }

    if (this.prototypes.length >= this.capacity) {
      let oldest = 0;
      for (let i = 1; i < this.prototypes.length; i++) {
        if (this.prototypes[i].lastSeen < this.prototypes[oldest].lastSeen) oldest = i;
      }
      this.prototypes.splice(oldest, 1);
    }
    const founded: Prototype = { id: this.nextId++, units: Array.from(engram), exposures: 1, lastSeen: timestamp, unseenSleeps: 0 };
    this.prototypes.push(founded);
    return this.describe(founded, 0, true);
  }

  /**
   * Best-matching category for an engram WITHOUT learning from it — used to
   * name a code that was reinstated from memory rather than perceived.
   *
   * @returns The category and its overlap with the engram, or `null` if nothing overlaps
   */
  match(engram: ArrayLike<number>): { id: number; label: string; overlap: number; exposures: number } | null {
    let best: Prototype | null = null;
    let bestOverlap = 0;
    for (const prototype of this.prototypes) {
      const overlap = PrototypeMemory.overlap(engram, prototype.units);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = prototype;
      }
    }
    return best
      ? { id: best.id, label: `${this.options.labelPrefix}-${best.id}`, overlap: bestOverlap, exposures: best.exposures }
      : null;
  }

  /** The categories learned so far. */
  list(): CategoryView[] {
    return this.prototypes.map((p) => ({ id: p.id, label: `${this.options.labelPrefix}-${p.id}`, units: p.units, exposures: p.exposures, lastSeen: p.lastSeen }));
  }

  /**
   * How entrenched each unit is (0..1) by the exposures of the categories it
   * belongs to — the perceptual-narrowing term (Werker & Tees 1984): a
   * category met many times becomes a perceptual magnet (Kuhl 1991) that
   * captures nearby inputs instead of letting them found a category of their own.
   *
   * @param unitCount - Neurons of the cortex
   * @param tau - Exposures at which entrenchment is ~63%
   */
  entrenchment(unitCount: number, tau: number): Float32Array {
    const out = new Float32Array(unitCount);
    for (const p of this.prototypes) {
      const e = 1 - Math.exp(-p.exposures / tau);
      for (const u of p.units) if (u < unitCount && e > out[u]) out[u] = e;
    }
    return out;
  }

  /** A sleep has passed: every category is one sleep older since it was last seen. */
  age(): void {
    for (const p of this.prototypes) p.unseenSleeps++;
  }

  /**
   * Postnatal pruning: a category met fewer than `minExposures` times and not
   * seen for `maxUnseenSleeps` sleeps is dropped. What is never repeated is
   * not worth keeping neurons for.
   *
   * @returns The pruned categories (their units can be released by the cortex)
   */
  prune(minExposures: number, maxUnseenSleeps: number): CategoryView[] {
    const pruned: CategoryView[] = [];
    this.prototypes = this.prototypes.filter((p) => {
      const drop = p.exposures < minExposures && p.unseenSleeps >= maxUnseenSleeps;
      if (drop) pruned.push({ id: p.id, label: `${this.options.labelPrefix}-${p.id}`, units: p.units, exposures: p.exposures, lastSeen: p.lastSeen });
      return !drop;
    });
    return pruned;
  }

  /** Units that belong to some surviving category. */
  unitsInUse(): Set<number> {
    const used = new Set<number>();
    for (const p of this.prototypes) for (const u of p.units) used.add(u);
    return used;
  }

  private describe(prototype: Prototype, familiarity: number, isNew: boolean): Recognition {
    return {
      id: prototype.id,
      label: `${this.options.labelPrefix}-${prototype.id}`,
      familiarity,
      exposures: prototype.exposures,
      isNew,
      timestamp: prototype.lastSeen,
    };
  }

  serialize(): SerializedPrototypes {
    return { nextId: this.nextId, prototypes: this.prototypes.map((p) => ({ ...p, units: [...p.units] })) };
  }

  /** Restores from disk; entries that do not validate are dropped. */
  deserialize(data: unknown): void {
    const d = data as Partial<SerializedPrototypes> | null;
    if (!d || !Array.isArray(d.prototypes)) return;

    const restored: Prototype[] = [];
    for (const raw of d.prototypes.slice(-this.capacity)) {
      const p = raw as Partial<Prototype>;
      if (!Number.isInteger(p?.id) || !Array.isArray(p.units) || p.units.length === 0) continue;
      if (!p.units.every((u) => Number.isInteger(u) && u >= 0 && u < this.options.unitCount)) continue;
      restored.push({
        id: p.id as number,
        units: [...p.units],
        exposures: Number.isInteger(p.exposures) && (p.exposures as number) > 0 ? (p.exposures as number) : 1,
        lastSeen: typeof p.lastSeen === 'number' && Number.isFinite(p.lastSeen) ? p.lastSeen : 0,
        unseenSleeps: Number.isInteger(p.unseenSleeps) && (p.unseenSleeps as number) >= 0 ? (p.unseenSleeps as number) : 0,
      });
    }
    this.prototypes = restored;
    const maxId = restored.reduce((max, p) => Math.max(max, p.id), 0);
    this.nextId = Math.max(maxId + 1, Number.isInteger(d.nextId) ? (d.nextId as number) : 1);
  }
}

/**
 * Segments a region's tick-by-tick output into presentations and extracts the
 * engram of each: the `engramSize` neurons that fired most while the stimulus
 * was present. A presentation ends after `gapTicks` ticks without input.
 */
export class PresentationTracker {
  private readonly counts: Int32Array;
  private drivenTicks = 0;
  private silentTicks = 0;

  constructor(
    unitCount: number,
    private readonly engramSize: number,
    private readonly gapTicks: number = 10,
    private readonly minDrivenTicks: number = 3,
  ) {
    this.counts = new Int32Array(unitCount);
  }

  /** Whether a presentation is in progress. */
  get active(): boolean {
    return this.drivenTicks > 0;
  }

  /**
   * Feeds one tick.
   *
   * @param driven - Whether the region received input this tick
   * @param fired - Indices of the neurons that fired this tick
   * @returns The engram when this tick closes a presentation, otherwise `null`
   */
  tick(driven: boolean, fired: ArrayLike<number>): Int32Array | null {
    if (driven) {
      this.drivenTicks++;
      this.silentTicks = 0;
      for (let i = 0; i < fired.length; i++) this.counts[fired[i]]++;
      return null;
    }
    if (this.drivenTicks === 0) return null;
    if (++this.silentTicks < this.gapTicks) return null;

    const enough = this.drivenTicks >= this.minDrivenTicks;
    const engram = enough ? this.extract() : null;
    this.counts.fill(0);
    this.drivenTicks = 0;
    this.silentTicks = 0;
    return engram;
  }

  private extract(): Int32Array {
    const candidates: number[] = [];
    for (let i = 0; i < this.counts.length; i++) if (this.counts[i] > 0) candidates.push(i);
    candidates.sort((a, b) => this.counts[b] - this.counts[a] || a - b);
    return Int32Array.from(candidates.slice(0, this.engramSize).sort((a, b) => a - b));
  }
}
