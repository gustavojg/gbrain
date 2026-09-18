/**
 * SEQUENCE MEMORY — what tends to follow what
 * ===========================================================================
 * Experience is not a bag of things but an order: the cross is shown, THEN
 * its name is said; "coche", THEN "azul". The hippocampus and the cortex
 * learn such transitions (heteroassociative chains: pattern t predicts
 * pattern t+1; Levy 1996, Lisman 1999), and once learned, the first member
 * of a pair brings an EXPECTATION of the second — which is what makes the
 * second one unsurprising when it comes, and its absence surprising.
 *
 * This is the multi-second scale: percepts one after another. Order inside
 * a sound (a syllable, a word) lives in the spectrogram window of the ear.
 */

export interface Prediction {
  /** The percept expected next (`auditory:Sound-1`). */
  key: string;
  /** How often it has followed the current one (0..1). */
  probability: number;
  /** Typical delay, in ticks. */
  expectedInTicks: number;
}

interface Successor {
  count: number;
  gapSum: number;
}

/** Fewest occurrences of a predecessor before it predicts anything. */
const MIN_OBSERVATIONS = 2;

export class SequenceMemory {
  /** predecessor → successor → how often, and after how long. */
  private transitions = new Map<string, Map<string, Successor>>();
  private totals = new Map<string, number>();
  private last: { key: string; tick: number } | null = null;
  /** The prediction made from the last percept. */
  private current: Prediction | null = null;

  constructor(private readonly windowTicks: number) {}

  /**
   * A percept has just happened. If another one came shortly before, the
   * transition between them is learned; then what usually follows this one
   * becomes the current expectation.
   */
  observe(key: string, tick: number): { learned: { from: string; to: string } | null; prediction: Prediction | null } {
    let learned: { from: string; to: string } | null = null;
    if (this.last && this.last.key !== key && tick - this.last.tick <= this.windowTicks) {
      let successors = this.transitions.get(this.last.key);
      if (!successors) {
        successors = new Map();
        this.transitions.set(this.last.key, successors);
      }
      const successor = successors.get(key) ?? { count: 0, gapSum: 0 };
      successor.count++;
      successor.gapSum += tick - this.last.tick;
      successors.set(key, successor);
      this.totals.set(this.last.key, (this.totals.get(this.last.key) ?? 0) + 1);
      learned = { from: this.last.key, to: key };
    }
    this.last = { key, tick };
    this.current = this.predict(key);
    return { learned, prediction: this.current };
  }

  /** What usually follows `key`, if it has been seen often enough to say. */
  predict(key: string): Prediction | null {
    const total = this.totals.get(key) ?? 0;
    const successors = this.transitions.get(key);
    if (!successors || total < MIN_OBSERVATIONS) return null;
    let best: [string, Successor] | null = null;
    for (const entry of successors) if (!best || entry[1].count > best[1].count) best = entry;
    if (!best) return null;
    return { key: best[0], probability: best[1].count / total, expectedInTicks: Math.round(best[1].gapSum / best[1].count) };
  }

  /** How strongly `key` was expected right now (0 if nothing, or something else, was). */
  expectedness(key: string, tick: number): number {
    if (!this.current || !this.last || tick - this.last.tick > this.windowTicks) return 0;
    return this.current.key === key ? this.current.probability : 0;
  }

  /** The expectation in force, if any. */
  get expectation(): Prediction | null {
    return this.current;
  }

  /** Number of transitions learned (predecessor–successor pairs). */
  get size(): number {
    let n = 0;
    for (const successors of this.transitions.values()) n += successors.size;
    return n;
  }

  serialize(): unknown {
    return {
      transitions: Array.from(this.transitions.entries()).map(([from, successors]) => [
        from,
        Array.from(successors.entries()).map(([to, s]) => [to, s.count, s.gapSum]),
      ]),
    };
  }

  deserialize(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const list = (data as { transitions?: unknown }).transitions;
    if (!Array.isArray(list)) return;
    this.transitions = new Map();
    this.totals = new Map();
    for (const entry of list.slice(0, 5000)) {
      if (!Array.isArray(entry) || entry.length !== 2 || typeof entry[0] !== 'string' || !Array.isArray(entry[1])) continue;
      const from = entry[0].slice(0, 80);
      const successors = new Map<string, Successor>();
      let total = 0;
      for (const s of entry[1].slice(0, 500)) {
        if (!Array.isArray(s) || typeof s[0] !== 'string') continue;
        const count = Number.isInteger(s[1]) && s[1] > 0 ? Math.min(1e6, s[1] as number) : 0;
        const gapSum = typeof s[2] === 'number' && Number.isFinite(s[2]) && s[2] >= 0 ? s[2] : 0;
        if (count === 0) continue;
        successors.set(String(s[0]).slice(0, 80), { count, gapSum });
        total += count;
      }
      if (successors.size > 0) {
        this.transitions.set(from, successors);
        this.totals.set(from, total);
      }
    }
  }
}
