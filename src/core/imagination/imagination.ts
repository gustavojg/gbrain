/**
 * IMAGINATION — what the brain makes up on its own
 * ===========================================================================
 * Without external input the cortex does not go quiet: the default-mode
 * network (Raichle 2001) keeps reactivating memories and recombining them —
 * mind-wandering (Smallwood & Schooler 2015). Awake, the hippocampus replays
 * fragments of what it knows (Foster & Wilson 2006) and the same completion
 * that recalls a memory from a partial cue completes a MIXED cue into
 * something never experienced: the constructive episodic simulation of
 * Schacter & Addis (2007). Asleep, in REM, the same machinery runs with high
 * acetylcholine, low norepinephrine and serotonin and no prefrontal control,
 * and the chimeras are dreams (Hobson; Stickgold & Walker).
 *
 * This module keeps the bookkeeping the brain needs around that:
 *
 *   - which combinations it has imagined, and how often (the novelty of an
 *     imagining habituates like any other, so it does not loop on the same
 *     daydream for the dopamine);
 *   - REALITY MONITORING (Johnson & Raye 1981): what was imagined is tagged
 *     as internal and never filed as an episode of the world; and when a
 *     combination it imagined later turns up for real, the brain recognises
 *     it as something it had foreseen.
 */

/** Where an imagining came from. */
export type ImaginationOrigin = 'daydream' | 'dream';

/** A record of a combination the brain has imagined. */
export interface ImaginedCombination {
  /** Sorted source keys ('colour:Colour-2', 'visual:Visual-1'). */
  sources: string[];
  /** Times it has been imagined. */
  count: number;
  /** Tick of the last time. */
  lastTick: number;
  /** Whether it has since been experienced for real (rewarded once). */
  realized: boolean;
}

/** Half the units of one code and half of the other: a chimera. */
export function mixUnits(a: readonly number[], b: readonly number[], random: () => number = Math.random): number[] {
  const pick = (units: readonly number[], n: number): number[] => {
    const pool = [...units];
    const out: number[] = [];
    while (out.length < n && pool.length > 0) out.push(pool.splice(Math.floor(random() * pool.length), 1)[0]);
    return out;
  };
  const half = Math.max(1, Math.round(Math.max(a.length, b.length) / 2));
  const mixed = new Set<number>([...pick(a, half), ...pick(b, half)]);
  return [...mixed].sort((x, y) => x - y);
}

export class ImaginationMemory {
  private combinations = new Map<string, ImaginedCombination>();

  static keyOf(sources: readonly string[]): string {
    return [...sources].sort().join('+');
  }

  /**
   * The brain has just imagined this combination: how many times, this one
   * included. A combination it had already experienced when it imagined it
   * cannot later be "foreseen": only what was new when imagined counts.
   */
  imagined(sources: readonly string[], tick: number, novel: boolean = true): ImaginedCombination {
    const key = ImaginationMemory.keyOf(sources);
    let record = this.combinations.get(key);
    if (!record) {
      record = { sources: [...sources].sort(), count: 0, lastTick: tick, realized: false };
      this.combinations.set(key, record);
    }
    record.count++;
    record.lastTick = tick;
    if (!novel) record.realized = true;
    return record;
  }

  /**
   * An experience has just bound these things together. If the brain had
   * imagined that very combination (every source of the imagining present in
   * the experience) and never met it for real, this is it: foreseen.
   */
  foresaw(experienced: readonly string[]): ImaginedCombination | null {
    const present = new Set(experienced);
    for (const record of this.combinations.values()) {
      if (record.realized || record.sources.length < 2) continue;
      if (record.sources.every((s) => present.has(s))) {
        record.realized = true;
        return record;
      }
    }
    return null;
  }

  get size(): number {
    return this.combinations.size;
  }

  /** The most recently imagined combinations, newest first. */
  recent(limit: number = 8): ImaginedCombination[] {
    return [...this.combinations.values()].sort((a, b) => b.lastTick - a.lastTick).slice(0, limit);
  }

  serialize(): unknown {
    return [...this.combinations.values()].map((r) => [r.sources, r.count, r.lastTick, r.realized]);
  }

  deserialize(data: unknown): void {
    if (!Array.isArray(data)) return;
    this.combinations = new Map();
    for (const entry of data.slice(0, 2000)) {
      if (!Array.isArray(entry) || entry.length !== 4 || !Array.isArray(entry[0])) continue;
      const sources = (entry[0] as unknown[]).filter((s): s is string => typeof s === 'string').map((s) => s.slice(0, 64)).sort();
      if (sources.length < 2 || !Number.isInteger(entry[1]) || (entry[1] as number) <= 0) continue;
      const lastTick = typeof entry[2] === 'number' && Number.isFinite(entry[2]) ? (entry[2] as number) : 0;
      this.combinations.set(ImaginationMemory.keyOf(sources), { sources, count: Math.min(1e6, entry[1] as number), lastTick, realized: entry[3] === true });
    }
  }
}
