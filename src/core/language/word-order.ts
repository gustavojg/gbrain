/**
 * WORD ORDER — which kind of word comes first
 * ===========================================================================
 * "Coche azul", not "azul coche". A child does not learn that as a rule: it
 * hears the noun before the adjective, time after time, and comes to put
 * them in that order itself (statistical learning of sequence; Saffran
 * 1996). What is ordered here is not words but the KINDS the words refer
 * to — the shape's name, the colour's name, the sound's — which is what the
 * word–referent record already knows about a word. Each utterance heard
 * adds a vote for the order of the kinds in it; producing, the kinds are
 * put in the order that has won.
 */

export interface OrderVote {
  first: string;
  then: string;
  count: number;
}

const MAX_PAIRS = 64;

export class WordOrderMemory {
  /** first → then → times heard in that order */
  private counts = new Map<string, Map<string, number>>();

  /** An utterance was heard: the kinds of its words, in the order they came. */
  observe(kinds: readonly string[]): void {
    for (let i = 0; i < kinds.length; i++) {
      for (let j = i + 1; j < kinds.length; j++) {
        if (kinds[i] === kinds[j]) continue;
        let then = this.counts.get(kinds[i]);
        if (!then) {
          if (this.counts.size >= MAX_PAIRS) return;
          this.counts.set(kinds[i], (then = new Map()));
        }
        then.set(kinds[j], (then.get(kinds[j]) ?? 0) + 1);
      }
    }
  }

  /** How many more times `a` came before `b` than after it. */
  private lead(a: string, b: string): number {
    return (this.counts.get(a)?.get(b) ?? 0) - (this.counts.get(b)?.get(a) ?? 0);
  }

  /** Whether the order of two kinds has been heard at all. */
  knows(a: string, b: string): boolean {
    return (this.counts.get(a)?.get(b) ?? 0) + (this.counts.get(b)?.get(a) ?? 0) > 0;
  }

  /** Items put in the order their kinds have been heard in (unknown orders keep their own). */
  order<T>(items: readonly T[], kindOf: (item: T) => string): T[] {
    return [...items].sort((x, y) => {
      const a = kindOf(x), b = kindOf(y);
      if (a === b) return 0;
      return -this.lead(a, b);
    });
  }

  /** The orders heard, strongest first. */
  votes(): OrderVote[] {
    const out: OrderVote[] = [];
    for (const [first, then] of this.counts) for (const [t, count] of then) out.push({ first, then: t, count });
    return out.sort((x, y) => y.count - x.count);
  }

  get size(): number {
    return this.votes().length;
  }

  serialize(): unknown {
    return this.votes().map((v) => [v.first, v.then, v.count]);
  }

  deserialize(data: unknown): void {
    if (!Array.isArray(data)) return;
    this.counts = new Map();
    for (const entry of data.slice(0, MAX_PAIRS * MAX_PAIRS)) {
      if (!Array.isArray(entry) || entry.length !== 3 || typeof entry[0] !== 'string' || typeof entry[1] !== 'string') continue;
      if (!Number.isInteger(entry[2]) || (entry[2] as number) <= 0) continue;
      const first = (entry[0] as string).slice(0, 32), then = (entry[1] as string).slice(0, 32);
      let map = this.counts.get(first);
      if (!map) this.counts.set(first, (map = new Map()));
      map.set(then, Math.min(1e6, entry[2] as number));
    }
  }
}
