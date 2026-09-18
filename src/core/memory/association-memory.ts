/**
 * ASSOCIATION MEMORY — Binding what is perceived together
 * =======================================================
 * A heteroassociative memory that links the codes of different modalities
 * (what is seen, heard and read) when they occur together, and later recalls
 * the missing ones from any one of them: see the ball → the word "pelota".
 *
 * Model (convergence zone; Damasio, 1989):
 *   A layer of CONJUNCTION units sits above the modalities. An event that
 *   spans several modalities activates a sparse set of conjunction units, and
 *   Hebbian synapses grow between those units and every active channel of
 *   every modality present. A later cue from a single modality reactivates the
 *   conjunction units through the synapses it learned, and they project back
 *   onto the other modalities, reinstating an approximation of their codes.
 *
 * Learning rules:
 *   - LTP toward the channel's strength: w += lr · (x − w). Each repetition of
 *     a pairing closes part of the gap (25%, 44%, 58%, 68%…); one exposure is
 *     not enough to answer with confidence, a few are. Association is learned
 *     by repetition, and what is reinstated converges to the original code.
 *   - Anti-Hebbian LTD: a channel that is active while a conjunction that
 *     carries it is NOT, weakens that synapse. A channel present in every event
 *     ("this", "is", "a") thereby loses its grip on all of them, and the
 *     channels that co-occur consistently ("ball" ↔ the ball) win —
 *     cross-situational learning (Smith & Yu, 2008).
 *
 * A pairing gets its own conjunction units unless existing ones already match
 * EVERY modality of the event; so one drawing can be bound to two words, and
 * one word to two drawings, without the pairs collapsing into each other.
 *
 * Storage is sparse (only synapses that were ever potentiated exist), so the
 * memory costs what it has learned, and serializes to a short list.
 */

/** A modality's code for one percept: active channels and their strengths (0–1]. */
export interface ModalCode {
  indices: number[];
  values: number[];
}

/** What one modality contributes to a recall. */
export interface RecalledCode {
  /** Reinstated channel strengths (0–1), indexed by channel. */
  pattern: Map<number, number>;
  /** How strongly this modality is associated with the cue (0–1). */
  strength: number;
}

export interface RecallResult {
  /** Conjunction units the cue reactivated (what `reinforce` acts upon). */
  units: number[];
  /** Reinstated code per modality (the cue's own modality excluded). */
  recalled: Record<string, RecalledCode>;
  /** How well the cue matches a learned conjunction (0–1). */
  match: number;
}

export interface AssociationMemoryOptions {
  /** Number of conjunction units. */
  units?: number;
  /** Conjunction units active per event. */
  activeUnits?: number;
  /** Base learning rate of the LTP rule. */
  learningRate?: number;
  /**
   * LTD rate, relative to the learning rate. At 1, a channel that shows up in
   * every event settles at about half the strength of one that belongs to a
   * single pairing — enough for the consistent pairing to stand out.
   */
  depressionRatio?: number;
  /**
   * Minimum match — in EVERY modality of the event — for it to reuse existing
   * conjunction units. Low on purpose: after a single pairing the synapses are
   * at 25%, and a slightly different view of the same thing must still land on
   * them; unrelated events score ~0 in at least one modality.
   */
  reuseMatch?: number;
  /** Seed of the fixed random projection that recruits fresh conjunction units. */
  seed?: number;
}

/** Serialized synapse: [modality, conjunction unit, channel, weight]. */
type SerializedSynapse = [string, number, number, number];

export interface SerializedAssociations {
  units: number;
  synapses: SerializedSynapse[];
  bindings: number;
}

export class AssociationMemory {
  private readonly units: number;
  private readonly activeUnits: number;
  private readonly learningRate: number;
  private readonly depressionRatio: number;
  private readonly reuseMatch: number;
  private readonly seed: number;

  /** modality → conjunction unit → channel → weight */
  private readonly forward: Map<string, Map<number, Map<number, number>>> = new Map();
  /** modality → channel → conjunction units that carry it (inverted index) */
  private readonly carriers: Map<string, Map<number, Set<number>>> = new Map();
  /** Number of multimodal events bound so far. */
  private bindingCount = 0;
  /** Cache of `channelTotals`, invalidated by learning. */
  private readonly totalsCache: Map<string, Map<number, number>> = new Map();

  constructor(options: AssociationMemoryOptions = {}) {
    this.units = options.units ?? 2000;
    this.activeUnits = options.activeUnits ?? 20;
    this.learningRate = options.learningRate ?? 0.25;
    this.depressionRatio = options.depressionRatio ?? 1.0;
    this.reuseMatch = options.reuseMatch ?? 0.15;
    this.seed = options.seed ?? 0x5eed_a550;
  }

  /** Multimodal events bound so far. */
  get bindings(): number {
    return this.bindingCount;
  }

  /** Whether some conjunction carries this channel of a modality (the unit is in use by an association). */
  usesChannel(modality: string, channel: number): boolean {
    const units = this.carriers.get(modality)?.get(channel);
    return units !== undefined && units.size > 0;
  }

  /** Number of synapses learned (for monitoring). */
  get synapseCount(): number {
    let n = 0;
    for (const byUnit of this.forward.values()) for (const synapses of byUnit.values()) n += synapses.size;
    return n;
  }

  // ----------------------------------------------------------------
  // Learning
  // ----------------------------------------------------------------

  /**
   * Binds the codes of an event that spans two or more modalities.
   *
   * @param event - Code of each modality present in the event
   * @param gain - Neuromodulatory gain on the learning rate (1 = neutral)
   * @returns Whether anything was bound (needs ≥ 2 non-empty modalities)
   */
  bind(event: Record<string, ModalCode>, gain: number = 1.0): boolean {
    const modalities = Object.keys(event).filter((m) => event[m].indices.length > 0).sort();
    if (modalities.length < 2) return false;

    const conjunction = this.selectConjunction(event, modalities);
    const active = new Set(conjunction);
    const lr = Math.min(1, this.learningRate * Math.max(0, gain));

    for (const modality of modalities) {
      const code = event[modality];
      const byUnit = this.modalityMap(this.forward, modality);
      const carriers = this.modalityMap(this.carriers, modality);

      for (let c = 0; c < code.indices.length; c++) {
        const channel = code.indices[c];
        const x = Math.max(0, Math.min(1, code.values[c]));
        if (x === 0) continue;

        // LTD: conjunctions that carry this channel but are not part of this event.
        const carrying = carriers.get(channel);
        if (carrying) {
          for (const unit of carrying) {
            if (active.has(unit)) continue;
            const synapses = byUnit.get(unit)!;
            const w = synapses.get(channel)! * (1 - lr * this.depressionRatio * x);
            if (w < 1e-3) {
              synapses.delete(channel);
              carrying.delete(unit);
            } else {
              synapses.set(channel, w);
            }
          }
        }

        // LTP toward the channel's strength, onto the event's conjunction units.
        for (const unit of conjunction) {
          let synapses = byUnit.get(unit);
          if (!synapses) byUnit.set(unit, (synapses = new Map()));
          const w = synapses.get(channel) ?? 0;
          synapses.set(channel, w + lr * (x - w));
          let set = carriers.get(channel);
          if (!set) carriers.set(channel, (set = new Set()));
          set.add(unit);
        }
      }
    }

    this.bindingCount++;
    this.totalsCache.clear();
    return true;
  }

  /**
   * Conjunction units for an event: the ones that already match EVERY modality
   * of the event (a repetition strengthens the same units), or else a fresh
   * set recruited by a fixed random projection of the whole event.
   */
  private selectConjunction(event: Record<string, ModalCode>, modalities: string[]): number[] {
    let joint: Map<number, number> | null = null;
    for (const modality of modalities) {
      const drive = this.drive(modality, event[modality]);
      if (joint === null) {
        joint = drive;
      } else {
        // A unit must be supported by all modalities: keep the weakest support.
        for (const [unit, value] of joint) {
          const other = drive.get(unit);
          if (other === undefined) joint.delete(unit);
          else if (other < value) joint.set(unit, other);
        }
      }
    }

    const matching = [...(joint ?? new Map<number, number>())]
      .filter(([, match]) => match >= this.reuseMatch)
      .sort((a, b) => b[1] - a[1] || a[0] - b[0])
      .slice(0, this.activeUnits)
      .map(([unit]) => unit);
    if (matching.length >= this.activeUnits / 2) return matching;

    return this.recruit(event, modalities);
  }

  /** Deterministic pseudo-random choice of fresh units, keyed by the event's content. */
  private recruit(event: Record<string, ModalCode>, modalities: string[]): number[] {
    let hash = this.seed >>> 0;
    const mix = (n: number): void => {
      hash ^= n + 0x9e3779b9 + ((hash << 6) >>> 0) + (hash >>> 2);
      hash >>>= 0;
    };
    for (const modality of modalities) {
      for (let i = 0; i < modality.length; i++) mix(modality.charCodeAt(i));
      for (const channel of [...event[modality].indices].sort((a, b) => a - b)) mix(channel);
    }

    const chosen = new Set<number>();
    let state = hash || 1;
    while (chosen.size < this.activeUnits) {
      // xorshift32
      state ^= state << 13; state >>>= 0;
      state ^= state >>> 17;
      state ^= state << 5; state >>>= 0;
      chosen.add(state % this.units);
    }
    return [...chosen];
  }

  // ----------------------------------------------------------------
  // Recall
  // ----------------------------------------------------------------

  /**
   * Normalized drive each conjunction unit receives from a code: the cosine
   * between the code and the unit's learned synapses for that modality.
   */
  private drive(modality: string, code: ModalCode): Map<number, number> {
    const drive = new Map<number, number>();
    const byUnit = this.forward.get(modality);
    const carriers = this.carriers.get(modality);
    if (!byUnit || !carriers) return drive;

    let codeNorm2 = 0;
    for (let c = 0; c < code.indices.length; c++) {
      const x = code.values[c];
      codeNorm2 += x * x;
      const carrying = carriers.get(code.indices[c]);
      if (!carrying) continue;
      for (const unit of carrying) {
        drive.set(unit, (drive.get(unit) ?? 0) + x * byUnit.get(unit)!.get(code.indices[c])!);
      }
    }
    if (codeNorm2 === 0) return new Map();

    // Normalize by the code's norm and by a full-strength synapse vector over
    // the same channels, so the match reflects BOTH overlap and how well
    // learned the synapses are (it grows with repetition, up to 1).
    for (const [unit, dot] of drive) drive.set(unit, dot / codeNorm2);
    return drive;
  }

  /**
   * Recalls the other modalities from a single-modality cue.
   *
   * @param modality - Modality of the cue
   * @param code - The cue
   * @returns Reinstated codes and the cue's match, or `null` if the cue evokes nothing
   */
  recall(modality: string, code: ModalCode): RecallResult | null {
    const drive = this.drive(modality, code);
    if (drive.size === 0) return null;

    // Every conjunction the cue drives about as strongly as its best one takes
    // part: a cue bound to two things reinstates both. (Not the merely similar
    // ones: "cuadrado" shares letters with "cruz", and at a lower bar the cross
    // leaked into what "cuadrado" brought to mind.)
    const ranked = [...drive].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    const best = ranked[0][1];
    if (best <= 0) return null;
    const winners = ranked.filter(([, value]) => value >= best * 0.8).slice(0, this.activeUnits * 4);
    const top = winners.slice(0, this.activeUnits);
    const match = top.reduce((sum, [, value]) => sum + value, 0) / this.activeUnits;

    const recalled: Record<string, RecalledCode> = {};
    for (const [other, byUnit] of this.forward) {
      if (other === modality) continue;
      const pattern = new Map<number, number>();
      const winnerTotals = new Map<number, number>();
      let winnersHere = 0;
      for (const [unit, value] of winners) {
        const synapses = byUnit.get(unit);
        if (!synapses) continue;
        winnersHere++;
        // Each conjunction unit votes for its channels, weighted by how
        // strongly the cue drives it.
        for (const [channel, w] of synapses) {
          pattern.set(channel, (pattern.get(channel) ?? 0) + (w * value) / this.activeUnits);
          winnerTotals.set(channel, (winnerTotals.get(channel) ?? 0) + w);
        }
      }
      if (pattern.size === 0) continue;

      // Contrast against what EVERY OTHER conjunction would reinstate: a
      // channel that comes back whatever the cue ("this is a …") says nothing
      // about this cue, and is discounted; what is specific to it stands out.
      const totals = this.channelTotals(other);
      const otherUnits = byUnit.size - winnersHere;
      if (otherUnits > 0) {
        const scale = match; // same scale as the votes above (≈ weight × match)
        for (const [channel, value] of pattern) {
          const baseline = ((totals.get(channel) ?? 0) - (winnerTotals.get(channel) ?? 0)) / otherUnits;
          const specific = value - baseline * scale;
          if (specific > 1e-4) pattern.set(channel, specific);
          else pattern.delete(channel);
        }
        if (pattern.size === 0) continue;
      }

      let strength = 0;
      for (const v of pattern.values()) if (v > strength) strength = v;
      recalled[other] = { pattern, strength };
    }

    return Object.keys(recalled).length > 0 ? { units: winners.map(([unit]) => unit), recalled, match } : null;
  }

  /**
   * Reward-modulated plasticity on the conjunction units of a recall (the
   * third factor of a three-factor rule: the synapses that were just used are
   * still "eligible", and the outcome decides their fate).
   *
   * @param units - Conjunction units of the recall being judged (`RecallResult.units`)
   * @param reward - > 0 strengthens the association (toward saturation),
   *   < 0 weakens it (toward extinction); magnitude 0–1
   */
  reinforce(units: readonly number[], reward: number): void {
    const r = Math.max(-1, Math.min(1, reward));
    if (r === 0 || units.length === 0) return;
    for (const [modality, byUnit] of this.forward) {
      const carriers = this.carriers.get(modality);
      for (const unit of units) {
        const synapses = byUnit.get(unit);
        if (!synapses) continue;
        for (const [channel, w] of synapses) {
          const next = r > 0 ? w + r * 0.5 * (1 - w) * w : w * (1 + r * 0.5);
          if (next < 1e-3) {
            synapses.delete(channel);
            carriers?.get(channel)?.delete(unit);
          } else {
            synapses.set(channel, Math.min(1, next));
          }
        }
      }
    }
    this.totalsCache.clear();
  }

  /** Sum of the synaptic weights onto each channel of a modality (cached until the next `bind`). */
  private channelTotals(modality: string): Map<number, number> {
    let totals = this.totalsCache.get(modality);
    if (totals) return totals;
    totals = new Map();
    for (const synapses of (this.forward.get(modality) ?? new Map<number, Map<number, number>>()).values()) {
      for (const [channel, w] of synapses) totals.set(channel, (totals.get(channel) ?? 0) + w);
    }
    this.totalsCache.set(modality, totals);
    return totals;
  }

  // ----------------------------------------------------------------
  // Persistence
  // ----------------------------------------------------------------

  private modalityMap<V>(store: Map<string, Map<number, V>>, modality: string): Map<number, V> {
    let map = store.get(modality);
    if (!map) store.set(modality, (map = new Map()));
    return map;
  }

  serialize(): SerializedAssociations {
    const synapses: SerializedSynapse[] = [];
    for (const [modality, byUnit] of this.forward) {
      for (const [unit, channels] of byUnit) {
        for (const [channel, w] of channels) synapses.push([modality, unit, channel, Math.round(w * 1e4) / 1e4]);
      }
    }
    return { units: this.units, synapses, bindings: this.bindingCount };
  }

  /** Restores from disk; synapses that do not validate are dropped. */
  deserialize(data: unknown): void {
    const d = data as Partial<SerializedAssociations> | null;
    if (!d || !Array.isArray(d.synapses) || d.units !== this.units) return;

    this.forward.clear();
    this.carriers.clear();
    this.totalsCache.clear();
    for (const raw of d.synapses.slice(0, 2_000_000)) {
      if (!Array.isArray(raw) || raw.length !== 4) continue;
      const [modality, unit, channel, w] = raw as [unknown, unknown, unknown, unknown];
      if (typeof modality !== 'string' || modality.length > 32) continue;
      if (!Number.isInteger(unit) || (unit as number) < 0 || (unit as number) >= this.units) continue;
      if (!Number.isInteger(channel) || (channel as number) < 0 || (channel as number) > 1_000_000) continue;
      if (typeof w !== 'number' || !(w > 0 && w <= 1)) continue;

      const byUnit = this.modalityMap(this.forward, modality);
      let synapses = byUnit.get(unit as number);
      if (!synapses) byUnit.set(unit as number, (synapses = new Map()));
      synapses.set(channel as number, w);
      const carriers = this.modalityMap(this.carriers, modality);
      let set = carriers.get(channel as number);
      if (!set) carriers.set(channel as number, (set = new Set()));
      set.add(unit as number);
    }
    this.bindingCount = Number.isInteger(d.bindings) && (d.bindings as number) >= 0 ? (d.bindings as number) : 0;
  }
}
