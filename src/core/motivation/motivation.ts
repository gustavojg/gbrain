/**
 * MOTIVATION — why the brain does anything
 * ===========================================================================
 * Dopamine is not "reward": it is reward PREDICTION ERROR — what arrived minus
 * what was expected (Schultz, Dayan & Montague, 1997). The same praise, once
 * expected, moves nothing; praise withheld where it was expected is a dip.
 * That is where habituation of pleasure comes from, and it is not a hack.
 *
 * Two intrinsic rewards make the brain act on its own, as an infant does:
 *   - NOVELTY: a thing never seen brings a bonus that shrinks with every
 *     sighting (Kakade & Dayan, 2002);
 *   - LEARNING PROGRESS: an activity is worth doing while it is teaching
 *     something — while the error of the map it trains keeps falling
 *     (Oudeyer & Kaplan, 2007). Mastered, it stops being interesting.
 *
 * Actions (babbling, scribbling) carry a learned value: the progress they
 * tend to bring (actor–critic, the striatum's job). Drives colour all of it:
 * boredom (nothing rewarding for a while) pushes to act, the need for contact
 * (no voice for a while) pushes to call.
 *
 * Everything here counts in ticks; the brain converts its real-time windows.
 */

export type ActivityKind = 'babble' | 'scribble' | 'daydream';

/** Every activity, for iteration. */
export const ACTIVITY_KINDS: readonly ActivityKind[] = ['babble', 'scribble', 'daydream'];

export interface RewardEvent {
  kind: 'novelty' | 'progress' | 'external' | 'omission' | 'imagination';
  /** The cue in context (`visual:Visual-1`), or `null` when none. */
  key: string | null;
  /** What arrived (novelty bonus, learning progress, praise…), −1..1. */
  reward: number;
  /** What was expected for this cue. */
  expected: number;
  /** The prediction error: what dopamine encodes. */
  error: number;
  tick: number;
}

export interface Drives {
  /** Expected learning progress available: how much there is to learn by acting (0..1). */
  curiosity: number;
  /** Time without anything rewarding (0..1). */
  boredom: number;
  /** Time without a voice (0..1). */
  contact: number;
}

export interface MotivationConfig {
  /** Ticks after a percept within which an external reward is credited to it. */
  rewardWindowTicks: number;
  /** Ticks of nothing rewarding at which boredom is ~63%. */
  boredomTicks: number;
  /** Ticks without a voice at which the need for contact is ~63%. */
  contactTicks: number;
}

/** Learning rate of a cue's expected reward. */
const ALPHA = 0.3;
/** Learning rate of the context-free expectation (what the world tends to bring). */
const ALPHA_GLOBAL = 0.05;
/** Bonus of a first sighting; halves with the second, thirds with the third… */
const NOVELTY_BONUS = 0.5;
/** Reward per unit of confidence gained in a recall (learning progress). */
const PROGRESS_GAIN = 1.0;
/** Smallest confidence gain that counts as progress. */
const MIN_PROGRESS = 0.02;
/** Value per unit of growth of a motor map's knowledge (learning 5% of the map in one go is worth 1). */
const ACTIVITY_PROGRESS_GAIN = 20.0;
/** Learning rate of an activity's value. */
const ACTIVITY_ALPHA = 0.2;
/** Optimistic initial value: everything is worth trying once. */
const ACTIVITY_INITIAL = 0.3;
/** Softmax temperature of the choice among activities. */
const CHOICE_TEMPERATURE = 0.15;
/** Expected reward below which an omission is not worth a dip. */
const MIN_EXPECTATION = 0.1;

export class Motivation {
  /** Expected external reward per cue (what praise or reprimand a thing tends to bring). */
  private values = new Map<string, number>();
  /** Times each cue has been perceived. */
  private familiarity = new Map<string, number>();
  /** Confidence of the last recall from each cue, for learning progress. */
  private lastConfidence = new Map<string, number>();
  /** What the world tends to bring, whatever the cue. */
  private globalValue = 0;
  /** Value of each activity: the learning progress it tends to bring. */
  readonly activityValues: Record<ActivityKind, number> = { babble: ACTIVITY_INITIAL, scribble: ACTIVITY_INITIAL, daydream: ACTIVITY_INITIAL };
  /** What each activity's map knew after its last experience. */
  private mapKnowledge: Record<ActivityKind, number | null> = { babble: null, scribble: null, daydream: null };
  /** A cue was perceived: an external reward within the window is credited to it. */
  private pending: { key: string; expected: number; untilTick: number } | null = null;
  /** Born now: boredom and the need for contact grow from zero. */
  private lastRewardTick = 0;
  private lastVoiceTick = 0;
  lastEvent: RewardEvent | null = null;
  /** The last few events, newest last (several can happen in one tick). */
  readonly recentEvents: RewardEvent[] = [];
  private static readonly RECENT = 24;
  /** Counters, for the dashboard. */
  events = 0;

  constructor(private readonly cfg: MotivationConfig) {}

  // ── Rewards ──────────────────────────────────────────────────────────────

  /**
   * A cue has just been perceived. Its novelty is an intrinsic reward that
   * habituates by itself; and it opens a window in which an external reward
   * is credited to it (and its absence, if one was expected, is an omission).
   */
  perceive(key: string, tick: number, expectedness: number = 0, surprise: number = 1): RewardEvent {
    this.resolveOmission(tick);
    const expected = this.values.get(key) ?? this.globalValue;
    this.pending = { key, expected, untilTick: tick + this.cfg.rewardWindowTicks };
    return this.novelty(key, tick, expectedness, surprise);
  }

  /**
   * Something new (or not so new any more): the novelty bonus, habituating
   * with every occurrence — and discounted by how much it was EXPECTED to
   * come next (a thing that always follows another is no surprise).
   */
  novelty(key: string, tick: number, expectedness: number = 0, surprise: number = 1): RewardEvent {
    const seen = (this.familiarity.get(key) ?? 0) + 1;
    this.familiarity.set(key, seen);
    // The bonus habituates with the sightings, and a sighting that the cortex
    // predicted well (low surprise) is worth half of one it did not.
    const bonus = (NOVELTY_BONUS / seen) * (0.5 + 0.5 * Math.max(0, Math.min(1, surprise)));
    const anticipated = Math.max(0, Math.min(1, expectedness)) * bonus;
    return this.record({ kind: 'novelty', key, reward: bonus, expected: anticipated, error: bonus - anticipated, tick });
  }

  /**
   * Something rewarding or punishing arrived from outside (a warm or harsh
   * voice, a 👍 / 👎, a face). It is credited to the cue in context, if one,
   * and compared with what that cue had led to expect.
   */
  external(reward: number, tick: number): RewardEvent {
    const r = Math.max(-1, Math.min(1, reward));
    const key = this.pending && tick <= this.pending.untilTick ? this.pending.key : null;
    const expected = key ? (this.values.get(key) ?? this.globalValue) : this.globalValue;
    const error = r - expected;
    if (key) {
      this.values.set(key, (this.values.get(key) ?? this.globalValue) + ALPHA * error);
      this.pending = null; // credited: no omission
    }
    this.globalValue += ALPHA_GLOBAL * error;
    return this.record({ kind: 'external', key, reward: r, expected, error, tick });
  }

  /**
   * A recall from a cue came back with some confidence: if it is higher than
   * last time, the brain is learning that thing — an intrinsic reward that
   * vanishes once the thing is mastered.
   */
  progress(key: string, confidence: number, tick: number): RewardEvent | null {
    const previous = this.lastConfidence.get(key);
    this.lastConfidence.set(key, confidence);
    if (previous === undefined) return null;
    const gain = confidence - previous;
    if (gain < MIN_PROGRESS) return null;
    const reward = Math.min(1, gain * PROGRESS_GAIN);
    return this.record({ kind: 'progress', key, reward, expected: 0, error: reward, tick });
  }

  /**
   * An activity's map has just learned from one experience; `knowledge` is
   * what it knows now (0..1). Progress is knowledge GROWING: each experience
   * that teaches the map something is worth doing again; when the postures
   * or strokes only repeat what is known, the growth stops and so does the
   * value (Oudeyer & Kaplan, 2007). The activity's value tracks it.
   */
  activityLearned(kind: ActivityKind, knowledge: number, tick: number): RewardEvent | null {
    const previous = this.mapKnowledge[kind];
    this.mapKnowledge[kind] = knowledge;
    const progress = previous === null ? 0 : Math.max(0, Math.min(1, (knowledge - previous) * ACTIVITY_PROGRESS_GAIN));
    this.activityValues[kind] = Math.max(0, Math.min(1, this.activityValues[kind] + ACTIVITY_ALPHA * (progress - this.activityValues[kind])));
    if (progress <= 0) return null;
    return this.record({ kind: 'progress', key: `activity:${kind}`, reward: progress, expected: 0, error: progress, tick });
  }

  /**
   * An activity brought a reward of its own — an imagining that was new, or
   * one that later turned out to be true. The activity's value tracks it, as
   * for the motor maps; the event is the prediction error the reward makes.
   * A reward of 0 still teaches the activity that it brought nothing.
   */
  activityRewarded(kind: ActivityKind, reward: number, tick: number, key: string | null = null): RewardEvent | null {
    const r = Math.max(0, Math.min(1, reward));
    this.activityValues[kind] = Math.max(0, Math.min(1, this.activityValues[kind] + ACTIVITY_ALPHA * (r - this.activityValues[kind])));
    if (r <= 0) return null;
    return this.record({ kind: 'imagination', key: key ?? `activity:${kind}`, reward: r, expected: 0, error: r, tick });
  }

  /** The reward window of the cue in context closed with nothing: disappointment. */
  resolveOmission(tick: number): RewardEvent | null {
    if (!this.pending || tick <= this.pending.untilTick) return null;
    const { key, expected } = this.pending;
    this.pending = null;
    if (expected < MIN_EXPECTATION) return null;
    this.values.set(key, (this.values.get(key) ?? this.globalValue) + ALPHA * (0 - expected));
    this.globalValue += ALPHA_GLOBAL * (0 - expected);
    return this.record({ kind: 'omission', key, reward: 0, expected, error: -expected, tick });
  }

  private record(event: RewardEvent): RewardEvent {
    this.lastEvent = event;
    this.recentEvents.push(event);
    if (this.recentEvents.length > Motivation.RECENT) this.recentEvents.shift();
    this.events++;
    if (event.reward > 0) this.lastRewardTick = event.tick;
    return event;
  }

  // ── Actions and drives ───────────────────────────────────────────────────

  /** Chooses among the available activities by their values (softmax). */
  choose(available: ActivityKind[], random: () => number = Math.random): ActivityKind | null {
    if (available.length === 0) return null;
    if (available.length === 1) return available[0];
    const weights = available.map((kind) => Math.exp(this.activityValues[kind] / CHOICE_TEMPERATURE));
    const total = weights.reduce((a, b) => a + b, 0);
    let r = random() * total;
    for (let i = 0; i < available.length; i++) {
      r -= weights[i];
      if (r <= 0) return available[i];
    }
    return available[available.length - 1];
  }

  /** A voice was heard: contact. */
  heardVoice(tick: number): void {
    this.lastVoiceTick = tick;
  }

  drives(tick: number): Drives {
    const since = (last: number, tau: number): number => 1 - Math.exp(-Math.max(0, tick - last) / tau);
    return {
      curiosity: Math.max(0, Math.min(1, Math.max(...ACTIVITY_KINDS.map((k) => this.activityValues[k])))),
      boredom: since(this.lastRewardTick, this.cfg.boredomTicks),
      contact: since(this.lastVoiceTick, this.cfg.contactTicks),
    };
  }

  /** What a cue leads the brain to expect (0 if unknown). */
  expectation(key: string): number {
    return this.values.get(key) ?? 0;
  }

  /** How many times a cue has been perceived. */
  seen(key: string): number {
    return this.familiarity.get(key) ?? 0;
  }

  /** Sleep: some of the familiarity wears off — a thing seen long ago is a little new again. */
  sleep(): void {
    for (const [key, n] of this.familiarity) this.familiarity.set(key, Math.max(1, n * 0.8));
  }

  // ── Persistence ──────────────────────────────────────────────────────────

  serialize(): unknown {
    return {
      values: Array.from(this.values.entries()),
      familiarity: Array.from(this.familiarity.entries()),
      globalValue: this.globalValue,
      activityValues: { ...this.activityValues },
    };
  }

  deserialize(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as { values?: unknown; familiarity?: unknown; globalValue?: unknown; activityValues?: unknown };
    const finite = (x: unknown, lo: number, hi: number): number | null =>
      typeof x === 'number' && Number.isFinite(x) ? Math.max(lo, Math.min(hi, x)) : null;
    const entries = (list: unknown, lo: number, hi: number): Array<[string, number]> =>
      Array.isArray(list)
        ? list
            .filter((e): e is [unknown, unknown] => Array.isArray(e) && e.length === 2 && typeof e[0] === 'string')
            .map(([k, v]) => [String(k).slice(0, 80), finite(v, lo, hi)] as [string, number | null])
            .filter((e): e is [string, number] => e[1] !== null)
            .slice(0, 5000)
        : [];
    this.values = new Map(entries(d.values, -1, 1));
    this.familiarity = new Map(entries(d.familiarity, 0, 1e6));
    this.globalValue = finite(d.globalValue, -1, 1) ?? 0;
    if (typeof d.activityValues === 'object' && d.activityValues !== null) {
      for (const kind of ACTIVITY_KINDS) {
        const v = finite((d.activityValues as Record<string, unknown>)[kind], 0, 1);
        if (v !== null) this.activityValues[kind] = v;
      }
    }
  }
}
