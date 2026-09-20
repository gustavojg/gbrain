/**
 * TICK CLOCK — one clock for every tick the brain takes
 * ===========================================================================
 * The brain is meant to live at a fixed rate of ticks per second of real
 * time: everything on a human timescale (how long a stimulus stays in view,
 * how long a percept waits for its name, how often it sleeps, how long a
 * burst of dopamine lasts) is written in real seconds and converted to ticks
 * at that rate. But a perception is propagated AHEAD of time by the
 * perception scheduler (its ~50 ticks run at once, so the dashboard answers
 * at once), and with a camera and a microphone streaming, those extra ticks
 * made the brain live four times faster than it believed: sleeps every few
 * seconds, words pruned before they were repeated, dopamine gone in half a
 * second.
 *
 * This clock counts every tick, wherever it came from, against real time.
 * Ticks run ahead of time are a lead the timer waits out; ticks the timer
 * could not run (a stall, a long save) are a lag it catches up in bounded
 * steps. Both are forgiven beyond a limit: a long lesson does not freeze the
 * brain for minutes afterwards, and a laptop asleep for an hour does not
 * wake to an hour of ticks.
 */
export interface TickClockOptions {
  /** Real ms per tick. */
  intervalMs: number;
  /** Ticks per interval (the simulation speed). */
  speed?: number;
  /** The most the brain may run ahead of real time (ticks) before the excess is forgiven. */
  maxLeadTicks: number;
  /** The most the brain may lag real time (ticks) before the rest is forgiven. */
  maxLagTicks: number;
  /** The most ticks the timer runs at once when catching up. */
  maxCatchUp?: number;
  /** The clock's origin (ms); defaults to now. */
  now?: number;
}

export class TickClock {
  private readonly intervalMs: number;
  private readonly speed: number;
  private readonly maxLeadTicks: number;
  private readonly maxLagTicks: number;
  private readonly maxCatchUp: number;
  private readonly start: number;
  /** Ticks the brain has taken since the clock started. */
  private run = 0;

  constructor(options: TickClockOptions) {
    this.intervalMs = Math.max(1, options.intervalMs);
    this.speed = Math.max(1, options.speed ?? 1);
    this.maxLeadTicks = Math.max(0, options.maxLeadTicks);
    this.maxLagTicks = Math.max(0, options.maxLagTicks);
    this.maxCatchUp = Math.max(1, options.maxCatchUp ?? this.speed * 3);
    this.start = options.now ?? Date.now();
  }

  /** Records `n` ticks the brain has taken (by the timer, or ahead of time). */
  credit(n: number = 1): void {
    this.run += Math.max(0, n);
  }

  /** Ticks real time has called for so far. */
  expected(now: number = Date.now()): number {
    return Math.floor(Math.max(0, now - this.start) / this.intervalMs) * this.speed;
  }

  /** Ticks the brain is ahead of real time (negative: behind). */
  lead(now: number = Date.now()): number {
    return this.run - this.expected(now);
  }

  /**
   * How many ticks the timer should run now: the deficit against real time,
   * in a bounded step; none while the brain is ahead.
   */
  due(now: number = Date.now()): number {
    const expected = this.expected(now);
    // Forgive a lead or a lag beyond the limits.
    if (this.run > expected + this.maxLeadTicks) this.run = expected + this.maxLeadTicks;
    if (this.run < expected - this.maxLagTicks) this.run = expected - this.maxLagTicks;
    const deficit = expected - this.run;
    return deficit <= 0 ? 0 : Math.min(deficit, this.maxCatchUp);
  }
}
