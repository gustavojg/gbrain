/**
 * PERCEPTION SCHEDULER — Non-blocking propagation of sensory input
 * =================================================================
 * A perception needs ~50 brain ticks to propagate through the connectome.
 * Running them inline blocks the Node event loop for ~0.5 s per input, so a
 * single client streaming audio could stall the tick loop, the broadcasts and
 * every other client.
 *
 * The scheduler runs those ticks in short slices, yielding to the event loop
 * between slices, one perception at a time. Streaming inputs (webcam, mic) are
 * coalesced: a newer frame replaces an older one that has not started yet.
 */

/** Thrown (as a rejection) when the queue is full. */
export class SchedulerBusyError extends Error {
  constructor() {
    super('Perception queue is full');
  }
}

export interface PerceptionJob<R> {
  /**
   * Jobs sharing a key are coalesced: a newer one replaces a queued older one
   * (whose promise resolves to `null`). Omit for inputs that must not be lost.
   */
  coalesceKey?: string;
  /** Injects the stimulus into the brain (must not run ticks itself). */
  inject: () => void;
  /** Builds the result once propagation has finished. */
  finish: () => R;
}

export interface SchedulerOptions {
  /** Ticks to run per perception. */
  ticksPerJob?: number;
  /** Max ticks per slice before yielding to the event loop. */
  sliceTicks?: number;
  /**
   * Max wall-clock time per slice (ms). On a slow host a tick can take several
   * times longer than in development; the time budget keeps the longest stall
   * bounded regardless (at least one tick always runs).
   */
  sliceBudgetMs?: number;
  /** Max queued (not yet started) jobs. */
  maxQueue?: number;
  /** Yield primitive (injectable for tests). */
  defer?: (fn: () => void) => void;
}

interface QueuedJob {
  job: PerceptionJob<unknown>;
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
}

export class PerceptionScheduler {
  private readonly ticksPerJob: number;
  private readonly sliceTicks: number;
  private readonly sliceBudgetMs: number;
  private readonly maxQueue: number;
  private readonly defer: (fn: () => void) => void;
  private readonly queue: QueuedJob[] = [];
  private running = false;

  constructor(private readonly tick: () => void, options: SchedulerOptions = {}) {
    this.ticksPerJob = options.ticksPerJob ?? 50;
    this.sliceTicks = Math.max(1, options.sliceTicks ?? 5);
    this.sliceBudgetMs = options.sliceBudgetMs ?? 25;
    this.maxQueue = options.maxQueue ?? 32;
    this.defer = options.defer ?? ((fn) => setImmediate(fn));
  }

  /** Jobs waiting to start. */
  get pending(): number {
    return this.queue.length;
  }

  /**
   * Queues a perception. Resolves with the job's result, or `null` if it was
   * superseded by a newer job with the same `coalesceKey`. Rejects with
   * `SchedulerBusyError` when the queue is full.
   */
  submit<R>(job: PerceptionJob<R>): Promise<R | null> {
    return new Promise<R | null>((resolve, reject) => {
      const entry: QueuedJob = {
        job,
        resolve: resolve as (value: unknown) => void,
        reject,
      };

      if (job.coalesceKey !== undefined) {
        const idx = this.queue.findIndex((q) => q.job.coalesceKey === job.coalesceKey);
        if (idx !== -1) {
          // Keep the older job's place in line, but with the newer stimulus.
          this.queue[idx].resolve(null);
          this.queue[idx] = entry;
          return;
        }
      }

      if (this.queue.length >= this.maxQueue) {
        reject(new SchedulerBusyError());
        return;
      }

      this.queue.push(entry);
      this.pump();
    });
  }

  private pump(): void {
    if (this.running) return;
    const next = this.queue.shift();
    if (!next) return;
    this.running = true;

    let remaining = this.ticksPerJob;
    const done = (settle: () => void): void => {
      this.running = false;
      settle();
      this.defer(() => this.pump());
    };

    const runSlice = (): void => {
      try {
        const sliceStart = performance.now();
        let n = 0;
        do {
          this.tick();
          n++;
          remaining--;
        } while (
          remaining > 0 &&
          n < this.sliceTicks &&
          performance.now() - sliceStart < this.sliceBudgetMs
        );
        if (remaining > 0) {
          this.defer(runSlice);
        } else {
          const result = next.job.finish();
          done(() => next.resolve(result));
        }
      } catch (err) {
        done(() => next.reject(err as Error));
      }
    };

    try {
      next.job.inject();
    } catch (err) {
      done(() => next.reject(err as Error));
      return;
    }
    runSlice();
  }
}
