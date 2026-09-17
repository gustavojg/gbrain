/**
 * Reproducible randomness for tests.
 *
 * The simulation draws from `Math.random` (weight initialization, synaptic
 * noise, Bernoulli spike encoding). Unseeded, every run builds a different
 * brain and threshold-based assertions become flaky. `seedRandom()` swaps
 * `Math.random` for a deterministic PRNG; call it BEFORE constructing a brain.
 */

/** mulberry32 — small, fast, well-distributed 32-bit PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Replaces `Math.random` with a PRNG seeded by `seed`. */
export function seedRandom(seed: number): void {
  Math.random = mulberry32(seed);
}

/** Runs `fn` with the brain's console output silenced. */
export function quiet<T>(fn: () => T): T {
  const { log, warn } = console;
  console.log = () => {};
  console.warn = () => {};
  try {
    return fn();
  } finally {
    console.log = log;
    console.warn = warn;
  }
}
