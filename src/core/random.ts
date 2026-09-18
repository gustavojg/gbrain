/**
 * Deterministic pseudo-random source (mulberry32): small, fast, well
 * distributed. Used where a subsystem must draw its own randomness without
 * touching the global `Math.random` — so that adding it leaves the seeded
 * trajectories of the rest of the brain untouched.
 */
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
