/**
 * VERIFICATION EXPERIMENT — Visual Cortex Learning
 * ===============================================================
 * Demonstrates DETERMINISTICALLY that the visual cortex really learns
 * with the Izhikevich + STDP core:
 *
 *   1. STABILITY: repeating pattern A consolidates its engram (the overlap
 *      between successive presentations rises toward ~1.0) and the weights converge
 *      (Σ|Δw| per presentation decays).
 *   2. DISCRIMINATION: a distinct pattern B activates a distinct engram
 *      (low A/B overlap).
 *
 * Math.random is seeded with a reproducible PRNG so that the result
 * is identical on every run.
 *
 * Run:  npx tsx tests/visual-learning.test.ts
 */

import { VisualCortex } from '../src/regions/visual-cortex/visual-cortex.js';

// ── Reproducible PRNG (mulberry32) — replaces Math.random for determinism ──
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(12345);
Math.random = rng;

// ── Utilities ──
function makeSparsePattern(size: number, activeFraction: number, rate: number): Float32Array {
  const p = new Float32Array(size);
  const active = Math.floor(size * activeFraction);
  const taken = new Set<number>();
  while (taken.size < active) {
    taken.add(Math.floor(rng() * size));
  }
  for (const idx of taken) p[idx] = rate;
  return p;
}

function fmt(x: number, d = 3): string {
  return x.toFixed(d).padStart(d + 4);
}

// ── Experiment configuration ──
const INPUT = 300;
const NEURONS = 800;
const K = 15;
const REPS = 16;

const cortex = new VisualCortex({
  neuronCount: NEURONS,
  inputCount: INPUT,
  kWinners: K,
  presentationTicks: 50,
});

const A = makeSparsePattern(INPUT, 0.12, 0.85);
const B = makeSparsePattern(INPUT, 0.12, 0.85);

console.log('\n🧪 EXPERIMENTO: Aprendizaje de la Corteza Visual');
console.log('═'.repeat(64));
console.log(`Neuronas: ${NEURONS} | Inputs: ${INPUT} | k-WTA: ${K} | Repeticiones: ${REPS}`);
console.log(`Patrones A y B: ${Math.floor(INPUT * 0.12)} entradas activas cada uno`);
function activeIndices(p: Float32Array): Int32Array {
  const idx: number[] = [];
  for (let i = 0; i < p.length; i++) if (p[i] > 0) idx.push(i);
  return Int32Array.from(idx);
}
const inputOverlap = VisualCortex.engramOverlap(activeIndices(A), activeIndices(B));
console.log(`Overlap de ENTRADA A/B: ${fmt(inputOverlap)} (cuánto se parecen los estímulos)\n`);

// ── 1. Baseline (without learning): initial engrams ──
const engramA0 = cortex.predict(A, 1, 0).winners;
const engramB0 = cortex.predict(B, 1, 0).winners;
const discrim0 = VisualCortex.engramOverlap(engramA0, engramB0);
console.log('── Antes de aprender ──');
console.log(`  Engrama A inicial: ${engramA0.length} neuronas`);
console.log(`  Overlap A/B inicial: ${fmt(discrim0)}\n`);

// ── 2. Learning phase: present A repeatedly ──
console.log('── Aprendizaje (presentando A repetidamente) ──');
console.log('  rep |  Σ|Δw|  | estabilidad(A_t vs A_{t-1}) | actividad');
console.log('  ----+---------+----------------------------+----------');

let prevEngram = cortex.predict(A, 1, 0).winners;
const stabilityCurve: number[] = [];
const weightChangeCurve: number[] = [];

for (let r = 0; r < REPS; r++) {
  const res = cortex.present(A, { learn: true });
  // Probe without learning to measure A's current engram
  const probe = cortex.predict(A, 1, 0).winners;
  const stability = VisualCortex.engramOverlap(prevEngram, probe);
  stabilityCurve.push(stability);
  weightChangeCurve.push(res.weightChange);
  prevEngram = probe;
  console.log(
    `  ${String(r + 1).padStart(3)} | ${fmt(res.weightChange, 2)} |          ${fmt(stability)}          | ${fmt(res.activity)}`,
  );
}

// ── 3. Final state: discrimination A vs B ──
const engramAf = cortex.predict(A, 1, 0).winners;
const engramBf = cortex.predict(B, 1, 0).winners;
const discrimF = VisualCortex.engramOverlap(engramAf, engramBf);

console.log('\n── Después de aprender ──');
console.log(`  Engrama A final: [${Array.from(engramAf).slice(0, 12).join(', ')}${engramAf.length > 12 ? ', …' : ''}]`);
console.log(`  Engrama B final: [${Array.from(engramBf).slice(0, 12).join(', ')}${engramBf.length > 12 ? ', …' : ''}]`);
console.log(`  Overlap A/B final: ${fmt(discrimF)}`);

// ── 4. Verdict ──
const finalStability = stabilityCurve.slice(-3).reduce((s, v) => s + v, 0) / 3;
const firstChange = weightChangeCurve.slice(0, 2).reduce((s, v) => s + v, 0) / 2;
const lastChange = weightChangeCurve.slice(-2).reduce((s, v) => s + v, 0) / 2;
const converged = lastChange < firstChange;

console.log('\n═'.repeat(64));
console.log('VEREDICTO');
console.log('═'.repeat(64));

const checks: [string, boolean, string][] = [
  ['Estabilidad del engrama A (media últimas 3)', finalStability >= 0.6, `${fmt(finalStability)} (objetivo ≥ 0.60)`],
  ['Convergencia de pesos (Σ|Δw| decae)', converged, `${fmt(firstChange, 2)} → ${fmt(lastChange, 2)}`],
  ['Discriminación A vs B (engramas distintos)', discrimF <= 0.5, `${fmt(discrimF)} (objetivo ≤ 0.50)`],
];

let allPass = true;
for (const [name, pass, detail] of checks) {
  console.log(`  ${pass ? '✅' : '❌'} ${name}: ${detail}`);
  if (!pass) allPass = false;
}

console.log('═'.repeat(64));
console.log(allPass ? '\n🎉 LA CORTEZA VISUAL APRENDE DE VERDAD.\n' : '\n⚠️  Aún no converge — requiere tuning de parámetros.\n');

process.exit(allPass ? 0 : 1);
