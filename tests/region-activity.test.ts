/**
 * Verifies that the "cognitive" regions (Wernicke, Broca, PFC) no longer fire
 * a constant fraction at rest. Previously k-WTA turned on exactly
 * k = sparsity·N neurons every tick (≈13.8%) because the gate compared against
 * _modulatedThreshold (−55 mV) on dimensionless potentials ~0.
 *
 * Expected after the fix:
 *   (1) Rest (zero input, several ticks): firingRate == 0.
 *   (2) With signal: firingRate > 0, variable across patterns, and ≤ sparsity (k-WTA cap).
 */

import { WernickeArea } from '../src/regions/broca-wernicke/wernicke.js';
import { BrocaArea } from '../src/regions/broca-wernicke/broca.js';
import { PrefrontalCortex } from '../src/regions/prefrontal-cortex/prefrontal-cortex.js';
import { Lexicon } from '../src/regions/broca-wernicke/lexicon.js';
import type { ModulationEffects } from '../src/core/neuromodulators/modulator-system.js';

const NEUTRAL: ModulationEffects = {
  learningRateMultiplier: 1,
  thresholdMultiplier: 1,
  attentionGain: 1,
  consolidationRate: 1,
  spikeGainMultiplier: 1,
  socialWeightBoost: 1,
};

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(0xbeef);

function randomPattern(n: number, density: number): Float32Array {
  const p = new Float32Array(n);
  for (let i = 0; i < n; i++) if (rng() < density) p[i] = 0.5 + rng() * 0.5;
  return p;
}

interface Region {
  feedInput(d: Float32Array, t?: number): void;
  step(dt: number, m: ModulationEffects): { firingRate: number };
}

function idleRate(r: Region, ticks: number): number {
  let last = 0;
  for (let t = 0; t < ticks; t++) last = r.step(10, NEUTRAL).firingRate;
  return last;
}

function drivenRates(r: Region, inputCount: number, trials: number): number[] {
  const rates: number[] = [];
  for (let t = 0; t < trials; t++) {
    r.feedInput(randomPattern(inputCount, 0.1));
    rates.push(r.step(10, NEUTRAL).firingRate);
  }
  return rates;
}

const lex = new Lexicon();
const cases: { name: string; region: Region; inputCount: number }[] = [
  { name: 'Wernicke', region: new WernickeArea(lex, 1000, 1000) as unknown as Region, inputCount: 1000 },
  { name: 'Broca', region: new BrocaArea(lex, 1000, 1000) as unknown as Region, inputCount: 1000 },
  { name: 'PFC', region: new PrefrontalCortex(1000, 1000) as unknown as Region, inputCount: 1000 },
];

console.log('── Actividad por región: reposo vs señal ──\n');

let ok = true;
for (const c of cases) {
  const idle = idleRate(c.region, 30);
  const driven = drivenRates(c.region, c.inputCount, 8);
  const variable = new Set(driven.map((x) => x.toFixed(4))).size > 1;
  const meanDriven = driven.reduce((a, b) => a + b, 0) / driven.length;

  const restOk = idle === 0;
  const activeOk = meanDriven > 0 && variable;
  ok = ok && restOk && activeOk;

  console.log(`${c.name}:`);
  console.log(`  reposo (30 ticks):  ${(idle * 100).toFixed(1)}%   ${restOk ? '✅' : '❌ debería ser 0%'}`);
  console.log(`  señal (8 patrones): media ${(meanDriven * 100).toFixed(1)}%, variable=${variable}   ${activeOk ? '✅' : '❌'}`);
  console.log(`    muestras: ${driven.map((x) => (x * 100).toFixed(1) + '%').join(', ')}\n`);
}

if (ok) {
  console.log('✅ La actividad por región es REAL: 0% en reposo, dinámica con señal.');
  process.exit(0);
} else {
  console.log('❌ FALLO: alguna región sigue con actividad constante o muerta.');
  process.exit(1);
}
