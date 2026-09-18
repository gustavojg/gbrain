/**
 * Verification of the honest activity metrics (`drive`, `firingRate`).
 * Before: firingRate was constant (= sparsity) with k-WTA → frozen bars, and
 * the brain was never at rest (noise winners + endless reverberation).
 * Now: at rest every region is silent; reading makes `drive` and the firing
 * rates RISE in the regions that receive the cascade; then activity fades.
 */

import { DigitalBrain } from '../src/brain.js';
import { seedRandom } from './helpers/seed.js';

// Reproducible brain: weights, noise and spike encoding all draw from Math.random.
seedRandom(103);

function meanDrive(brain: DigitalBrain): Record<string, number> {
  const out: Record<string, number> = {};
  const s = brain.getState();
  for (const [id, a] of Object.entries(s.regions)) out[id] = a.drive;
  return out;
}

function snapshot(brain: DigitalBrain): Record<string, { d: number; f: number; n: number }> {
  const out: Record<string, { d: number; f: number; n: number }> = {};
  const s = brain.getState();
  for (const [id, a] of Object.entries(s.regions)) out[id] = { d: a.drive, f: a.firingRate, n: a.novelty };
  return out;
}

console.log('── Métrica de actividad honesta (drive) ──\n');

const brain = new DigitalBrain();

// 1. Rest: let several ticks run with no input.
for (let i = 0; i < 60; i++) brain.tick();
const idleSnap = snapshot(brain);
console.log('REPOSO (drive / firingRate / novelty):');
for (const [id, v] of Object.entries(idleSnap)) {
  console.log(`  ${id.padEnd(16)} d=${(v.d * 100).toFixed(1)}%  f=${(v.f * 100).toFixed(1)}%  n=${(v.n * 100).toFixed(1)}%`);
}
const idleNov: Record<string, number> = {};
for (const [id, v] of Object.entries(idleSnap)) idleNov[id] = v.n;
const idle = meanDrive(brain);

// 2. Inject text and capture the PEAK of drive and firingRate per region.
const active: Record<string, number> = {};
const activeF: Record<string, number> = {};
const activeN: Record<string, number> = {};
for (let r = 0; r < 5; r++) {
  // Inject only, and observe the WHOLE wave tick by tick (read() would run its
  // 50 propagation ticks inline, before we can look: most of the wave is over by then).
  brain.read('hola cerebro como estas hoy', { propagate: false });
  for (let i = 0; i < 120; i++) {
    brain.tick();
    const s = snapshot(brain);
    for (const [id, v] of Object.entries(s)) {
      active[id] = Math.max(active[id] ?? 0, v.d);
      activeF[id] = Math.max(activeF[id] ?? 0, v.f);
      activeN[id] = Math.max(activeN[id] ?? 0, v.n);
    }
  }
}
console.log('\nPICO durante lectura (drive / firingRate / novelty):');
for (const id of Object.keys(active)) {
  console.log(`  ${id.padEnd(16)} d=${(active[id] * 100).toFixed(1)}%  f=${((activeF[id] ?? 0) * 100).toFixed(1)}%  n=${((activeN[id] ?? 0) * 100).toFixed(1)}%`);
}
console.log('\nNOVEDAD reposo → pico:');
for (const id of Object.keys(activeN)) {
  console.log(`  ${id.padEnd(16)} ${((idleNov[id] ?? 0) * 100).toFixed(1)}% → ${((activeN[id] ?? 0) * 100).toFixed(1)}%`);
}

const pct = (x: number) => (x * 100).toFixed(1) + '%';
console.log('Región            reposo →  activo');
let anyRose = false;
for (const id of Object.keys(active)) {
  const before = idle[id] ?? 0;
  const after = active[id] ?? 0;
  const arrow = after > before + 0.001 ? '▲' : after < before - 0.001 ? '▼' : ' ';
  if (after > before + 0.01) anyRose = true;
  console.log(`${id.padEnd(16)}  ${pct(before).padStart(6)} → ${pct(after).padStart(6)}  ${arrow}`);
}

// Verdict. The activity panel is honest when:
//  (a) AT REST the brain is silent: no region fires and no drive is reported
//      (k-WTA must not conjure winners out of noise);
//  (b) WHILE READING more than 3 regions fire for real, with DIFFERENTIATED
//      peak rates (not the old frozen "0/0/0/0/0/14/14/14"), and drive rises;
//  (c) AFTERWARDS the reverberation fades and the brain returns to rest
//      (synaptic depression + LIF reset), instead of echoing forever.
const idleSilent = Object.values(idleSnap).every((v) => v.f === 0 && v.d < 0.001);
const nonZero = Object.values(activeF).filter((f) => f > 0.001).length;
const distinctF = new Set(Object.values(activeF).map((f) => f.toFixed(3))).size;

for (let i = 0; i < 300; i++) brain.tick();
const after = snapshot(brain);
const backToRest = Object.values(after).every((v) => v.f === 0);

console.log('\nfiringRate por región (reposo → pico leyendo → 300 ticks después):');
for (const id of Object.keys(activeF)) {
  console.log(`  ${id.padEnd(16)} ${pct(idleSnap[id].f).padStart(6)} → ${pct(activeF[id]).padStart(6)} → ${pct(after[id].f).padStart(6)}`);
}
console.log('');
const ok = idleSilent && nonZero > 3 && distinctF > 3 && anyRose && backToRest;
if (ok) {
  console.log(`✅ PANEL HONESTO: silencio en reposo, ${nonZero}/10 regiones activas al leer (${distinctF} tasas distintas) y vuelta al reposo después.`);
  process.exit(0);
} else {
  console.log(`❌ FALLO: idleSilent=${idleSilent} nonZero=${nonZero} distinctF=${distinctF} driveRose=${anyRose} backToRest=${backToRest}`);
  process.exit(1);
}
