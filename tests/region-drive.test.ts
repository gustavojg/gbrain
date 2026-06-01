/**
 * Verification of the honest activity metric (`drive`).
 * Before: firingRate was constant (= sparsity) with k-WTA → frozen bars.
 * Now: `drive` (EMA of the input signal) should be ~0 at rest and RISE
 * in the regions that receive the cascade when reading a text.
 */

import { DigitalBrain } from '../src/brain.js';

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
  brain.read('hola cerebro como estas hoy');
  for (let i = 0; i < 20; i++) {
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

// Check of the main fix (this.spikes consistent in step()):
// the panel is no longer frozen at "0/0/0/0/0/14/14/14". We verify that
//  (a) MORE than 3 regions report real output (previously only PFC/Wernicke/Broca);
//  (b) the firingRate values are DIFFERENTIATED across regions.
const firing = snapshot(brain);
const nonZero = Object.values(firing).filter((v) => v.f > 0.001).length;
const distinctF = new Set(Object.values(firing).map((v) => v.f.toFixed(3))).size;

console.log('\nfiringRate real por región (output, vía this.spikes):');
for (const [id, v] of Object.entries(firing)) {
  console.log(`  ${id.padEnd(16)} ${(v.f * 100).toFixed(1)}%`);
}
console.log('');
const ok = nonZero > 3 && distinctF > 3;
if (ok) {
  console.log(`✅ PANEL HONESTO: ${nonZero}/8 regiones con output real y ${distinctF} valores distintos (ya no es 0/14 congelado).`);
  process.exit(0);
} else {
  console.log(`❌ FALLO: nonZero=${nonZero} distinctF=${distinctF}`);
  process.exit(1);
}
