/**
 * Verificación del hipocampo CA3 autoasociativo.
 * Prueba: (1) separación de patrones (DG), (2) completación desde pista
 * degradada (atractor), (3) discriminación entre episodios, y (4) que el
 * aprendizaje (pesos recurrentes) sobrevive a save/load.
 */

import { Hippocampus } from '../src/regions/hippocampus/hippocampus.js';

const N = 1000;
const P = 6; // número de episodios distintos
const rng = mulberry32(0xc0ffee);

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Patrón continuo aleatorio con ~density fracción de entradas activas. */
function randomPattern(density: number): Float32Array {
  const p = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (rng() < density) p[i] = 0.4 + rng() * 0.6;
  }
  return p;
}

/** Degrada una pista: pone a 0 una fracción `drop` de sus entradas activas. */
function corrupt(pattern: Float32Array, drop: number): Float32Array {
  const c = new Float32Array(pattern);
  for (let i = 0; i < N; i++) {
    if (c[i] > 0 && rng() < drop) c[i] = 0;
  }
  return c;
}

const overlap = Hippocampus.overlapBinary;

console.log('── Hipocampo CA3: verificación ──\n');

const hippo = new Hippocampus(N, N);

// 1. Generar y almacenar P episodios distintos.
const patterns: Float32Array[] = [];
for (let i = 0; i < P; i++) patterns.push(randomPattern(0.1));
for (const p of patterns) hippo.store(p, { sourceRegion: 'test' });
console.log(`Episodios almacenados: ${hippo.memoryCount}`);

// 2. Atractor objetivo de cada episodio (completar desde el patrón limpio).
const targets = patterns.map((p) => hippo.patternCompletion(p));

// 3. Separación de patrones (DG): solape medio entre atractores distintos.
let sepSum = 0;
let sepPairs = 0;
for (let i = 0; i < P; i++) {
  for (let j = i + 1; j < P; j++) {
    sepSum += overlap(targets[i], targets[j]);
    sepPairs++;
  }
}
const meanSeparationOverlap = sepSum / sepPairs;

// 4. Completación desde pista degradada (50% de la información eliminada).
const cues = patterns.map((p) => corrupt(p, 0.5));
const recovered = cues.map((c) => hippo.patternCompletion(c));

let selfSum = 0;
let crossSum = 0;
let crossPairs = 0;
for (let i = 0; i < P; i++) {
  selfSum += overlap(recovered[i], targets[i]);
  for (let j = 0; j < P; j++) {
    if (i === j) continue;
    crossSum += overlap(recovered[i], targets[j]);
    crossPairs++;
  }
}
const meanSelfRecovery = selfSum / P;
const meanCrossRecovery = crossSum / crossPairs;

// 5. Persistencia: extraer pesos → hipocampo nuevo → loadWeights → completar.
const savedWeights = new Float32Array(hippo.getNetworkConfig().weights);
const hippo2 = new Hippocampus(N, N);
hippo2.loadWeights(savedWeights);
let persistSum = 0;
for (let i = 0; i < P; i++) {
  const rec2 = hippo2.patternCompletion(cues[i]);
  persistSum += overlap(rec2, recovered[i]);
}
const meanPersistence = persistSum / P;

// ── Resultados ──
const pct = (x: number) => (x * 100).toFixed(1) + '%';
console.log(`\nSeparación DG (solape entre episodios):  ${pct(meanSeparationOverlap)}  (debe ser BAJO)`);
console.log(`Recuperación self (pista 50% → atractor): ${pct(meanSelfRecovery)}  (debe ser ALTO)`);
console.log(`Recuperación cross (a otros episodios):   ${pct(meanCrossRecovery)}  (debe ser BAJO)`);
console.log(`Persistencia (tras save/load de pesos):   ${pct(meanPersistence)}  (debe ser ≈100%)`);

const ok =
  meanSelfRecovery > 0.6 &&
  meanCrossRecovery < 0.2 &&
  meanSelfRecovery > meanCrossRecovery + 0.4 &&
  meanPersistence > 0.999;

console.log('');
if (ok) {
  console.log('✅ HIPOCAMPO APRENDE DE VERDAD: completa pistas degradadas, discrimina episodios y persiste.');
  process.exit(0);
} else {
  console.log('❌ FALLO: revisar dinámica de atractores / parámetros.');
  process.exit(1);
}
