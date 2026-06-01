/**
 * TEST DE VERIFICACIÓN — Bucle asociativo del lenguaje (texto→Wernicke→Broca)
 * ===========================================================================
 * Antes: el TextEncoder genérico producía vectores ortogonales al léxico
 *   (cos≈0), Wernicke no comprendía nada y Broca devolvía ensalada aleatoria,
 *   distinta en cada ronda.
 *
 * Ahora: el texto se codifica en el espacio del léxico, Wernicke lo comprende
 *   y Broca regenera la respuesta de forma determinista. Verificamos 3
 *   propiedades con criterios cuantitativos:
 *
 *   1. REPRODUCIBILIDAD — el mismo texto produce (casi) la misma respuesta.
 *   2. DISCRIMINACIÓN   — textos distintos producen respuestas distintas.
 *   3. RELEVANCIA       — la respuesta contiene palabras del propio input.
 */

import { DigitalBrain } from '../src/brain.js';

function normalizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\w\sáéíóúüñ]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

function ask(brain: DigitalBrain, text: string): string[] {
  brain.read(text);
  const r = brain.speak();
  return r.text ? normalizeWords(r.text) : [];
}

/** Índice de Jaccard entre dos conjuntos de palabras. */
function jaccard(a: string[], b: string[]): number {
  const sa = new Set(a);
  const sb = new Set(b);
  let inter = 0;
  for (const w of sa) if (sb.has(w)) inter++;
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : inter / union;
}

const texts = [
  'hola cerebro como estas',
  'tengo mucho miedo y tristeza',
  'el sol brilla feliz alegria',
  'quiero aprender y recordar memoria',
];

console.log('── Verificación del bucle asociativo del lenguaje ──\n');

const brain = new DigitalBrain();
for (let i = 0; i < 20; i++) brain.tick(); // calentar

// Capturar dos respuestas por texto (rondas separadas) para reproducibilidad.
const round1: Record<string, string[]> = {};
const round2: Record<string, string[]> = {};
for (const t of texts) round1[t] = ask(brain, t);
for (let i = 0; i < 10; i++) brain.tick();
for (const t of texts) round2[t] = ask(brain, t);

console.log('Respuestas por texto:');
for (const t of texts) {
  console.log(`  IN : "${t}"`);
  console.log(`    R1: [${round1[t].join(', ')}]`);
  console.log(`    R2: [${round2[t].join(', ')}]`);
}

// 1. REPRODUCIBILIDAD: Jaccard(R1, R2) del mismo texto debe ser alto.
let reproSum = 0;
for (const t of texts) reproSum += jaccard(round1[t], round2[t]);
const reproAvg = reproSum / texts.length;

// 2. DISCRIMINACIÓN: Jaccard entre respuestas de textos DISTINTOS debe ser bajo.
let discrimSum = 0;
let pairs = 0;
for (let i = 0; i < texts.length; i++) {
  for (let j = i + 1; j < texts.length; j++) {
    discrimSum += jaccard(round1[texts[i]], round1[texts[j]]);
    pairs++;
  }
}
const discrimAvg = discrimSum / pairs;

// 3. RELEVANCIA: cada respuesta debe contener ≥1 palabra de su propio input.
let relevantCount = 0;
for (const t of texts) {
  const inputWords = new Set(normalizeWords(t));
  const hit = round1[t].some((w) => inputWords.has(w));
  if (hit) relevantCount++;
}
const relevanceRatio = relevantCount / texts.length;

console.log('\n── Métricas ──');
console.log(`  Reproducibilidad (Jaccard R1↔R2, mismo texto): ${(reproAvg * 100).toFixed(0)}%  (objetivo ≥60%)`);
console.log(`  Discriminación  (Jaccard entre textos):        ${(discrimAvg * 100).toFixed(0)}%  (objetivo ≤40%)`);
console.log(`  Relevancia      (respuesta∩input ≥1):          ${(relevanceRatio * 100).toFixed(0)}%  (objetivo =100%)`);

const okRepro = reproAvg >= 0.6;
const okDiscrim = discrimAvg <= 0.4;
const okRelevance = relevanceRatio >= 0.999;

console.log('');
if (okRepro && okDiscrim && okRelevance) {
  console.log('✅ BUCLE ASOCIATIVO VERIFICADO: respuestas reproducibles, discriminativas y relevantes.');
  process.exit(0);
} else {
  console.log(`❌ FALLO: repro=${okRepro} discrim=${okDiscrim} relevance=${okRelevance}`);
  process.exit(1);
}
