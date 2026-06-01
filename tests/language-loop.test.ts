/**
 * VERIFICATION TEST — Associative language loop (text→Wernicke→Broca)
 * ===========================================================================
 * Before: the generic TextEncoder produced vectors orthogonal to the lexicon
 *   (cos≈0), Wernicke understood nothing, and Broca returned a random salad,
 *   different on each round.
 *
 * Now: the text is encoded in the lexicon space, Wernicke understands it
 *   and Broca regenerates the response deterministically. We verify 3
 *   properties with quantitative criteria:
 *
 *   1. REPRODUCIBILITY — the same text produces (almost) the same response.
 *   2. DISCRIMINATION  — different texts produce different responses.
 *   3. RELEVANCE       — the response contains words from the input itself.
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

/** Jaccard index between two sets of words. */
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
for (let i = 0; i < 20; i++) brain.tick(); // warm up

// Capture two responses per text (separate rounds) for reproducibility.
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

// 1. REPRODUCIBILITY: Jaccard(R1, R2) for the same text must be high.
let reproSum = 0;
for (const t of texts) reproSum += jaccard(round1[t], round2[t]);
const reproAvg = reproSum / texts.length;

// 2. DISCRIMINATION: Jaccard between responses of DIFFERENT texts must be low.
let discrimSum = 0;
let pairs = 0;
for (let i = 0; i < texts.length; i++) {
  for (let j = i + 1; j < texts.length; j++) {
    discrimSum += jaccard(round1[texts[i]], round1[texts[j]]);
    pairs++;
  }
}
const discrimAvg = discrimSum / pairs;

// 3. RELEVANCE: each response must contain ≥1 word from its own input.
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
