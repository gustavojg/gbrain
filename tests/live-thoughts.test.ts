/**
 * VERIFICATION TEST — Live "thoughts" (stream of consciousness)
 * ===========================================================================
 * The dashboard streams `brain.think()`, which decodes the brain's CURRENT
 * internal activation (Wernicke + Broca output spikes + a decaying trace of
 * the last thing it read) into a short, emotion-framed phrase.
 *
 * We verify three honest properties:
 *
 *   1. STRUCTURE     — think() always returns a well-formed thought (text,
 *                      words[], emotion, emoji, valence/arousal, color).
 *   2. REACTIVITY    — reading something changes the thought: an idle brain
 *                      thinks nothing (empty), but right after reading the
 *                      thought becomes non-empty (perception drives the stream).
 *   3. DISCRIMINATION — two very different inputs produce different thoughts
 *                      (low word overlap), i.e. the stream isn't a fixed loop.
 *
 * HONESTY NOTE: a "thought" is an associative read-out of the SNN's activation
 * in lexicon space — it surfaces words *co-activated by* / *neighbouring* the
 * input (n-gram + semantic neighbours), NOT a literal echo and NOT symbolic
 * reasoning. So we verify that perception SHAPES the stream and that distinct
 * inputs yield distinct streams — not that the exact input words come back.
 */

import { DigitalBrain } from '../src/brain.js';

function thoughtAfterReading(brain: DigitalBrain, text: string): { words: string[]; text: string; emotion: string } {
  brain.read(text);
  for (let i = 0; i < 5; i++) brain.tick(); // let activation settle while trace is fresh
  const t = brain.think();
  return { words: t.words, text: t.text, emotion: t.emotion };
}

function overlap(a: string[], b: string[]): number {
  const sa = new Set(a);
  const shared = b.filter((w) => sa.has(w)).length;
  const denom = Math.max(1, Math.min(a.length, b.length));
  return shared / denom;
}

console.log('── Verification: live thoughts (stream of consciousness) ──\n');

const brain = new DigitalBrain();
for (let i = 0; i < 10; i++) brain.tick(); // warm up

// 1. STRUCTURE — a thought is always well-formed, even when idle.
const idle = brain.think();
const structureOk =
  typeof idle.text === 'string' &&
  Array.isArray(idle.words) &&
  typeof idle.emotion === 'string' &&
  typeof idle.emoji === 'string' &&
  Number.isFinite(idle.valence) &&
  Number.isFinite(idle.arousal) &&
  typeof idle.color === 'string';
console.log(`Idle thought: "${idle.text}"`);
console.log(`Structure well-formed: ${structureOk}\n`);

// 2. REACTIVITY — idle thought is empty; reading drives a non-empty thought.
const idleEmpty = idle.words.length === 0;

const fearText = 'tengo mucho miedo y mucha tristeza';
const joyText = 'que alegria siento amor y felicidad';

const fear = thoughtAfterReading(brain, fearText);
console.log(`After "${fearText}"`);
console.log(`  thought: "${fear.text}"`);

const joy = thoughtAfterReading(brain, joyText);
console.log(`After "${joyText}"`);
console.log(`  thought: "${joy.text}"\n`);

const reactivityOk = idleEmpty && fear.words.length > 0 && joy.words.length > 0;

// 3. DISCRIMINATION — the two thoughts must not be the same word loop.
const wordOverlap = overlap(fear.words, joy.words);
const sameText = fear.text === joy.text;
const discriminationOk = !sameText && wordOverlap < 0.6;

console.log('── Metrics ──');
console.log(`  Structure well-formed:                  ${structureOk}`);
console.log(`  Reactivity (idle empty → read non-empty):${reactivityOk}  (idleEmpty=${idleEmpty}, fear=${fear.words.length}w, joy=${joy.words.length}w)`);
console.log(`  Discrimination (word overlap <0.60):     ${discriminationOk}  (overlap=${(wordOverlap * 100).toFixed(0)}%)`);

console.log('');
if (structureOk && reactivityOk && discriminationOk) {
  console.log('✅ LIVE THOUGHTS VERIFIED: well-formed, perception-driven, and discriminative.');
  process.exit(0);
} else {
  console.log(`❌ FAIL: structure=${structureOk} reactivity=${reactivityOk} discrimination=${discriminationOk}`);
  process.exit(1);
}
