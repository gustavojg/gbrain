/**
 * VERIFICATION TEST — Stage 2: learning to use its voice
 * ===========================================================================
 * An infant does not know what its mouth can do. It babbles, hears what comes
 * out, and that is how it learns to make — on purpose — the sounds it hears.
 *
 * The brain has a vocal tract (two formants) and a vocal motor cortex fed by
 * the auditory cortex. Nothing tells it which command makes which sound.
 *
 *   1. BEFORE BABBLING — it hears a vowel and cannot repeat it.
 *   2. BABBLING      — with the voice switched on it babbles on its own; with
 *                      it off it stays silent. The more it has babbled, the
 *                      more of the vowels it hears it can repeat.
 *   3. ACCURACY      — what it repeats is close to what it heard (formants).
 *   4. NO ECHO       — it repeats a sound once; hearing ITS OWN repetition
 *                      does not make it repeat again.
 *   5. SELECTIVE     — it only answers sounds: text and drawings leave it silent.
 *   6. MEMORY        — the skill survives a restart, without babbling again.
 */

import { existsSync, rmSync } from 'fs';
import { DigitalBrain, type Vocalization } from '../src/brain.js';
import { synthesizeSpectrum, VOCAL_RANGE } from '../src/core/voice/vocal-tract.js';
import { quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

/** Vowels "spoken by someone else" — here, by an identical vocal tract. */
const VOWELS: Record<string, [number, number]> = {
  a: [700, 1200], e: [500, 1900], i: [300, 2300], o: [500, 900], u: [350, 800],
};

/** Ticks for a sound to be heard, recognized and (maybe) repeated. */
const ANSWER_TICKS = 140;
/** One babble: the utterance, its way back through the ear, a breath. */
const BABBLE_TICKS = 60;

const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };

function newBrain(seed: number): DigitalBrain {
  seedRandom(seed);
  const brain = quiet(() => new DigitalBrain());
  brain.setVoice({ imitate: true });
  return brain;
}

/** Plays a vowel to the brain; returns its repetition, or `null` if it stayed silent. */
function sayTo(brain: DigitalBrain, f1: number, f2: number): Vocalization | null {
  const before = brain.getLastVocalization()?.serial ?? 0;
  quiet(() => brain.hearFrame(synthesizeSpectrum({ f1, f2, amplitude: 0.9 }), 48000, { propagate: false }));
  wait(brain, ANSWER_TICKS);
  const after = brain.getLastVocalization();
  return after && after.serial !== before && after.source === 'imitation' ? after : null;
}

/** Formant error as a fraction of the vocal range (0 = exact, ~0.33 = random). */
function error(v: Vocalization, f1: number, f2: number): number {
  const span1 = VOCAL_RANGE.f1[1] - VOCAL_RANGE.f1[0];
  const span2 = VOCAL_RANGE.f2[1] - VOCAL_RANGE.f2[0];
  return (Math.abs(v.command.f1 - f1) / span1 + Math.abs(v.command.f2 - f2) / span2) / 2;
}

function examine(brain: DigitalBrain): { repeated: number; meanError: number; detail: string } {
  const errors: number[] = [];
  const detail: string[] = [];
  for (const [name, [f1, f2]] of Object.entries(VOWELS)) {
    const answer = sayTo(brain, f1, f2);
    if (answer) {
      errors.push(error(answer, f1, f2));
      detail.push(`${name}→${answer.command.f1.toFixed(0)}/${answer.command.f2.toFixed(0)}`);
    } else {
      detail.push(`${name}→—`);
    }
  }
  const meanError = errors.length > 0 ? errors.reduce((s, e) => s + e, 0) / errors.length : NaN;
  return { repeated: errors.length, meanError, detail: detail.join(' ') };
}

function babble(brain: DigitalBrain, times: number): void {
  for (let i = 0; i < times; i++) {
    brain.babbleOnce();
    wait(brain, BABBLE_TICKS);
  }
}

console.log('── Verification: stage 2 — learning to use its voice ──\n');
const brain = newBrain(SEED);

// ── 1. BEFORE BABBLING ──────────────────────────────────────────────────────
console.log('1. BEFORE BABBLING');
const naive = examine(brain);
check('it cannot repeat any vowel it hears', naive.repeated === 0, naive.detail);

// ── 2. BABBLING ─────────────────────────────────────────────────────────────
console.log('\n2. BABBLING');
{
  wait(brain, 300);
  check('voice off → it stays silent', brain.getLastVocalization() === null && brain.getState().voice!.babbles === 0);

  brain.setVoice({ babble: true });
  wait(brain, 400);
  const spontaneous = brain.getState().voice!.babbles;
  brain.setVoice({ babble: false });
  wait(brain, 150);
  const afterOff = brain.getState().voice!.babbles;
  wait(brain, 300);
  check('voice on → it babbles on its own; off → it stops',
    spontaneous >= 3 && brain.getState().voice!.babbles === afterOff, `${spontaneous} babbles in 400 ticks`);
}
babble(brain, 40);
const early = examine(brain);
babble(brain, 110);
const later = examine(brain);
{
  check('the more it has babbled, the more vowels it can repeat',
    later.repeated > naive.repeated && later.repeated >= early.repeated && later.repeated >= 4,
    `0 babbles: ${naive.repeated}/5 · ~45: ${early.repeated}/5 · ~155: ${later.repeated}/5`);
}

// ── 3. ACCURACY ─────────────────────────────────────────────────────────────
console.log('\n3. ACCURACY');
check('what it repeats is close to what it heard', later.meanError <= 0.1,
  `mean formant error ${(later.meanError * 100).toFixed(1)}% of the vocal range (random ≈ 33%) — ${later.detail}`);

// ── 4. NO ECHO ──────────────────────────────────────────────────────────────
console.log('\n4. NO ECHO');
{
  let answer: Vocalization | null = null;
  for (const [f1, f2] of Object.values(VOWELS)) {
    answer = sayTo(brain, f1, f2);
    if (answer) break;
  }
  wait(brain, 400);
  check('it repeats a sound once — its own repetition does not set it off again',
    answer !== null && brain.getLastVocalization()!.serial === answer.serial);
}

// ── 5. SELECTIVE ────────────────────────────────────────────────────────────
console.log('\n5. SELECTIVE');
{
  const before = brain.getLastVocalization()!.serial;
  quiet(() => brain.read('hola cerebro como estas', { propagate: false }));
  wait(brain, 200);
  const image = new Array<number>(64 * 64).fill(27).map((v, i) => (i % 64 === 32 || Math.floor(i / 64) === 32 ? 255 : v));
  quiet(() => brain.see(image, 64, 64, { propagate: false }));
  wait(brain, 200);
  check('text and drawings leave the voice silent', brain.getLastVocalization()!.serial === before);
}

// ── 6. MEMORY ───────────────────────────────────────────────────────────────
console.log('\n6. MEMORY');
{
  const statePath = `/tmp/gbrain-vocal-test-${process.pid}.bin`;
  const files = ['', '.bak', '.tmp', '.lexicon.json'].map((suffix) => statePath + suffix);
  const cleanup = (): void => { for (const f of files) if (existsSync(f)) rmSync(f); };
  cleanup();

  quiet(() => brain.saveState(statePath));
  const restored = newBrain(SEED + 1);
  quiet(() => restored.loadState(statePath));
  const exam = examine(restored);
  check('a restored brain still repeats what it hears, without babbling again',
    restored.getState().voice!.babbles === 0 && exam.repeated >= later.repeated - 1 && exam.meanError <= 0.12,
    `${exam.repeated}/5, error ${(exam.meanError * 100).toFixed(1)}%`);
  cleanup();
}

// ── Verdict ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log('');
if (failed.length === 0) {
  console.log(`✅ VOCAL LEARNING VERIFIED: ${results.length}/${results.length} checks — by babbling and listening to itself, it learns to repeat the sounds it hears.`);
  process.exit(0);
} else {
  console.log(`❌ FAIL: ${failed.map(([name]) => name).join(' | ')}`);
  process.exit(1);
}
