/**
 * VERIFICATION TEST — Stage 1: learning what goes with what
 * ===========================================================================
 * The way an infant learns the word for a thing: someone shows the ball and
 * says "ball", again and again. Nobody explains anything; things that occur
 * together get bound together, and each one starts to bring the other to mind.
 *
 * The brain is driven through its public entry points only (`see`, `read`,
 * `hearFrame`), as the dashboard does.
 *
 *   1. REPETITION   — the association grows with every pairing; after ONE it is
 *                     too weak to be acted upon, after a few it is confident.
 *   2. NAMING       — seeing the drawing brings its word to mind: it shows up in
 *                     the recall, in the stream of thought and in what it says.
 *   3. IMAGERY      — reading the word brings the drawing's category to mind.
 *   4. SPECIFICITY  — each drawing recalls ITS word; a drawing never paired with
 *                     anything recalls nothing.
 *   5. SOUND        — the same between a sound and a drawing, in both directions.
 *   6. CROSS-SITUATIONAL — taught with whole sentences ("esto es una cruz"), it
 *                     singles out the word that goes with the drawing, not the
 *                     words that go with everything.
 *   7. REWARD       — dopamine makes the association form faster.
 *   8. MEMORY       — associations survive a restart.
 */

import { existsSync, rmSync } from 'fs';
import { DigitalBrain, type AssociationRecall } from '../src/brain.js';
import { ModulatorType } from '../src/core/neuromodulators/modulator-system.js';
import { mulberry32, quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

// ── Stimuli (same conventions as sensory-learning.test.ts) ──────────────────
const SIDE = 64;
const BACKGROUND = 27;
const INK = 255;

function drawing(draw: (plot: (x: number, y: number) => void) => void): number[] {
  const pixels = new Array<number>(SIDE * SIDE).fill(BACKGROUND);
  draw((x, y) => {
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const px = x + dx;
        const py = y + dy;
        if (px >= 0 && px < SIDE && py >= 0 && py < SIDE) pixels[py * SIDE + px] = INK;
      }
    }
  });
  return pixels;
}

const CROSS = drawing((plot) => { for (let i = 8; i < 56; i++) { plot(i, 32); plot(32, i); } });
const SQUARE = drawing((plot) => { for (let i = 12; i < 52; i++) { plot(i, 12); plot(i, 51); plot(12, i); plot(51, i); } });
const DIAGONAL = drawing((plot) => { for (let i = 6; i < 58; i++) plot(i, i); });

const stimulusNoise = mulberry32(11);
function vowel(formant1: number, formant2: number): number[] {
  return Array.from({ length: 512 }, (_, bin) => {
    const hz = bin * 46.875;
    const peak = (centre: number, width: number): number => Math.exp(-((hz - centre) ** 2) / (2 * width * width));
    return Math.min(1, 0.05 * stimulusNoise() + 0.9 * peak(formant1, 90) + 0.7 * peak(formant2, 120));
  });
}
const VOWEL_A = vowel(700, 1200);
const VOWEL_I = vowel(300, 2300);

// ── Driving the brain ───────────────────────────────────────────────────────
/** Between the two halves of a pairing: the first percept is complete, and still in mind. */
const TOGETHER_TICKS = 50;
/** How long a percept stays in mind here (the live default, 300 ticks, only makes the test slower). */
const WINDOW_TICKS = 120;
/** Between experiences: longer than the association window, so they stay separate events. */
const APART_TICKS = 180;

function newBrain(seed: number = SEED): DigitalBrain {
  seedRandom(seed);
  const brain = quiet(() => new DigitalBrain());
  brain.associationWindowTicks = WINDOW_TICKS;
  return brain;
}

const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const see = (brain: DigitalBrain, image: number[]): void => { quiet(() => brain.see(image, SIDE, SIDE, { propagate: false })); };
const read = (brain: DigitalBrain, text: string): void => { quiet(() => brain.read(text, { propagate: false })); };
const hear = (brain: DigitalBrain, frame: number[]): void => { quiet(() => brain.hearFrame(frame, 48000, { propagate: false })); };

/** One teaching episode: two stimuli together, then a pause. */
function teach(brain: DigitalBrain, first: () => void, second: () => void): void {
  first();
  wait(brain, TOGETHER_TICKS);
  second();
  wait(brain, APART_TICKS);
}

/** Presents a single stimulus and returns what it brought back from memory (if anything). */
function probe(brain: DigitalBrain, present: () => void): AssociationRecall | null {
  const before = brain.getLastRecall();
  present();
  wait(brain, TOGETHER_TICKS);
  const after = brain.getLastRecall();
  const recalled = after !== before ? after : null;
  wait(brain, APART_TICKS - TOGETHER_TICKS);
  return recalled;
}

const topWord = (recall: AssociationRecall | null): string => recall?.words[0]?.word ?? '—';

console.log('── Verification: stage 1 — learning what goes with what ──\n');

// ── 1. REPETITION ───────────────────────────────────────────────────────────
console.log('1. REPETITION');
const brain = newBrain();
const confidenceBySession: number[] = [];
const confidentBySession: boolean[] = [];
const PAIRINGS = 5;
for (let session = 1; session <= PAIRINGS; session++) {
  teach(brain, () => see(brain, CROSS), () => read(brain, 'cruz'));
  teach(brain, () => see(brain, SQUARE), () => read(brain, 'cuadrado'));
  const recall = probe(brain, () => see(brain, CROSS));
  confidenceBySession.push(recall?.confidence ?? 0);
  confidentBySession.push(recall?.confident ?? false);
}
{
  const rising = confidenceBySession.every((c, i) => i === 0 || c > confidenceBySession[i - 1]);
  check('the association grows with every pairing', rising,
    confidenceBySession.map((c) => c.toFixed(2)).join(' → '));
  check('one pairing is not enough to act on; a few are',
    !confidentBySession[0] && confidentBySession[2] && confidenceBySession[PAIRINGS - 1] >= 0.75,
    confidentBySession.map((c) => (c ? 'confident' : 'unsure')).join(', '));
}

// ── 2. NAMING ───────────────────────────────────────────────────────────────
console.log('\n2. NAMING');
{
  see(brain, CROSS);
  wait(brain, TOGETHER_TICKS);
  const recall = brain.getLastRecall();
  const thought = brain.think();
  const said = brain.speak().text;
  wait(brain, APART_TICKS);
  check('seeing the drawing recalls its word', topWord(recall) === 'cruz' && recall!.confident,
    `recalled "${topWord(recall)}" (${recall?.confidence.toFixed(2)})`);
  check('the word comes to mind (stream of thought)', thought.words.includes('cruz'), thought.text);
  check('…and it can say it', said.split(' ').includes('cruz'), `says "${said}"`);
}

// ── 3. IMAGERY ──────────────────────────────────────────────────────────────
console.log('\n3. IMAGERY');
const crossCategory = (() => {
  see(brain, CROSS);
  wait(brain, APART_TICKS);
  return brain.getRecognition().visual!.label;
})();
const squareCategory = (() => {
  see(brain, SQUARE);
  wait(brain, APART_TICKS);
  return brain.getRecognition().visual!.label;
})();
{
  const fromCruz = probe(brain, () => read(brain, 'cruz'));
  const fromCuadrado = probe(brain, () => read(brain, 'cuadrado'));
  check('reading the word recalls the drawing it names',
    fromCruz?.visual?.label === crossCategory && fromCuadrado?.visual?.label === squareCategory && crossCategory !== squareCategory,
    `"cruz"→${fromCruz?.visual?.label ?? '—'} "cuadrado"→${fromCuadrado?.visual?.label ?? '—'}`);
}

// ── 4. SPECIFICITY ──────────────────────────────────────────────────────────
console.log('\n4. SPECIFICITY');
{
  const fromSquare = probe(brain, () => see(brain, SQUARE));
  check('each drawing recalls ITS word', topWord(fromSquare) === 'cuadrado' &&
    !fromSquare!.words.some((w) => w.word === 'cruz'), `square → "${topWord(fromSquare)}"`);

  const fromUnpaired = probe(brain, () => see(brain, DIAGONAL));
  const fromUnknownWord = probe(brain, () => read(brain, 'montaña'));
  check('what was never paired with anything recalls nothing', fromUnpaired === null && fromUnknownWord === null,
    `diagonal → ${fromUnpaired ? topWord(fromUnpaired) : 'nothing'}`);
}

// ── 5. SOUND ────────────────────────────────────────────────────────────────
console.log('\n5. SOUND');
{
  for (let i = 0; i < 4; i++) {
    teach(brain, () => hear(brain, VOWEL_A), () => see(brain, DIAGONAL));
  }
  const diagonalCategory = brain.getRecognition().visual!.label;
  const soundCategory = brain.getRecognition().auditory!.label;

  const fromSound = probe(brain, () => hear(brain, VOWEL_A));
  const fromSight = probe(brain, () => see(brain, DIAGONAL));
  const fromOtherSound = probe(brain, () => hear(brain, VOWEL_I));
  check('a sound recalls the drawing it came with', fromSound?.visual?.label === diagonalCategory && fromSound.confident,
    `"a" → ${fromSound?.visual?.label ?? '—'} (${fromSound?.confidence.toFixed(2)})`);
  check('…and the drawing recalls the sound', fromSight?.auditory?.label === soundCategory,
    `diagonal → ${fromSight?.auditory?.label ?? '—'}`);
  check('a different sound recalls nothing', fromOtherSound === null);
}

// ── 6. CROSS-SITUATIONAL ────────────────────────────────────────────────────
console.log('\n6. CROSS-SITUATIONAL');
{
  const learner = newBrain(SEED + 1);
  for (let i = 0; i < 5; i++) {
    teach(learner, () => see(learner, CROSS), () => read(learner, 'esto es una cruz'));
    teach(learner, () => see(learner, SQUARE), () => read(learner, 'esto es un cuadrado'));
  }
  const fromCross = probe(learner, () => see(learner, CROSS));
  const fromSquare = probe(learner, () => see(learner, SQUARE));
  check('taught with whole sentences, it singles out the word that goes with each drawing',
    topWord(fromCross) === 'cruz' && topWord(fromSquare) === 'cuadrado',
    `cross → [${fromCross?.words.map((w) => w.word).join(', ')}]  square → [${fromSquare?.words.map((w) => w.word).join(', ')}]`);
}

// ── 7. REWARD ───────────────────────────────────────────────────────────────
console.log('\n7. REWARD');
{
  const confidenceAfterTwo = (dopamine: number): number => {
    const learner = newBrain(SEED + 2);
    for (let i = 0; i < 2; i++) {
      learner.getModulators().release(ModulatorType.Dopamine, dopamine);
      teach(learner, () => see(learner, CROSS), () => read(learner, 'cruz'));
    }
    return probe(learner, () => see(learner, CROSS))?.confidence ?? 0;
  };
  const neutral = confidenceAfterTwo(0);
  const rewarded = confidenceAfterTwo(0.6);
  check('dopamine makes the association form faster', rewarded > neutral * 1.1,
    `after 2 pairings: ${neutral.toFixed(2)} → ${rewarded.toFixed(2)} with dopamine`);
}

// ── 8. MEMORY ───────────────────────────────────────────────────────────────
console.log('\n8. MEMORY');
{
  const statePath = `/tmp/gbrain-association-test-${process.pid}.bin`;
  const files = ['', '.bak', '.tmp', '.lexicon.json'].map((suffix) => statePath + suffix);
  const cleanup = (): void => { for (const f of files) if (existsSync(f)) rmSync(f); };
  cleanup();

  quiet(() => brain.saveState(statePath));
  const restored = newBrain(SEED + 3);
  quiet(() => restored.loadState(statePath));
  const fromCross = probe(restored, () => see(restored, CROSS));
  const fromWord = probe(restored, () => read(restored, 'cuadrado'));
  check('a restored brain still names the drawing and pictures the word',
    topWord(fromCross) === 'cruz' && fromWord?.visual?.label === squareCategory,
    `cross → "${topWord(fromCross)}", "cuadrado" → ${fromWord?.visual?.label ?? '—'}`);
  cleanup();
}

// ── Verdict ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log('');
if (failed.length === 0) {
  console.log(`✅ ASSOCIATION LEARNING VERIFIED: ${results.length}/${results.length} checks — what is perceived together is learned together, by repetition.`);
  process.exit(0);
} else {
  console.log(`❌ FAIL: ${failed.map(([name]) => name).join(' | ')}`);
  process.exit(1);
}
