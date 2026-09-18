/**
 * VERIFICATION TEST — Stage 0: senses that learn by exposure
 * ===========================================================================
 * The first step of "learning like a baby": nobody tells the brain what it is
 * looking at or listening to. It is simply shown drawings and played vowels,
 * through the same entry points the dashboard uses (`see`, `hearFrame`), and it
 * must form perceptual categories on its own.
 *
 *   1. SIGHT     — distinct drawings found distinct categories; a drawing seen
 *                  again is recognized (familiarity ≥ 0.8) and its exposure
 *                  count grows.
 *   2. GENERALIZATION — a noisy or half-occluded drawing is recognized as the
 *                  drawing it comes from — not as the other one — while a
 *                  genuinely new drawing founds a new category.
 *   3. HEARING   — the same for vowels: five vowels → five categories; a noisy
 *                  or slightly shifted vowel is recognized, a new one is new.
 *   4. SELECTIVITY — typed text founds no visual or auditory category, and
 *                  silence / noise found no sound category.
 *   5. MEMORY    — what the senses learned survives a restart: a restored
 *                  brain recognizes the drawing and the vowel it knew.
 *
 * Everything is seeded; `TEST_SEED=n` checks that it is not seed-specific.
 */

import { existsSync, rmSync } from 'fs';
import { DigitalBrain } from '../src/brain.js';
import type { Recognition } from '../src/core/memory/prototype-memory.js';
import { mulberry32, quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

// ── Stimuli ─────────────────────────────────────────────────────────────────
// Drawings as the dashboard's whiteboard produces them: 64×64, dark background
// (#111827 → grey 27), white 3-px strokes.
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
const withNoise = (image: number[], fraction: number): number[] =>
  image.map((v) => (stimulusNoise() < fraction ? (v === INK ? BACKGROUND : INK) : v));
const occludeRight = (image: number[], fraction: number): number[] =>
  image.map((v, i) => (i % SIDE > SIDE * (1 - fraction) ? BACKGROUND : v));

// Vowels as one microphone frame: 512 linear FFT bins at 48 kHz (fftSize 1024),
// two formant peaks over a little broadband noise.
const VOWELS = { a: [700, 1200], e: [500, 1900], i: [300, 2300], o: [500, 900], u: [350, 800] } as const;
function vowel(formant1: number, formant2: number, noise = 0.05): number[] {
  return Array.from({ length: 512 }, (_, bin) => {
    const hz = bin * 46.875;
    const peak = (centre: number, width: number): number => Math.exp(-((hz - centre) ** 2) / (2 * width * width));
    return Math.min(1, noise * stimulusNoise() + 0.9 * peak(formant1, 90) + 0.7 * peak(formant2, 120));
  });
}
const HISS = Array.from({ length: 512 }, (_, bin) => (bin > 160 ? 0.6 : 0.02));

// ── Driving the brain ───────────────────────────────────────────────────────
/** Long enough for the presentation, its wave and the recovery of the pathways. */
const SETTLE_TICKS = 250;

function newBrain(seed: number): DigitalBrain {
  seedRandom(seed);
  return quiet(() => new DigitalBrain());
}

function settle(brain: DigitalBrain): void {
  for (let t = 0; t < SETTLE_TICKS; t++) brain.tick();
}

function show(brain: DigitalBrain, image: number[]): Recognition {
  quiet(() => brain.see(image, SIDE, SIDE, { propagate: false }));
  settle(brain);
  return brain.getRecognition().visual!;
}

function play(brain: DigitalBrain, frame: number[]): Recognition | null {
  quiet(() => brain.hearFrame(frame, 48000, { propagate: false }));
  settle(brain);
  return brain.getRecognition().auditory;
}

const describe = (r: Recognition | null): string =>
  r ? `${r.label}${r.isNew ? ' NEW' : ` ×${r.exposures} ${Math.round(r.familiarity * 100)}%`}` : 'nothing';

console.log('── Verification: stage 0 — senses that learn by exposure ──\n');
const brain = newBrain(SEED);

// ── 1. SIGHT ────────────────────────────────────────────────────────────────
console.log('1. SIGHT');
const REPETITIONS = 4;
const crossSeen: Recognition[] = [];
const squareSeen: Recognition[] = [];
for (let i = 0; i < REPETITIONS; i++) {
  crossSeen.push(show(brain, CROSS));
  squareSeen.push(show(brain, SQUARE));
}
{
  check('each new drawing founds its own category, unprompted',
    crossSeen[0].isNew && squareSeen[0].isNew && crossSeen[0].id !== squareSeen[0].id,
    `${describe(crossSeen[0])} / ${describe(squareSeen[0])}`);

  const recognized = (seen: Recognition[]): boolean =>
    seen.slice(1).every((r) => !r.isNew && r.id === seen[0].id && r.familiarity >= 0.8);
  check('a drawing seen again is recognized', recognized(crossSeen) && recognized(squareSeen),
    `cross: ${crossSeen.map((r) => Math.round(r.familiarity * 100)).join(',')}%`);
  check('exposures add up', crossSeen[REPETITIONS - 1].exposures === REPETITIONS &&
    brain.getRecognition().visualCategories === 2, `categories=${brain.getRecognition().visualCategories}`);
}

// ── 2. GENERALIZATION ───────────────────────────────────────────────────────
console.log('\n2. GENERALIZATION');
{
  const crossId = crossSeen[0].id;
  const squareId = squareSeen[0].id;
  const squareObject = brain.getRecognition().object?.label ?? null;
  const noisyCross = show(brain, withNoise(CROSS, 0.15));
  const halfCross = show(brain, occludeRight(CROSS, 0.5));
  const halfSquare = show(brain, occludeRight(SQUARE, 0.5));
  check('a noisy drawing (15% of pixels flipped) is recognized', noisyCross.id === crossId && !noisyCross.isNew, describe(noisyCross));
  check('half a drawing is not mistaken for another', halfCross.id === crossId && halfSquare.id !== crossId,
    `${describe(halfCross)} / ${describe(halfSquare)}`);
  // Half a square, re-centred by foveation, is a bracket to V1 (a whole-image
  // template has nothing to complete it with); the parts cortex above it
  // completes the square from the parts present (block 3, hierarchy).
  const halfSquareObject = brain.getRecognition().object;
  check('half a square is still the square — at the object level', halfSquareObject !== null && halfSquareObject.label === squareObject && !halfSquareObject.isNew,
    `V1: ${describe(halfSquare)}; object: ${halfSquareObject?.label ?? '—'}${halfSquareObject?.isNew ? ' NEW' : ''} (square = ${squareObject})`);
  console.log(`   ${halfSquare.id === squareId ? '🎉' : 'ℹ️ '} V1 alone ${halfSquare.id === squareId ? 'sees the square too' : 'sees a bracket'}  (${describe(halfSquare)})`);

  const novel = show(brain, DIAGONAL);
  check('a genuinely new drawing founds a new category', novel.isNew && novel.id !== crossId && novel.id !== squareId, describe(novel));
  const again = show(brain, CROSS);
  check('…without disturbing the known ones', again.id === crossId && again.familiarity >= 0.8, describe(again));
}

// ── 3. HEARING ──────────────────────────────────────────────────────────────
console.log('\n3. HEARING');
const heard: Record<string, Array<Recognition | null>> = {};
for (let rep = 0; rep < 3; rep++) {
  for (const [name, [f1, f2]] of Object.entries(VOWELS)) {
    (heard[name] ??= []).push(play(brain, vowel(f1, f2)));
  }
}
{
  const ids = Object.values(heard).map((rs) => rs[0]?.id);
  check('five vowels found five sound categories, unprompted',
    Object.values(heard).every((rs) => rs[0]?.isNew === true) && new Set(ids).size === 5,
    Object.entries(heard).map(([name, rs]) => `${name}→${rs[0]?.label}`).join(' '));
  check('a vowel heard again is recognized',
    Object.values(heard).every((rs) => rs.slice(1).every((r) => r !== null && !r.isNew && r.id === rs[0]!.id && r.familiarity >= 0.5)));

  const aId = heard.a[0]!.id;
  const noisy = play(brain, vowel(700, 1200, 0.3));
  const shifted = play(brain, vowel(740, 1240));
  check('a noisy or slightly shifted vowel is recognized', noisy?.id === aId && shifted?.id === aId,
    `${describe(noisy)} / ${describe(shifted)}`);
  check('no spurious categories', brain.getRecognition().auditoryCategories === 5,
    `categories=${brain.getRecognition().auditoryCategories}`);
}

// ── 4. SELECTIVITY ──────────────────────────────────────────────────────────
console.log('\n4. SELECTIVITY');
{
  const before = brain.getRecognition();
  quiet(() => brain.read('el perro corre por el parque', { propagate: false }));
  settle(brain);
  play(brain, HISS);
  const after = brain.getRecognition();
  check('typed text and broadband noise found no perceptual category',
    after.visualCategories === before.visualCategories && after.auditoryCategories === before.auditoryCategories,
    `sight ${after.visualCategories}, hearing ${after.auditoryCategories}`);
}

// ── 5. MEMORY ───────────────────────────────────────────────────────────────
console.log('\n5. MEMORY');
{
  const statePath = `/tmp/gbrain-sensory-test-${process.pid}.bin`;
  const files = ['', '.bak', '.tmp', '.lexicon.json'].map((suffix) => statePath + suffix);
  const cleanup = (): void => { for (const f of files) if (existsSync(f)) rmSync(f); };
  cleanup();

  const crossBefore = show(brain, CROSS);
  const vowelBefore = play(brain, vowel(...VOWELS.a))!;
  quiet(() => brain.saveState(statePath));

  const restored = newBrain(SEED + 1); // different seed ⇒ a different brain until it loads
  quiet(() => restored.loadState(statePath));
  const crossAfter = show(restored, CROSS);
  const vowelAfter = play(restored, vowel(...VOWELS.a));
  check('a restored brain recognizes the drawing it knew',
    !crossAfter.isNew && crossAfter.id === crossBefore.id && crossAfter.exposures === crossBefore.exposures + 1,
    describe(crossAfter));
  check('…and the vowel it knew',
    vowelAfter !== null && !vowelAfter.isNew && vowelAfter.id === vowelBefore.id, describe(vowelAfter));
  cleanup();
}

// ── Verdict ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log('');
if (failed.length === 0) {
  console.log(`✅ SENSORY LEARNING VERIFIED: ${results.length}/${results.length} checks — the senses form and recognize categories from exposure alone.`);
  process.exit(0);
} else {
  console.log(`❌ FAIL: ${failed.map(([name]) => name).join(' | ')}`);
  process.exit(1);
}
