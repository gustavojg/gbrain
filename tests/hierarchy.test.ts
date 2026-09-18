/**
 * VERIFICATION TEST — Block 3: the parts hierarchy (V2 → IT)
 * ===========================================================================
 * Above the whole-image cortex sits a level that sees objects as arrangements
 * of local parts, pooled over position. A partial view still drives most of
 * an object's parts, so the object is completed — which a single template
 * over the whole retina could not do.
 *
 *   1. PARTS      — seeing the cross and the square, the parts cortex learns
 *                   a few local parts and one object for each.
 *   2. COMPLETION — half a square is a bracket to V1 and the square to the
 *                   object level. (Half a cross — a T — is reported: it is
 *                   as much a square's corner as a cross's arm to this level.)
 *   3. WORDS      — taught "cuadrado", half a square brings the word back
 *                   through the object level.
 *   4. PERSISTENCE — parts and objects survive a restart.
 */
import { existsSync, rmSync } from 'fs';
import { DigitalBrain, type BrainEvent } from '../src/brain.js';
import { quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);
const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

const SIDE = 64;
function drawing(draw: (plot: (x: number, y: number) => void) => void): number[] {
  const pixels = new Array<number>(SIDE * SIDE).fill(27);
  draw((x, y) => {
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const px = x + dx, py = y + dy;
      if (px >= 0 && px < SIDE && py >= 0 && py < SIDE) pixels[py * SIDE + px] = 255;
    }
  });
  return pixels;
}
const CROSS = drawing((plot) => { for (let i = 8; i < 56; i++) { plot(i, 32); plot(32, i); } });
const SQUARE = drawing((plot) => { for (let i = 12; i < 52; i++) { plot(i, 12); plot(i, 51); plot(12, i); plot(51, i); } });
/** The right part of a drawing hidden (background). */
const occludeRight = (image: number[], fraction: number): number[] =>
  image.map((v, i) => ((i % SIDE) >= SIDE * (1 - fraction) ? 27 : v));

const newBrain = (): DigitalBrain => { seedRandom(SEED); const b = quiet(() => new DigitalBrain()); b.associationWindowTicks = 120; return b; };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const see = (brain: DigitalBrain, image: number[]): { v1: string; v1New: boolean; object: string | null; objectNew: boolean } => {
  quiet(() => brain.see(image, SIDE, SIDE, { propagate: false }));
  wait(brain, 60);
  const r = brain.getRecognition();
  wait(brain, 200);
  return { v1: r.visual?.label ?? '—', v1New: r.visual?.isNew ?? true, object: r.object?.label ?? null, objectNew: r.object?.isNew ?? true };
};

console.log('── Verification: block 3 — the parts hierarchy ──\n');

// ── 1. PARTS ────────────────────────────────────────────────────────────────
console.log('1. PARTS');
const brain = newBrain();
let cross = see(brain, CROSS);
let square = see(brain, SQUARE);
for (let i = 0; i < 2; i++) { cross = see(brain, CROSS); square = see(brain, SQUARE); }
{
  const r = brain.getRecognition();
  check('local parts are learned from the drawings', r.partsKnown >= 2 && r.partsKnown <= 24, `${r.partsKnown} parts`);
  check('each drawing is one object, seen again as the same', r.objectCategories === 2 && cross.object !== null && square.object !== null && cross.object !== square.object && !cross.objectNew && !square.objectNew,
    `${cross.object} (cross), ${square.object} (square); ${r.objectCategories} objects`);
}

// ── 2. COMPLETION ───────────────────────────────────────────────────────────
console.log('\n2. COMPLETION');
{
  const halfSquare = see(brain, occludeRight(SQUARE, 0.5));
  const halfCross = see(brain, occludeRight(CROSS, 0.5));
  check('half a square is the square at the object level', halfSquare.object === square.object && !halfSquare.objectNew,
    `object ${halfSquare.object}${halfSquare.objectNew ? ' NEW' : ''}; V1 ${halfSquare.v1}${halfSquare.v1New ? ' NEW' : ''}`);
  console.log(`   ${halfCross.object === cross.object ? '🎉' : 'ℹ️ '} half a cross (a T) → object ${halfCross.object}${halfCross.objectNew ? ' NEW' : ''}, V1 ${halfCross.v1}${halfCross.v1New ? ' NEW' : ''}${halfCross.object === cross.object ? '' : ' (the T is ambiguous at this level; V1 keeps it a cross)'}`);
  check('no object was founded for the halves', brain.getRecognition().objectCategories === 2, `${brain.getRecognition().objectCategories} objects`);
}

// ── 3. WORDS ────────────────────────────────────────────────────────────────
console.log('\n3. WORDS');
{
  const written: string[] = [];
  brain.on('response', (e: BrainEvent) => { if (e.data.kind === 'writing') written.push(String(e.data.text)); });
  const teach = (image: number[], word: string): void => {
    quiet(() => brain.see(image, SIDE, SIDE, { propagate: false }));
    wait(brain, 60);
    quiet(() => brain.read(word, { propagate: false }));
    wait(brain, 180);
  };
  for (let i = 0; i < 3; i++) { teach(SQUARE, 'cuadrado'); teach(CROSS, 'cruz'); }
  written.length = 0;
  quiet(() => brain.see(occludeRight(SQUARE, 0.5), SIDE, SIDE, { propagate: false }));
  wait(brain, 160);
  const half = written.join(' ');
  const halfRecall = brain.getLastRecall();
  written.length = 0;
  wait(brain, 100);
  quiet(() => brain.see(occludeRight(CROSS, 0.5), SIDE, SIDE, { propagate: false }));
  wait(brain, 160);
  const halfC = written.join(' ');
  check('half a square brings "cuadrado" back', half.includes('cuadrado') && !half.includes('cruz'),
    `${half || 'nothing written'}; recall ${halfRecall ? `${halfRecall.confidence.toFixed(2)} [${halfRecall.words.map((w) => w.word).join(' ')}] visual ${halfRecall.visual?.label ?? '—'} cue ${halfRecall.cue.label}` : '—'}`);
  check('half a cross brings "cruz" back', halfC.includes('cruz') && !halfC.includes('cuadrado'), halfC || 'nothing written');
}

// ── 4. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n4. PERSISTENCE');
{
  const path = `/tmp/gbrain-hierarchy-${process.pid}.bin`;
  quiet(() => brain.saveState(path));
  const restored = newBrain();
  quiet(() => restored.loadState(path));
  for (const p of [path, `${path}.lexicon.json`]) if (existsSync(p)) rmSync(p);
  const parts = restored.getRecognition().partsKnown;
  const objects = restored.getRecognition().objectCategories;
  const half = see(restored, occludeRight(SQUARE, 0.5));
  check('parts and objects survive a restart', parts >= 2 && objects === 2 && half.object === square.object && !half.objectNew,
    `${parts} parts, ${objects} objects; half a square → ${half.object}${half.objectNew ? ' NEW' : ''}`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
