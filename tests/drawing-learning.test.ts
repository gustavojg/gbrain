/**
 * VERIFICATION TEST — Stage 3: a hand that learns to draw, and a teacher's verdict
 * ===========================================================================
 * Infants scribble long before they draw: each scribble teaches the hand which
 * marks its movements leave. The brain has a whiteboard and a hand motor
 * cortex fed by the retinotopic visual signal; nothing tells it which command
 * inks which place.
 *
 *   1. BEFORE SCRIBBLING — it sees a drawing and cannot copy it.
 *   2. SCRIBBLING     — with the hand on it scribbles on its own; off, it is
 *                       still. The more it has scribbled, the better its copies.
 *   3. SPECIFICITY    — each copy looks like ITS model, not like the others.
 *   4. NO LOOP        — it copies a drawing once; seeing its own copy does not
 *                       make it copy again.
 *   5. FROM MEMORY    — taught that a word goes with a drawing, it DRAWS the
 *                       drawing when it reads the word — and writes the word
 *                       when it sees the drawing.
 *   6. FEEDBACK       — "no, that's not it" weakens what it has just recalled;
 *                       "yes" strengthens it.
 *   7. MEMORY         — the skill survives a restart.
 */

import { existsSync, rmSync } from 'fs';
import { DigitalBrain, type HandDrawing } from '../src/brain.js';
import { inkedCells, likeness } from '../src/core/hand/whiteboard.js';
import { quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

// ── Models to copy (as drawn on the dashboard's whiteboard) ─────────────────
const SIDE = 64;
function drawing(draw: (plot: (x: number, y: number) => void) => void): number[] {
  const pixels = new Array<number>(SIDE * SIDE).fill(27);
  draw((x, y) => {
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const px = x + dx;
        const py = y + dy;
        if (px >= 0 && px < SIDE && py >= 0 && py < SIDE) pixels[py * SIDE + px] = 255;
      }
    }
  });
  return pixels;
}
const MODELS: Record<string, number[]> = {
  cross: drawing((plot) => { for (let i = 8; i < 56; i++) { plot(i, 32); plot(32, i); } }),
  square: drawing((plot) => { for (let i = 12; i < 52; i++) { plot(i, 12); plot(i, 51); plot(12, i); plot(51, i); } }),
  diagonal: drawing((plot) => { for (let i = 6; i < 58; i++) plot(i, i); }),
};
const CELLS = Object.fromEntries(Object.entries(MODELS).map(([name, image]) => [name, inkedCells(image, SIDE, SIDE)]));

// ── Driving the brain ───────────────────────────────────────────────────────
const SCRIBBLE_TICKS = 60;
const ANSWER_TICKS = 150;

const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };

function newBrain(seed: number): DigitalBrain {
  seedRandom(seed);
  const brain = quiet(() => new DigitalBrain());
  brain.setHand({ copy: true });
  brain.associationWindowTicks = 120;
  return brain;
}

/** Shows a drawing; returns what the hand drew in answer, or `null` if it stayed still. */
function showTo(brain: DigitalBrain, image: number[]): HandDrawing | null {
  const before = brain.getLastDrawing()?.serial ?? 0;
  quiet(() => brain.see(image, SIDE, SIDE, { propagate: false }));
  wait(brain, ANSWER_TICKS);
  const after = brain.getLastDrawing();
  return after && after.serial !== before ? after : null;
}

function examine(brain: DigitalBrain): { copies: Record<string, HandDrawing | null>; mean: number; detail: string } {
  const copies: Record<string, HandDrawing | null> = {};
  let total = 0;
  const detail: string[] = [];
  for (const [name, image] of Object.entries(MODELS)) {
    const copy = showTo(brain, image);
    copies[name] = copy;
    const score = copy ? likeness(CELLS[name], copy.cells) : 0;
    total += score;
    detail.push(`${name} ${score.toFixed(2)}`);
  }
  return { copies, mean: total / Object.keys(MODELS).length, detail: detail.join(' · ') };
}

function scribble(brain: DigitalBrain, times: number): void {
  for (let i = 0; i < times; i++) {
    brain.scribbleOnce();
    wait(brain, SCRIBBLE_TICKS);
  }
}

console.log('── Verification: stage 3 — a hand that learns to draw ──\n');
const brain = newBrain(SEED);

// ── 1. BEFORE SCRIBBLING ────────────────────────────────────────────────────
console.log('1. BEFORE SCRIBBLING');
const naive = examine(brain);
check('it cannot copy any drawing it sees', Object.values(naive.copies).every((c) => c === null));

// ── 2. SCRIBBLING ───────────────────────────────────────────────────────────
console.log('\n2. SCRIBBLING');
{
  wait(brain, 200);
  check('hand off → it is still', brain.getLastDrawing() === null);
  brain.setHand({ scribble: true });
  wait(brain, 400);
  const spontaneous = brain.getState().hand!.scribbles;
  brain.setHand({ scribble: false });
  wait(brain, 150);
  const afterOff = brain.getState().hand!.scribbles;
  wait(brain, 300);
  check('hand on → it scribbles on its own; off → it stops',
    spontaneous >= 3 && brain.getState().hand!.scribbles === afterOff, `${spontaneous} scribbles in 400 ticks`);
}
scribble(brain, 35);
const early = examine(brain);
scribble(brain, 120);
const later = examine(brain);
check('the more it has scribbled, the better its copies',
  later.mean > early.mean && early.mean > naive.mean && later.mean >= 0.35,
  `likeness to the model — 0 scribbles: ${naive.mean.toFixed(2)} · ~40: ${early.mean.toFixed(2)} (${early.detail}) · ~160: ${later.mean.toFixed(2)} (${later.detail})`);

// ── 3. SPECIFICITY ──────────────────────────────────────────────────────────
console.log('\n3. SPECIFICITY');
{
  const names = Object.keys(MODELS);
  const table: string[] = [];
  const specific = names.every((name) => {
    const copy = later.copies[name];
    if (!copy) { table.push(`${name}: no copy`); return false; }
    const own = likeness(CELLS[name], copy.cells);
    table.push(`${name} copy → ${names.map((m) => `${m} ${likeness(CELLS[m], copy.cells).toFixed(2)}`).join(', ')}`);
    return names.every((other) => other === name || own > likeness(CELLS[other], copy.cells));
  });
  check('each copy looks like its own model, not like the others', specific, table.join(' | '));
}

// ── 4. NO LOOP ──────────────────────────────────────────────────────────────
console.log('\n4. NO LOOP');
{
  const copy = showTo(brain, MODELS.cross);
  wait(brain, 400);
  check('it copies a drawing once — seeing its own copy does not set it off again',
    copy !== null && copy.source === 'copy' && brain.getLastDrawing()!.serial === copy.serial);
}

// ── 5. FROM MEMORY ──────────────────────────────────────────────────────────
console.log('\n5. FROM MEMORY');
const written: string[] = [];
brain.on('response', (event) => { if (event.data.kind === 'writing') written.push(String(event.data.text)); });
{
  brain.setHand({ copy: false }); // while being taught it just watches
  for (let i = 0; i < 4; i++) {
    for (const [word, model] of [['cruz', MODELS.cross], ['cuadrado', MODELS.square]] as const) {
      quiet(() => brain.see(model, SIDE, SIDE, { propagate: false }));
      wait(brain, 50);
      quiet(() => brain.read(word, { propagate: false }));
      wait(brain, 180);
    }
  }
  brain.setHand({ copy: true });

  const before = brain.getLastDrawing()!.serial;
  quiet(() => brain.read('cruz', { propagate: false }));
  wait(brain, ANSWER_TICKS);
  const fromMemory = brain.getLastDrawing()!;
  const own = likeness(CELLS.cross, fromMemory.cells);
  const other = likeness(CELLS.square, fromMemory.cells);
  check('reading the word, it draws the drawing that goes with it',
    fromMemory.serial !== before && fromMemory.source === 'from-memory' && own >= 0.3 && own > other * 1.5,
    `"cruz" → likeness to the cross ${own.toFixed(2)}, to the square ${other.toFixed(2)}`);

  written.length = 0;
  brain.setHand({ copy: false });
  quiet(() => brain.see(MODELS.cross, SIDE, SIDE, { propagate: false }));
  wait(brain, ANSWER_TICKS);
  brain.setHand({ copy: true });
  check('seeing the drawing, it writes its word', written.includes('cruz'), `wrote: ${written.join(' ') || '—'}`);
}

// ── 6. FEEDBACK ─────────────────────────────────────────────────────────────
console.log('\n6. FEEDBACK');
{
  brain.setHand({ copy: false });
  const confidenceOnSeeing = (image: number[]): number => {
    quiet(() => brain.see(image, SIDE, SIDE, { propagate: false }));
    wait(brain, ANSWER_TICKS);
    return brain.getLastRecall()?.confidence ?? 0;
  };
  const before = confidenceOnSeeing(MODELS.square);
  brain.giveFeedback(false);
  const afterNo = confidenceOnSeeing(MODELS.square);
  brain.giveFeedback(true);
  const afterYes = confidenceOnSeeing(MODELS.square);
  check('"no" weakens what it has just recalled; "yes" strengthens it',
    afterNo < before * 0.8 && afterYes > afterNo * 1.1,
    `square → "cuadrado": ${before.toFixed(2)} → 👎 ${afterNo.toFixed(2)} → 👍 ${afterYes.toFixed(2)}`);
  brain.setHand({ copy: true });
}

// ── 7. MEMORY ───────────────────────────────────────────────────────────────
console.log('\n7. MEMORY');
{
  const statePath = `/tmp/gbrain-drawing-test-${process.pid}.bin`;
  const files = ['', '.bak', '.tmp', '.lexicon.json'].map((suffix) => statePath + suffix);
  const cleanup = (): void => { for (const f of files) if (existsSync(f)) rmSync(f); };
  cleanup();

  quiet(() => brain.saveState(statePath));
  const restored = newBrain(SEED + 1);
  quiet(() => restored.loadState(statePath));
  const copy = showTo(restored, MODELS.cross);
  const score = copy ? likeness(CELLS.cross, copy.cells) : 0;
  check('a restored brain still copies what it sees, without scribbling again',
    restored.getState().hand!.scribbles === 0 && score >= 0.35, `cross ${score.toFixed(2)}`);
  cleanup();
}

// ── Verdict ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log('');
if (failed.length === 0) {
  console.log(`✅ DRAWING LEARNING VERIFIED: ${results.length}/${results.length} checks — by scribbling and watching its marks, it learns to draw what it sees and what it remembers.`);
  process.exit(0);
} else {
  console.log(`❌ FAIL: ${failed.map(([name]) => name).join(' | ')}`);
  process.exit(1);
}
