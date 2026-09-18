/**
 * VERIFICATION TEST — Habits: a practised response stops needing a reason
 * ===========================================================================
 * The same response can be goal-directed (made for what it brings; it stops
 * as soon as the outcome is devalued) or habitual (stamped in by repetition;
 * it keeps running after devaluation and extinguishes slowly). Dickinson's
 * outcome-devaluation test, on the brain's naming of what it sees.
 *
 *   1. PRACTICE     — the car's name, written and praised many times, becomes
 *                     a habit; the cross's, written twice, does not.
 *   2. DEVALUATION  — reprimanded a few times, the car is still named (habit)
 *                     and the cross no longer is (goal-directed).
 *   3. EXTINCTION   — reprimanded many more times, the habit finally stops.
 *   4. PERSISTENCE  — habits survive a restart.
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
const CAR = drawing((plot) => {
  for (let x = 4; x < 60; x++) { plot(x, 30); plot(x, 42); }
  for (let y = 30; y < 42; y++) { plot(4, y); plot(59, y); }
  for (let x = 20; x < 44; x++) plot(x, 20);
  for (let y = 20; y < 30; y++) { plot(20, y); plot(43, y); }
  for (const cx of [16, 48]) for (let a = 0; a < 16; a++) plot(Math.round(cx + 5 * Math.cos(a / 16 * 2 * Math.PI)), Math.round(48 + 5 * Math.sin(a / 16 * 2 * Math.PI)));
});
/** The second thing: a cross — in grey, the filled card and the outlined car are too alike for a single-layer cortex. */
const CROSS = drawing((plot) => { for (let i = 8; i < 56; i++) { plot(i, 32); plot(32, i); } });

const newBrain = (): DigitalBrain => { seedRandom(SEED); const b = quiet(() => new DigitalBrain()); b.associationWindowTicks = 120; return b; };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };

console.log('── Verification: habits ──\n');
const brain = newBrain();
const written: Array<{ text: string; habit: boolean }> = [];
brain.on('response', (e: BrainEvent) => { if (e.data.kind === 'writing') written.push({ text: String(e.data.text), habit: e.data.habit === true }); });
const show = (pixels: number[]): void => { quiet(() => brain.see(pixels, SIDE, SIDE, { propagate: false })); wait(brain, 60); };
const say = (text: string): void => { quiet(() => brain.read(text, { propagate: false })); wait(brain, 180); };
/** Shows a thing and returns what it wrote on seeing it (and whether out of habit). */
const nameOf = (pixels: number[]): { text: string; habit: boolean } | null => {
  written.length = 0;
  show(pixels);
  wait(brain, 100);
  return written[0] ?? null;
};

for (let round = 0; round < 3; round++) {
  show(CAR); say('coche');
  show(CROSS); say('cruz');
}

// ── 1. PRACTICE ─────────────────────────────────────────────────────────────
console.log('1. PRACTICE');
{
  let carNamed = 0;
  for (let i = 0; i < 10; i++) {
    const r = nameOf(CAR);
    if (r?.text === 'coche') { carNamed++; brain.giveFeedback(true); }
    wait(brain, 200);
  }
  let cardNamed = 0;
  for (let i = 0; i < 2; i++) {
    const r = nameOf(CROSS);
    if (r?.text === 'cruz') { cardNamed++; brain.giveFeedback(true); }
    wait(brain, 200);
  }
  const car = nameOf(CAR);
  const card = nameOf(CROSS);
  const links = brain.getState().habits?.links ?? [];
  check('named and praised many times, the car\'s name becomes a habit', carNamed >= 8 && car?.text === 'coche' && car.habit,
    `named ${carNamed}/10 times; now ${car ? `"${car.text}"${car.habit ? ' out of habit' : ''}` : 'silent'}; strongest link ${links[0] ? `${links[0].response} ${links[0].strength.toFixed(2)}` : '—'}`);
  const recall = brain.getLastRecall();
  check('named twice, the cross\'s name is not', cardNamed >= 1 && card?.text === 'cruz' && !card.habit,
    `named ${cardNamed}/2 times; now ${card ? `"${card.text}"${card.habit ? ' out of habit' : ' (goal-directed)'}` : 'silent'}; recall ${recall ? `${recall.confidence.toFixed(2)} [${recall.words.map((w) => w.word).join(' ')}] cue ${recall.cue.label}` : '—'}; expects ${brain.expectationOf(`visual:${recall?.cue.label ?? ''}`).expected.toFixed(2)}`);
  wait(brain, 200);
}

// ── 2. DEVALUATION ──────────────────────────────────────────────────────────
console.log('\n2. DEVALUATION');
{
  for (let i = 0; i < 3; i++) {
    if (nameOf(CAR)) brain.giveFeedback(false);
    wait(brain, 200);
    if (nameOf(CROSS)) brain.giveFeedback(false);
    wait(brain, 200);
  }
  const car = nameOf(CAR);
  wait(brain, 200);
  const card = nameOf(CROSS);
  wait(brain, 200);
  check('reprimanded three times, the car is still named — out of habit', car?.text === 'coche' && car.habit,
    car ? `"${car.text}"${car.habit ? ' out of habit' : ''}` : 'silent');
  check('…and the cross no longer is: its name was goal-directed', card === null, card ? `still "${card.text}"` : 'silent');
}

// ── 3. EXTINCTION ───────────────────────────────────────────────────────────
console.log('\n3. EXTINCTION');
{
  let lasted = 0;
  let stopped = false;
  for (let i = 0; i < 14 && !stopped; i++) {
    const r = nameOf(CAR);
    if (r) { lasted++; brain.giveFeedback(false); } else stopped = true;
    wait(brain, 200);
  }
  check('reprimanded many more times, the habit finally goes', stopped && lasted >= 2, `lasted ${lasted} more reprimands`);
}

// ── 4. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n4. PERSISTENCE');
{
  const fresh = newBrain();
  const seen: Array<{ text: string; habit: boolean }> = [];
  fresh.on('response', (e: BrainEvent) => { if (e.data.kind === 'writing') seen.push({ text: String(e.data.text), habit: e.data.habit === true }); });
  for (let round = 0; round < 3; round++) { quiet(() => fresh.see(CAR, SIDE, SIDE, { propagate: false })); wait(fresh, 60); quiet(() => fresh.read('coche', { propagate: false })); wait(fresh, 180); }
  for (let i = 0; i < 10; i++) { seen.length = 0; quiet(() => fresh.see(CAR, SIDE, SIDE, { propagate: false })); wait(fresh, 160); if (seen[0]) fresh.giveFeedback(true); wait(fresh, 200); }
  const before = fresh.getState().habits?.count ?? 0;
  const path = `/tmp/gbrain-habits-${process.pid}.bin`;
  quiet(() => fresh.saveState(path));
  const restored = newBrain();
  quiet(() => restored.loadState(path));
  for (const p of [path, `${path}.lexicon.json`]) if (existsSync(p)) rmSync(p);
  check('habits survive a restart', before >= 1 && (restored.getState().habits?.count ?? 0) === before, `${before} → ${restored.getState().habits?.count ?? 0} habits`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
