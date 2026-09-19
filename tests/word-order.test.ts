/**
 * VERIFICATION TEST — Block 2: word order in production
 * ===========================================================================
 * The words a thing brings back go out together, in the order their kinds
 * have been heard in. Taught "coche verde", the blue car — never seen — is
 * "coche azul", in one breath; taught "verde coche", it is "azul coche".
 * The order is learned from what is heard, not given.
 *
 *   1. IN ORDER    — noun then colour, as taught.
 *   2. THE OTHER   — colour then noun, as taught the other way.
 *   3. PERSISTENCE — the order survives a restart.
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
type Rgb = [number, number, number];
const BLUE: Rgb = [40, 90, 230];
const GREEN: Rgb = [40, 200, 80];
const WHITE: Rgb = [235, 235, 235];
interface Picture { pixels: number[]; rgb: number[] }
function picture(colour: Rgb, draw: (plot: (x: number, y: number) => void) => void): Picture {
  const pixels = new Array<number>(SIDE * SIDE).fill(27);
  const rgb = new Array<number>(SIDE * SIDE * 3).fill(0);
  for (let i = 0; i < SIDE * SIDE; i++) { rgb[i * 3] = 17; rgb[i * 3 + 1] = 24; rgb[i * 3 + 2] = 39; }
  draw((x, y) => {
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const px = x + dx, py = y + dy;
      if (px < 0 || px >= SIDE || py < 0 || py >= SIDE) continue;
      const i = py * SIDE + px;
      pixels[i] = 255;
      rgb[i * 3] = colour[0]; rgb[i * 3 + 1] = colour[1]; rgb[i * 3 + 2] = colour[2];
    }
  });
  return { pixels, rgb };
}
const car = (colour: Rgb): Picture => picture(colour, (plot) => {
  for (let x = 4; x < 60; x++) { plot(x, 30); plot(x, 42); }
  for (let y = 30; y < 42; y++) { plot(4, y); plot(59, y); }
  for (let x = 20; x < 44; x++) plot(x, 20);
  for (let y = 20; y < 30; y++) { plot(20, y); plot(43, y); }
  for (const cx of [16, 48]) for (let a = 0; a < 16; a++) plot(Math.round(cx + 5 * Math.cos(a / 16 * 2 * Math.PI)), Math.round(48 + 5 * Math.sin(a / 16 * 2 * Math.PI)));
});
const cardOf = (colour: Rgb): Picture => picture(colour, (plot) => {
  for (let y = 10; y < 54; y += 2) for (let x = 22; x < 42; x += 2) plot(x, y);
});
const WHITE_CAR = car(WHITE);
const GREEN_CAR = car(GREEN);
const BLUE_CAR = car(BLUE);
const GREEN_CARD = cardOf(GREEN);
const BLUE_CARD = cardOf(BLUE);

const newBrain = (): DigitalBrain => { seedRandom(SEED); const b = quiet(() => new DigitalBrain()); b.associationWindowTicks = 120; return b; };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const show = (brain: DigitalBrain, p: Picture): void => { quiet(() => brain.see(p.pixels, SIDE, SIDE, { propagate: false, rgb: p.rgb })); wait(brain, 60); };
const say = (brain: DigitalBrain, text: string): void => { quiet(() => brain.read(text, { propagate: false })); wait(brain, 180); };

/** Teaches the words and a two-word phrase in the given order; returns what the blue car, never seen, is called in one breath. */
function raise(phrase: string): { utterances: string[]; brain: DigitalBrain } {
  const brain = newBrain();
  const utterances: string[] = [];
  brain.on('response', (e: BrainEvent) => { if (e.data.kind === 'writing') utterances.push(String(e.data.text)); });
  for (let round = 0; round < 3; round++) {
    show(brain, WHITE_CAR); say(brain, 'coche');
    show(brain, GREEN_CARD); say(brain, 'verde');
    show(brain, BLUE_CARD); say(brain, 'azul');
  }
  for (let round = 0; round < 3; round++) { show(brain, GREEN_CAR); say(brain, phrase); }
  utterances.length = 0;
  show(brain, BLUE_CAR);
  wait(brain, 120);
  return { utterances: [...utterances], brain };
}

console.log('── Verification: block 2 — word order in production ──\n');

// ── 1. IN ORDER ─────────────────────────────────────────────────────────────
console.log('1. IN ORDER');
const taught = raise('coche verde');
{
  const order = taught.brain.getState().language?.order ?? [];
  check('the order heard is learned: shape before colour', order.some((v) => v.first === 'visual' && v.then === 'colour' && v.count >= 2),
    order.map((v) => `${v.first}→${v.then}×${v.count}`).join(', ') || 'no order learned');
  check('the blue car, never seen, is "coche azul" — in one breath', taught.utterances.includes('coche azul'), taught.utterances.join(' | ') || 'nothing written');
}

// ── 2. THE OTHER WAY ────────────────────────────────────────────────────────
console.log('\n2. THE OTHER WAY');
{
  const other = raise('verde coche');
  const order = other.brain.getState().language?.order ?? [];
  check('taught colour before shape, it learns that', order.some((v) => v.first === 'colour' && v.then === 'visual' && v.count >= 2),
    order.map((v) => `${v.first}→${v.then}×${v.count}`).join(', ') || 'no order learned');
  check('…and says "azul coche"', other.utterances.includes('azul coche'), other.utterances.join(' | ') || 'nothing written');
}

// ── 3. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n3. PERSISTENCE');
{
  const path = `/tmp/gbrain-order-${process.pid}.bin`;
  quiet(() => taught.brain.saveState(path));
  const restored = newBrain();
  quiet(() => restored.loadState(path));
  for (const p of [path, `${path}.lexicon.json`]) if (existsSync(p)) rmSync(p);
  const utterances: string[] = [];
  restored.on('response', (e: BrainEvent) => { if (e.data.kind === 'writing') utterances.push(String(e.data.text)); });
  show(restored, BLUE_CAR);
  wait(restored, 120);
  check('the order survives a restart', (restored.getState().language?.order.length ?? 0) >= 1 && utterances.includes('coche azul'), utterances.join(' | ') || 'nothing written');
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
