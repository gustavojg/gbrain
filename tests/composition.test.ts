/**
 * VERIFICATION TEST — Block 3: colour, and putting words together
 * ===========================================================================
 * Colour parts ways with shape (V4), so a word can attach to a colour across
 * objects and another to a shape across colours. Taught the way a child is —
 * a car, "coche"; a green card, "verde"; a blue card, "azul"; a green car,
 * "coche verde" — a blue car it has never seen brings back "coche" and "azul".
 *
 *   1. COLOUR      — the blue card and the blue car are the same colour; the
 *                    green card is another; the achromatic car has no colour.
 *   2. WORDS       — "azul" attaches to the colour, "coche" to the shape.
 *   3. COMPOSITION — the blue car, never seen, is "coche" and "azul".
 *   4. QUESTIONS   — asked and answered a few times, "¿de qué color es?" comes
 *                    to ask for the colour and "¿qué es?" for the shape of what
 *                    is in front, and it answers.
 *   5. PERSISTENCE — colour categories survive a restart.
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

// ── Coloured drawings: grey pixels for the shape, r,g,b for the colour ──────
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
/**
 * A car: a long low body, a cabin and two wheels. (The retina is 14×14 and
 * the visual cortex a single layer: shapes are told apart by their footprint
 * and strokes, so the car is wide and low and the card tall and narrow, as
 * they would be on a table. Parts-based recognition awaits the hierarchy.)
 */
const car = (colour: Rgb): Picture => picture(colour, (plot) => {
  for (let x = 4; x < 60; x++) { plot(x, 30); plot(x, 42); }
  for (let y = 30; y < 42; y++) { plot(4, y); plot(59, y); }
  for (let x = 20; x < 44; x++) plot(x, 20);
  for (let y = 20; y < 30; y++) { plot(20, y); plot(43, y); }
  for (const cx of [16, 48]) for (let a = 0; a < 16; a++) plot(Math.round(cx + 5 * Math.cos(a / 16 * 2 * Math.PI)), Math.round(48 + 5 * Math.sin(a / 16 * 2 * Math.PI)));
});
/** A card: a filled, tall rectangle. */
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
const colourOf = (brain: DigitalBrain): string | null => brain.getRecognition().colour?.label ?? null;

console.log('── Verification: block 3 — colour and composition ──\n');

// ── 1. COLOUR ───────────────────────────────────────────────────────────────
console.log('1. COLOUR');
{
  const brain = newBrain();
  show(brain, BLUE_CARD);
  const blueCard = colourOf(brain);
  wait(brain, 200);
  show(brain, GREEN_CARD);
  const greenCard = colourOf(brain);
  wait(brain, 200);
  show(brain, BLUE_CAR);
  const blueCar = colourOf(brain);
  wait(brain, 200);
  const before = brain.getRecognition().colourCategories;
  quiet(() => brain.see(WHITE_CAR.pixels, SIDE, SIDE, { propagate: false }));
  wait(brain, 60);
  check('a colour is perceived', blueCard !== null && greenCard !== null, `${blueCard} ${greenCard}`);
  check('blue is not green', blueCard !== greenCard, `${blueCard} ≠ ${greenCard}`);
  check('the blue car is the same colour as the blue card', blueCar === blueCard, `${blueCar} = ${blueCard}`);
  check('an image without colour founds no colour', brain.getRecognition().colourCategories === before);
  check('shape and colour are different things', brain.getRecognition().visualCategories >= 2, `${brain.getRecognition().visualCategories} shapes`);
}

// ── 2 & 3. WORDS AND COMPOSITION ────────────────────────────────────────────
console.log('\n2. WORDS');
const brain = newBrain();
const written: string[] = [];
brain.on('response', (e: BrainEvent) => { if (e.data.kind === 'writing') written.push(...String(e.data.text).split(' ')); });
const teach = (p: Picture, text: string): void => { show(brain, p); say(brain, text); };
for (let round = 0; round < 3; round++) {
  teach(WHITE_CAR, 'coche');
  teach(GREEN_CARD, 'verde');
  teach(BLUE_CARD, 'azul');
}
for (let round = 0; round < 3; round++) teach(GREEN_CAR, 'coche verde');
{
  written.length = 0;
  show(brain, BLUE_CARD);
  wait(brain, 100);
  check('"azul" attaches to the colour: the blue card brings it back', written.includes('azul'), written.join(' ') || 'nothing written');
  written.length = 0;
  show(brain, WHITE_CAR);
  wait(brain, 100);
  check('"coche" attaches to the shape: the car brings it back', written.includes('coche'), written.join(' ') || 'nothing written');
}

console.log('\n3. COMPOSITION');
{
  written.length = 0;
  show(brain, BLUE_CAR); // never seen
  wait(brain, 100);
  const said = written.join(' ');
  check('the blue car, never seen, is "coche"', written.includes('coche'), said || 'nothing written');
  check('…and "azul"', written.includes('azul'), said || 'nothing written');
  check('…and not "verde"', !written.includes('verde'), said || 'nothing written');
}

// ── 4. QUESTIONS ────────────────────────────────────────────────────────────
console.log('\n4. QUESTIONS');
{
  // The way a child learns what "¿de qué color es?" asks for: the adult asks
  // and answers, a few times, with the thing in front.
  const ask = (p: Picture, question: string, answer: string): void => {
    show(brain, p);
    quiet(() => brain.read(question, { propagate: false }));
    wait(brain, 30);
    quiet(() => brain.read(answer, { propagate: false }));
    wait(brain, 180);
  };
  // Asked about several things: a question that always came with the blue
  // card would name blue as much as "azul" does.
  for (let i = 0; i < 2; i++) {
    ask(BLUE_CARD, 'de que color es', 'azul');
    ask(GREEN_CARD, 'de que color es', 'verde');
    ask(GREEN_CAR, 'de que color es', 'verde');
    ask(WHITE_CAR, 'que es', 'coche');
    ask(GREEN_CAR, 'que es', 'coche');
  }
  check('it learns what each question asks for', (brain.getState().questions?.known ?? 0) >= 2,
    `${brain.getState().questions?.known ?? 0} questions known`);

  const answerTo = (p: Picture, question: string): string | null => {
    show(brain, p);
    written.length = 0;
    quiet(() => brain.read(question, { propagate: false }));
    wait(brain, 30);
    const answer = brain.getState().questions?.lastAnswer;
    const fresh = answer && written.includes(answer.word) ? answer.word : null;
    wait(brain, 180);
    return fresh;
  };
  check('"¿de qué color es?" on the green card: "verde"', answerTo(GREEN_CARD, 'de que color es') === 'verde', written.join(' ') || 'no answer');
  check('"¿qué es?" on the blue car: "coche"', answerTo(BLUE_CAR, 'que es') === 'coche', written.join(' ') || 'no answer');
  check('"¿de qué color es?" on the blue car: "azul"', answerTo(BLUE_CAR, 'de que color es') === 'azul', written.join(' ') || 'no answer');
}

// ── 5. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n5. PERSISTENCE');
{
  const path = `/tmp/gbrain-colour-${process.pid}.bin`;
  quiet(() => brain.saveState(path));
  const restored = newBrain();
  quiet(() => restored.loadState(path));
  for (const p of [path, `${path}.lexicon.json`]) if (existsSync(p)) rmSync(p);
  const colours = restored.getRecognition().colourCategories;
  show(restored, BLUE_CARD);
  const r = restored.getRecognition().colour;
  check('colour categories survive a restart', colours >= 2 && r !== null && !r.isNew, `${colours} colours; blue card ${r?.isNew ? 'new' : 'recognized'}`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
