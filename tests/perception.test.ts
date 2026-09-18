/**
 * VERIFICATION TEST — Block 3 (part 2): prediction and a closer look
 * ===========================================================================
 * The cortex predicts its input: the error between what a category's neurons
 * expect and what arrives is SURPRISE, and it falls as the thing repeats.
 * And a small thing is looked at more closely, so it is the same thing at any
 * size and wherever it is.
 *
 *   1. SURPRISE   — the first sight of a drawing surprises; the fifth much
 *                   less; a new drawing surprises again; a repeated one earns
 *                   less novelty than an unpredicted one.
 *   2. INVARIANCE — the cross at half size, in a corner, is still the cross;
 *                   the small square is the square, not the cross.
 */
import { DigitalBrain } from '../src/brain.js';
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
/** The cross at half size, up in the top-left corner. */
const SMALL_CROSS = drawing((plot) => { for (let i = 4; i < 28; i++) { plot(i, 16); plot(16, i); } });
/** The square at half size, down in the bottom-right corner. */
const SMALL_SQUARE = drawing((plot) => { for (let i = 36; i < 56; i++) { plot(i, 36); plot(i, 55); plot(36, i); plot(55, i); } });

const newBrain = (): DigitalBrain => { seedRandom(SEED); return quiet(() => new DigitalBrain()); };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const see = (brain: DigitalBrain, image: number[]): { label: string; isNew: boolean; surprise: number } => {
  quiet(() => brain.see(image, SIDE, SIDE, { propagate: false }));
  wait(brain, 60);
  const r = brain.getRecognition().visual!;
  wait(brain, 200);
  return { label: r.label, isNew: r.isNew, surprise: r.surprise ?? 1 };
};

console.log('── Verification: block 3 (part 2) — surprise and a closer look ──\n');

// ── 1. SURPRISE ─────────────────────────────────────────────────────────────
console.log('1. SURPRISE');
{
  const brain = newBrain();
  const crosses = Array.from({ length: 5 }, () => see(brain, CROSS));
  const square = see(brain, SQUARE);
  const surprises = crosses.map((c) => c.surprise);
  // (Random synapses reconstruct an image a little; a freshly recruited engram
  // is not blind to it, only bad at it.)
  check('the first sight of a drawing surprises', surprises[0] >= 0.3, `surprise=${surprises[0].toFixed(2)}`);
  check('the fifth much less', surprises[4] <= surprises[0] * 0.3 && surprises[4] < surprises[1],
    surprises.map((s) => s.toFixed(2)).join(' → '));
  check('a new drawing surprises again', square.isNew && square.surprise >= 0.3, `surprise=${square.surprise.toFixed(2)}`);
  const events = brain.getMotivation().recentEvents.filter((e) => e.kind === 'novelty' && e.key === 'visual:Visual-1');
  const first = events[0]?.error ?? 0;
  const fifth = events[4]?.error ?? 0;
  check('what the cortex predicts well earns less novelty', first >= 0.3 && fifth < first * 0.2, `dopamine ${first.toFixed(2)} → ${fifth.toFixed(2)}`);
}

// ── 2. INVARIANCE ───────────────────────────────────────────────────────────
console.log('\n2. INVARIANCE');
{
  const brain = newBrain();
  const cross = see(brain, CROSS);
  see(brain, CROSS);
  const square = see(brain, SQUARE);
  see(brain, SQUARE);
  const smallCross = see(brain, SMALL_CROSS);
  const smallSquare = see(brain, SMALL_SQUARE);
  check('the cross at half size, in a corner, is still the cross', !smallCross.isNew && smallCross.label === cross.label,
    `${smallCross.label}${smallCross.isNew ? ' NEW' : ''}`);
  check('the small square is the square, not the cross', !smallSquare.isNew && smallSquare.label === square.label,
    `${smallSquare.label}${smallSquare.isNew ? ' NEW' : ''}`);
  check('two things, not four', brain.getRecognition().visualCategories === 2, `${brain.getRecognition().visualCategories} categories`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
