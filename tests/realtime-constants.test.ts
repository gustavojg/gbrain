/**
 * VERIFICATION TEST — human time is measured in seconds, not ticks
 * ===========================================================================
 * The neural dynamics advance `dt` simulated ms per tick whatever the server's
 * timer does. Everything on a human timescale — how long a stimulus stays in
 * view, how long a percept waits to be bound with the next one, the pause
 * between babbles — is defined in real milliseconds and converted to ticks
 * with the tick rate, so a server driven at 100 Hz behaves per second of wall
 * clock like one driven at 10 Hz, only living through ten times more neural
 * time in that second.
 *
 *   1. SCALING   — the tick counts follow the rate (10× at 100 Hz), and a
 *                  nonsense rate is refused.
 *   2. REST      — driven at 100 Hz the brain is still silent without input.
 *   3. LEARNING  — driven at 100 Hz, shown a drawing and its name a few times
 *                  (the same seconds apart as a person would), the drawing
 *                  brings the word to mind, and the brain returns to rest.
 */
import { DigitalBrain, type AssociationRecall } from '../src/brain.js';
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
const CROSS = drawing((plot) => { for (let i = 8; i < 56; i++) { plot(i, 32); plot(32, i); } });

const newBrain = (tickRate: number): DigitalBrain => {
  seedRandom(SEED);
  return quiet(() => new DigitalBrain({ tickRate }));
};
const isSilent = (brain: DigitalBrain): boolean =>
  Object.values(brain.getState().regions).every((r) => r.firingRate === 0);
/** Lets `ms` of real time pass at the brain's rate. */
const wait = (brain: DigitalBrain, ms: number): void => { for (let t = 0, n = brain.ticksFor(ms); t < n; t++) brain.tick(); };

console.log('── Verification: human time in seconds, whatever the tick rate ──\n');

// ── 1. SCALING ──────────────────────────────────────────────────────────────
console.log('1. SCALING');
{
  const slow = newBrain(10);
  const fast = newBrain(100);
  check('at 10 Hz the constants are what they always were',
    slow.presentationTicks === 30 && slow.perceptionTicks === 50 && slow.associationWindowTicks === 300,
    `presentation=${slow.presentationTicks} perception=${slow.perceptionTicks} window=${slow.associationWindowTicks}`);
  check('at 100 Hz they span the same seconds in ten times the ticks',
    fast.presentationTicks === 300 && fast.perceptionTicks === 500 && fast.associationWindowTicks === 3000,
    `presentation=${fast.presentationTicks} perception=${fast.perceptionTicks} window=${fast.associationWindowTicks}`);
  check('one second is 10 ticks at 10 Hz and 100 at 100 Hz', slow.ticksFor(1000) === 10 && fast.ticksFor(1000) === 100);
  check('a fraction of a tick still runs one tick', fast.ticksFor(1) === 1 && slow.ticksFor(0) === 1);
  let refused = false;
  try { quiet(() => new DigitalBrain({ tickRate: 0 })); } catch { refused = true; }
  check('a zero tick rate is refused', refused);
}

// ── 2. REST ─────────────────────────────────────────────────────────────────
console.log('\n2. REST (100 Hz)');
const brain = newBrain(100);
{
  wait(brain, 5000);
  const state = brain.getState();
  check('no region fires without input', isSilent(brain));
  check('no episode is stored without input', state.memoriesCount === 0, `episodes=${state.memoriesCount}`);
}

// ── 3. LEARNING ─────────────────────────────────────────────────────────────
console.log('\n3. LEARNING (100 Hz)');
{
  const see = (): void => { quiet(() => brain.see(CROSS, SIDE, SIDE, { propagate: false })); };
  const read = (): void => { quiet(() => brain.read('cruz', { propagate: false })); };
  const probe = (): AssociationRecall | null => {
    const before = brain.getLastRecall();
    see();
    wait(brain, 5000);
    const after = brain.getLastRecall();
    return after !== before ? after : null;
  };
  const confidence: number[] = [];
  const started = Date.now();
  for (let session = 1; session <= 3; session++) {
    see();
    wait(brain, 5000);
    read();
    wait(brain, 20_000);
    const recall = probe();
    confidence.push(recall?.confidence ?? 0);
    wait(brain, 20_000);
  }
  const seconds = ((Date.now() - started) / 1000).toFixed(1);
  const last = brain.getLastRecall();
  check('the association grows with every pairing',
    confidence.every((c, i) => i === 0 || c > confidence[i - 1]),
    confidence.map((c) => c.toFixed(2)).join(' → '));
  check('after a few, the drawing brings its word to mind, confidently',
    last?.confident === true && last.words[0]?.word === 'cruz',
    `${last?.words[0]?.word ?? '—'} ${(100 * (last?.confidence ?? 0)).toFixed(0)}% · ${seconds} s of CPU for ~3 min of brain time`);
  check('the brain returns to rest', isSilent(brain) && brain.getBus().pendingCount === 0);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
