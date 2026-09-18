/**
 * VERIFICATION TEST — Block 2: time
 * ===========================================================================
 * Experience has an order. Inside a sound, the order of its parts is what
 * tells "pa-ta" from "ta-pa"; between percepts, what tends to follow what is
 * learned, so that the first brings an expectation of the second. What was
 * surprising or rewarding is held in mind; the routine passes through.
 *
 *   1. ORDER IN A SOUND   — two vowels in one order and in the other are two
 *                           different sounds; the same order twice is the same sound.
 *   2. WHAT FOLLOWS WHAT  — shown the cross and then played /a/ a few times, the
 *                           cross alone makes it expect /a/; /a/ does not make it
 *                           expect the cross.
 *   3. EXPECTED IS NO SURPRISE — the /a/ that was expected earns less novelty
 *                           than an unexpected new sound.
 *   4. WORKING MEMORY     — the new and the praised enter and stay in mind; the
 *                           routine does not.
 *   5. PERSISTENCE        — the orders learned survive a restart.
 */
import { existsSync, rmSync } from 'fs';
import { DigitalBrain } from '../src/brain.js';
import { synthesizeSpectrum } from '../src/core/voice/vocal-tract.js';
import type { VoiceContour } from '../src/regions/amygdala/prosody.js';
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
const A = synthesizeSpectrum({ f1: 700, f2: 1200, amplitude: 0.9 });
const I = synthesizeSpectrum({ f1: 300, f2: 2300, amplitude: 0.9 });
const U = synthesizeSpectrum({ f1: 350, f2: 800, amplitude: 0.9 });
const contour = (frames: number, rms: (t: number) => number, f0: (t: number) => number): VoiceContour => ({
  rms: Array.from({ length: frames }, (_, i) => rms(i / (frames - 1))),
  f0: Array.from({ length: frames }, (_, i) => f0(i / (frames - 1))),
  frameMs: 50,
});
const WARM = contour(24, (t) => 0.02 + 0.06 * Math.sin(Math.PI * t), (t) => 200 + 90 * Math.sin(Math.PI * t));

const newBrain = (): DigitalBrain => { seedRandom(SEED); return quiet(() => new DigitalBrain()); };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
/** A sound streamed as frames 200 ms apart, then silence until the ear has closed it. */
function play(brain: DigitalBrain, frames: Float32Array[]): string | null {
  const before = brain.getRecognition().auditory;
  for (const frame of frames) {
    quiet(() => brain.hearFrame(frame, 48000, { propagate: false }));
    wait(brain, brain.ticksFor(200));
  }
  wait(brain, brain.presentationTicks + 20);
  const after = brain.getRecognition().auditory;
  return after && after !== before ? after.label : null;
}
const see = (brain: DigitalBrain, image: number[]): void => { quiet(() => brain.see(image, SIDE, SIDE, { propagate: false })); wait(brain, 60); };

console.log('── Verification: block 2 — time ──\n');

// ── 1. ORDER IN A SOUND ─────────────────────────────────────────────────────
console.log('1. ORDER IN A SOUND');
{
  const brain = newBrain();
  const ai1 = play(brain, [A, I]);
  const ai2 = play(brain, [A, I]);
  const ia = play(brain, [I, A]);
  const a = play(brain, [A]);
  check('a sound is heard as one thing', ai1 !== null && ai2 !== null && ia !== null, `${ai1} ${ai2} ${ia}`);
  check('the same order twice is the same sound', ai1 !== null && ai1 === ai2, `${ai1} = ${ai2}`);
  check('the other order is another sound', ia !== null && ia !== ai1, `${ai1} ≠ ${ia}`);
  check('a part alone is not the whole', a !== null && a !== ai1 && a !== ia, `${a}`);
}

// ── 2. WHAT FOLLOWS WHAT ────────────────────────────────────────────────────
console.log('\n2. WHAT FOLLOWS WHAT');
const brain = newBrain();
{
  for (let i = 0; i < 4; i++) {
    see(brain, CROSS);
    wait(brain, brain.ticksFor(1500));
    play(brain, [A]);
    wait(brain, brain.ticksFor(15_000)); // long enough for the next cross not to follow the /a/
  }
  see(brain, CROSS);
  const expectation = brain.getState().sequence?.expectation ?? null;
  check('the cross makes it expect the sound that followed it', expectation !== null && expectation.key.startsWith('auditory:') && expectation.probability >= 0.6,
    expectation ? `${expectation.key} ${Math.round(expectation.probability * 100)}% in ~${expectation.expectedInTicks} ticks` : 'nothing');
  // (Measured from percept to percept, so it includes the time the ear takes
  // to close a sound: more than the 1.5 s between the stimuli, well under the window.)
  check('the delay it expects is plausible', expectation !== null && expectation.expectedInTicks >= brain.ticksFor(1500) && expectation.expectedInTicks <= brain.ticksFor(10_000),
    `${expectation?.expectedInTicks} ticks`);
  wait(brain, brain.ticksFor(15_000));
  play(brain, [A]);
  const reverse = brain.getState().sequence?.expectation ?? null;
  check('the sound does not make it expect the cross', reverse === null || !reverse.key.startsWith('visual:'), reverse ? reverse.key : 'nothing');
}

// ── 3. EXPECTED IS NO SURPRISE ──────────────────────────────────────────────
console.log('\n3. EXPECTED IS NO SURPRISE');
{
  wait(brain, brain.ticksFor(15_000));
  see(brain, CROSS);
  wait(brain, brain.ticksFor(1500));
  const before = brain.getMotivation().events;
  play(brain, [A]);
  const events = brain.getMotivation().recentEvents.slice(-(brain.getMotivation().events - before));
  const expectedA = events.find((e) => e.kind === 'novelty' && e.key !== null && e.key.startsWith('auditory:'));
  wait(brain, brain.ticksFor(15_000));
  const before2 = brain.getMotivation().events;
  play(brain, [U]); // never heard before, nothing announced it
  const events2 = brain.getMotivation().recentEvents.slice(-(brain.getMotivation().events - before2));
  const newU = events2.find((e) => e.kind === 'novelty' && e.key !== null && e.key.startsWith('auditory:'));
  check('the expected sound earns little novelty', expectedA !== undefined && expectedA.expected > 0 && expectedA.error < expectedA.reward,
    expectedA ? `reward ${expectedA.reward.toFixed(2)} expected ${expectedA.expected.toFixed(2)} → dopamine ${expectedA.error.toFixed(2)}` : 'no event');
  check('an unexpected new sound earns the full bonus', newU !== undefined && newU.expected === 0 && newU.error >= 0.4,
    newU ? `dopamine ${newU.error.toFixed(2)}` : 'no event');
}

// ── 4. WORKING MEMORY ───────────────────────────────────────────────────────
console.log('\n4. WORKING MEMORY');
{
  const fresh = newBrain();
  const held = (): string[] => (fresh.getState().workingMemory ?? []).map((s) => s.label);
  see(fresh, CROSS);
  check('something new enters working memory', held().includes('Visual-1'), held().join(', ') || 'empty');
  for (let i = 0; i < 6; i++) { wait(fresh, 300); see(fresh, CROSS); }
  wait(fresh, 400); // the slot, unrefreshed by anything surprising, fades
  const routine = held();
  check('the routine does not stay in mind', !routine.includes('Visual-1'), routine.join(', ') || 'empty');
  see(fresh, SQUARE);
  fresh.hearVoice(WARM); // praised right after
  const praised = (fresh.getState().workingMemory ?? []).find((s) => s.label === 'Visual-2');
  check('the praised enters with a high priority', praised !== undefined && praised.priority >= 0.5,
    praised ? `priority ${praised.priority.toFixed(2)}` : 'not held');
}

// ── 5. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n5. PERSISTENCE');
{
  const path = `/tmp/gbrain-sequence-${process.pid}.bin`;
  quiet(() => brain.saveState(path));
  const restored = newBrain();
  quiet(() => restored.loadState(path));
  for (const p of [path, `${path}.lexicon.json`]) if (existsSync(p)) rmSync(p);
  const before = restored.getState().sequence?.transitions ?? 0;
  see(restored, CROSS);
  const expectation = restored.getState().sequence?.expectation ?? null;
  check('the orders learned survive a restart', before > 0 && expectation !== null && expectation.key.startsWith('auditory:'),
    `${before} transitions; after the cross expects ${expectation?.key ?? 'nothing'}`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
