/**
 * VERIFICATION TEST — Block 1: motivation and body
 * ===========================================================================
 * Dopamine is reward prediction error, not reward: what was expected moves
 * nothing, what was better or worse than expected does. Novelty and learning
 * progress are rewards in themselves — that is why an infant looks, babbles
 * and scribbles without anyone asking. Drives (boredom, the need for contact)
 * push it to act and to call.
 *
 *   1. NOVELTY      — the first sight of a thing is a dopamine burst; the
 *                     fifth is almost nothing; a new thing brings it back.
 *   2. EXPECTATION  — praise for the same thing habituates; praise withheld
 *                     where it was expected is a dip; a harsh voice after
 *                     praise is a big negative error.
 *   3. PROGRESS     — while a lesson is working, each recall is rewarding;
 *                     once mastered, no more.
 *   4. ADAPTATION   — a sustained high normalizes: the tonic baseline drifts
 *                     up, and comes back down.
 *   5. CHOICE       — with both a mastered and an unmastered activity, it
 *                     spends its time on the one that still teaches it.
 *   6. DRIVES       — bored, it acts more; alone, it calls; a voice settles it.
 */
import { DigitalBrain } from '../src/brain.js';
import { ModulatorType, NeuromodulatorSystem } from '../src/core/neuromodulators/modulator-system.js';
import type { VoiceContour } from '../src/regions/amygdala/prosody.js';
import { quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);
const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

const contour = (frames: number, rms: (t: number) => number, f0: (t: number) => number): VoiceContour => ({
  rms: Array.from({ length: frames }, (_, i) => rms(i / (frames - 1))),
  f0: Array.from({ length: frames }, (_, i) => f0(i / (frames - 1))),
  frameMs: 50,
});
const WARM = contour(24, (t) => 0.02 + 0.06 * Math.sin(Math.PI * t), (t) => 200 + 90 * Math.sin(Math.PI * t));
const HARSH = contour(7, (t) => (Math.round(t * 6) % 2 === 0 ? 0.32 : 0.1), (t) => 115 - 20 * t);

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

const newBrain = (): DigitalBrain => { seedRandom(SEED); return quiet(() => new DigitalBrain()); };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const see = (brain: DigitalBrain, image: number[]): void => { quiet(() => brain.see(image, SIDE, SIDE)); };
const dopamine = (brain: DigitalBrain): number => brain.getModulators().getLevel(ModulatorType.Dopamine);
/** The last reward prediction error of a given kind, if it is the last event. */
const lastError = (brain: DigitalBrain, kind: string): number | null => {
  const e = brain.getMotivation().lastEvent;
  return e && e.kind === kind ? e.error : null;
};

console.log('── Verification: block 1 — motivation and body ──\n');

// ── 1. NOVELTY ──────────────────────────────────────────────────────────────
console.log('1. NOVELTY');
{
  const brain = newBrain();
  const bursts: number[] = [];
  const levels: number[] = [];
  for (let i = 0; i < 5; i++) {
    wait(brain, 300);
    const before = dopamine(brain);
    see(brain, CROSS);
    bursts.push(lastError(brain, 'novelty') ?? 0);
    levels.push(dopamine(brain) - before);
  }
  wait(brain, 300);
  see(brain, SQUARE);
  const fresh = lastError(brain, 'novelty') ?? 0;
  // (The bonus is scaled by the cortex's surprise; a first sight surprises a
  // freshly recruited engram about a third — see perception.test.ts.)
  check('the first sight of a thing is a dopamine burst', bursts[0] >= 0.3 && levels[0] > 0.1,
    `error=${bursts[0].toFixed(2)} Δdopamine=${levels[0].toFixed(2)}`);
  check('it habituates with every sighting', bursts.every((b, i) => i === 0 || b < bursts[i - 1]) && bursts[4] <= 0.12,
    bursts.map((b) => b.toFixed(2)).join(' → '));
  check('a new thing brings it back', fresh >= 0.3, `error=${fresh.toFixed(2)}`);
  check('it knows how often it has seen each', brain.expectationOf('visual:Visual-1').seen === 5);
}

// ── 2. EXPECTATION ──────────────────────────────────────────────────────────
console.log('\n2. EXPECTATION');
{
  const brain = newBrain();
  const praise: number[] = [];
  for (let i = 0; i < 6; i++) {
    see(brain, CROSS);
    brain.hearVoice(WARM); // "¡muy bien!" every time it looks at the cross
    praise.push(lastError(brain, 'external') ?? 0);
    wait(brain, 300);
  }
  check('praise for the same thing habituates', praise[0] >= 0.5 && praise[5] < praise[0] * 0.35,
    praise.map((p) => p.toFixed(2)).join(' → '));
  check('…because the thing has come to predict it', brain.expectationOf('visual:Visual-1').expected >= 0.5,
    `expected=${brain.expectationOf('visual:Visual-1').expected.toFixed(2)}`);

  // Now the cross, and silence.
  const before = dopamine(brain);
  see(brain, CROSS);
  wait(brain, brain.ticksFor(9000));
  const omission = lastError(brain, 'omission');
  check('praise withheld where it was expected is a dip', omission !== null && omission <= -0.3 && dopamine(brain) < before,
    `error=${omission?.toFixed(2)} dopamine ${before.toFixed(2)} → ${dopamine(brain).toFixed(2)}`);

  wait(brain, 300);
  see(brain, CROSS);
  brain.hearVoice(HARSH);
  const reprimand = lastError(brain, 'external');
  check('a harsh voice where praise was expected is a large negative error', reprimand !== null && reprimand <= -1,
    `error=${reprimand?.toFixed(2)}`);
}

// ── 3. PROGRESS ─────────────────────────────────────────────────────────────
console.log('\n3. PROGRESS');
{
  const brain = newBrain();
  brain.associationWindowTicks = 120;
  const lesson = (): number | null => {
    const before = brain.getMotivation().events;
    quiet(() => brain.see(CROSS, SIDE, SIDE, { propagate: false }));
    wait(brain, 50);
    const since = brain.getMotivation().recentEvents.slice(-(brain.getMotivation().events - before));
    const progress = since.find((e) => e.kind === 'progress' && e.key === 'visual:Visual-1');
    quiet(() => brain.read('cruz', { propagate: false }));
    wait(brain, 180);
    return progress ? progress.error : null;
  };
  const rewards: Array<number | null> = [];
  for (let i = 0; i < 8; i++) rewards.push(lesson());
  const early = rewards.slice(1, 4).filter((r): r is number => r !== null);
  const late = rewards.slice(5).filter((r): r is number => r !== null);
  check('while the lesson is working, each recall is rewarding', early.length >= 2 && early.every((r) => r > 0),
    rewards.map((r) => (r === null ? '—' : r.toFixed(2))).join(' '));
  check('once mastered, the reward fades', late.length === 0 || Math.max(...late) < Math.max(...early) * 0.5,
    `late: ${late.map((r) => r.toFixed(2)).join(' ') || 'none'}`);
}

// ── 4. ADAPTATION ───────────────────────────────────────────────────────────
console.log('\n4. ADAPTATION');
{
  const mods = new NeuromodulatorSystem();
  const setPoint = mods.getBaseline(ModulatorType.Dopamine);
  for (let t = 0; t < 20000; t++) {
    if (t % 200 === 0) mods.release(ModulatorType.Dopamine, 0.3);
    mods.decay(1);
  }
  const adapted = mods.getBaseline(ModulatorType.Dopamine);
  for (let t = 0; t < 300000; t++) mods.decay(1);
  const recovered = mods.getBaseline(ModulatorType.Dopamine);
  check('a sustained high raises the tonic baseline', adapted >= setPoint + 0.05, `${setPoint.toFixed(2)} → ${adapted.toFixed(2)}`);
  check('it never leaves the band around the set point', adapted <= setPoint + 0.15);
  check('and it comes back once the high is over', recovered < setPoint + 0.02, `→ ${recovered.toFixed(2)}`);
}

// ── 5. CHOICE ───────────────────────────────────────────────────────────────
console.log('\n5. CHOICE');
{
  const brain = newBrain();
  // The hand practises until its map has settled…
  brain.setHand({ scribble: false, copy: false });
  for (let i = 0; i < 60; i++) { quiet(() => brain.scribbleOnce()); wait(brain, 70); }
  const values = brain.getMotivation().activityValues;
  // …then both activities are available, and the voice has everything to learn.
  brain.setVoice({ babble: true });
  brain.setHand({ scribble: true });
  wait(brain, 4000);
  const chosen = brain.getMotivation().chosen;
  check('a mastered activity has lost its value; an untried one keeps it', values.scribble < values.babble,
    `scribble=${values.scribble.toFixed(2)} babble=${values.babble.toFixed(2)}`);
  check('it spends its time on the activity that still teaches it', chosen.babble > chosen.scribble * 1.5,
    `babbles=${chosen.babble} scribbles=${chosen.scribble}`);
}

// ── 6. DRIVES ───────────────────────────────────────────────────────────────
console.log('\n6. DRIVES');
{
  const bored = newBrain();
  wait(bored, 3000); // nothing happens for five minutes of brain time
  const drives = bored.getMotivation().drives;
  check('nothing rewarding for a while: boredom', drives.boredom >= 0.9, `boredom=${drives.boredom.toFixed(2)}`);
  const firstBabble = (brain: DigitalBrain): number => {
    brain.setVoice({ babble: true });
    for (let t = 1; t <= 400; t++) { brain.tick(); if (brain.getState().voice!.babbles > 0) return t; }
    return Infinity;
  };
  const boredLatency = firstBabble(bored);

  const fresh = newBrain();
  see(fresh, CROSS); // something just happened
  const freshLatency = firstBabble(fresh);
  // (Babbling itself relieves boredom: it is the first move that comes sooner.)
  check('bored, it acts sooner', boredLatency < freshLatency * 0.7, `first babble after ${boredLatency} vs ${freshLatency} ticks`);

  const alone = newBrain();
  wait(alone, 4000); // no voice for almost seven minutes
  check('no voice for a while: the need for contact', alone.getMotivation().drives.contact >= 0.7,
    `contact=${alone.getMotivation().drives.contact.toFixed(2)}`);
  alone.setVoice({ babble: true });
  wait(alone, 3);
  check('alone, it calls', alone.getMotivation().calls === 1 && alone.getLastVocalization()?.source === 'call');
  alone.hearVoice(WARM);
  check('a voice settles it', alone.getMotivation().drives.contact < 0.05, `contact=${alone.getMotivation().drives.contact.toFixed(2)}`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
