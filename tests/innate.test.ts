/**
 * VERIFICATION TEST — Block 0: what the brain brings to the world unlearned
 * ===========================================================================
 * A newborn knows no words, but reads the TONE of a voice, startles at a
 * sudden loud sound, is alarmed by what looms, looks at faces, learns to fear
 * in one pairing (and to stop fearing slowly), and sleeps when it has to.
 * Words acquire their emotion by being heard alongside one of these events.
 *
 *   1. PROSODY      — the innate reading of a voice: warm vs harsh, calm vs alarming.
 *   2. VOICE        — a warm voice comforts the brain, a harsh one alarms it; a
 *                     sudden loud one startles it before the cortex has heard it.
 *   3. LOOMING      — something growing fast on the retina alarms; shrinking does not.
 *   4. FACES        — a face-like pattern gets attention; the same strokes upside down do not.
 *   5. WORDS        — "miedo" read alone means nothing; read with a harsh voice it
 *                     comes to mean fear.
 *   6. FEAR         — the figure present at a harsh voice is feared afterwards; safe
 *                     exposures extinguish the fear slowly; stress brings it back.
 *   7. VERDICT      — a warm voice after a recall strengthens what was recalled.
 *   8. SLEEP        — a stimulated brain sleeps sooner than one at rest.
 */
import { DigitalBrain } from '../src/brain.js';
import { ModulatorType } from '../src/core/neuromodulators/modulator-system.js';
import { appraiseProsody, type VoiceContour } from '../src/regions/amygdala/prosody.js';
import { quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);
const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

// ── Voices (envelope + pitch track, 50 ms frames) ───────────────────────────
const FRAME_MS = 50;
const contour = (frames: number, rms: (t: number) => number, f0: (t: number) => number): VoiceContour => ({
  rms: Array.from({ length: frames }, (_, i) => rms(i / (frames - 1))),
  f0: Array.from({ length: frames }, (_, i) => f0(i / (frames - 1))),
  frameMs: FRAME_MS,
});
/** "¡Muy bien!": 1.2 s, soft, smooth, high and bell-shaped. */
const WARM = contour(24, (t) => 0.02 + 0.06 * Math.sin(Math.PI * t), (t) => 200 + 90 * Math.sin(Math.PI * t));
/** "¡No!": 0.35 s, loud from the first frame, staccato, low and falling. */
const HARSH = contour(7, (t) => (Math.round(t * 6) % 2 === 0 ? 0.32 : 0.1), (t) => 115 - 20 * t);
/** Talking to oneself: 0.8 s, medium, flat. */
const NEUTRAL = contour(16, () => 0.05, () => 150);

// ── Drawings ────────────────────────────────────────────────────────────────
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
const SQUARE = drawing((plot) => { for (let i = 12; i < 52; i++) { plot(i, 12); plot(i, 51); plot(12, i); plot(51, i); } });
const blob = (radius: number): number[] => drawing((plot) => {
  for (let y = 32 - radius; y <= 32 + radius; y++) for (let x = 32 - radius; x <= 32 + radius; x++) {
    if ((x - 32) ** 2 + (y - 32) ** 2 <= radius * radius) plot(x, y);
  }
});
/** Two eyes above a mouth… */
const FACE = drawing((plot) => {
  for (let d = -2; d <= 2; d++) for (let e = -2; e <= 2; e++) { plot(20 + d, 20 + e); plot(44 + d, 20 + e); }
  for (let x = 22; x <= 42; x++) plot(x, 44);
});
/** …and the same strokes upside down (Johnson & Morton's control). */
const INVERTED_FACE = drawing((plot) => {
  for (let d = -2; d <= 2; d++) for (let e = -2; e <= 2; e++) { plot(20 + d, 44 + e); plot(44 + d, 44 + e); }
  for (let x = 22; x <= 42; x++) plot(x, 20);
});

const newBrain = (config: Record<string, unknown> = {}): DigitalBrain => {
  seedRandom(SEED);
  return quiet(() => new DigitalBrain(config));
};
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const see = (brain: DigitalBrain, image: number[]): void => { quiet(() => brain.see(image, SIDE, SIDE)); };
const level = (brain: DigitalBrain, type: ModulatorType): number => brain.getModulators().getLevel(type);

console.log('── Verification: block 0 — the innate layer ──\n');

// ── 1. PROSODY ──────────────────────────────────────────────────────────────
console.log('1. PROSODY');
{
  const warm = appraiseProsody(WARM)!;
  const harsh = appraiseProsody(HARSH)!;
  const neutral = appraiseProsody(NEUTRAL)!;
  check('a soft, high, bell-shaped voice reads as warm', warm.valence >= 0.3, `valence=${warm.valence.toFixed(2)}`);
  check('a loud, low, staccato voice reads as harsh', harsh.valence <= -0.3, `valence=${harsh.valence.toFixed(2)}`);
  check('the harsh voice is the alarming one', harsh.arousal > warm.arousal + 0.2,
    `arousal harsh=${harsh.arousal.toFixed(2)} warm=${warm.arousal.toFixed(2)}`);
  check('a flat, medium voice reads as neither', Math.abs(neutral.valence) < 0.3, `valence=${neutral.valence.toFixed(2)}`);
  check('only the sudden loud one startles', harsh.startle && !warm.startle && !neutral.startle);
  check('a click is not a voice', appraiseProsody({ rms: [0.3], f0: [0], frameMs: FRAME_MS }) === null);
}

// ── 2. VOICE ────────────────────────────────────────────────────────────────
console.log('\n2. VOICE');
{
  const brain = newBrain();
  const before = brain.feel().valence;
  const oxytocinBefore = level(brain, ModulatorType.Oxytocin);
  brain.hearVoice(WARM);
  check('a warm voice comforts', brain.feel().valence > before + 0.05 && level(brain, ModulatorType.Oxytocin) > oxytocinBefore,
    `valence ${before.toFixed(2)} → ${brain.feel().valence.toFixed(2)}`);

  const alarmed = newBrain();
  const cortisolBefore = level(alarmed, ModulatorType.Cortisol);
  const valenceBefore = alarmed.feel().valence;
  alarmed.hearVoice(HARSH);
  const state = alarmed.getState();
  check('a harsh voice alarms', alarmed.feel().valence < valenceBefore - 0.05 && level(alarmed, ModulatorType.Cortisol) > cortisolBefore,
    `valence ${valenceBefore.toFixed(2)} → ${alarmed.feel().valence.toFixed(2)}`);
  check('the startle fires before the auditory cortex has heard anything',
    state.innate?.startles === 1 && state.regions.auditoryCortex.activeNeurons.length === 0 && level(alarmed, ModulatorType.Norepinephrine) > 0.4,
    `NE=${level(alarmed, ModulatorType.Norepinephrine).toFixed(2)}`);
  check('it knows a warm voice from a harsh one', brain.getInnate().lastVoice?.kind === 'warm' && alarmed.getInnate().lastVoice?.kind === 'harsh');
}

// ── 3. LOOMING ──────────────────────────────────────────────────────────────
console.log('\n3. LOOMING');
{
  // Two webcam frames two seconds apart: the first is only injected, the
  // second arrives while the first is still in view.
  const brain = newBrain();
  quiet(() => brain.see(blob(6), SIDE, SIDE, { propagate: false }));
  wait(brain, brain.ticksFor(2000));
  const cortisolBefore = level(brain, ModulatorType.Cortisol);
  see(brain, blob(28));
  check('something growing fast on the retina alarms', brain.getInnate().loomings === 1 && level(brain, ModulatorType.Cortisol) > cortisolBefore,
    `cortisol ${cortisolBefore.toFixed(2)} → ${level(brain, ModulatorType.Cortisol).toFixed(2)}`);
  const calm = newBrain();
  quiet(() => calm.see(blob(28), SIDE, SIDE, { propagate: false }));
  wait(calm, calm.ticksFor(2000));
  see(calm, blob(6));
  check('something shrinking does not', calm.getInnate().loomings === 0);
}

// ── 4. FACES ────────────────────────────────────────────────────────────────
console.log('\n4. FACES');
{
  const brain = newBrain();
  see(brain, FACE);
  const face = brain.getInnate().faceMatch;
  const faces = brain.getInnate().facesSeen;
  wait(brain, 200);
  see(brain, INVERTED_FACE);
  const inverted = brain.getInnate().faceMatch;
  check('two eyes above a mouth look like a face', faces === 1 && face >= 0.35, `match=${face.toFixed(2)}`);
  check('the same strokes upside down do not', brain.getInnate().facesSeen === 1 && inverted < face - 0.15, `match=${inverted.toFixed(2)}`);
}

// ── 5. WORDS ────────────────────────────────────────────────────────────────
console.log('\n5. WORDS');
{
  const brain = newBrain();
  const before = brain.feel().valence;
  quiet(() => brain.read('miedo'));
  check('a word read alone carries no emotion', Math.abs(brain.feel().valence - before) < 0.02 && brain.getInnate().affectiveWords === 0,
    `Δvalence=${(brain.feel().valence - before).toFixed(3)}`);
  wait(brain, 600);
  brain.hearVoice(HARSH);
  quiet(() => brain.read('miedo'));
  check('read alongside a harsh voice, it takes on its emotion', brain.getInnate().affectiveWords === 1);
  wait(brain, 1200); // the voice's effect fades…
  const rested = brain.feel().valence;
  const cortisolRested = level(brain, ModulatorType.Cortisol);
  quiet(() => brain.read('miedo'));
  check('…and from then on the word alone evokes it', brain.feel().valence < rested - 0.03 && level(brain, ModulatorType.Cortisol) > cortisolRested,
    `valence ${rested.toFixed(2)} → ${brain.feel().valence.toFixed(2)}`);
  wait(brain, 1200);
  const neutralBefore = brain.feel().valence;
  quiet(() => brain.read('mesa'));
  check('a word never paired with anything still evokes nothing', Math.abs(brain.feel().valence - neutralBefore) < 0.02);
}

// ── 6. FEAR ─────────────────────────────────────────────────────────────────
console.log('\n6. FEAR');
{
  const brain = newBrain();
  see(brain, CROSS);
  brain.hearVoice(HARSH); // the cross was there when the harsh voice came
  check('one pairing conditions the figure', brain.getInnate().conditionedCues === 1);
  wait(brain, 1500);
  const cortisolBefore = level(brain, ModulatorType.Cortisol);
  see(brain, CROSS);
  const feared = brain.getInnate().lastCueResponse;
  check('seeing it afterwards brings the fear back', feared !== null && feared.valence < 0 && level(brain, ModulatorType.Cortisol) > cortisolBefore,
    `arousal=${feared?.arousal.toFixed(2)} cortisol ${cortisolBefore.toFixed(2)} → ${level(brain, ModulatorType.Cortisol).toFixed(2)}`);
  wait(brain, 1500);
  see(brain, SQUARE);
  check('another figure evokes nothing', brain.getInnate().lastCueResponse?.timestamp === feared?.timestamp);

  // Extinction: the cross, again and again, and nothing happens.
  const first = feared?.arousal ?? 0;
  for (let i = 0; i < 8; i++) { wait(brain, 400); see(brain, CROSS); }
  const extinguished = brain.getInnate().lastCueResponse?.arousal ?? 0;
  check('safe exposures extinguish the fear, slowly', extinguished < first * 0.6, `arousal ${first.toFixed(2)} → ${extinguished.toFixed(2)}`);

  // Spontaneous recovery under stress: once the cortisol of those exposures
  // has worn off, the same figure under fresh stress frightens again.
  wait(brain, 30_000);
  see(brain, CROSS);
  const calm = brain.getInnate().lastCueResponse?.arousal ?? 0;
  wait(brain, 400);
  brain.getModulators().release(ModulatorType.Cortisol, 0.5);
  see(brain, CROSS);
  const recovered = brain.getInnate().lastCueResponse?.arousal ?? 0;
  check('under stress the fear comes back', recovered > calm * 1.2, `arousal ${calm.toFixed(2)} → ${recovered.toFixed(2)} (cortisol ${level(brain, ModulatorType.Cortisol).toFixed(2)})`);
}

// ── 7. VERDICT ──────────────────────────────────────────────────────────────
console.log('\n7. VERDICT');
{
  const brain = newBrain();
  brain.associationWindowTicks = 120;
  const teach = (): void => {
    quiet(() => brain.see(CROSS, SIDE, SIDE, { propagate: false }));
    wait(brain, 50);
    quiet(() => brain.read('cruz', { propagate: false }));
    wait(brain, 180);
  };
  const probe = (): number => {
    quiet(() => brain.see(CROSS, SIDE, SIDE, { propagate: false }));
    wait(brain, 50);
    return brain.getLastRecall()?.confidence ?? 0;
  };
  teach();
  teach();
  const before = probe();
  brain.hearVoice(WARM); // "¡muy bien!" right after it recalled
  const judged = brain.getInnate().lastVoice?.judged === true;
  wait(brain, 180);
  const after = probe();
  check('a warm voice after a recall is taken as approval', judged);
  check('…and strengthens what was recalled', after > before, `confidence ${before.toFixed(2)} → ${after.toFixed(2)}`);
}

// ── 8. SLEEP ────────────────────────────────────────────────────────────────
console.log('\n8. SLEEP');
{
  const firstSleepTick = (brain: DigitalBrain, stimulate: (tick: number) => void): number => {
    let peak = 0;
    for (let t = 1; t <= 6000; t++) {
      stimulate(t);
      quiet(() => brain.tick());
      const pressure = brain.getInnate().sleepPressure;
      if (pressure < peak - 0.5) return t; // pressure cleared: it slept
      peak = Math.max(peak, pressure);
    }
    return Infinity;
  };
  const idle = newBrain({ memory: { sensoryBufferMs: 250, workingMemorySlots: 7, hippocampalCapacity: 10000, consolidationIntervalMs: 4000, consolidationReplays: 10 } });
  const idleSleep = firstSleepTick(idle, () => {});
  const busy = newBrain({ memory: { sensoryBufferMs: 250, workingMemorySlots: 7, hippocampalCapacity: 10000, consolidationIntervalMs: 4000, consolidationReplays: 10 } });
  const busySleep = firstSleepTick(busy, (t) => { if (t % 100 === 1) quiet(() => busy.read(t % 200 === 1 ? 'el perro corre' : 'la casa grande', { propagate: false })); });
  check('at rest it sleeps when the wake pressure is up', idleSleep >= 3900 && idleSleep <= 4100, `tick ${idleSleep}`);
  // Sleep pressure is mostly time awake; activity brings it forward by a
  // fraction (continuous stimulation here: about a fifth).
  check('stimulated, it sleeps sooner', busySleep < idleSleep * 0.9, `tick ${busySleep} vs ${idleSleep}`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
