/**
 * VERIFICATION TEST — Block 5: development — imagination, dreams, critical periods, pruning
 * ===========================================================================
 * Left alone, the brain does not go blank: it recombines what it knows into
 * things it never met (default mode), rewards itself for the new ones, and
 * knows they are its own. Asleep, the same machinery dreams. A category met
 * many times becomes a magnet for nearby inputs (perceptual narrowing), and
 * a category met once and never again is pruned.
 *
 *   1. DAYDREAM   — with nothing coming in, it imagines: combinations never
 *                   experienced, with words and an image in the mind's eye.
 *   2. ITS OWN    — the world imagined founds no category, no episode, no
 *                   association (reality monitoring).
 *   3. DOPAMINE   — imagining something new rewards; the same imagining
 *                   again rewards less.
 *   4. FORESEEN   — what it imagined, met for real, is recognized as foreseen.
 *   5. DREAMS     — asleep it dreams: chimeras of what it knows, with words.
 *   6. NARROWING  — a contrast heard from the start is two sounds; heard only
 *                   after one of them is well worn, it is assimilated.
 *   7. PRUNING    — a thing seen once and never again is gone after sleeping;
 *                   a thing seen often stays.
 *   8. PERSISTENCE — what it imagined survives a restart.
 */
import { existsSync, rmSync } from 'fs';
import { DigitalBrain, type BrainEvent, type Imagined } from '../src/brain.js';
import { synthesizeSpectrum } from '../src/core/voice/vocal-tract.js';
import { quiet, seedRandom, mulberry32 } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);
const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

// ── Coloured drawings (as in the composition test) ──────────────────────────
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
const BLUE_CAR = car(BLUE);
const GREEN_CARD = cardOf(GREEN);
const BLUE_CARD = cardOf(BLUE);
const grey = (draw: (plot: (x: number, y: number) => void) => void): number[] => picture(WHITE, draw).pixels;
const CROSS = grey((plot) => { for (let i = 8; i < 56; i++) { plot(i, 32); plot(32, i); } });
const SQUARE = grey((plot) => { for (let i = 12; i < 52; i++) { plot(i, 12); plot(i, 51); plot(12, i); plot(51, i); } });

const newBrain = (): DigitalBrain => { seedRandom(SEED); const b = quiet(() => new DigitalBrain()); b.associationWindowTicks = 120; return b; };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const show = (brain: DigitalBrain, p: Picture): void => { quiet(() => brain.see(p.pixels, SIDE, SIDE, { propagate: false, rgb: p.rgb })); wait(brain, 60); };
const showGrey = (brain: DigitalBrain, pixels: number[]): void => { quiet(() => brain.see(pixels, SIDE, SIDE, { propagate: false })); wait(brain, 60); };
const say = (brain: DigitalBrain, text: string): void => { quiet(() => brain.read(text, { propagate: false })); wait(brain, 180); };
const teach = (brain: DigitalBrain, p: Picture, text: string): void => { show(brain, p); say(brain, text); };

console.log('── Verification: block 5 — development ──\n');

// ── 1. DAYDREAM ─────────────────────────────────────────────────────────────
console.log('1. DAYDREAM');
const brain = newBrain();
const imagined: Imagined[] = [];
brain.on('response', (e: BrainEvent) => { if (e.data.kind === 'imagination') imagined.push(e.data as unknown as Imagined); });
for (let round = 0; round < 3; round++) {
  teach(brain, WHITE_CAR, 'coche');
  teach(brain, GREEN_CARD, 'verde');
  teach(brain, BLUE_CARD, 'azul');
}
show(brain, WHITE_CAR);
const carLabel = brain.getRecognition().visual!.label;
wait(brain, 200);
show(brain, BLUE_CARD);
const blueLabel = brain.getRecognition().colour!.label;
wait(brain, 200);
const before = {
  visual: brain.getRecognition().visualCategories,
  colour: brain.getRecognition().colourCategories,
  episodes: brain.getState().memoriesCount,
  bindings: brain.getState().association?.bindings ?? 0,
};
{
  imagined.length = 0;
  wait(brain, brain.ticksFor(60_000)); // left alone for a minute
  const daydreams = imagined.filter((im) => im.origin === 'daydream');
  const novel = daydreams.filter((im) => im.novelty >= 0.5);
  const worded = daydreams.filter((im) => im.words.length > 0);
  const pictured = daydreams.filter((im) => im.image !== null && im.image.some((v) => v > 0.2));
  check('left alone, the mind wanders', daydreams.length >= 2, `${daydreams.length} daydreams in a minute`);
  check('it imagines combinations it never experienced', novel.length >= 1,
    novel.slice(0, 3).map((im) => im.sources.map((s) => s.replace(/^[a-z]+:/, '')).join('+')).join(', ') || 'none');
  check('…with words that come to mind', worded.length >= 1, [...new Set(worded.flatMap((im) => im.words))].join(' ') || 'none');
  check("…and an image in the mind's eye", pictured.length >= 1, `${pictured.length} with an image`);
}

// ── 2. ITS OWN ──────────────────────────────────────────────────────────────
console.log('\n2. ITS OWN (reality monitoring)');
{
  const after = {
    visual: brain.getRecognition().visualCategories,
    colour: brain.getRecognition().colourCategories,
    episodes: brain.getState().memoriesCount,
    bindings: brain.getState().association?.bindings ?? 0,
  };
  check('what it imagined founded no category', after.visual === before.visual && after.colour === before.colour,
    `${before.visual}→${after.visual} shapes, ${before.colour}→${after.colour} colours`);
  check('…and no episode of the world', after.episodes <= before.episodes, `${before.episodes}→${after.episodes} episodes`);
  check('…and bound nothing', after.bindings === before.bindings, `${before.bindings}→${after.bindings} bindings`);
  check('it is tagged as its own', imagined.every((im) => im.origin === 'daydream' || im.origin === 'dream'));
}

// ── 3. DOPAMINE ─────────────────────────────────────────────────────────────
console.log('\n3. DOPAMINE of its own');
{
  const same = mulberry32(7);
  const first = quiet(() => brain.imagineOnce('daydream', mulberry32(7)));
  const r1 = brain.getMotivation().lastEvent;
  const again = quiet(() => brain.imagineOnce('daydream', same));
  const r2 = brain.getMotivation().lastEvent;
  check('imagining something new is rewarding', first !== null && r1 !== null && r1.kind === 'imagination' && r1.error > 0,
    r1 ? `${r1.key} dopamine +${r1.error.toFixed(2)}` : 'no event');
  check('the same imagining again rewards less', again !== null && r2 !== null && r2.kind === 'imagination' && r1 !== null && r2.error < r1.error,
    r1 && r2 ? `${r1.error.toFixed(2)} → ${r2.error.toFixed(2)}` : 'no event');
  check('the value of daydreaming is learned', brain.getMotivation().activityValues.daydream > 0, `worth ${brain.getMotivation().activityValues.daydream.toFixed(2)}`);
}

// ── 4. FORESEEN ─────────────────────────────────────────────────────────────
console.log('\n4. FORESEEN');
{
  // Make it imagine the car in blue — a combination it has never seen.
  let blueCar: Imagined | null = null;
  const random = mulberry32(11);
  for (let i = 0; i < 60 && !blueCar; i++) {
    const im = quiet(() => brain.imagineOnce('daydream', random));
    if (im && im.sources.includes(`visual:${carLabel}`) && im.sources.includes(`colour:${blueLabel}`)) blueCar = im;
  }
  check('it can imagine the car in blue, never seen', blueCar !== null && blueCar.novelty >= 0.5,
    blueCar ? `words: ${blueCar.words.join(' ') || '—'}, novelty ${blueCar.novelty.toFixed(2)}` : 'never came up');
  const foreseenBefore = brain.getImagination().foreseen;
  const affects: Array<Record<string, unknown>> = [];
  const onAffect = (e: BrainEvent): void => { affects.push(e.data); };
  brain.on('affect', onAffect);
  wait(brain, 200);
  show(brain, BLUE_CAR); // for real, for the first time
  wait(brain, 200);
  const foreseen = affects.find((a) => a.kind === 'foreseen');
  check('the blue car, met for real, is what it had imagined', foreseen !== undefined && brain.getImagination().foreseen === foreseenBefore + 1,
    foreseen ? String((foreseen.sources as string[]).join(' + ')) : 'not recognized as foreseen');
  const rewarded = brain.getMotivation().recentEvents.find((e) => e.kind === 'imagination' && e.key !== null && e.key.startsWith('foreseen:'));
  check('…and the imagining that proved true is rewarded', rewarded !== undefined && rewarded.error > 0, rewarded ? `dopamine +${rewarded.error.toFixed(2)}` : 'no reward');
}

// ── 5. DREAMS ───────────────────────────────────────────────────────────────
console.log('\n5. DREAMS');
{
  imagined.length = 0;
  const episodes = brain.getState().memoriesCount;
  const stats = quiet(() => brain.sleep());
  const dreams = imagined.filter((im) => im.origin === 'dream');
  check('asleep, it dreams', (stats.dreams ?? 0) >= 1 && dreams.length === stats.dreams, `${stats.dreams} dreams: ${(stats.dreamed ?? []).map((d) => d || '(wordless)').join(' | ')}`);
  check('a dream is made of things it knows, recombined', dreams.every((im) => im.sources.length === 2) && dreams.some((im) => im.words.length > 0 || im.visual !== null || im.colour !== null));
  check('dreaming founds no episode', brain.getState().memoriesCount <= episodes, `${episodes}→${brain.getState().memoriesCount}`);
  wait(brain, 5);
  const thought = brain.think();
  const remembered = dreams.some((im) => im.words.some((w) => thought.words.includes(w)));
  check('on waking, the dream is what is on its mind', remembered || dreams.every((im) => im.words.length === 0), thought.words.join(' · ') || '…');
}

// ── 6. NARROWING ────────────────────────────────────────────────────────────
console.log('\n6. NARROWING (critical period)');
{
  const S1 = synthesizeSpectrum({ f1: 700, f2: 1200, amplitude: 0.9 });
  const S2 = synthesizeSpectrum({ f1: 780, f2: 1350, amplitude: 0.9 });
  const play = (b: DigitalBrain, frame: Float32Array): string | null => {
    const prior = b.getRecognition().auditory;
    quiet(() => b.hearFrame(frame, 48000, { propagate: false }));
    wait(b, b.ticksFor(200));
    wait(b, b.presentationTicks + 20);
    const now = b.getRecognition().auditory;
    return now && now !== prior ? now.label : null;
  };
  // Heard from the start, side by side: two sounds.
  const young = newBrain();
  const y1 = play(young, S1);
  const y2 = play(young, S2);
  play(young, S1); play(young, S2);
  // One sound worn in first; the other only then.
  const older = newBrain();
  let o1: string | null = null;
  for (let i = 0; i < 16; i++) o1 = play(older, S1) ?? o1;
  const o2 = play(older, S2);
  const o2b = play(older, S2);
  check('heard from the start, the two sounds are two', y1 !== null && y2 !== null && y1 !== y2, `${y1} ≠ ${y2}`);
  check('heard only once one of them is well worn, the other is assimilated', o1 !== null && o2 !== null && o2 === o1 && o2b === o1, `${o1} ← ${o2}, ${o2b}`);
  const youngSounds = young.getRecognition().auditoryCategories ?? -1;
  const olderSounds = older.getRecognition().auditoryCategories ?? -1;
  check('the older brain has fewer sound categories', olderSounds < youngSounds, `${olderSounds} vs ${youngSounds}`);
}

// ── 7. PRUNING ──────────────────────────────────────────────────────────────
console.log('\n7. PRUNING');
{
  const b = newBrain();
  for (let i = 0; i < 3; i++) { showGrey(b, CROSS); wait(b, 200); }
  showGrey(b, SQUARE);
  const squareLabel = b.getRecognition().visual!.label;
  wait(b, 200);
  const founded = b.getRecognition().visualCategories;
  const s1 = quiet(() => b.sleep());
  wait(b, 50);
  const s2 = quiet(() => b.sleep());
  wait(b, 50);
  const pruned = [...(s1.prunedCategories ?? []), ...(s2.prunedCategories ?? [])];
  check('a thing seen once and never again is pruned after sleeping', pruned.includes(squareLabel) && b.getRecognition().visualCategories === founded - 1,
    `pruned: ${pruned.join(', ') || 'nothing'}; ${founded}→${b.getRecognition().visualCategories} categories`);
  showGrey(b, CROSS);
  const cross = b.getRecognition().visual!;
  wait(b, 200);
  check('a thing seen often stays', !cross.isNew && cross.exposures >= 4, `${cross.label} ${cross.exposures} exposures`);
  showGrey(b, SQUARE);
  const square = b.getRecognition().visual!;
  check('seen again, the pruned thing is new again', square.isNew, `${square.label}${square.isNew ? ' NEW' : ''}`);
}

// ── 8. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n8. PERSISTENCE');
{
  const path = `/tmp/gbrain-development-${process.pid}.bin`;
  quiet(() => brain.saveState(path));
  const restored = newBrain();
  quiet(() => restored.loadState(path));
  for (const p of [path, `${path}.lexicon.json`]) if (existsSync(p)) rmSync(p);
  const was = brain.getImagination();
  const is = restored.getImagination();
  check('what it imagined survives a restart', is.combinations === was.combinations && is.foreseen === was.foreseen && is.dreams === was.dreams,
    `${is.combinations} combinations, ${is.foreseen} foreseen, ${is.dreams} dreams`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
