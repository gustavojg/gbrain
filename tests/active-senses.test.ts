/**
 * VERIFICATION TEST — Block 4: active senses
 * ===========================================================================
 * The senses are not passive. A scene with several things in it is not seen
 * at once: the eye fixates one, the cortex sees it alone, and a saccade
 * takes the eye to the next.
 *
 *   1. THE MOVING EYE — a scene with a cross and a square, both known and
 *                       named, is looked at in two fixations, each named;
 *                       a single thing is looked at once.
 *   2. CONSONANTS     — an utterance unfolds in time: having babbled "ma"s
 *                       and "pa"s, it repeats "ma" with the lips closed and
 *                       the nose open, "pa" with a release, "a" with neither.
 */
import { DigitalBrain, type BrainEvent, type Vocalization } from '../src/brain.js';
import { synthesizeFrames, UTTERANCE_FRAME_MS, VOCAL_RANGE, type VocalCommand } from '../src/core/voice/vocal-tract.js';
import { quiet, seedRandom } from './helpers/seed.js';

const SEED = Number(process.env.TEST_SEED ?? 20260917);
const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

function canvas(w: number, h: number, draw: (plot: (x: number, y: number) => void) => void): number[] {
  const pixels = new Array<number>(w * h).fill(27);
  draw((x, y) => {
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      const px = x + dx, py = y + dy;
      if (px >= 0 && px < w && py >= 0 && py < h) pixels[py * w + px] = 255;
    }
  });
  return pixels;
}
const cross = (plot: (x: number, y: number) => void, cx: number, cy: number, r: number): void => { for (let i = -r; i <= r; i++) { plot(cx + i, cy); plot(cx, cy + i); } };
const square = (plot: (x: number, y: number) => void, cx: number, cy: number, r: number): void => { for (let i = -r; i <= r; i++) { plot(cx + i, cy - r); plot(cx + i, cy + r); plot(cx - r, cy + i); plot(cx + r, cy + i); } };
const SIDE = 64;
const CROSS = canvas(SIDE, SIDE, (plot) => cross(plot, 32, 32, 24));
const SQUARE = canvas(SIDE, SIDE, (plot) => square(plot, 32, 32, 20));
/** A wide scene: the cross on the left, the square on the right. */
const SCENE_W = 128, SCENE_H = 64;
const SCENE = canvas(SCENE_W, SCENE_H, (plot) => { cross(plot, 30, 32, 16); square(plot, 96, 32, 14); });

const newBrain = (): DigitalBrain => { seedRandom(SEED); const b = quiet(() => new DigitalBrain()); b.associationWindowTicks = 120; return b; };
const wait = (brain: DigitalBrain, ticks: number): void => { for (let t = 0; t < ticks; t++) brain.tick(); };
const see = (brain: DigitalBrain, pixels: number[], w = SIDE, h = SIDE): void => { quiet(() => brain.see(pixels, w, h, { propagate: false })); wait(brain, 60); };
const say = (brain: DigitalBrain, text: string): void => { quiet(() => brain.read(text, { propagate: false })); wait(brain, 180); };

console.log('── Verification: block 4 — active senses ──\n');

// ── 1. THE MOVING EYE ───────────────────────────────────────────────────────
console.log('1. THE MOVING EYE');
{
  const brain = newBrain();
  const written: string[] = [];
  const saccades: Array<Record<string, unknown>> = [];
  const seen: string[] = [];
  brain.on('response', (e: BrainEvent) => { if (e.data.kind === 'writing') written.push(String(e.data.text)); });
  brain.on('affect', (e: BrainEvent) => { if (e.data.kind === 'saccade') saccades.push(e.data); });
  brain.on('memory', (e: BrainEvent) => { if (e.data.kind === 'recognition' && e.data.modality === 'visual') seen.push(String(e.data.label)); });
  for (let i = 0; i < 4; i++) {
    see(brain, CROSS); say(brain, 'cruz');
    see(brain, SQUARE); say(brain, 'cuadrado');
  }
  see(brain, CROSS);
  const crossLabel = brain.getRecognition().visual!.label;
  wait(brain, 200);
  see(brain, SQUARE);
  const squareLabel = brain.getRecognition().visual!.label;
  wait(brain, 200);
  const categories = brain.getRecognition().visualCategories;

  // A single thing: one look, no saccade.
  saccades.length = 0;
  see(brain, CROSS);
  wait(brain, 200);
  check('a single thing is looked at once', saccades.length === 0, `${saccades.length} saccades`);

  // The scene: two fixations, each recognized and named.
  written.length = 0;
  saccades.length = 0;
  const labels: string[] = [];
  quiet(() => brain.see(SCENE, SCENE_W, SCENE_H, { propagate: false }));
  wait(brain, 60);
  labels.push(brain.getRecognition().visual?.label ?? '—');
  wait(brain, brain.presentationTicks + brain.ticksFor(1200) + 80);
  labels.push(brain.getRecognition().visual?.label ?? '—');
  wait(brain, 200);
  check('a scene with two things is looked at in two fixations', saccades.length === 2 && saccades.every((s) => Number(s.count) === 2),
    saccades.map((s) => `${s.index}/${s.count} at x=${(s.box as { x: number }).x}`).join(', ') || 'none');
  check('each fixation is one of the things it knows', labels.includes(crossLabel) && labels.includes(squareLabel) && brain.getRecognition().visualCategories === categories,
    `${labels.join(' → ')} (${crossLabel} = cross, ${squareLabel} = square); ${brain.getRecognition().visualCategories} categories`);
  check('…and it names them one by one', written.includes('cruz') && written.includes('cuadrado'), written.join(' ') || 'nothing written');
}

// ── 2. CONSONANTS ───────────────────────────────────────────────────────────
console.log('\n2. CONSONANTS');
{
  const brain = newBrain();
  brain.setVoice({ imitate: true });
  /** Someone says an utterance: its frames reach the ear one every 200 ms. */
  const sayTo = (command: VocalCommand): Vocalization | null => {
    const before = brain.getLastVocalization()?.serial ?? 0;
    for (const frame of synthesizeFrames(command)) {
      quiet(() => brain.hearFrame(frame, 48000, { propagate: false }));
      wait(brain, brain.ticksFor(UTTERANCE_FRAME_MS));
    }
    wait(brain, 140);
    const after = brain.getLastVocalization();
    return after && after.serial !== before && after.source === 'imitation' ? after : null;
  };
  const error = (v: Vocalization, c: VocalCommand): number =>
    (Math.abs(v.command.f1 - c.f1) / (VOCAL_RANGE.f1[1] - VOCAL_RANGE.f1[0]) + Math.abs(v.command.f2 - c.f2) / (VOCAL_RANGE.f2[1] - VOCAL_RANGE.f2[0])) / 2;
  const VOWELS: Record<string, [number, number]> = { a: [700, 1200], i: [300, 2300], o: [500, 900], u: [350, 800] };
  for (let i = 0; i < 480; i++) { brain.babbleOnce(); wait(brain, 60); }
  const said = (c: VocalCommand): string => `${c.onset === 'nasal' ? 'm' : c.onset === 'stop' ? 'p' : ''}${c.f1.toFixed(0)}/${c.f2.toFixed(0)}`;
  /** Each syllable of a kind, said to it once: what it repeats, and with what lips. */
  const trial = (onset: 'nasal' | 'stop' | undefined): { repeated: number; right: number; detail: string } => {
    let repeated = 0, right = 0;
    const detail: string[] = [];
    for (const [vowel, [f1, f2]] of Object.entries(VOWELS)) {
      const command: VocalCommand = { f1, f2, amplitude: 0.9 };
      if (onset) command.onset = onset;
      const v = sayTo(command);
      const name = `${onset === 'nasal' ? 'm' : onset === 'stop' ? 'p' : ''}${vowel}`;
      if (!v) { detail.push(`${name}→—`); continue; }
      repeated++;
      // The lips are what this is about; the vowel's accuracy is the vocal test's business.
      const ok = v.command.onset === onset;
      if (ok) right++;
      detail.push(`${name}→${said(v.command)}${ok ? '' : ' ✗'}${error(v, command) >= 0.2 ? ' (vowel off)' : ''}`);
    }
    return { repeated, right, detail: detail.join(' ') };
  };
  const nasal = trial('nasal');
  const stop = trial('stop');
  const bare = trial(undefined);
  check('"ma", "mi", "mo", "mu": repeated with the lips closed and the nose open', nasal.repeated >= 3 && nasal.right === nasal.repeated, nasal.detail);
  check('"pa", "pi", "po", "pu": repeated with a release', stop.repeated >= 3 && stop.right === stop.repeated, stop.detail);
  // (/i/ and the nasal murmur share a low first formant; a bare /i/ may be
  // taken for a closed-lips onset — a confusion infants make too.)
  check('the bare vowels: repeated with the lips open', bare.repeated >= 3 && bare.right >= 3, bare.detail);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
