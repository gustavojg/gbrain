/**
 * VERIFICATION TEST — The whole brain, end to end
 * ===========================================================================
 * The other tests exercise one region or one subsystem in isolation. This one
 * drives the assembled `DigitalBrain` (8 regions + connectome + bus +
 * neuromodulators) the way the server does, with a seeded PRNG so every number
 * below is reproducible.
 *
 * HARD CHECKS (must hold — they fail the suite):
 *   1. REST          — with no input nothing fires, nothing is stored, nothing
 *                      travels on the bus, and the thought stream is empty.
 *   2. CASCADE       — a stimulus reaches all 8 regions, entering through the
 *                      thalamus first, and then the brain returns to rest.
 *   3. SENSORY CODE  — the thalamic response is reproducible for the same text
 *                      and different for a different text.
 *   4. INTEGRATORS   — in isolation, the leaky-integrator regions (Wernicke,
 *                      Broca, PFC) give reproducible, input-specific responses
 *                      (guards the LIF reset: without it the same neurons won
 *                      for every input).
 *   5. AFFECT        — positive vs negative Spanish text moves valence apart.
 *   6. PERSISTENCE   — save → fresh brain → load restores weights exactly,
 *                      plus neuromodulators and learned vocabulary.
 *   7. DETERMINISM   — same seed + same inputs ⇒ identical brain.
 *
 * KNOWN GAPS (measured and reported, but they do NOT fail the suite):
 *   Defects found by the audit that are not fixed yet. Each one prints its
 *   metric next to the target. When a fix lands, the gap reports "CLOSED":
 *   move it to the hard checks so it can never regress.
 */

import { existsSync, rmSync } from 'fs';
import { DigitalBrain } from '../src/brain.js';
import type { BrainRegion } from '../src/core/brain-region.js';
import { ModulatorType, NeuromodulatorSystem } from '../src/core/neuromodulators/modulator-system.js';
import { BrocaArea } from '../src/regions/broca-wernicke/broca.js';
import { Lexicon } from '../src/regions/broca-wernicke/lexicon.js';
import { encodeSentenceToLexiconSpace, seedSpanishLexicon } from '../src/regions/broca-wernicke/spanish-lexicon.js';
import { WernickeArea } from '../src/regions/broca-wernicke/wernicke.js';
import { PrefrontalCortex } from '../src/regions/prefrontal-cortex/prefrontal-cortex.js';
import { quiet, seedRandom } from './helpers/seed.js';

/** Override with TEST_SEED=n to check that the results are not seed-specific. */
const SEED = Number(process.env.TEST_SEED ?? 20260917);
const TEXT_A = 'el perro corre por el parque';
const TEXT_B = 'la musica suena en la noche';
const REGION_IDS = [
  'thalamus', 'visualCortex', 'auditoryCortex', 'hippocampus',
  'amygdala', 'prefrontalCortex', 'broca', 'wernicke',
] as const;

/** Ticks a perception wave is observed for, and the rest that follows it. */
const WAVE_TICKS = 150;
/** Long enough for every pathway's synaptic resources to recover. */
const RECOVERY_TICKS = 600;

// ── Reporting ───────────────────────────────────────────────────────────────
const hard: Array<[string, boolean]> = [];
const gaps: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  hard.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};
const gap = (name: string, closed: boolean, detail: string): void => {
  gaps.push([name, closed]);
  console.log(`   ${closed ? '🎉 CLOSED' : '⚠️  OPEN  '} ${name}  (${detail})`);
};

// ── Helpers ─────────────────────────────────────────────────────────────────
const newBrain = (seed: number = SEED): DigitalBrain => {
  seedRandom(seed);
  return quiet(() => new DigitalBrain());
};

const jaccard = (a: Set<number>, b: Set<number>): number => {
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 1 : inter / union;
};

interface Wave {
  /** First non-empty firing pattern of each region. */
  first: Record<string, Set<number>>;
  /** Tick (1-based) at which each region first fired. */
  latency: Record<string, number>;
}

/** Reads `text`, observes the wave tick by tick, then lets the brain recover. */
function perceive(brain: DigitalBrain, text: string): Wave {
  const wave: Wave = { first: {}, latency: {} };
  quiet(() => brain.read(text, { propagate: false }));
  for (let t = 1; t <= WAVE_TICKS; t++) {
    brain.tick();
    const { regions } = brain.getState();
    for (const id of REGION_IDS) {
      if (wave.first[id] === undefined && regions[id].activeNeurons.length > 0) {
        wave.first[id] = new Set(regions[id].activeNeurons);
        wave.latency[id] = t;
      }
    }
  }
  for (let t = 0; t < RECOVERY_TICKS; t++) brain.tick();
  return wave;
}

const isSilent = (brain: DigitalBrain): boolean =>
  Object.values(brain.getState().regions).every((r) => r.firingRate === 0);

console.log('── Verification: the whole brain, end to end ──\n');

// ── 1. REST ─────────────────────────────────────────────────────────────────
console.log('1. REST');
{
  const brain = newBrain();
  for (let t = 0; t < 300; t++) brain.tick();
  const state = brain.getState();
  check('no region fires without input', isSilent(brain));
  check('no episode is stored without input', state.memoriesCount === 0, `episodes=${state.memoriesCount}`);
  check('nothing travels on the bus', brain.getBus().pendingCount === 0);
  check('the thought stream is empty', brain.think().words.length === 0);
}

// ── 2. CASCADE ──────────────────────────────────────────────────────────────
console.log('\n2. CASCADE');
const mainBrain = newBrain();
const waveA1 = perceive(mainBrain, TEXT_A);
{
  const reached = REGION_IDS.filter((id) => waveA1.latency[id] !== undefined);
  check('a stimulus reaches all 8 regions', reached.length === REGION_IDS.length, `${reached.length}/8`);

  const thalamusFirst = REGION_IDS.every((id) => (waveA1.latency[id] ?? Infinity) >= waveA1.latency.thalamus);
  check('the thalamus is the gateway (fires first)', thalamusFirst,
    REGION_IDS.map((id) => `${id.slice(0, 5)}@${waveA1.latency[id] ?? '—'}`).join(' '));

  check('the brain returns to rest after the wave', isSilent(mainBrain) && mainBrain.getBus().pendingCount === 0);
}

// ── 3. SENSORY CODE ─────────────────────────────────────────────────────────
console.log('\n3. SENSORY CODE');
const waveA2 = perceive(mainBrain, TEXT_A);
const waveB = perceive(mainBrain, TEXT_B);
{
  const same = jaccard(waveA1.first.thalamus, waveA2.first.thalamus);
  const diff = jaccard(waveA1.first.thalamus, waveB.first.thalamus);
  check('thalamic code is reproducible for the same text', same >= 0.6, `J(A,A')=${same.toFixed(2)}`);
  check('thalamic code differs for a different text', diff <= 0.3, `J(A,B)=${diff.toFixed(2)}`);
}

// ── 4. INTEGRATORS (in isolation) ───────────────────────────────────────────
console.log('\n4. INTEGRATORS');
{
  seedRandom(SEED);
  const lexicon = new Lexicon(1000);
  quiet(() => seedSpanishLexicon(lexicon));
  const effects = new NeuromodulatorSystem().getEffects();
  const regions: BrainRegion[] = quiet(() => [
    new WernickeArea(lexicon, 1000, 1000),
    new BrocaArea(lexicon, 1000, 1000),
    new PrefrontalCortex(1000, 1000),
  ]);

  for (const region of regions) {
    /** Presents a pattern for one tick, returns the response, then rests. */
    const present = (text: string, ticks = 1): Set<number> => {
      const pattern = encodeSentenceToLexiconSpace(text, 1000);
      let response = new Set<number>();
      for (let t = 0; t < ticks; t++) {
        region.feedInput(pattern);
        response = new Set(quiet(() => region.step(1, effects)).activeNeurons);
      }
      for (let t = 0; t < 300; t++) region.step(1, effects);
      return response;
    };

    present('hola amigo como estas', 60); // warm-up: membranes start hyperpolarized
    const a1 = present(TEXT_A);
    const a2 = present(TEXT_A);
    const b1 = present(TEXT_B);
    const same = jaccard(a1, a2);
    const diff = jaccard(a1, b1);
    check(`${region.id}: reproducible and input-specific`,
      a1.size > 0 && same >= 0.8 && diff <= 0.5, `J(A,A')=${same.toFixed(2)} J(A,B)=${diff.toFixed(2)}`);
  }
}

// ── 5. AFFECT ───────────────────────────────────────────────────────────────
console.log('\n5. AFFECT');
const valenceAfter = (text: string): number => {
  const brain = newBrain();
  quiet(() => { for (let i = 0; i < 3; i++) brain.read(text); });
  return brain.feel().valence;
};
{
  const positive = valenceAfter('estoy feliz con alegria y amor');
  const negative = valenceAfter('tengo miedo tristeza y odio');
  check('Spanish: positive text ≫ negative text in valence', positive - negative >= 0.5,
    `+${positive.toFixed(2)} vs ${negative.toFixed(2)}`);
}

// ── 6. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n6. PERSISTENCE');
const statePath = `/tmp/gbrain-whole-test-${process.pid}.bin`;
const stateFiles = ['', '.bak', '.tmp', '.lexicon.json'].map((suffix) => statePath + suffix);
const cleanup = (): void => { for (const f of stateFiles) if (existsSync(f)) rmSync(f); };
let restoredEpisodes = -1;
let savedEpisodes = -1;
{
  cleanup();
  const NOVEL = 'zarpalio';
  quiet(() => { for (let i = 0; i < 3; i++) mainBrain.read(`el ${NOVEL} corre feliz`); });
  mainBrain.getModulators().release(ModulatorType.Oxytocin, 0.4);
  savedEpisodes = mainBrain.getState().memoriesCount;
  quiet(() => mainBrain.saveState(statePath));

  const fresh = newBrain(SEED + 1); // different seed ⇒ different initial weights
  const result = quiet(() => fresh.loadState(statePath));
  restoredEpisodes = fresh.getState().memoriesCount;

  let weightsEqual = true;
  for (const [id, region] of mainBrain.getRegions()) {
    const a = region.getNetworkConfig().weights;
    const b = fresh.getRegion(id)!.getNetworkConfig().weights;
    for (let i = 0; i < a.length && weightsEqual; i++) if (a[i] !== b[i]) weightsEqual = false;
  }
  check('all 8 regions restored with bit-identical weights', result.loaded.length === 8 && weightsEqual,
    `loaded=${result.loaded.length} skipped=${result.skipped.length}`);

  const oxySaved = mainBrain.getModulators().getLevel(ModulatorType.Oxytocin);
  const oxyLoaded = fresh.getModulators().getLevel(ModulatorType.Oxytocin);
  check('neuromodulator levels restored', Math.abs(oxySaved - oxyLoaded) < 1e-6, `oxytocin=${oxyLoaded.toFixed(3)}`);
  check('learned vocabulary restored', mainBrain.knowsWord(NOVEL) && fresh.knowsWord(NOVEL));
  cleanup();
}

// ── 7. DETERMINISM ──────────────────────────────────────────────────────────
console.log('\n7. DETERMINISM');
{
  const fingerprint = (): string => {
    const brain = newBrain(SEED + 2);
    quiet(() => { brain.read(TEXT_A); brain.read(TEXT_B); });
    const state = brain.getState();
    return JSON.stringify([
      state.memoriesCount,
      state.modulators,
      Object.values(state.regions).map((r) => r.activeNeurons),
      brain.speak().text,
      brain.think().text,
    ]);
  };
  check('same seed + same inputs ⇒ identical brain', fingerprint() === fingerprint());
}

// ── KNOWN GAPS ──────────────────────────────────────────────────────────────
console.log('\nKNOWN GAPS (audit findings not fixed yet — reported, not enforced)');
{
  // G1. In the assembled brain the language areas' response is dominated by
  // feedback traffic and by Hebbian learning on it, not by what was read.
  const same = jaccard(waveA1.first.wernicke, waveA2.first.wernicke);
  const diff = jaccard(waveA1.first.wernicke, waveB.first.wernicke);
  gap('G1 Wernicke response in the whole brain is content-specific', same >= 0.6 && diff <= 0.3,
    `J(A,A')=${same.toFixed(2)} J(A,B)=${diff.toFixed(2)}; target ≥0.60 / ≤0.30`);

  // G2. The hippocampal novelty gate only compares with the last stored code,
  // so one stimulus is stored as dozens of near-duplicate episodes.
  const brain = newBrain();
  perceive(brain, TEXT_A);
  const episodes = brain.getState().memoriesCount;
  gap('G2 one stimulus is stored as ~one episode', episodes <= 3, `episodes=${episodes}; target ≤3`);

  // G3. Emotional words only exist in Spanish (EMOTIONAL_WORDS); the amygdala's
  // own network does not evaluate content.
  const positive = valenceAfter('i am happy with joy and love');
  const negative = valenceAfter('i feel fear sadness and hate');
  gap('G3 English text moves valence too', positive - negative >= 0.5,
    `+${positive.toFixed(2)} vs ${negative.toFixed(2)}; target Δ≥0.50`);

  // G4. Thalamus.processInput ignores the modulation effects on the live path.
  const thalamicSpikes = (acetylcholine: number): number => {
    const b = newBrain();
    b.getModulators().release(ModulatorType.Acetylcholine, acetylcholine);
    quiet(() => b.read(TEXT_A, { propagate: false }));
    let spikes = 0;
    for (let t = 0; t < 60; t++) {
      b.tick();
      spikes += b.getState().regions.thalamus.activeNeurons.length;
    }
    return spikes;
  };
  const base = thalamicSpikes(0);
  const boosted = thalamicSpikes(0.6);
  gap('G4 acetylcholine widens thalamic gating in the running brain', boosted > base,
    `spikes ${base} → ${boosted}; target: increase`);

  // G5. Only weights are persisted: the episodic index is lost on restart.
  gap('G5 hippocampal episodes survive save/load', savedEpisodes > 0 && restoredEpisodes === savedEpisodes,
    `saved=${savedEpisodes} restored=${restoredEpisodes}`);
}

// ── Verdict ─────────────────────────────────────────────────────────────────
const failed = hard.filter(([, ok]) => !ok);
const open = gaps.filter(([, closed]) => !closed);
const closed = gaps.filter(([, isClosed]) => isClosed);
console.log('');
if (closed.length > 0) {
  console.log(`🎉 ${closed.length} known gap(s) now CLOSED — promote to hard checks: ${closed.map(([n]) => n.slice(0, 2)).join(', ')}`);
}
if (failed.length === 0) {
  console.log(`✅ WHOLE BRAIN VERIFIED: ${hard.length}/${hard.length} hard checks; ${open.length} known gap(s) still open.`);
  process.exit(0);
} else {
  console.log(`❌ FAIL: ${failed.map(([name]) => name).join(' | ')}`);
  process.exit(1);
}
