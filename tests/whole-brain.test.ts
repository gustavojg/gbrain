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
 *   2. CASCADE       — a stimulus enters through the thalamus, reaches every
 *                      region of ITS pathway and no other sensory cortex (text
 *                      is neither seen nor heard; a voice reaches the auditory
 *                      cortex, reproducibly and specifically; noise is gated
 *                      out), and then the brain returns to rest.
 *   3. CONTENT CODE  — in the running brain, from the very first stimulus, the
 *                      thalamus and the language areas respond reproducibly to
 *                      the same text and differently to a different text
 *                      (feedback pathways modulate; they do not drive).
 *   4. INTEGRATORS   — in isolation, the leaky-integrator regions (Wernicke,
 *                      Broca, PFC) give reproducible, input-specific responses
 *                      (guards the LIF reset: without it the same neurons won
 *                      for every input).
 *   5. AFFECT        — the amygdala appraises emotional words in Spanish AND
 *                      English and drives the neuromodulators accordingly.
 *   6. NEUROMODULATION — in the running brain, acetylcholine widens the
 *                      thalamic gate and serotonin/cortisol raise the firing
 *                      threshold (fewer cortical neurons recruited).
 *   7. MEMORY        — one stimulus is encoded as one episode, a re-experience
 *                      is recognized (no duplicate) and recalled, different
 *                      stimuli get separate episodes and evoke different
 *                      hippocampal and prefrontal responses; sleep replays
 *                      episodes into the cortex and leaves the brain at rest.
 *   8. PERSISTENCE   — save → fresh brain → load restores weights exactly,
 *                      plus neuromodulators, vocabulary, the simulation clock
 *                      and the episodic index: a restored brain RECOGNIZES
 *                      what it had experienced before the restart.
 *   9. DETERMINISM   — same seed + same inputs ⇒ identical brain.
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
import { Hippocampus } from '../src/regions/hippocampus/hippocampus.js';
import { Lexicon } from '../src/regions/broca-wernicke/lexicon.js';
import { encodeSentenceToLexiconSpace, seedSpanishLexicon } from '../src/regions/broca-wernicke/spanish-lexicon.js';
import { WernickeArea } from '../src/regions/broca-wernicke/wernicke.js';
import { PrefrontalCortex } from '../src/regions/prefrontal-cortex/prefrontal-cortex.js';
import { mulberry32, quiet, seedRandom } from './helpers/seed.js';

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
  /** Spikes per neuron of each region over the whole wave. */
  counts: Record<string, Float32Array>;
  /** First non-empty firing pattern of each region. */
  first: Record<string, Set<number>>;
  /** Tick (1-based) at which each region first fired. */
  latency: Record<string, number>;
}

/** Reads `text`, observes the wave tick by tick, then lets the brain recover. */
function perceive(brain: DigitalBrain, text: string): Wave {
  return observe(brain, () => brain.read(text, { propagate: false }));
}

/** Injects a stimulus, observes the wave tick by tick, then lets the brain recover. */
function observe(brain: DigitalBrain, inject: () => void): Wave {
  const wave: Wave = { counts: {}, first: {}, latency: {} };
  quiet(inject);
  for (let t = 1; t <= WAVE_TICKS; t++) {
    brain.tick();
    const { regions } = brain.getState();
    for (const id of REGION_IDS) {
      const active = regions[id].activeNeurons;
      wave.counts[id] ??= new Float32Array(regions[id].outputSpikes.length);
      for (const n of active) wave.counts[id][n]++;
      if (wave.first[id] === undefined && active.length > 0) {
        wave.first[id] = new Set(active);
        wave.latency[id] = t;
      }
    }
  }
  for (let t = 0; t < RECOVERY_TICKS; t++) brain.tick();
  return wave;
}

/** Pearson correlation of two spike-count vectors. */
const correlation = (a: Float32Array, b: Float32Array): number => {
  const n = a.length;
  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < n; i++) { meanA += a[i]; meanB += b[i]; }
  meanA /= n;
  meanB /= n;
  let cov = 0;
  let varA = 0;
  let varB = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - meanA;
    const y = b[i] - meanB;
    cov += x * y;
    varA += x * x;
    varB += y * y;
  }
  return varA > 0 && varB > 0 ? cov / Math.sqrt(varA * varB) : 0;
};

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
  const sensoryCortices: string[] = ['visualCortex', 'auditoryCortex'];
  const textPathway = REGION_IDS.filter((id) => !sensoryCortices.includes(id));
  check('text reaches its whole pathway', textPathway.every((id) => reached.includes(id)),
    `${reached.filter((id) => !sensoryCortices.includes(id)).length}/6`);
  check('typed text is neither seen nor heard', sensoryCortices.every((id) => !(reached as string[]).includes(id)));

  const thalamusFirst = REGION_IDS.every((id) => (waveA1.latency[id] ?? Infinity) >= waveA1.latency.thalamus);
  check('the thalamus is the gateway (fires first)', thalamusFirst,
    REGION_IDS.map((id) => `${id.slice(0, 5)}@${waveA1.latency[id] ?? '—'}`).join(' '));

  check('the brain returns to rest after the wave', isSilent(mainBrain) && mainBrain.getBus().pendingCount === 0);

  // One microphone frame: 128 linear FFT bins at 48 kHz (187.5 Hz per bin).
  const rng = mulberry32(3);
  const vowel = (formant1: number, formant2: number): number[] =>
    Array.from({ length: 128 }, (_, bin) => {
      const hz = bin * 187.5;
      const peak = (centre: number, width: number): number => Math.exp(-((hz - centre) ** 2) / (2 * width * width));
      return Math.min(1, 0.05 * rng() + 0.9 * peak(formant1, 250) + 0.7 * peak(formant2, 350));
    });
  const hiss = Array.from({ length: 128 }, (_, bin) => (bin > 40 ? 0.6 + 0.2 * rng() : 0.02));

  const earBrain = newBrain();
  const hear = (frame: number[]): Wave => observe(earBrain, () => earBrain.hearFrame(frame, 48000, { propagate: false }));
  const voiceA1 = hear(vowel(500, 1500));
  const voiceA2 = hear(vowel(500, 1500));
  const voiceB = hear(vowel(300, 2500));
  const noise = hear(hiss);

  check('a voice reaches the auditory cortex — and not the visual one',
    voiceA1.latency.auditoryCortex !== undefined && voiceA1.latency.visualCortex === undefined,
    `first spike @${voiceA1.latency.auditoryCortex ?? '—'}`);
  const sameVoice = jaccard(voiceA1.first.auditoryCortex ?? new Set(), voiceA2.first.auditoryCortex ?? new Set());
  const otherVoice = jaccard(voiceA1.first.auditoryCortex ?? new Set(), voiceB.first.auditoryCortex ?? new Set());
  check('the auditory code is reproducible and sound-specific', sameVoice >= 0.8 && otherVoice <= 0.3,
    `J(same vowel)=${sameVoice.toFixed(2)} J(other vowel)=${otherVoice.toFixed(2)}`);
  check('broadband noise is gated out (no voice)', noise.latency.auditoryCortex === undefined);
}

// ── 3. CONTENT CODE ─────────────────────────────────────────────────────────
console.log('\n3. CONTENT CODE');
const waveA2 = perceive(mainBrain, TEXT_A);
const waveB = perceive(mainBrain, TEXT_B);
{
  // waveA1 is the very first stimulus this brain ever received.
  for (const id of ['thalamus', 'wernicke', 'broca'] as const) {
    const same = jaccard(waveA1.first[id], waveA2.first[id]);
    const diff = jaccard(waveA1.first[id], waveB.first[id]);
    check(`${id}: first response is reproducible and content-specific`, same >= 0.8 && diff <= 0.3,
      `J(A,A')=${same.toFixed(2)} J(A,B)=${diff.toFixed(2)}`);
  }
  const same = correlation(waveA1.counts.wernicke, waveA2.counts.wernicke);
  const diff = correlation(waveA1.counts.wernicke, waveB.counts.wernicke);
  check('wernicke: the WHOLE wave stays content-specific (feedback does not swamp it)',
    same >= 0.8 && diff <= 0.3, `r(A,A')=${same.toFixed(2)} r(A,B)=${diff.toFixed(2)}`);
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
  const pairs: Array<[string, string, string]> = [
    ['Spanish', 'estoy feliz con alegria y amor', 'tengo miedo tristeza y odio'],
    ['English', 'i am happy with joy and love', 'i feel fear sadness and hate'],
  ];
  for (const [language, positiveText, negativeText] of pairs) {
    const positive = valenceAfter(positiveText);
    const negative = valenceAfter(negativeText);
    check(`${language}: positive text ≫ negative text in valence`, positive - negative >= 0.5 && negative < 0,
      `+${positive.toFixed(2)} vs ${negative.toFixed(2)}`);
  }
  const neutral = valenceAfter('the hat is on the table');
  check('a neutral look-alike ("hat" ≠ "hate") evokes no negative affect', neutral > 0, `valence=${neutral.toFixed(2)}`);
}

// ── 6. NEUROMODULATION (in the running brain) ───────────────────────────────
console.log('\n6. NEUROMODULATION');
{
  // A long sentence: it activates more lexical channels than the thalamic
  // bottleneck lets through, so widening the gate is observable downstream.
  const longText =
    'el perro corre por el parque mientras la musica suena en la noche y los amigos ' +
    'caminan hacia la casa grande para comer juntos cerca del rio con mucha calma';

  const see = (modulator: ModulatorType | null, amount = 0): { relayedDrive: number; pfcRecruited: number } => {
    const brain = newBrain();
    if (modulator) brain.getModulators().release(modulator, amount);
    quiet(() => brain.read(longText, { propagate: false }));
    let relayedDrive = 0;
    let pfcRecruited = 0;
    for (let t = 0; t < 120; t++) {
      brain.tick();
      const { regions } = brain.getState();
      relayedDrive = Math.max(relayedDrive, regions.wernicke.drive);
      pfcRecruited = Math.max(pfcRecruited, regions.prefrontalCortex.activeNeurons.length);
    }
    return { relayedDrive, pfcRecruited };
  };

  const baseline = see(null);
  const acetylcholine = see(ModulatorType.Acetylcholine, 0.6);
  const serotonin = see(ModulatorType.Serotonin, 0.5);
  const cortisol = see(ModulatorType.Cortisol, 0.6);

  check('acetylcholine widens the thalamic gate (more signal reaches the cortex)',
    acetylcholine.relayedDrive > baseline.relayedDrive * 1.05,
    `Wernicke drive ${baseline.relayedDrive.toFixed(3)} → ${acetylcholine.relayedDrive.toFixed(3)}`);
  check('serotonin raises the firing threshold (fewer PFC neurons recruited)',
    serotonin.pfcRecruited < baseline.pfcRecruited * 0.95, `${baseline.pfcRecruited} → ${serotonin.pfcRecruited}`);
  check('cortisol raises the firing threshold (fewer PFC neurons recruited)',
    cortisol.pfcRecruited < baseline.pfcRecruited * 0.95, `${baseline.pfcRecruited} → ${cortisol.pfcRecruited}`);
}

// ── 7. MEMORY ───────────────────────────────────────────────────────────────
console.log('\n7. MEMORY');
{
  const brain = newBrain();
  const hippocampus = brain.getRegion('hippocampus') as Hippocampus;
  const episodesNow = (): number => brain.getState().memoriesCount;

  const a1 = perceive(brain, TEXT_A);
  const afterOne = episodesNow();
  const b1 = perceive(brain, TEXT_B);
  const afterTwo = episodesNow();
  check('one stimulus is encoded as one episode (at the event boundary)',
    afterOne === 1 && afterTwo === 2, `episodes: ${afterOne}, then ${afterTwo}`);

  // Re-experience both stimuli, twice: recall (2nd) vs recall (3rd).
  const a2 = perceive(brain, TEXT_A);
  const b2 = perceive(brain, TEXT_B);
  const a3 = perceive(brain, TEXT_A);
  const b3 = perceive(brain, TEXT_B);
  check('a re-experience is recognized, not stored again', episodesNow() === 2, `episodes after 6 stimuli: ${episodesNow()}`);

  const [episodeA, episodeB] = hippocampus.replay(2)
    .sort((x, y) => x.context.timestamp - y.context.timestamp)
    .map((episode) => episode.pattern);
  const overlap = Hippocampus.overlapBinary(episodeA, episodeB);
  check('different stimuli get well-separated episode codes', overlap <= 0.1, `overlap=${overlap.toFixed(2)}`);

  // The PFC legitimately shares part of its input across stimuli (the amygdala's
  // affect population, ~a quarter of its drive), so its ceiling for "different"
  // is higher than the hippocampus's — but it must stay far below "same".
  for (const [id, maxDifferent] of [['hippocampus', 0.3], ['prefrontalCortex', 0.5]] as const) {
    const same = Math.min(correlation(a2.counts[id], a3.counts[id]), correlation(b2.counts[id], b3.counts[id]));
    const diff = correlation(a3.counts[id], b3.counts[id]);
    check(`${id}: recalls the same stimulus alike, and different stimuli differently`,
      same >= 0.8 && diff <= maxDifferent && same - diff >= 0.4,
      `r(same)=${same.toFixed(2)} r(different)=${diff.toFixed(2)}`);
  }
  const novelVsRecalled = correlation(a1.counts.hippocampus, a2.counts.hippocampus);
  check('a first experience looks different from its recall (novelty vs memory)',
    novelVsRecalled < correlation(a2.counts.hippocampus, a3.counts.hippocampus) && b1.counts.hippocampus.length > 0,
    `r(1st,2nd)=${novelVsRecalled.toFixed(2)}`);

  const pfcWeights = brain.getRegion('prefrontalCortex')!.getNetworkConfig().weights;
  const before = pfcWeights.slice();
  const stats = quiet(() => brain.sleep());
  let change = 0;
  for (let i = 0; i < pfcWeights.length; i++) change += Math.abs(pfcWeights[i] - before[i]);
  check('sleep replays the stored episodes into the cortex',
    stats.consolidatedLabels.length === 2 && stats.memoriesReplayed > 0 && change > 0,
    `episodes=${stats.consolidatedLabels.length} replays=${stats.memoriesReplayed} PFC Σ|Δw|=${change.toFixed(1)}`);

  for (let t = 0; t < 50; t++) brain.tick();
  check('the brain wakes up at rest, with its episodes intact', isSilent(brain) && episodesNow() === 2);

  const afterSleep = perceive(brain, TEXT_A);
  check('an episode is still recalled after sleep (downscaling + replay keep the engram)',
    episodesNow() === 2 && correlation(afterSleep.counts.hippocampus, a3.counts.hippocampus) >= 0.8,
    `r=${correlation(afterSleep.counts.hippocampus, a3.counts.hippocampus).toFixed(2)}`);
}

// ── 8. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n8. PERSISTENCE');
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
  check('all 10 regions restored with bit-identical weights', result.loaded.length === 10 && weightsEqual,
    `loaded=${result.loaded.length} skipped=${result.skipped.length}`);

  const oxySaved = mainBrain.getModulators().getLevel(ModulatorType.Oxytocin);
  const oxyLoaded = fresh.getModulators().getLevel(ModulatorType.Oxytocin);
  check('neuromodulator levels restored', Math.abs(oxySaved - oxyLoaded) < 1e-6, `oxytocin=${oxyLoaded.toFixed(3)}`);
  check('learned vocabulary restored', mainBrain.knowsWord(NOVEL) && fresh.knowsWord(NOVEL));
  check('episodic index and simulation clock restored',
    savedEpisodes > 0 && restoredEpisodes === savedEpisodes && fresh.time === mainBrain.time,
    `episodes ${restoredEpisodes}/${savedEpisodes}, t=${fresh.time.toFixed(0)}ms`);

  // The real point of persisting memory: the restored brain recognizes what
  // it lived before the restart instead of storing it as something new.
  perceive(fresh, TEXT_A);
  check('a restored brain recognizes a pre-restart experience', fresh.getState().memoriesCount === savedEpisodes,
    `episodes after re-reading: ${fresh.getState().memoriesCount}`);
  perceive(fresh, 'los amigos caminan hacia la casa grande');
  check('…and still stores a genuinely new one', fresh.getState().memoriesCount === savedEpisodes + 1);
  cleanup();
}

// ── 9. DETERMINISM ──────────────────────────────────────────────────────────
console.log('\n9. DETERMINISM');
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
  console.log('   (none — every audited defect covered by this test is now a hard check)');
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
