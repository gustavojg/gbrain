/**
 * VERIFICATION TEST — the native cortex: a region on the engine
 * ===========================================================================
 * A sheet of 10 000 spiking neurons with sparse recurrent synapses and real
 * interneurons, fed through an afferent projection, must behave as a
 * cortex should without any algorithmic competition or template:
 *
 *   1. ASSEMBLIES  — the same input twice drives much the same neurons;
 *                    two different inputs drive different ones.
 *   2. SPARSE      — the interneurons keep the response to a few percent.
 *   3. ASSEMBLY    — after an input has been repeated, the recurrent synapses
 *                    between the neurons of its assembly have grown more than
 *                    the rest (Hebb: what fires together wires together) and
 *                    structural plasticity has grown synapses between them.
 *   4. COMPLETION  — the afferents are topographic, so half the input drives
 *                    the half of the sheet that listens to it; the assembly's
 *                    other half, which the cue does not reach, must come back
 *                    through the recurrent synapses — and only for its own
 *                    cue, not for another input, and not in silence (short-
 *                    term depression lets the assembly ignite and fade).
 *   5. PERSISTENCE — the recurrent synapses, rewired ones included, survive
 *                    a restart.
 *   6. IN THE BRAIN — with GBRAIN_NATIVE=1 the brain has the region, fed by
 *                    the visual relay: it fires when something is seen and
 *                    is quiet otherwise.
 *
 * Needs `npm run build:native`.
 */
import { DigitalBrain } from '../src/brain.js';
import { NativeCortex } from '../src/regions/native-cortex/native-cortex.js';
import { quiet, seedRandom } from './helpers/seed.js';

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

console.log('── Verification: the native cortex ──\n');
if (!NativeCortex.available()) {
  console.log('❌ The native engine is not built. Run: npm run build:native');
  process.exit(1);
}

const INPUTS = 1000;
const effects = { learningRateMultiplier: 1, thresholdMultiplier: 1, attentionGain: 1, consolidationRate: 1, spikeGainMultiplier: 1, socialWeightBoost: 1 };
/** An input pattern: a fixed random tenth of the channels lit, the same share in each half. */
const pattern = (seed: number, fraction = 0.1): Float32Array => {
  let r = seed >>> 0;
  const next = (): number => { r = (r * 1664525 + 1013904223) >>> 0; return r / 4294967296; };
  const p = new Float32Array(INPUTS);
  const perHalf = Math.round((INPUTS / 2) * fraction);
  for (const offset of [0, INPUTS / 2]) {
    let lit = 0;
    while (lit < perHalf) { const i = offset + Math.floor(next() * (INPUTS / 2)); if (p[i] === 0) { p[i] = 1; lit++; } }
  }
  return p;
};
const half = (p: Float32Array): Float32Array => p.map((v, i) => (i < INPUTS / 2 ? v : 0));
const correlation = (x: Float32Array, y: Float32Array): number => {
  const n = x.length;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; syy += y[i] * y[i]; sxy += x[i] * y[i]; }
  const cov = sxy / n - (sx / n) * (sy / n);
  return cov / Math.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2) + 1e-12);
};
/**
 * Presents a pattern for `ticks` ticks and returns how many times each
 * EXCITATORY neuron fired over the presentation (the interneurons fire to
 * everything and carry no representation). An assembly ignites and fades, so
 * the whole presentation counts. Correlating these counts across
 * presentations is the same-vs-different criterion the whole-brain test uses
 * for the hippocampus and the prefrontal cortex.
 */
const present = (cortex: NativeCortex, p: Float32Array, ticks: number, learn: boolean, silence = 100): Float32Array => {
  const counts = new Float32Array(cortex.population);
  for (let t = 0; t < ticks; t++) {
    cortex.processInput(p, { ...effects, learningRateMultiplier: learn ? 1 : 0 });
    for (const n of cortex.fired()) if (!cortex.isInhibitory(n)) counts[n]++;
  }
  // Silence between presentations: membranes and synaptic resources recover.
  const quiet = new Float32Array(INPUTS);
  for (let t = 0; t < silence; t++) cortex.processInput(quiet, { ...effects, learningRateMultiplier: 0 });
  return counts;
};
const total = (counts: Float32Array): number => counts.reduce((a, b) => a + b, 0);

// ── 1 & 2. ASSEMBLIES, SPARSE ──────────────────────────────────────────────
console.log('1. ASSEMBLIES');
const cortex = new NativeCortex({ neurons: 10_000, inputCount: INPUTS, seed: 11, plastic: true });
const A = pattern(1), B = pattern(2);
const a1 = present(cortex, A, 60, false);
const a2 = present(cortex, A, 60, false);
const b1 = present(cortex, B, 60, false);
const same = correlation(a1, a2), different = correlation(a1, b1);
check('the same input twice drives much the same neurons', same >= 0.5, `r = ${same.toFixed(2)}`);
// Topographic afferents: where two random inputs light the same stretch of channels, the same neurons answer both.
check('two different inputs drive different neurons', different <= 0.35 && different <= same / 2, `r = ${different.toFixed(2)}`);
console.log('\n2. SPARSE');
let excitatory = 0, active = 0;
for (let i = 0; i < cortex.population; i++) if (!cortex.isInhibitory(i)) { excitatory++; if (a1[i] > 0) active++; }
{
  // Sparse per tick, and few neurons held up: the onset makes many fire once
  // before the interneurons catch up, but the input keeps only a few going.
  const perTick = total(a1) / 60 / excitatory;
  let sustained = 0;
  for (let i = 0; i < cortex.population; i++) if (!cortex.isInhibitory(i) && a1[i] >= 3) sustained++;
  const heldUp = sustained / excitatory, fraction = active / excitatory;
  check('the interneurons keep the response sparse', perTick > 0.002 && perTick < 0.05 && heldUp < 0.1, `${(perTick * 100).toFixed(1)}% of the excitatory neurons fire per tick; ${(heldUp * 100).toFixed(1)}% fire three times or more over the presentation (${(fraction * 100).toFixed(0)}% at least once)`);
}

// ── 3. ASSEMBLY ─────────────────────────────────────────────────────────────
console.log('\n3. ASSEMBLY');
{
  const h0 = present(cortex, half(A), 60, false);
  const w0 = cortex.recurrentWeights();
  for (let i = 0; i < 40; i++) present(cortex, A, 60, true);
  const w1 = cortex.recurrentWeights();
  const { rowPtr, targets } = cortex.recurrentSynapses();
  let changed = 0;
  for (let s = 0; s < w1.length; s++) if (w1[s] !== w0[s]) changed++;
  // The assembly: the twentieth of the excitatory neurons the input holds up most.
  const ranked = Array.from({ length: cortex.population }, (_, i) => i).filter((i) => !cortex.isInhibitory(i)).sort((x, y) => a1[y] + a2[y] - a1[x] - a2[x]);
  const assembly = new Set(ranked.slice(0, Math.floor(ranked.length / 20)));
  let inSum = 0, inN = 0, outSum = 0, outN = 0;
  for (let i = 0; i < cortex.population; i++) {
    if (cortex.isInhibitory(i)) continue;
    for (let s = rowPtr[i]; s < rowPtr[i + 1]; s++) {
      if (cortex.isInhibitory(targets[s])) continue;
      const d = w1[s] - w0[s];
      if (assembly.has(i) && assembly.has(targets[s])) { inSum += d; inN++; } else { outSum += d; outN++; }
    }
  }
  const within = inSum / Math.max(1, inN), rest = outSum / Math.max(1, outN);
  check('repetition moves the recurrent synapses', changed > 1000, `${changed} synapses moved`);
  check('the synapses within the assembly grow more than the rest', within > rest && within > 0,
    `within ${within >= 0 ? '+' : ''}${within.toFixed(3)} (${inN} of ${inN + outN} excitatory synapses, assembly of ${assembly.size}) vs rest ${rest >= 0 ? '+' : ''}${rest.toFixed(3)}`);
  check('structural plasticity rewires synapses toward coactive neurons', cortex.rewired > 100, `${cortex.rewired} synapses rewired`);

  // ── 4. COMPLETION ─────────────────────────────────────────────────────────
  console.log('\n4. COMPLETION');
  const full = present(cortex, A, 60, false);
  const h1 = present(cortex, half(A), 60, false);
  const bAfter = present(cortex, B, 60, false);
  const idle = present(cortex, new Float32Array(INPUTS), 200, false, 0);
  // The assembly after learning: excitatory neurons the full input holds up
  // (three or more spikes). With receptive fields a fifth of the channels
  // wide, a neuron in the last two fifths of the sheet listens only to the
  // unlit half: the cue cannot reach it, only the recurrent synapses can.
  const n = cortex.population;
  const unlitSide = (i: number): boolean => (i + 0.5) / n >= 0.6, litSide = (i: number): boolean => (i + 0.5) / n < 0.4;
  const inAssembly = (i: number): boolean => !cortex.isInhibitory(i) && full[i] >= 3;
  const all = Array.from({ length: n }, (_, i) => i).filter((i) => !cortex.isInhibitory(i));
  const unlitAssembly = all.filter((i) => inAssembly(i) && unlitSide(i)), litAssembly = all.filter((i) => inAssembly(i) && litSide(i));
  const unlitOthers = all.filter((i) => !inAssembly(i) && unlitSide(i));
  const share = (set: number[], counts: Float32Array): number => (set.length ? set.filter((i) => counts[i] >= 1).length / set.length : 0);
  const back0 = share(unlitAssembly, h0), back1 = share(unlitAssembly, h1), others = share(unlitOthers, h1);
  check('the cue drives its own side of the assembly', share(litAssembly, h1) >= 0.8 && litAssembly.length >= 5 && unlitAssembly.length >= 5, `${litAssembly.length} on the lit side, ${unlitAssembly.length} on the unlit side`);
  check('the side the cue does not reach comes back through the recurrent synapses', back1 >= 0.6 && back1 >= back0 + 0.4, `${(back0 * 100).toFixed(0)}% → ${(back1 * 100).toFixed(0)}% of the unlit side of the assembly fires to half the input`);
  check('…and mostly the assembly, not the rest of that side', others <= 0.15 && others < back1 / 3, `${(others * 100).toFixed(0)}% of the other neurons there fire`);
  check('the completed response is the input\'s own: another input does not bring it back', correlation(h1, bAfter) <= 0.4, `r(half A, B) = ${correlation(h1, bAfter).toFixed(2)}`);
  check('it ignites and fades: nothing keeps firing in silence', total(idle) / 200 < 1, `${(total(idle) / 200).toFixed(2)} excitatory spikes per tick with no input`);
}

// ── 5. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n5. PERSISTENCE');
{
  // Structural plasticity rewired synapses above; a restored cortex must have the same connectivity AND weights.
  const extra = JSON.parse(JSON.stringify(cortex.serializeExtra())) as unknown;
  const restored = new NativeCortex({ neurons: 10_000, inputCount: INPUTS, seed: 11, plastic: true });
  restored.deserializeExtra(extra);
  const w0 = cortex.recurrentWeights(), w1 = restored.recurrentWeights();
  const t0 = cortex.recurrentSynapses().targets, t1 = restored.recurrentSynapses().targets;
  let weightsOff = 0, targetsOff = 0;
  for (let s = 0; s < w0.length; s++) { if (w0[s] !== w1[s]) weightsOff++; if (t0[s] !== t1[s]) targetsOff++; }
  check('the recurrent synapses survive a restart, rewired ones included', weightsOff === 0 && targetsOff === 0 && cortex.rewired > 0, `${weightsOff} weights and ${targetsOff} targets differ after restore (${cortex.rewired} rewired)`);
  const h2 = present(restored, half(A), 60, false);
  check('…and the restored cortex completes the pattern too', correlation(h2, present(cortex, half(A), 60, false)) >= 0.5, `r = ${correlation(h2, present(cortex, half(A), 60, false)).toFixed(2)} between the two cortices' responses to half the input`);
}

// ── 6. IN THE BRAIN ─────────────────────────────────────────────────────────
console.log('\n6. IN THE BRAIN');
{
  process.env.GBRAIN_NATIVE = '1';
  seedRandom(20260917);
  const brain = quiet(() => new DigitalBrain());
  const region = brain.getRegion('nativeCortex') as NativeCortex | undefined;
  check('with GBRAIN_NATIVE=1 the brain has the native cortex', region !== undefined && region.population === 5000, region ? `${region.population} neurons, ${region.synapses} synapses` : 'missing');
  const SIDE = 64;
  const pixels = new Array<number>(SIDE * SIDE).fill(27);
  for (let i = 8; i < 56; i++) for (let d = -1; d <= 1; d++) { pixels[(32 + d) * SIDE + i] = 255; pixels[i * SIDE + 32 + d] = 255; }
  const firedWhileSeeing = (): number => {
    let total = 0;
    for (let t = 0; t < 40; t++) { brain.tick(); total += region?.fired().length ?? 0; }
    return total;
  };
  const quietBefore = firedWhileSeeing();
  quiet(() => brain.see(pixels, SIDE, SIDE, { propagate: false }));
  const seeing = firedWhileSeeing();
  for (let t = 0; t < 80; t++) brain.tick();
  const quietAfter = firedWhileSeeing();
  check('it fires when something is seen and is quiet otherwise', seeing > 20 && seeing > quietBefore * 3 && quietAfter < seeing / 2,
    `${quietBefore} spikes at rest, ${seeing} while seeing the cross, ${quietAfter} after`);
  const state = brain.getState();
  check('it shows in the state as a region', state.regions.nativeCortex !== undefined && Object.keys(state.regions).length === 13, `${Object.keys(state.regions).length} regions`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
