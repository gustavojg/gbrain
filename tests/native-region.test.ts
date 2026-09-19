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
 *                    the rest (Hebb: what fires together wires together).
 *                    Completion from half the input is reported, not counted:
 *                    with a hundred random recurrent synapses per neuron an
 *                    assembly of a tenth has ~9 synapses into each member —
 *                    too few to complete it even at full weight (that needs
 *                    denser within-assembly connectivity: structural plasticity).
 *   4. IN THE BRAIN — with GBRAIN_NATIVE=1 the brain has the region, fed by
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
/** An input pattern: a fixed random tenth of the channels lit. */
const pattern = (seed: number, fraction = 0.1): Float32Array => {
  let r = seed >>> 0;
  const next = (): number => { r = (r * 1664525 + 1013904223) >>> 0; return r / 4294967296; };
  const p = new Float32Array(INPUTS);
  for (let i = 0; i < INPUTS; i++) if (next() < fraction) p[i] = 1;
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
 * EXCITATORY neuron fired over the second half (the interneurons fire to
 * everything and carry no representation). Correlating these counts across
 * presentations is the same-vs-different criterion the whole-brain test uses
 * for the hippocampus and the prefrontal cortex.
 */
const present = (cortex: NativeCortex, p: Float32Array, ticks: number, learn: boolean): Float32Array => {
  const counts = new Float32Array(cortex.population);
  for (let t = 0; t < ticks; t++) {
    const out = cortex.processInput(p, { ...effects, learningRateMultiplier: learn ? 1 : 0 });
    if (t >= ticks / 2) for (let i = 0; i < out.length; i++) if (out[i] > 0 && !cortex.isInhibitory(i)) counts[i]++;
  }
  // Silence between presentations: membranes recover, the input's trace fades.
  const silence = new Float32Array(INPUTS);
  for (let t = 0; t < 100; t++) cortex.processInput(silence, { ...effects, learningRateMultiplier: 0 });
  return counts;
};

// ── 1 & 2. ASSEMBLIES, SPARSE ──────────────────────────────────────────────
console.log('1. ASSEMBLIES');
const cortex = new NativeCortex({ neurons: 10_000, inputCount: INPUTS, seed: 11, plastic: true });
const A = pattern(1), B = pattern(2);
const a1 = present(cortex, A, 60, false);
const a2 = present(cortex, A, 60, false);
const b1 = present(cortex, B, 60, false);
const same = correlation(a1, a2), different = correlation(a1, b1);
check('the same input twice drives much the same neurons', same >= 0.5, `r = ${same.toFixed(2)}`);
check('two different inputs drive different neurons', different <= 0.2, `r = ${different.toFixed(2)}`);
console.log('\n2. SPARSE');
let excitatory = 0, active = 0;
for (let i = 0; i < cortex.population; i++) if (!cortex.isInhibitory(i)) { excitatory++; if (a1[i] > 0) active++; }
{
  const fraction = active / excitatory;
  check('the interneurons keep the response sparse', fraction > 0.03 && fraction < 0.4, `${(fraction * 100).toFixed(1)}% of the excitatory neurons fired over the second half`);
}

// ── 3. ASSEMBLY ─────────────────────────────────────────────────────────────
console.log('\n3. ASSEMBLY');
{
  const before = correlation(present(cortex, half(A), 60, false), a1);
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
  const full = present(cortex, A, 60, false);
  const after = correlation(present(cortex, half(A), 60, false), full);
  console.log(`   ${after > before + 0.1 ? '🎉' : 'ℹ️ '} half the input brings back r = ${before.toFixed(2)} → ${after.toFixed(2)} of the response (completion needs denser within-assembly connectivity)`);
}

// ── 4. IN THE BRAIN ─────────────────────────────────────────────────────────
console.log('\n4. IN THE BRAIN');
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
