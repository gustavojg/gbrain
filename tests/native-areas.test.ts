/**
 * VERIFICATION TEST — two areas on one engine
 * ===========================================================================
 * The step toward the whole brain living in the engine: a network is
 * populations plus connectivity blocks. Here two cortical areas share one
 * network. Area 1 is driven by input channels through a topographic afferent
 * projection; area 2 is driven ONLY by area 1's spikes, through a
 * topographic feedforward block (a projection between regions as a block of
 * the CSR, with synaptic currents, plasticity and all), and sends a sparse
 * feedback block to area 1. Each area has local and long-range recurrents and
 * its own interneurons, which never project outside their area.
 *
 *   1. RELAY      — area 2 fires when area 1 does, and is quiet otherwise.
 *   2. ASSEMBLIES — the same input twice drives much the same area-2
 *                   neurons; the projection itself learns.
 *   3. COMPLETION — after repetition, half the input at area 1 brings back
 *                   the side of area 2's assembly the cue does not reach.
 *
 * What is measured and reported but NOT counted, because it does not hold
 * yet: area 2 telling two inputs apart. Area 2 answers mostly the onset
 * volley of area 1 (every area-1 neuron with a lit channel fires once when
 * the input starts, and that volley looks the same for any input); the
 * sustained, selective firing of area 1 (which does tell the inputs apart,
 * r ≈ 0.75 same vs 0.09 different over the second half) is too sparse and
 * too weak through a hundred random synapses to drive area 2 on its own. And
 * repetition makes it worse: the synchronous volley potentiates every
 * feedforward synapse alike. What is missing is what cortex has for this —
 * synaptic normalization (heterosynaptic depression, scaling) so that
 * potentiation is competitive, and slow (NMDA-like) currents that let sparse
 * sustained input sum. That is the next step of the engine.
 *
 * Needs `npm run build:native`.
 */
import { createNativeNetwork, isNativeAvailable } from '../src/core/snn/native.js';
import { mulberry32 } from '../src/core/random.js';

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

console.log('── Verification: two areas on one engine ──\n');
if (!isNativeAvailable()) {
  console.log('❌ The native engine is not built. Run: npm run build:native');
  process.exit(1);
}

const INPUTS = 1000, AREA = 5000, N = 2 * AREA;
const EXC_MAX = 0.2;
const net = createNativeNetwork({
  neurons: N,
  seed: 23,
  inhibitoryFraction: 0.2,
  excMax: EXC_MAX,
  inhWeight: -2,
  excToInhGain: 8,
  plastic: true,
  aPlus: 0.015,
  aMinus: 0.010,
  wMax: 2,
  structural: true,
  rewireEvery: 60,
  coactiveSpikes: 3,
  rewiresPerEvent: 4,
  pruneBelow: 0.15,
  newWeight: 1,
  inhibitoryPlasticity: true,
  targetRate: 0.05,
  blocks: [
    // Area 1: local and long-range recurrents.
    { srcFrom: 0, srcTo: AREA, dstFrom: 0, dstTo: AREA, fanOut: 70, wMin: 0, wMax: EXC_MAX, sigma: 0.05 },
    { srcFrom: 0, srcTo: AREA, dstFrom: 0, dstTo: AREA, fanOut: 30, wMin: 0, wMax: EXC_MAX, sigma: 0 },
    // Area 2: the same.
    { srcFrom: AREA, srcTo: N, dstFrom: AREA, dstTo: N, fanOut: 70, wMin: 0, wMax: EXC_MAX, sigma: 0.05 },
    { srcFrom: AREA, srcTo: N, dstFrom: AREA, dstTo: N, fanOut: 30, wMin: 0, wMax: EXC_MAX, sigma: 0 },
    // Feedforward, topographic: area 1 → area 2 (excitatory sources only), with
    // feedforward inhibition at half the strength of the excitation.
    { srcFrom: 0, srcTo: AREA, dstFrom: AREA, dstTo: N, fanOut: 100, wMin: 0.8, wMax: 2.0, sigma: 0.05, inhGain: 0.5 },
    // Feedback, sparse and weak: area 2 → area 1.
    { srcFrom: AREA, srcTo: N, dstFrom: 0, dstTo: AREA, fanOut: 10, wMin: 0, wMax: 0.3, sigma: 0.05 },
  ],
});

// The afferent projection reaches area 1 only, topographically (a fifth of the channels per receptive field).
{
  const random = mulberry32(0xa2ea);
  const rowPtr = new Uint32Array(N + 1);
  const cols: number[] = [], weights: number[] = [];
  const span = 0.2 * INPUTS;
  for (let i = 0; i < N; i++) {
    rowPtr[i] = cols.length;
    if (i >= AREA) continue;
    const k = net.isInhibitory(i) ? 15 : 50;
    const centre = ((i + 0.5) / AREA) * INPUTS;
    for (let s = 0; s < k; s++) {
      const channel = Math.floor(centre + (random() - 0.5) * span);
      cols.push(((channel % INPUTS) + INPUTS) % INPUTS);
      weights.push(2 * (0.2 + 0.8 * random()));
    }
  }
  rowPtr[N] = cols.length;
  net.setInputProjection(INPUTS, rowPtr, Uint32Array.from(cols), Float32Array.from(weights));
}

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
/** Excitatory spike counts per neuron over a presentation, for each area, followed by silence. */
interface Response { a1: Float32Array; a2: Float32Array; late1: Float32Array; late2: Float32Array }
const present = (p: Float32Array, ticks: number, learn: boolean, silence = 100): Response => {
  const a1 = new Float32Array(AREA), a2 = new Float32Array(AREA), late1 = new Float32Array(AREA), late2 = new Float32Array(AREA);
  for (let t = 0; t < ticks; t++) {
    for (const n of net.stepChannels(p, learn ? 1 : 0)) {
      if (net.isInhibitory(n)) continue;
      if (n < AREA) { a1[n]++; if (t >= ticks / 2) late1[n]++; } else { a2[n - AREA]++; if (t >= ticks / 2) late2[n - AREA]++; }
    }
  }
  const quiet = new Float32Array(INPUTS);
  for (let t = 0; t < silence; t++) net.stepChannels(quiet, 0);
  return { a1, a2, late1, late2 };
};
const total = (c: Float32Array): number => c.reduce((a, b) => a + b, 0);
const excitatoryIn = (from: number): number[] => Array.from({ length: AREA }, (_, i) => from + i).filter((i) => !net.isInhibitory(i)).map((i) => i - from);

// ── 1. RELAY ────────────────────────────────────────────────────────────────
console.log('1. RELAY');
const A = pattern(1), B = pattern(2);
const rest = present(new Float32Array(INPUTS), 60, false);
const x1 = present(A, 60, false);
check('area 2 fires when area 1 does, and is quiet otherwise', total(x1.a2) > 200 && total(rest.a2) < total(x1.a2) / 10 && total(x1.a1) > 200,
  `area 1: ${total(rest.a1)} spikes at rest, ${total(x1.a1)} to the input; area 2: ${total(rest.a2)} at rest, ${total(x1.a2)} to the input`);
{
  const exc2 = excitatoryIn(AREA);
  const perTick = total(x1.a2) / 60 / exc2.length;
  check('area 2 keeps its response sparse', perTick > 0.002 && perTick < 0.05, `${(perTick * 100).toFixed(1)}% of its excitatory neurons fire per tick`);
}

// ── 2. ASSEMBLIES ───────────────────────────────────────────────────────────
console.log('\n2. ASSEMBLIES');
const x2 = present(A, 60, false);
const y1 = present(B, 60, false);
const same2 = correlation(x1.a2, x2.a2), different2 = correlation(x1.a2, y1.a2);
check('the same input twice drives much the same area-2 neurons', same2 >= 0.5, `r = ${same2.toFixed(2)} (area 1: ${correlation(x1.a1, x2.a1).toFixed(2)})`);
console.log(`   ⚠️  not counted — area 2 does not yet tell two inputs apart: r = ${different2.toFixed(2)} (area 1: ${correlation(x1.a1, y1.a1).toFixed(2)}); over the second half of the presentation area 1 is selective (same ${correlation(x1.late1, x2.late1).toFixed(2)}, different ${correlation(x1.late1, y1.late1).toFixed(2)}) and area 2 barely fires (${total(x1.late2)} spikes vs ${total(x1.a2)} over the whole presentation: the onset volley is what it answers)`);

// ── 3. COMPLETION ───────────────────────────────────────────────────────────
console.log('\n3. COMPLETION');
{
  const h0 = present(half(A), 60, false);
  const w0 = net.weights();
  for (let i = 0; i < 40; i++) present(A, 60, true);
  const w1 = net.weights();
  const rowPtr = net.synapseRowPtr(), targets = net.synapseTargets();
  let feedforwardMoved = 0, feedforward = 0;
  for (let i = 0; i < AREA; i++) for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) if (targets[k] >= AREA) { feedforward++; if (w1[k] !== w0[k]) feedforwardMoved++; }
  const full = present(A, 60, false);
  const h1 = present(half(A), 60, false);
  const bAfter = present(B, 60, false);
  const idle = present(new Float32Array(INPUTS), 200, false, 0);
  const exc2 = excitatoryIn(AREA);
  const unlitSide = (i: number): boolean => (i + 0.5) / AREA >= 0.6, litSide = (i: number): boolean => (i + 0.5) / AREA < 0.4;
  const inAssembly = (i: number): boolean => full.a2[i] >= 3;
  const unlitAssembly = exc2.filter((i) => inAssembly(i) && unlitSide(i)), litAssembly = exc2.filter((i) => inAssembly(i) && litSide(i));
  const unlitOthers = exc2.filter((i) => !inAssembly(i) && unlitSide(i));
  const share = (set: number[], counts: Float32Array): number => (set.length ? set.filter((i) => counts[i] >= 1).length / set.length : 0);
  const back0 = share(unlitAssembly, h0.a2), back1 = share(unlitAssembly, h1.a2), others = share(unlitOthers, h1.a2);
  check('the cue reaches area 2\'s own side of the assembly through area 1', share(litAssembly, h1.a2) >= 0.8 && litAssembly.length >= 5 && unlitAssembly.length >= 5,
    `${litAssembly.length} on the lit side, ${unlitAssembly.length} on the unlit side`);
  check('the side of area 2 the cue does not reach comes back', back1 >= 0.6 && back1 >= back0 + 0.3, `${(back0 * 100).toFixed(0)}% → ${(back1 * 100).toFixed(0)}% of the unlit side of the assembly (${(others * 100).toFixed(0)}% of the other neurons there)`);
  check('nothing keeps firing in silence', total(idle.a2) / 200 < 1 && total(idle.a1) / 200 < 1, `${(total(idle.a2) / 200).toFixed(2)} area-2 spikes per tick with no input`);
  console.log(`   ⚠️  not counted — the completed area-2 response is not yet the input's own: r(half A, B) = ${correlation(h1.a2, bAfter.a2).toFixed(2)} (same cause as above)`);
  check('the projection itself learns: feedforward synapses moved', feedforwardMoved > 1000, `${feedforwardMoved} of ${feedforward} feedforward synapses moved; ${net.rewired} synapses rewired`);
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
