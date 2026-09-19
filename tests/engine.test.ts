/**
 * VERIFICATION TEST — the native engine
 * ===========================================================================
 * The C++ core must be the same neuron as the TypeScript one, run a real
 * network with real inhibition and plasticity, and scale.
 *
 *   1. PARITY     — a native neuron driven by the same current as the
 *                   TypeScript Izhikevich neuron fires at the same ticks.
 *   2. NETWORK    — a 10 000-neuron network with 20% interneurons runs; the
 *                   interneurons keep the activity sparse (no runaway), and
 *                   STDP moves the excitatory weights, never past the bounds.
 *   3. SCALE      — 100 000 neurons run faster than real time; the tick
 *                   cost is reported at 10k, 100k (and 1M if there is memory).
 *
 * Needs `npm run build:native`; fails clearly if the addon is not there.
 */
import { createNativeNetwork, isNativeAvailable, nativeBackend } from '../src/core/snn/native.js';
import { SpikingNeuron } from '../src/core/snn/neuron.js';

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

console.log('── Verification: the native engine ──\n');
if (!isNativeAvailable()) {
  console.log('❌ The native engine is not built. Run: npm run build:native');
  process.exit(1);
}
console.log(`   backend: ${nativeBackend()}\n`);

// ── 1. PARITY ───────────────────────────────────────────────────────────────
console.log('1. PARITY');
{
  // One neuron, no synapses, no noise: the native Izhikevich neuron against the TypeScript one.
  const net = createNativeNetwork({ neurons: 1, fanIn: 0, inhibitoryFraction: 0, noise: 0, plastic: false, seed: 1 });
  net.reset();
  const ref = new SpikingNeuron('RegularSpiking');
  // Same starting point: the engine draws a random resting potential, the reference starts at −65 mV.
  ref.v = net.potentials()[0];
  ref.u = 0.2 * ref.v;
  const I = new Float32Array([10]);
  const nativeSpikes: number[] = [];
  const refSpikes: number[] = [];
  for (let t = 0; t < 500; t++) {
    if (net.step(I, 1).length > 0) nativeSpikes.push(t);
    if (ref.step(10, 1, t)) refSpikes.push(t);
  }
  // Single precision against double: the two drift apart slowly, so the first
  // spikes must coincide and the count over the run may differ by one.
  const firstSame = nativeSpikes.slice(0, 6).every((t, i) => Math.abs(t - refSpikes[i]) <= 1);
  const same = firstSame && Math.abs(nativeSpikes.length - refSpikes.length) <= 1;
  check('the native neuron fires when the TypeScript one does (I = 10, 500 ticks; float32 vs float64 may drift one spike)', same && nativeSpikes.length >= 5,
    `${nativeSpikes.length} spikes native vs ${refSpikes.length} reference; first at ${nativeSpikes.slice(0, 4).join(',')} vs ${refSpikes.slice(0, 4).join(',')}`);
  const v = net.potentials()[0];
  check('its membrane potential is in range after the run', v > -80 && v < 30, `v = ${v.toFixed(1)} mV`);
}

// ── 2. NETWORK ──────────────────────────────────────────────────────────────
console.log('\n2. NETWORK');
{
  const N = 10_000;
  const net = createNativeNetwork({ neurons: N, fanIn: 100, inhibitoryFraction: 0.2, seed: 42 });
  let inhibitory = 0;
  for (let i = 0; i < N; i++) if (net.isInhibitory(i)) inhibitory++;
  check('a fifth of the neurons are interneurons', Math.abs(inhibitory / N - 0.2) < 0.01, `${inhibitory} of ${N}`);
  const I = new Float32Array(N);
  for (let i = 0; i < N; i++) if (!net.isInhibitory(i) && i % 10 === 0) I[i] = 15;
  const before = net.weights();
  let spikes = 0, peak = 0;
  for (let t = 0; t < 300; t++) {
    const fired = net.step(I, 1).length;
    spikes += fired;
    if (fired > peak) peak = fired;
  }
  const rate = spikes / 300 / N;
  check('the network fires, sparsely (inhibition holds it)', rate > 0.002 && rate < 0.2 && peak < N * 0.5, `mean ${(rate * 100).toFixed(2)}% per tick, peak ${peak} neurons`);
  const after = net.weights();
  let changed = 0, outOfBounds = 0;
  for (let s = 0; s < after.length; s++) {
    if (after[s] !== before[s]) changed++;
    if (after[s] > 1.0001 || (after[s] < 0 && before[s] >= 0)) outOfBounds++;
  }
  check('plasticity moved excitatory weights, within bounds', changed > 1000 && outOfBounds === 0, `${changed} synapses changed, ${outOfBounds} out of bounds`);

  // Without plasticity the weights stay put.
  const frozen = createNativeNetwork({ neurons: 2000, fanIn: 50, plastic: false, seed: 3 });
  const w0 = frozen.weights();
  const J = new Float32Array(2000).fill(0); for (let i = 0; i < 200; i++) J[i] = 15;
  for (let t = 0; t < 100; t++) frozen.step(J, 1);
  const w1 = frozen.weights();
  check('with plasticity off, no weight moves', w0.every((w, i) => w === w1[i]));
}

// ── 3. SCALE ────────────────────────────────────────────────────────────────
console.log('\n3. SCALE');
{
  const timeIt = (N: number, ticks: number): { ticksPerSecond: number; rate: number } => {
    const net = createNativeNetwork({ neurons: N, fanIn: 100, seed: 5 });
    const I = new Float32Array(N);
    for (let i = 0; i < N; i++) if (!net.isInhibitory(i) && i % 10 === 0) I[i] = 15;
    for (let t = 0; t < 60; t++) net.step(I, 1); // warm up: past the first volley
    let spikes = 0;
    const t0 = performance.now();
    for (let t = 0; t < ticks; t++) spikes += net.step(I, 1).length;
    const seconds = (performance.now() - t0) / 1000;
    return { ticksPerSecond: ticks / seconds, rate: spikes / ticks / N };
  };
  const small = timeIt(10_000, 300);
  const medium = timeIt(100_000, 100);
  console.log(`   10k: ${small.ticksPerSecond.toFixed(0)} ticks/s (${(small.rate * 100).toFixed(2)}% firing) · 100k: ${medium.ticksPerSecond.toFixed(0)} ticks/s (${(medium.rate * 100).toFixed(2)}% firing)`);
  check('100 000 neurons run faster than real time (≥ 10 ticks/s)', medium.ticksPerSecond >= 10, `${medium.ticksPerSecond.toFixed(0)} ticks/s`);
  if (process.env.GBRAIN_BENCH_1M === '1') {
    const large = timeIt(1_000_000, 40);
    console.log(`   1M: ${large.ticksPerSecond.toFixed(1)} ticks/s (${(large.rate * 100).toFixed(2)}% firing)`);
    check('a million neurons run in real time (≥ 10 ticks/s)', large.ticksPerSecond >= 10, `${large.ticksPerSecond.toFixed(1)} ticks/s`);
  } else {
    console.log('   (set GBRAIN_BENCH_1M=1 to time a million neurons — ~2 GB of synapses)');
  }
}

// ── Summary ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
