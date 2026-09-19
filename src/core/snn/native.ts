/**
 * NATIVE ENGINE — the C++ spiking core, from TypeScript
 * ===========================================================================
 * `native/` holds the engine written to scale to a million neurons
 * (Izhikevich neurons in arrays, sparse synapses, real inhibitory
 * interneurons, trace STDP, threads; CUDA kernels for a GPU). This module
 * loads its Node addon if it has been built (`npm run build:native`) and
 * says so if it has not, so that the brain can run on either.
 */
import { createRequire } from 'module';

/**
 * A block of connectivity: every neuron in [srcFrom, srcTo) sends `fanOut`
 * synapses into [dstFrom, dstTo), drawn around its relative position with a
 * Gaussian of width `sigma` (a fraction of the destination range; 0 =
 * anywhere). An area's local and long-range recurrents, a projection from one
 * area to the next, a feedback path: a network is populations plus blocks.
 */
export interface SynapseBlock {
  srcFrom: number;
  srcTo: number;
  dstFrom: number;
  dstTo: number;
  fanOut: number;
  wMin?: number;
  wMax?: number;
  sigma?: number;
  /** Gain on this block's synapses onto interneurons (feedforward inhibition); omitted = the network's `excToInhGain`. */
  inhGain?: number;
}

export interface NativeNetworkOptions {
  neurons: number;
  /** Random recurrent fan-in, used when no `blocks` are given. */
  fanIn?: number;
  /** Structured connectivity: the synapses are built from these blocks instead of at random. */
  blocks?: SynapseBlock[];
  inhibitoryFraction?: number;
  excMin?: number;
  excMax?: number;
  inhWeight?: number;
  excToInhGain?: number;
  /** Inhibitory plasticity (Vogels 2011): interneuron→excitatory synapses track each neuron's firing toward a target rate. */
  inhibitoryPlasticity?: boolean;
  iEta?: number;
  targetRate?: number;
  inhMax?: number;
  /** Structural plasticity: synapses grown between coactive neurons, the weakest pruned. */
  structural?: boolean;
  rewireEvery?: number;
  coactiveSpikes?: number;
  rewiresPerEvent?: number;
  pruneBelow?: number;
  newWeight?: number;
  dt?: number;
  /** Integration steps of `dt` per tick under the same input (the tick reports every neuron that fired in any of them). */
  substeps?: number;
  /** Synaptic current time constants (ms): excitatory (AMPA ≈ 5) and inhibitory (GABA_A ≈ 10); 0 = one-tick pulses. */
  tauSynExc?: number;
  tauSynInh?: number;
  /** A slow (NMDA-like) share of every excitatory synapse's weight, with its own time constant (ms); lets sparse sustained input sum. */
  nmdaShare?: number;
  tauSynNmda?: number;
  /** Synaptic normalization: every `normalizeEvery` ticks the excitatory synapses into each neuron are scaled back to their built sum (competitive potentiation). */
  normalize?: boolean;
  normalizeEvery?: number;
  /** The budget as a multiple of the built sum; potentiation is free below it. */
  normalizeGain?: number;
  /** Short-term depression on excitatory synapses (Tsodyks & Markram 1997): a spike spends `stdU` of the presynaptic resources, which recover with `stdTauRec` ms. */
  shortTermDepression?: boolean;
  stdU?: number;
  stdTauRec?: number;
  noise?: number;
  plastic?: boolean;
  aPlus?: number;
  aMinus?: number;
  tauPlus?: number;
  tauMinus?: number;
  wMax?: number;
  threads?: number;
  seed?: number;
}

/** The addon's network object (see native/addon.cc). */
export interface NativeNetwork {
  readonly neurons: number;
  readonly synapses: number;
  /** Synapses rewired so far by structural plasticity. */
  readonly rewired: number;
  /** One tick: external currents per neuron (or null) and the neuromodulatory gain on plasticity; returns the neurons that fired. */
  step(externalCurrent: Float32Array | null, modulation?: number): Uint32Array;
  /** One tick driven by input channels through the afferent projection. */
  stepChannels(channels: Float32Array, modulation?: number): Uint32Array;
  /** The afferent projection, CSR by neuron over input channels. */
  setInputProjection(channels: number, rowPtr: Uint32Array, cols: Uint32Array, weights: Float32Array): void;
  /** A random afferent projection: every neuron listens to `fanIn` channels. */
  buildRandomInputProjection(channels: number, fanIn: number, wMin?: number, wMax?: number): void;
  potentials(): Float32Array;
  weights(): Float32Array;
  /** The recurrent connectivity: CSR row pointers by presynaptic neuron, and the targets. */
  synapseRowPtr(): Uint32Array;
  synapseTargets(): Uint32Array;
  setWeights(weights: Float32Array): void;
  setSynapses(rowPtr: Uint32Array, targets: Uint32Array, weights: Float32Array): void;
  isInhibitory(neuron: number): boolean;
  reset(): void;
}

interface NativeModule {
  NativeNetwork: new (options: NativeNetworkOptions) => NativeNetwork;
  backend: string;
}

let loaded: NativeModule | null | undefined;

function load(): NativeModule | null {
  if (loaded !== undefined) return loaded;
  const require = createRequire(import.meta.url);
  // Compiled to dist/core/snn (the app) or dist-test/src/core/snn (the tests): the addon is at the repo root either way.
  for (const candidate of ['../../../build/Release/gbrain_native.node', '../../../../build/Release/gbrain_native.node']) {
    try {
      loaded = require(candidate) as NativeModule;
      return loaded;
    } catch {
      // try the next
    }
  }
  loaded = null;
  return loaded;
}

/** Whether the native engine has been built and can be loaded. */
export function isNativeAvailable(): boolean {
  return load() !== null;
}

/** The backend the native engine was built for ('cpu' or 'cuda'), or null if not built. */
export function nativeBackend(): string | null {
  return load()?.backend ?? null;
}

/** Creates a native network, or throws with the build instruction if the engine is not built. */
export function createNativeNetwork(options: NativeNetworkOptions): NativeNetwork {
  const mod = load();
  if (!mod) throw new Error('The native engine is not built: run `npm run build:native` (needs a C++17 compiler and Python for node-gyp).');
  return new mod.NativeNetwork(options);
}
