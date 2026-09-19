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

export interface NativeNetworkOptions {
  neurons: number;
  fanIn?: number;
  inhibitoryFraction?: number;
  excMin?: number;
  excMax?: number;
  inhWeight?: number;
  excToInhGain?: number;
  dt?: number;
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
