/**
 * NATIVE CORTEX — a region whose neurons live in the engine
 * ===========================================================================
 * The first region of the brain that runs on the native substrate: a sheet
 * of spiking neurons with sparse recurrent synapses and real inhibitory
 * interneurons, fed by an afferent projection from its input channels.
 * Nothing in it is algorithmic competition or a template: an input drives
 * the neurons whose afferents happen to sample it, the interneurons keep
 * the response sparse, and recurrent STDP stamps in the neurons that fire
 * together — a cell assembly (Hebb 1949) that completes from a partial cue
 * once it has been stamped in enough.
 *
 * It has the contract of any BrainRegion: it takes the afferent spikes the
 * bus (or the thalamus) delivers, and answers with the neurons that fired.
 * Its synapses persist through the region's extra state.
 */
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { packArray, unpackFloat32 } from '../../core/persistence/binary-protocol.js';
import { createNativeNetwork, isNativeAvailable, type NativeNetwork } from '../../core/snn/native.js';
import { mulberry32 } from '../../core/random.js';

export interface NativeCortexConfig {
  neurons: number;
  inputCount: number;
  /** Recurrent synapses per neuron. */
  fanIn: number;
  /** Input channels each neuron listens to. */
  inputFanIn: number;
  /** Current per unit of input, through the afferent weights. */
  inputGain: number;
  inhibitoryFraction: number;
  /** Interneurons get this fraction of the afferents an excitatory neuron gets: they are driven mostly by the sheet itself (feedback inhibition). */
  inhibitoryAfferents: number;
  /** Recurrent weights: excitatory range, the inhibitory weight, and the gain on excitatory synapses onto interneurons (dense, strong E→I coupling). */
  excMax: number;
  inhWeight: number;
  excToInhGain: number;
  seed: number;
  /** Whether recurrent synapses learn (STDP). */
  plastic: boolean;
  /**
   * STDP amplitudes. Potentiation dominates: neurons that fire together while
   * an input holds them up wire together (a Hebbian assembly); with LTD
   * dominant, as in the engine's default, uncorrelated timing within the
   * assembly nets out to depression and nothing is stamped in.
   */
  aPlus: number;
  aMinus: number;
  wMax: number;
  /** Structural plasticity: synapses grow between neurons that fire together, the weakest go (what completes an assembly from a part of it). */
  structural: boolean;
  rewireEvery: number;
  /** Spikes in a window that make a neuron part of the coactive core. */
  coactiveSpikes: number;
  rewiresPerEvent: number;
  pruneBelow: number;
  /** A new synapse between coactive neurons is born strong: it is there to carry the assembly. */
  newWeight: number;
  /** Inhibitory plasticity: inhibition learns to balance each neuron's excitation (Vogels 2011), so recurrent drive can be strong without runaway. */
  inhibitoryPlasticity: boolean;
  targetRate: number;
}

const DEFAULT_CONFIG: NativeCortexConfig = {
  neurons: 10_000,
  inputCount: 1000,
  fanIn: 100,
  inputFanIn: 50,
  inputGain: 2.0,
  inhibitoryFraction: 0.2,
  inhibitoryAfferents: 0.3,
  excMax: 0.2,
  inhWeight: -2.0,
  excToInhGain: 8.0,
  seed: 0x9a71,
  plastic: true,
  aPlus: 0.015,
  aMinus: 0.010,
  wMax: 2.0,
  structural: true,
  rewireEvery: 60,
  coactiveSpikes: 3,
  rewiresPerEvent: 4,
  pruneBelow: 0.15,
  newWeight: 1.0,
  inhibitoryPlasticity: true,
  targetRate: 0.05,
};

export class NativeCortex extends BrainRegion {
  private readonly cfg: NativeCortexConfig;
  private readonly net: NativeNetwork;
  private readonly channels: Float32Array;
  private lastFired: Uint32Array = new Uint32Array(0);
  /** Ticks stepped (for monitoring). */
  ticks = 0;

  /** Whether a NativeCortex can be built (the engine addon is there). */
  static available(): boolean {
    return isNativeAvailable();
  }

  constructor(config: Partial<NativeCortexConfig> = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    // The base class allocates neurons × inputs weights for the afferents; the
    // engine holds the real ones, so the base matrix is kept minimal (1 × inputs).
    super('nativeCortex', 'Corteza Nativa (motor C++)', 1, cfg.inputCount);
    this.cfg = cfg;
    this.net = createNativeNetwork({
      neurons: cfg.neurons,
      fanIn: cfg.fanIn,
      inhibitoryFraction: cfg.inhibitoryFraction,
      seed: cfg.seed,
      plastic: cfg.plastic,
      aPlus: cfg.aPlus,
      aMinus: cfg.aMinus,
      wMax: cfg.wMax,
      excMax: cfg.excMax,
      inhWeight: cfg.inhWeight,
      excToInhGain: cfg.excToInhGain,
      structural: cfg.structural,
      rewireEvery: cfg.rewireEvery,
      coactiveSpikes: cfg.coactiveSpikes,
      rewiresPerEvent: cfg.rewiresPerEvent,
      pruneBelow: cfg.pruneBelow,
      newWeight: cfg.newWeight,
      inhibitoryPlasticity: cfg.inhibitoryPlasticity,
      targetRate: cfg.targetRate,
    });
    // The afferent projection, from the region's own random source: every
    // excitatory neuron samples `inputFanIn` channels, an interneuron fewer.
    const random = mulberry32(cfg.seed ^ 0xaff);
    const rowPtr = new Uint32Array(cfg.neurons + 1);
    const cols: number[] = [];
    const weights: number[] = [];
    for (let i = 0; i < cfg.neurons; i++) {
      rowPtr[i] = cols.length;
      const k = this.net.isInhibitory(i) ? Math.round(cfg.inputFanIn * cfg.inhibitoryAfferents) : cfg.inputFanIn;
      for (let s = 0; s < k; s++) {
        cols.push(Math.floor(random() * cfg.inputCount));
        weights.push(0.2 + 0.8 * random());
      }
    }
    rowPtr[cfg.neurons] = cols.length;
    this.net.setInputProjection(cfg.inputCount, rowPtr, Uint32Array.from(cols), Float32Array.from(weights));
    this.channels = new Float32Array(cfg.inputCount);
    // The region's own arrays follow the engine's population size.
    this.spikes = new Float32Array(cfg.neurons);
    this.potentials = new Float32Array(cfg.neurons);
  }

  /** The base class draws random afferents at construction; the engine's are the real ones. */
  protected override initializeWeights(): void {}

  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    const output = new Float32Array(this.cfg.neurons);
    let any = false;
    for (let i = 0; i < this.channels.length; i++) {
      const v = i < spikes.length ? spikes[i] : 0;
      this.channels[i] = v * this.cfg.inputGain;
      if (v > 0) any = true;
    }
    // Plasticity follows the neuromodulators' say, as everywhere else.
    const modulation = this.cfg.plastic ? (modulationEffects.learningRateMultiplier ?? 1) : 0;
    this.lastFired = any || this.ticks > 0 ? this.net.stepChannels(this.channels, modulation) : new Uint32Array(0);
    this.ticks++;
    for (let k = 0; k < this.lastFired.length; k++) output[this.lastFired[k]] = 1;
    return output;
  }

  /** Neurons that fired on the last tick. */
  fired(): Uint32Array {
    return this.lastFired;
  }

  /** Whether a neuron is an interneuron (the representation lives in the excitatory ones). */
  isInhibitory(neuron: number): boolean {
    return this.net.isInhibitory(neuron);
  }

  /** The engine's population, not the base class's placeholder row. */
  override get neurons(): number {
    return this.cfg.neurons;
  }

  get population(): number {
    return this.cfg.neurons;
  }

  get synapses(): number {
    return this.net.synapses;
  }

  /** Synapses rewired so far by structural plasticity. */
  get rewired(): number {
    return this.net.rewired;
  }

  /** Recurrent weights, for inspection. */
  recurrentWeights(): Float32Array {
    return this.net.weights();
  }

  /** Recurrent connectivity (CSR by presynaptic neuron), for inspection. */
  recurrentSynapses(): { rowPtr: Uint32Array; targets: Uint32Array } {
    return { rowPtr: this.net.synapseRowPtr(), targets: this.net.synapseTargets() };
  }

  override serializeExtra(): unknown {
    return { recurrent: packArray(this.net.weights()) };
  }

  override deserializeExtra(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const w = unpackFloat32((data as Record<string, unknown>).recurrent, this.net.synapses);
    if (w) this.net.setWeights(w);
  }
}
