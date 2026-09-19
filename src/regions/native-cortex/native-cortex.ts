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
import { packArray, unpackFloat32, unpackInt32 } from '../../core/persistence/binary-protocol.js';
import { createNativeNetwork, isNativeAvailable, type NativeNetwork } from '../../core/snn/native.js';
import { mulberry32 } from '../../core/random.js';

export interface NativeCortexConfig {
  neurons: number;
  inputCount: number;
  /** Recurrent synapses per neuron. */
  fanIn: number;
  /**
   * Structured recurrents: this share of a neuron's synapses stay local
   * (Gaussian of width `localSigma`, a fraction of the sheet, around its
   * place: the dense horizontal connections within a cortical neighbourhood);
   * the rest reach anywhere (the sparse long-range horizontal connections
   * that bind distant parts of a pattern). 0 = all random, as before.
   */
  localShare: number;
  localSigma: number;
  /** Input channels each neuron listens to. */
  inputFanIn: number;
  /**
   * Topography: the width of a neuron's receptive field as a fraction of the
   * input channels (1 = it samples anywhere). Afferents in cortex are
   * topographic (retinotopy, tonotopy): neighbours listen to neighbouring
   * inputs, so a part of the input drives a part of the sheet fully — and
   * completing the rest is the recurrent synapses' work.
   */
  topography: number;
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
  /** Integration steps per brain tick (a tick is 100 real ms; one step is 1 simulated ms). */
  substeps: number;
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
  iEta: number;
  targetRate: number;
  inhMax: number;
  /** Synaptic current time constants (ms): what lets spikes sum in time. */
  tauSynExc: number;
  tauSynInh: number;
  /**
   * The slow (NMDA-like) share of excitatory synapses and its time constant.
   * A small share: at 0.3 the slower excitation reaches the interneurons too
   * and completion fails in 2 of 12 seed/pattern combinations; at 0.15 all 12
   * complete (60–100% of the unlit side).
   */
  nmdaShare: number;
  tauSynNmda: number;
  /**
   * Synaptic normalization: potentiation competes for a budget per neuron.
   * Off here: an assembly's members need their total excitatory weight to
   * grow many times over for a partial cue to bring the rest back, and any
   * budget that allows it (measured up to 3× the built sum) is no budget.
   */
  normalize: boolean;
  normalizeEvery: number;
  normalizeGain: number;
  /** Short-term depression: what makes an assembly ignite and fade instead of taking the sheet over. */
  shortTermDepression: boolean;
  stdU: number;
  stdTauRec: number;
}

const DEFAULT_CONFIG: NativeCortexConfig = {
  neurons: 10_000,
  inputCount: 1000,
  fanIn: 100,
  localShare: 0.7,
  localSigma: 0.05,
  inputFanIn: 50,
  topography: 0.2,
  inputGain: 2.0,
  inhibitoryFraction: 0.2,
  inhibitoryAfferents: 0.3,
  excMax: 0.2,
  inhWeight: -2.0,
  excToInhGain: 8.0,
  seed: 0x9a71,
  substeps: 1,
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
  iEta: 0.002,
  targetRate: 0.05,
  inhMax: 8.0,
  tauSynExc: 5.0,
  tauSynInh: 10.0,
  nmdaShare: 0.15,
  tauSynNmda: 50,
  normalize: false,
  normalizeEvery: 20,
  normalizeGain: 2.0,
  shortTermDepression: true,
  stdU: 0.3,
  stdTauRec: 200,
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
    const local = Math.round(cfg.fanIn * Math.max(0, Math.min(1, cfg.localShare)));
    this.net = createNativeNetwork({
      neurons: cfg.neurons,
      substeps: cfg.substeps,
      fanIn: cfg.fanIn,
      blocks: local > 0 ? [
        { srcFrom: 0, srcTo: cfg.neurons, dstFrom: 0, dstTo: cfg.neurons, fanOut: local, wMin: 0, wMax: cfg.excMax, sigma: cfg.localSigma },
        { srcFrom: 0, srcTo: cfg.neurons, dstFrom: 0, dstTo: cfg.neurons, fanOut: cfg.fanIn - local, wMin: 0, wMax: cfg.excMax, sigma: 0 },
      ] : undefined,
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
      iEta: cfg.iEta,
      targetRate: cfg.targetRate,
      inhMax: cfg.inhMax,
      tauSynExc: cfg.tauSynExc,
      tauSynInh: cfg.tauSynInh,
      nmdaShare: cfg.nmdaShare,
      tauSynNmda: cfg.tauSynNmda,
      normalize: cfg.normalize,
      normalizeEvery: cfg.normalizeEvery,
      normalizeGain: cfg.normalizeGain,
      shortTermDepression: cfg.shortTermDepression,
      stdU: cfg.stdU,
      stdTauRec: cfg.stdTauRec,
    });
    // The afferent projection, from the region's own random source: every
    // excitatory neuron samples `inputFanIn` channels within its receptive
    // field (a window of the channels around its place in the sheet), an
    // interneuron fewer.
    const random = mulberry32(cfg.seed ^ 0xaff);
    const rowPtr = new Uint32Array(cfg.neurons + 1);
    const cols: number[] = [];
    const weights: number[] = [];
    const span = Math.max(1, Math.min(1, cfg.topography) * cfg.inputCount);
    for (let i = 0; i < cfg.neurons; i++) {
      rowPtr[i] = cols.length;
      const k = this.net.isInhibitory(i) ? Math.round(cfg.inputFanIn * cfg.inhibitoryAfferents) : cfg.inputFanIn;
      const centre = ((i + 0.5) / cfg.neurons) * cfg.inputCount;
      for (let s = 0; s < k; s++) {
        const channel = Math.floor(centre + (random() - 0.5) * span);
        cols.push(((channel % cfg.inputCount) + cfg.inputCount) % cfg.inputCount);
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

  /**
   * The recurrent synapses persist as weights AND targets: structural
   * plasticity rewires synapses, so the weights alone, put back on the
   * connectivity a fresh network draws, would land on the wrong neurons.
   */
  override serializeExtra(): unknown {
    const { targets } = this.recurrentSynapses();
    return { recurrent: packArray(this.net.weights()), targets: packArray(Int32Array.from(targets)) };
  }

  override deserializeExtra(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const record = data as Record<string, unknown>;
    const w = unpackFloat32(record.recurrent, this.net.synapses);
    if (!w) return;
    const targets = unpackInt32(record.targets, this.net.synapses);
    if (targets && targets.every((t) => t >= 0 && t < this.cfg.neurons)) {
      this.net.setSynapses(this.net.synapseRowPtr(), Uint32Array.from(targets), w);
    } else {
      // A state written before targets were saved: the weights on the fresh connectivity are the best there is.
      this.net.setWeights(w);
    }
  }
}
