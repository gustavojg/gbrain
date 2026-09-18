/**
 * COLOUR CORTEX — what colour a thing is, whatever its shape
 * ===========================================================================
 * In the ventral stream, colour and shape part ways: V4 and the colour
 * patches of the temporal lobe represent hue and saturation of a surface
 * largely regardless of the surface's form or position (Zeki 1973; Conway
 * 2007). That separation is what lets a word attach to a colour across
 * objects: "azul" for the blue card, the blue car, the blue ball.
 *
 * The retina delivers the colour of what is in view as a small population
 * code (a hue × saturation histogram over the chromatic pixels; see
 * VisualEncoder.encodeColor). This region forms colour categories from it by
 * exposure alone — vigilance-gated competition and prototype memory, as the
 * auditory cortex does for sounds — and completes a colour percept when the
 * presentation ends. Its engram is then a code of its own modality
 * ('colour') for the association memory.
 *
 * Its synapses are initialized deterministically (no draw from the global
 * random source), so adding the region leaves the trajectories of the other
 * regions untouched.
 */
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { PresentationTracker, PrototypeMemory, type Recognition } from '../../core/memory/prototype-memory.js';
import { packArray, unpackFloat32, unpackInt32 } from '../../core/persistence/binary-protocol.js';

/** Channels of the retina's colour code (see VisualEncoder.encodeColor). */
export const COLOR_CHANNELS = 26;

export interface ColorCortexConfig {
  neuronCount: number;
  /** Winners per tick. */
  kWinners: number;
  /** Cosine match a TUNED neuron needs with the input to compete (vigilance). */
  vigilance: number;
  /** Overlap between engrams from which two colours are the same category. */
  categoryMatch: number;
  /** Ticks of silence that close a presentation. */
  gapTicks: number;
  learningRate: number;
}

const DEFAULT_CONFIG: ColorCortexConfig = {
  neuronCount: 120,
  kWinners: 6,
  vigilance: 0.85,
  categoryMatch: 0.5,
  gapTicks: 10,
  learningRate: 0.3,
};

/** mulberry32 with a fixed seed: deterministic, never touches Math.random. */
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class ColorCortex extends BrainRegion {
  private readonly cfg: ColorCortexConfig;
  /** 1 once a neuron has been tuned to some colour (vigilance applies to it). */
  private tuned: Int32Array;
  private winCounts: Float32Array;
  private readonly presentation: PresentationTracker;
  private presentationInput: Float32Array;
  private presentationTicks = 0;
  private readonly prototypes: PrototypeMemory;
  private lastRecognition: Recognition | null = null;
  private lastEngram: Int32Array = new Int32Array(0);
  private perceptCount = 0;
  private suppressPercept = false;

  constructor(config: Partial<ColorCortexConfig> = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    super('colorCortex', 'Corteza del Color (V4)', cfg.neuronCount, COLOR_CHANNELS);
    this.cfg = cfg;
    this.tuned = new Int32Array(cfg.neuronCount);
    this.winCounts = new Float32Array(cfg.neuronCount);
    this.presentation = new PresentationTracker(cfg.neuronCount, cfg.kWinners, cfg.gapTicks);
    this.presentationInput = new Float32Array(COLOR_CHANNELS);
    this.prototypes = new PrototypeMemory({ labelPrefix: 'Colour', matchThreshold: cfg.categoryMatch, unitCount: cfg.neuronCount });
  }

  /** Small random-looking synapses from a fixed seed: the global random source is not consumed. */
  protected override initializeWeights(): void {
    const random = seeded(0xc0104);
    for (let n = 0; n < this.neuronCount; n++) {
      const offset = n * this.inputCount;
      for (let i = 0; i < this.inputCount; i++) this.weights[offset + i] = random() * 0.05;
    }
  }

  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    const output = new Float32Array(this.neuronCount);
    let inputNorm2 = 0;
    for (let i = 0; i < this.inputCount; i++) inputNorm2 += spikes[i] * spikes[i];
    if (inputNorm2 < 1e-6) {
      this.closePresentation(this.presentation.tick(false, new Int32Array(0)), modulationEffects);
      return output;
    }
    const inputNorm = Math.sqrt(inputNorm2);

    const potentials = new Float32Array(this.neuronCount);
    for (let n = 0; n < this.neuronCount; n++) {
      const offset = n * this.inputCount;
      let sum = 0;
      let norm2 = 0;
      for (let i = 0; i < this.inputCount; i++) {
        sum += spikes[i] * this.weights[offset + i];
        norm2 += this.weights[offset + i] * this.weights[offset + i];
      }
      if (this.tuned[n] === 1) {
        // Vigilance: a neuron tuned to a colour only competes for that colour.
        const match = sum / (Math.sqrt(norm2) * inputNorm + 1e-9);
        if (match < this.cfg.vigilance) continue;
      }
      potentials[n] = Math.max(0, sum - this.winCounts[n] * 0.01);
    }
    const order = Array.from({ length: this.neuronCount }, (_, i) => i).sort((a, b) => potentials[b] - potentials[a] || a - b);
    const winners: number[] = [];
    for (let i = 0; i < this.cfg.kWinners && potentials[order[i]] > 0; i++) winners.push(order[i]);
    for (const w of winners) output[w] = 1;

    for (let i = 0; i < this.inputCount; i++) this.presentationInput[i] += spikes[i];
    this.presentationTicks++;
    this.presentation.tick(true, winners);
    return output;
  }

  /** End of a presentation: tune the winners to the colour seen and file it as a category. */
  private closePresentation(engram: Int32Array | null, effects: ModulationEffects): void {
    if (this.presentationTicks === 0) return;
    if (!engram) {
      if (!this.presentation.active) this.resetPresentation();
      return;
    }
    const mean = new Float32Array(this.inputCount);
    for (let i = 0; i < mean.length; i++) mean[i] = this.presentationInput[i] / this.presentationTicks;
    this.resetPresentation();
    // Surprise: 1 − cosine between the colour seen and what the engram's neurons expect, before they learn.
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < this.inputCount; i++) {
      let w = 0;
      for (let k = 0; k < engram.length; k++) w += this.weights[engram[k] * this.inputCount + i];
      w /= Math.max(1, engram.length);
      dot += mean[i] * w; na += mean[i] * mean[i]; nb += w * w;
    }
    const surprise = na > 0 && nb > 0 ? Math.max(0, Math.min(1, 1 - dot / Math.sqrt(na * nb))) : 1;

    const lr = Math.min(1, this.cfg.learningRate * (effects.learningRateMultiplier ?? 1));
    for (let k = 0; k < engram.length; k++) {
      const n = engram[k];
      const offset = n * this.inputCount;
      // A freshly recruited neuron drops its random synapses: it becomes THIS colour.
      if (this.tuned[n] === 0) this.weights.fill(0, offset, offset + this.inputCount);
      for (let i = 0; i < this.inputCount; i++) this.weights[offset + i] += lr * (mean[i] - this.weights[offset + i]);
      this.tuned[n] = 1;
      this.winCounts[n]++;
    }
    for (let n = 0; n < this.neuronCount; n++) this.winCounts[n] *= 0.9;

    if (this.suppressPercept) {
      this.suppressPercept = false;
      return;
    }
    const recognition = this.prototypes.observe(engram, this.currentTime);
    if (recognition) {
      recognition.surprise = surprise;
      this.lastRecognition = recognition;
      this.lastEngram = engram;
      this.perceptCount++;
    }
  }

  private resetPresentation(): void {
    this.presentationInput.fill(0);
    this.presentationTicks = 0;
  }

  /** The presentation that is starting is the brain's own drawing: learn, but no percept. */
  suppressNextPercept(): void {
    this.suppressPercept = true;
  }

  getRecognition(): Recognition | null {
    return this.lastRecognition;
  }

  getLastEngram(): Int32Array {
    return this.lastEngram;
  }

  /** Completed colour presentations so far — changes exactly when a new percept is available. */
  get percepts(): number {
    return this.perceptCount;
  }

  get categoryCount(): number {
    return this.prototypes.size;
  }

  /** Names a code reinstated from memory: the learned colour it overlaps most. */
  matchCategory(units: ArrayLike<number>): { id: number; label: string; overlap: number; exposures: number } | null {
    return this.prototypes.match(units);
  }

  override serializeExtra(): unknown {
    return { winCounts: packArray(this.winCounts), tuned: packArray(this.tuned), categories: this.prototypes.serialize() };
  }

  override deserializeExtra(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as Record<string, unknown>;
    const wins = unpackFloat32(d.winCounts, this.neuronCount);
    if (wins) this.winCounts.set(wins);
    const tuned = unpackInt32(d.tuned, this.neuronCount);
    if (tuned) this.tuned.set(tuned);
    this.prototypes.deserialize(d.categories);
  }
}
