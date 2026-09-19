/**
 * PARTS CORTEX (V2 → IT) — objects as arrangements of parts
 * ===========================================================================
 * The primary visual cortex here answers a drawing as a whole: one template
 * over the whole retina. That is why half a square, re-centred by the eye,
 * was a bracket to it — nothing in a whole-image template can complete
 * what is missing. The ventral stream does not work that way: V1/V2 neurons
 * have small receptive fields tuned to local features, complex cells pool
 * them over a neighbourhood so the feature counts wherever exactly it falls
 * (Hubel & Wiesel 1962), and further up (V4, IT) neurons respond to
 * ARRANGEMENTS of those parts (Tanaka 1996; Riesenhuber & Poggio 1999).
 * A partial view still drives most of an object's parts, so the object
 * neuron completes it — recognition by parts (Biederman 1987).
 *
 * Two levels, both learned by exposure (no labels):
 *
 *   LEVEL 1, parts: a small dictionary of local templates (`partCount`),
 *   shared across the retina, each over a `patch × patch` window of the
 *   contrast map, read as ink or no ink per cell (a stroke is a stroke
 *   whatever its thickness — a coarse, thickness-tolerant part). At each of a grid of
 *   positions the patch there competes among the parts (cosine, with
 *   vigilance for tuned parts; an untuned part is recruited for a patch no
 *   part matches). Pooling: for each part, the max over the positions of
 *   each pooled region (the four overlapping quadrants of the position
 *   grid) and over all positions — the complex-cell step that buys
 *   tolerance to where exactly a part sits.
 *
 *   LEVEL 2, objects: neurons over the pooled parts code, k-WTA with
 *   vigilance and commitment (as the colour cortex), prototypes by exposure
 *   ('Object-n'), surprise at close. The engram of an object is a code of
 *   the visual modality alongside V1's, so that what a partial view evokes
 *   at this level reaches the association memory.
 *
 * Deterministic initial synapses (fixed seed): adding the region leaves the
 * seeded trajectories of the other regions untouched.
 */
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { PERCEPTUAL_NARROWING, PresentationTracker, PrototypeMemory, RELEASE_KEEP, type CategoryView, type Recognition } from '../../core/memory/prototype-memory.js';
import { packArray, unpackFloat32, unpackInt32 } from '../../core/persistence/binary-protocol.js';
import { mulberry32 } from '../../core/random.js';

export interface PartsCortexConfig {
  /** Side of the retina (cells). */
  retinaSide: number;
  /** Retinal channel maps: contrast + edge maps, each retinaSide² long. */
  channelMaps: number;
  /** Retinal channels the region is fed (≥ retinaSide² × channelMaps; the rest is padding). */
  inputCount: number;
  /** Side of a part's window (cells) and the stride of the position grid. */
  patch: number;
  stride: number;
  /** Parts in the dictionary. */
  partCount: number;
  /** Cosine a TUNED part needs with a patch to claim it. */
  partVigilance: number;
  /** Learning rate of a part toward the patches it claims. */
  partLearningRate: number;
  /** Object neurons and winners per tick. */
  neuronCount: number;
  kWinners: number;
  /** Cosine a TUNED object neuron needs with the parts code to compete. */
  vigilance: number;
  /** Overlap between engrams from which two views are the same object. */
  categoryMatch: number;
  gapTicks: number;
  learningRate: number;
}

const DEFAULT_CONFIG: PartsCortexConfig = {
  retinaSide: 14,
  channelMaps: 5,
  inputCount: 1000,
  patch: 6,
  stride: 2,
  partCount: 24,
  partVigilance: 0.6,
  partLearningRate: 0.2,
  neuronCount: 600,
  kWinners: 8,
  vigilance: 0.5,
  categoryMatch: 0.35,
  gapTicks: 10,
  learningRate: 0.3,
};

/** Contrast from which a retinal cell counts as inked for a part. */
const PART_INK = 0.2;
/** A patch must have at least this many inked cells to be a part of anything (as a norm: √cells). */
const PATCH_FLOOR = 1.4;
/** Channel maps the parts read: the contrast map only (the edge maps double-count thick strokes). */
const PART_MAPS = 1;
/** The window is max-pooled 2×2 before comparison: a thick stroke and a thin one, a stroke a cell to the left or right, are the same part. */
const PART_POOL = 2;
/** Weight of the position-free pool (which parts are present at all) against the quadrant pools (where). */
const PRESENCE_WEIGHT = 1.5;

export class PartsCortex extends BrainRegion {
  private readonly cfg: PartsCortexConfig;
  /** Position grid: top-left cell of each position. */
  private readonly positions: Array<{ row: number; col: number }> = [];
  /** Pooled regions: which positions each pools over (four overlapping quadrants, then all). */
  private readonly pools: number[][] = [];
  /** Level 1: part templates, partCount × (patch² × channelMaps). */
  private readonly partWeights: Float32Array;
  private readonly partTuned: Int32Array;
  private readonly partUses: Float32Array;
  private readonly patchSize: number;
  private readonly pooledSide: number;
  /** Level 2 (the region's own weights, over the pooled code). */
  readonly codeSize: number;
  private tuned: Int32Array;
  private winCounts: Float32Array;
  private entrenched: Float32Array;
  private readonly presentation: PresentationTracker;
  private presentationCode: Float32Array;
  private presentationTicks = 0;
  /** Patches claimed during the presentation in progress, for tuning the parts at close: part → summed patch. */
  private claimed: Map<number, { sum: Float32Array; n: number }> = new Map();
  /** Parts recruited during the presentation in progress (not blended until the next). */
  private recruited = new Set<number>();
  private readonly prototypes: PrototypeMemory;
  private lastRecognition: Recognition | null = null;
  private lastEngram: Int32Array = new Int32Array(0);
  private lastCode: Float32Array;
  private perceptCount = 0;
  private suppressPercept = false;

  constructor(config: Partial<PartsCortexConfig> = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    super('partsCortex', 'Corteza de Partes (V2→IT)', cfg.neuronCount, cfg.inputCount);
    this.cfg = cfg;
    for (let row = 0; row + cfg.patch <= cfg.retinaSide; row += cfg.stride) {
      for (let col = 0; col + cfg.patch <= cfg.retinaSide; col += cfg.stride) this.positions.push({ row, col });
    }
    // Pools: the four overlapping quadrants of the position grid, then everything.
    const side = Math.round(Math.sqrt(this.positions.length));
    const half = Math.ceil(side / 2);
    for (const [r0, c0] of [[0, 0], [0, side - half], [side - half, 0], [side - half, side - half]]) {
      const pool: number[] = [];
      for (let r = r0; r < r0 + half; r++) for (let c = c0; c < c0 + half; c++) pool.push(r * side + c);
      this.pools.push(pool);
    }
    this.pools.push(this.positions.map((_, i) => i));
    this.pooledSide = Math.ceil(cfg.patch / PART_POOL);
    this.patchSize = this.pooledSide * this.pooledSide * PART_MAPS;
    this.codeSize = cfg.partCount * this.pools.length;
    this.partWeights = new Float32Array(cfg.partCount * this.patchSize);
    this.partTuned = new Int32Array(cfg.partCount);
    this.partUses = new Float32Array(cfg.partCount);
    this.tuned = new Int32Array(cfg.neuronCount);
    this.winCounts = new Float32Array(cfg.neuronCount);
    this.entrenched = new Float32Array(cfg.neuronCount);
    this.presentation = new PresentationTracker(cfg.neuronCount, cfg.kWinners, cfg.gapTicks);
    this.presentationCode = new Float32Array(this.codeSize);
    this.lastCode = new Float32Array(this.codeSize);
    this.prototypes = new PrototypeMemory({ labelPrefix: 'Object', matchThreshold: cfg.categoryMatch, unitCount: cfg.neuronCount });
    const random = mulberry32(0x9a275);
    for (let i = 0; i < this.partWeights.length; i++) this.partWeights[i] = random() * 0.05;
  }

  /** Small deterministic synapses for the object neurons over the pooled code (the rest of the row stays 0). */
  protected override initializeWeights(): void {
    const random = mulberry32(0x0b1ec7);
    const code = DEFAULT_CONFIG.partCount * 5; // codeSize is not known yet here; recomputed in the constructor's fields
    for (let n = 0; n < this.neuronCount; n++) {
      const offset = n * this.inputCount;
      for (let i = 0; i < Math.min(code, this.inputCount); i++) this.weights[offset + i] = random() * 0.05;
    }
  }

  // ── Level 1: parts ────────────────────────────────────────────────────────

  /** The patch of every channel map at a position, as one vector. */
  private patchAt(spikes: Float32Array, position: number, out: Float32Array): number {
    const { retinaSide, patch } = this.cfg;
    const { row, col } = this.positions[position];
    const map = retinaSide * retinaSide;
    let energy = 0;
    out.fill(0);
    const ps = this.pooledSide;
    for (let m = 0; m < PART_MAPS; m++) {
      for (let r = 0; r < patch; r++) {
        for (let c = 0; c < patch; c++) {
          const v = (spikes[m * map + (row + r) * retinaSide + col + c] ?? 0) >= PART_INK ? 1 : 0;
          if (v === 0) continue;
          const idx = m * ps * ps + Math.floor(r / PART_POOL) * ps + Math.floor(c / PART_POOL);
          if (out[idx] === 0) { out[idx] = 1; energy++; }
        }
      }
    }
    return Math.sqrt(energy);
  }

  /** The part that claims a patch (cosine, with vigilance for tuned parts), or −1 for an empty patch. */
  private partFor(patch: Float32Array, norm: number, learning: boolean): number {
    if (norm < PATCH_FLOOR) return -1;
    let best = -1;
    let bestMatch = 0;
    let untuned = -1;
    for (let p = 0; p < this.cfg.partCount; p++) {
      if (this.partTuned[p] === 0) {
        if (untuned < 0) untuned = p;
        continue;
      }
      const offset = p * this.patchSize;
      let dot = 0, n2 = 0;
      for (let i = 0; i < this.patchSize; i++) { dot += patch[i] * this.partWeights[offset + i]; n2 += this.partWeights[offset + i] ** 2; }
      const match = dot / (Math.sqrt(n2) * norm + 1e-9);
      if (match > bestMatch) { bestMatch = match; best = p; }
    }
    if (best >= 0 && bestMatch >= this.cfg.partVigilance) return best;
    // No tuned part matches: a fresh one is recruited (when learning) — or,
    // with none left, the nearest part answers anyway.
    if (learning && untuned >= 0) return untuned;
    return best;
  }

  /** The pooled parts code of a retinal input: max over each pool's positions, per part. */
  encodeParts(spikes: Float32Array, learning: boolean): Float32Array {
    const code = new Float32Array(this.codeSize);
    const patch = new Float32Array(this.patchSize);
    const perPosition = new Int32Array(this.positions.length).fill(-1);
    const strength = new Float32Array(this.positions.length);
    for (let pos = 0; pos < this.positions.length; pos++) {
      const norm = this.patchAt(spikes, pos, patch);
      const part = this.partFor(patch, norm, learning);
      if (part < 0) continue;
      perPosition[pos] = part;
      strength[pos] = Math.min(1, norm);
      if (learning) {
        // A freshly recruited part IS the patch that recruited it (so that it
        // claims the same patch at the next position instead of a new part
        // each time); it starts learning from other patches next time.
        if (this.partTuned[part] === 0) {
          this.partWeights.set(patch, part * this.patchSize);
          this.partTuned[part] = 1;
          this.recruited.add(part);
        } else if (!this.recruited.has(part)) {
          let c = this.claimed.get(part);
          if (!c) this.claimed.set(part, (c = { sum: new Float32Array(this.patchSize), n: 0 }));
          for (let i = 0; i < this.patchSize; i++) c.sum[i] += patch[i];
          c.n++;
        }
      }
    }
    this.pools.forEach((pool, q) => {
      for (const pos of pool) {
        const part = perPosition[pos];
        if (part < 0) continue;
        const idx = q * this.cfg.partCount + part;
        const v = strength[pos] * (q === this.pools.length - 1 ? PRESENCE_WEIGHT : 1);
        if (v > code[idx]) code[idx] = v;
      }
    });
    return code;
  }

  /** At the close of a presentation: each part moves toward the mean of the patches it claimed. */
  private tuneParts(lr: number): void {
    for (const [part, c] of this.claimed) {
      const offset = part * this.patchSize;
      for (let i = 0; i < this.patchSize; i++) {
        const target = c.sum[i] / c.n;
        this.partWeights[offset + i] += lr * (target - this.partWeights[offset + i]);
      }
      this.partUses[part]++;
    }
    this.claimed.clear();
    this.recruited.clear();
  }

  // ── Level 2: objects ──────────────────────────────────────────────────────

  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    const output = new Float32Array(this.neuronCount);
    let energy = 0;
    for (let i = 0; i < spikes.length; i++) energy += spikes[i] * spikes[i];
    if (energy < 1e-6) {
      this.closePresentation(this.presentation.tick(false, new Int32Array(0)), modulationEffects);
      return output;
    }
    const code = this.encodeParts(spikes, true);
    let codeNorm2 = 0;
    for (let i = 0; i < code.length; i++) codeNorm2 += code[i] * code[i];
    if (codeNorm2 < 1e-6) {
      this.closePresentation(this.presentation.tick(false, new Int32Array(0)), modulationEffects);
      return output;
    }
    const codeNorm = Math.sqrt(codeNorm2);

    const potentials = new Float32Array(this.neuronCount);
    for (let n = 0; n < this.neuronCount; n++) {
      const offset = n * this.inputCount;
      let sum = 0, norm2 = 0;
      for (let i = 0; i < code.length; i++) {
        sum += code[i] * this.weights[offset + i];
        norm2 += this.weights[offset + i] ** 2;
      }
      if (this.tuned[n] === 1) {
        const match = sum / (Math.sqrt(norm2) * codeNorm + 1e-9);
        if (match < this.cfg.vigilance * (1 - PERCEPTUAL_NARROWING.gain * this.entrenched[n])) continue;
      }
      potentials[n] = Math.max(0, sum - this.winCounts[n] * 0.01);
    }
    const order = Array.from({ length: this.neuronCount }, (_, i) => i).sort((a, b) => potentials[b] - potentials[a] || a - b);
    const winners: number[] = [];
    for (let i = 0; i < this.cfg.kWinners && potentials[order[i]] > 0; i++) winners.push(order[i]);
    for (const w of winners) output[w] = 1;

    for (let i = 0; i < code.length; i++) this.presentationCode[i] += code[i];
    this.presentationTicks++;
    this.presentation.tick(true, winners);
    return output;
  }

  private closePresentation(engram: Int32Array | null, effects: ModulationEffects): void {
    if (this.presentationTicks === 0) return;
    if (!engram) {
      if (!this.presentation.active) this.resetPresentation();
      return;
    }
    const mean = new Float32Array(this.codeSize);
    for (let i = 0; i < mean.length; i++) mean[i] = this.presentationCode[i] / this.presentationTicks;
    this.resetPresentation();
    const lr = Math.min(1, this.cfg.learningRate * (effects.learningRateMultiplier ?? 1));
    this.tuneParts(Math.min(1, this.cfg.partLearningRate * (effects.learningRateMultiplier ?? 1)));

    // Surprise: 1 − cosine between the parts seen and what the engram expects, before it learns.
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < this.codeSize; i++) {
      let w = 0;
      for (let k = 0; k < engram.length; k++) w += this.weights[engram[k] * this.inputCount + i];
      w /= Math.max(1, engram.length);
      dot += mean[i] * w; na += mean[i] * mean[i]; nb += w * w;
    }
    const surprise = na > 0 && nb > 0 ? Math.max(0, Math.min(1, 1 - dot / Math.sqrt(na * nb))) : 1;

    for (let k = 0; k < engram.length; k++) {
      const n = engram[k];
      const offset = n * this.inputCount;
      if (this.tuned[n] === 0) this.weights.fill(0, offset, offset + this.inputCount);
      for (let i = 0; i < this.codeSize; i++) this.weights[offset + i] += lr * (mean[i] - this.weights[offset + i]);
      this.tuned[n] = 1;
      this.winCounts[n]++;
    }
    for (let n = 0; n < this.neuronCount; n++) this.winCounts[n] *= 0.9;
    this.lastCode = mean;

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
      this.refreshEntrenchment();
    }
  }

  private resetPresentation(): void {
    this.presentationCode.fill(0);
    this.presentationTicks = 0;
  }

  /**
   * What an object looks like, from its engram: each part the object's
   * neurons expect, painted at the centre of the pool it is expected in —
   * the retina the object would evoke (feedback imagery, coarse).
   */
  imagine(units: ArrayLike<number>): Float32Array {
    const { retinaSide, patch, partCount } = this.cfg;
    const image = new Float32Array(retinaSide * retinaSide);
    if (units.length === 0) return image;
    const expected = new Float32Array(this.codeSize);
    for (let u = 0; u < units.length; u++) {
      const offset = units[u] * this.inputCount;
      for (let i = 0; i < this.codeSize; i++) expected[i] += this.weights[offset + i] / units.length;
    }
    let peak = 0;
    this.pools.forEach((pool, q) => {
      if (q === this.pools.length - 1) return; // the global pool says nothing about where
      let cr = 0, cc = 0;
      for (const pos of pool) { cr += this.positions[pos].row; cc += this.positions[pos].col; }
      cr = Math.round(cr / pool.length); cc = Math.round(cc / pool.length);
      for (let p = 0; p < partCount; p++) {
        const w = expected[q * partCount + p];
        if (w <= 0.05 || this.partTuned[p] === 0) continue;
        const offset = p * this.patchSize; // the contrast map is the first channel map
        for (let r = 0; r < patch; r++) for (let c = 0; c < patch; c++) {
          const rr = cr + r, cc2 = cc + c;
          if (rr >= retinaSide || cc2 >= retinaSide) continue;
          const v = w * this.partWeights[offset + Math.floor(r / PART_POOL) * this.pooledSide + Math.floor(c / PART_POOL)];
          const idx = rr * retinaSide + cc2;
          if (v > image[idx]) image[idx] = v;
          if (image[idx] > peak) peak = image[idx];
        }
      }
    });
    if (peak > 0) for (let i = 0; i < image.length; i++) image[i] /= peak;
    return image;
  }

  suppressNextPercept(): void {
    this.suppressPercept = true;
  }

  getRecognition(): Recognition | null {
    return this.lastRecognition;
  }

  getLastEngram(): Int32Array {
    return this.lastEngram;
  }

  get percepts(): number {
    return this.perceptCount;
  }

  get categoryCount(): number {
    return this.prototypes.size;
  }

  /** Parts learned so far. */
  get partsKnown(): number {
    let n = 0;
    for (let p = 0; p < this.cfg.partCount; p++) if (this.partTuned[p] === 1) n++;
    return n;
  }

  matchCategory(units: ArrayLike<number>): { id: number; label: string; overlap: number; exposures: number } | null {
    return this.prototypes.match(units);
  }

  categories(): CategoryView[] {
    return this.prototypes.list();
  }

  private refreshEntrenchment(): void {
    this.entrenched = this.prototypes.entrenchment(this.neuronCount, PERCEPTUAL_NARROWING.tau);
  }

  pruneCategories(minExposures: number, afterSleeps: number, inUse: (unit: number) => boolean = () => false): string[] {
    this.prototypes.age();
    const pruned = this.prototypes.prune(minExposures, afterSleeps);
    if (pruned.length === 0) return [];
    const used = this.prototypes.unitsInUse();
    for (const c of pruned) for (const u of c.units) if (!used.has(u) && !inUse(u)) this.release(u);
    this.refreshEntrenchment();
    return pruned.map((c) => c.label);
  }

  private release(n: number): void {
    this.tuned[n] = 0;
    this.winCounts[n] = 0;
    const offset = n * this.inputCount;
    for (let i = 0; i < this.inputCount; i++) this.weights[offset + i] *= RELEASE_KEEP;
  }

  override serializeExtra(): unknown {
    return {
      parts: packArray(this.partWeights),
      partTuned: packArray(this.partTuned),
      partUses: packArray(this.partUses),
      winCounts: packArray(this.winCounts),
      tuned: packArray(this.tuned),
      categories: this.prototypes.serialize(),
    };
  }

  override deserializeExtra(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const d = data as Record<string, unknown>;
    const parts = unpackFloat32(d.parts, this.partWeights.length);
    if (parts) this.partWeights.set(parts);
    const partTuned = unpackInt32(d.partTuned, this.cfg.partCount);
    if (partTuned) this.partTuned.set(partTuned);
    const uses = unpackFloat32(d.partUses, this.cfg.partCount);
    if (uses) this.partUses.set(uses);
    const wins = unpackFloat32(d.winCounts, this.neuronCount);
    if (wins) this.winCounts.set(wins);
    const tuned = unpackInt32(d.tuned, this.neuronCount);
    if (tuned) this.tuned.set(tuned);
    this.prototypes.deserialize(d.categories);
    this.refreshEntrenchment();
  }
}
