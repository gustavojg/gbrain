/**
 * HAND MOTOR CORTEX — Learning to draw what it sees
 * =================================================
 * A map of motor units over the whiteboard — one per cell the hand can ink —
 * fed by the retinotopic visual signal (the dorsal, "where/how" visual stream).
 * It learns from its own scribbling which marks its commands leave, and can
 * then run that knowledge backwards: given an image (seen, or imagined), ink
 * the cells that would produce it.
 *
 * Biological basis:
 *   - MOTOR BABBLING: infants scribble long before they draw. Each scribble is
 *     an experiment: a motor pattern is executed, its visual consequence is
 *     seen, and the two are associated (Hebb). This is the same sensorimotor
 *     scheme the vocal motor cortex uses with sounds.
 *   - The map is visuomotor and retinotopic (posterior parietal → premotor):
 *     what has to be learned is which retinal channels light up when a given
 *     cell is inked — intensity AND edge channels, whatever co-occurs.
 *   - The eye centres what it looks at, so the hand works in the same centred
 *     frame: a scribble is centred on the board, and so is a copy.
 *   - While drawing, what arrives from the eye is the consequence of its own
 *     command: it is used to learn, never as something to copy (no loop of
 *     copies of copies).
 *
 * The weights of this region ARE the visual → motor map (cells × retinal
 * channels), learned from scratch and persisted with the rest of the synapses.
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { GRID_SIDE, centred, type Drawing } from '../../core/hand/whiteboard.js';

export interface HandMotorConfig {
  /** Retinal channels (dimension of the input). */
  inputCount: number;
  /** Ticks a command is held: the drawing stays in view while its image comes back. */
  holdTicks: number;
  /** Minimum learned drive (0–1) on the best cell to attempt a drawing. */
  drawThreshold: number;
  /** A cell is inked if driven at least this fraction of the best cell's drive. */
  inkFraction: number;
  /** Ticks without visual input that end a seen image (then the copy is drawn). */
  planGapTicks: number;
  /** Retinal channels below this rate are treated as background. */
  inputFloor: number;
}

const DEFAULT_HAND_CONFIG: HandMotorConfig = {
  inputCount: 1000,
  holdTicks: 55,
  drawThreshold: 0.2,
  inkFraction: 0.7,
  planGapTicks: 10,
  inputFloor: 0.25,
};

/** A drawing issued by the hand motor cortex, with why it was issued. */
export interface HandOutput {
  cells: Drawing;
  source: 'scribble' | 'copy' | 'from-memory';
  /** For a copy or a drawing from memory: how strongly the image drove the map (0–1). */
  confidence: number;
}

export class HandMotorCortex extends BrainRegion {
  private readonly cfg: HandMotorConfig;

  /** Motor pattern being executed (1 per inked cell), or `null` when not drawing. */
  private held: Float32Array | null = null;
  private heldTicks = 0;

  /** Visual drive accumulated while an EXTERNAL image is being seen. */
  private readonly plan: Float32Array;
  private planTicks = 0;
  private planSilentTicks = 0;

  private pending: HandOutput | null = null;

  /** Running estimate of how often each cell is inked (its base rate). */
  private inkRate!: Float32Array;
  /** Ticks each retinal channel was lit during the drawing in view, and ticks with any input. */
  private seenTicks!: Float32Array;
  private seenTotal = 0;
  /** Experiences in which each channel was lit (sets that channel's learning step). */
  private channelCount!: Float32Array;
  private experiences = 0;
  /** Minimum strength (0–1) for a channel of an IMAGINED image to be drawn. */
  private static readonly IMAGERY_FLOOR = 0.5;
  /** Floor of the learning step: the map never stops adapting. */
  private static readonly MIN_STEP = 0.02;
  /** Pseudo-observations of "lit but not inked" every synapse starts with. */
  private static readonly PRIOR_COUNT = 3;

  /** Scribbles produced so far (for monitoring). */
  scribbleCount = 0;
  /**
   * Drawings learned from, and what the map knows after the last one: the
   * fraction of retinal channels it has seen lit by its own marks (0..1). Its
   * growth per drawing is learning progress; it saturates — mastery.
   */
  learnings = 0;
  knowledge = 0;

  /** Whether an image it sees (and knows how to make) is copied on the whiteboard. */
  copy = false;

  constructor(config: Partial<HandMotorConfig> = {}) {
    const cfg = { ...DEFAULT_HAND_CONFIG, ...config };
    super('handMotorCortex', 'Corteza Motora de la Mano', GRID_SIDE * GRID_SIDE, cfg.inputCount);
    this.cfg = cfg;
    this.plan = new Float32Array(this.neuronCount);
    this.inkRate = new Float32Array(this.neuronCount);
    this.seenTicks = new Float32Array(cfg.inputCount);
    this.channelCount = new Float32Array(cfg.inputCount);
    this.weights.fill(0);
  }

  /** The visual → motor map is learned from scratch; no random synapses. */
  protected override initializeWeights(): void {
    // Intentionally empty (see constructor).
  }

  // ----------------------------------------------------------------
  // Drawing
  // ----------------------------------------------------------------

  /** Whether a drawing is being executed (its image is expected from the eye). */
  get drawing(): boolean {
    return this.held !== null;
  }

  /**
   * A scribble: one or two random strokes, centred on the board. Its image
   * will come back through the eye while the pattern is still held, which is
   * when the map learns.
   */
  scribble(): HandOutput {
    this.scribbleCount++;
    const cells = new Set<number>();
    const strokes = 1 + Math.floor(Math.random() * 2);
    for (let s = 0; s < strokes; s++) {
      const x0 = Math.floor(Math.random() * GRID_SIDE);
      const y0 = Math.floor(Math.random() * GRID_SIDE);
      const x1 = Math.floor(Math.random() * GRID_SIDE);
      const y1 = Math.floor(Math.random() * GRID_SIDE);
      const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
      for (let i = 0; i <= steps; i++) {
        const x = Math.round(x0 + ((x1 - x0) * i) / steps);
        const y = Math.round(y0 + ((y1 - y0) * i) / steps);
        cells.add(y * GRID_SIDE + x);
      }
    }
    return this.execute(centred([...cells]), 'scribble', 0);
  }

  private execute(cells: Drawing, source: HandOutput['source'], confidence: number): HandOutput {
    const pattern = new Float32Array(this.neuronCount);
    for (const cell of cells) pattern[cell] = 1;
    this.held = pattern;
    this.heldTicks = this.cfg.holdTicks;
    this.resetPlan();
    return { cells, source, confidence };
  }

  /**
   * Draws an image that is only IMAGINED — a retinal pattern reinstated from
   * memory (what a word brings to the mind's eye), not one arriving through
   * the eye. It drives the map exactly as a seen image would.
   *
   * @returns The drawing being executed, or `null` if the hand is off, busy,
   *   or the map does not know how to make that image yet
   */
  drawImagined(retinalPattern: Float32Array): HandOutput | null {
    if (!this.copy || this.held) return null;
    // An imagined image is blurrier than a seen one (neurons shared between
    // engrams leak a faint trace of other images): only its clear parts count.
    const clear = new Float32Array(retinalPattern.length);
    for (let i = 0; i < clear.length; i++) clear[i] = retinalPattern[i] >= HandMotorCortex.IMAGERY_FLOOR ? retinalPattern[i] : 0;
    const drive = this.driveFrom(clear);
    return drive ? this.decide(drive.cells, drive.peak, 'from-memory') : null;
  }

  /** Takes the copy that became ready on this tick, if any. */
  takeCommand(): HandOutput | null {
    const output = this.pending;
    this.pending = null;
    return output;
  }

  // ----------------------------------------------------------------
  // Main processing
  // ----------------------------------------------------------------

  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    const seen: number[] = [];
    for (let i = 0; i < spikes.length; i++) if (spikes[i] > this.cfg.inputFloor) seen.push(i);

    // --- Drawing: what is seen is the consequence of the held command ---
    if (this.held) {
      if (seen.length > 0) {
        for (let s = 0; s < seen.length; s++) this.seenTicks[seen[s]]++;
        this.seenTotal++;
      }
      const output = new Float32Array(this.held);
      if (--this.heldTicks <= 0) {
        // The drawing has been looked at: learn from the whole experience once.
        this.learn(this.held, modulationEffects.learningRateMultiplier ?? 1);
        this.held = null;
      }
      this.spikes.set(output);
      return output;
    }

    // --- Looking: a seen image drives the cells that used to produce it ---
    const output = new Float32Array(this.neuronCount);
    if (seen.length > 0) {
      const drive = this.driveFrom(spikes);
      if (drive) {
        for (let m = 0; m < this.neuronCount; m++) {
          this.plan[m] += drive.cells[m];
          if (drive.cells[m] >= drive.peak * this.cfg.inkFraction) output[m] = 1;
        }
      }
      this.planTicks++;
      this.planSilentTicks = 0;
    } else if (this.planTicks > 0 && ++this.planSilentTicks >= this.cfg.planGapTicks) {
      // The image is gone: if the map knew it well enough, copy it.
      if (this.copy) {
        let peak = 0;
        for (let m = 0; m < this.neuronCount; m++) {
          this.plan[m] /= this.planTicks;
          if (this.plan[m] > peak) peak = this.plan[m];
        }
        this.pending = this.decide(this.plan, peak, 'copy');
      }
      this.resetPlan();
    }

    this.spikes.set(output);
    return output;
  }

  /** Drive of every cell by a retinal pattern (mean learned weight over the active channels). */
  private driveFrom(retina: Float32Array): { cells: Float32Array; peak: number } | null {
    const active: number[] = [];
    const n = Math.min(retina.length, this.inputCount);
    for (let i = 0; i < n; i++) if (retina[i] > this.cfg.inputFloor) active.push(i);
    if (active.length === 0) return null;

    const cells = new Float32Array(this.neuronCount);
    let peak = 0;
    for (let m = 0; m < this.neuronCount; m++) {
      const offset = m * this.inputCount;
      let best = 0;
      // A cell is wanted if SOME active channel strongly calls for it (its own
      // retinal position); averaging over all active channels would dilute it.
      for (let a = 0; a < active.length; a++) {
        const w = this.weights[offset + active[a]];
        if (w > best) best = w;
      }
      // The weight estimates P(cell inked | channel lit). What matters is how
      // much that EXCEEDS the cell's base rate: cells near the centre are inked
      // in most scribbles, so every channel "predicts" them a little — without
      // this correction any image came out as a blob in the middle.
      const base = this.inkRate[m];
      const lift = base < 0.999 ? (best - base) / (1 - base) : 0;
      cells[m] = lift > 0 ? lift : 0;
      if (cells[m] > peak) peak = cells[m];
    }
    return { cells, peak };
  }

  private decide(drive: Float32Array, peak: number, source: HandOutput['source']): HandOutput | null {
    if (peak < this.cfg.drawThreshold) return null;
    const cells: Drawing = [];
    for (let m = 0; m < this.neuronCount; m++) if (drive[m] >= peak * this.cfg.inkFraction) cells.push(m);
    return cells.length > 0 ? this.execute(centred(cells), source, Math.min(1, peak)) : null;
  }

  private resetPlan(): void {
    this.plan.fill(0);
    this.planTicks = 0;
    this.planSilentTicks = 0;
  }

  /**
   * Learns from one drawing experience (the command that was held and the
   * retinal channels that were lit while it was in view).
   *
   * Each synapse estimates P(cell inked | channel lit): it moves toward 1 when
   * the channel was lit and the cell inked, toward 0 when the channel was lit
   * and the cell was not. The step is 1/n for the n-th time the channel is seen
   * (every experience counts the same — a running mean), floored so the map
   * stays plastic. After enough scribbles each cell is left with the channels
   * that light up whenever, and only when, it is inked.
   *
   * A fast per-tick Hebbian rule does NOT work here: within a single scribble it
   * saturates every (inked cell, lit channel) pair and erases the rest, so the
   * map only ever reflects the last thing drawn.
   */
  private learn(pattern: Float32Array, gain: number): void {
    if (this.seenTotal === 0) return;
    const lit: number[] = [];
    for (let i = 0; i < this.inputCount; i++) {
      if (this.seenTicks[i] * 2 >= this.seenTotal) lit.push(i);
    }
    this.seenTicks.fill(0);
    this.seenTotal = 0;
    if (lit.length === 0) return;

    const plasticity = Math.max(0, gain);
    this.experiences++;
    const cellStep = Math.max(HandMotorCortex.MIN_STEP, 1 / this.experiences);
    for (let m = 0; m < this.neuronCount; m++) {
      const inked = pattern[m] > 0 ? 1 : 0;
      this.inkRate[m] += cellStep * (inked - this.inkRate[m]);
    }
    for (const channel of lit) {
      // Running mean with a sceptical prior: a channel seen once or twice says
      // little (its few co-occurrences would otherwise count as certainties and
      // whole old scribbles would reappear inside later drawings).
      const n = ++this.channelCount[channel];
      const step = Math.min(1, Math.max(HandMotorCortex.MIN_STEP, 1 / (n + HandMotorCortex.PRIOR_COUNT)) * plasticity);
      for (let m = 0; m < this.neuronCount; m++) {
        const idx = m * this.inputCount + channel;
        this.weights[idx] += step * ((pattern[m] > 0 ? 1 : 0) - this.weights[idx]);
      }
    }
    this.learnings++;
    let known = 0;
    for (let i = 0; i < this.inputCount; i++) if (this.channelCount[i] > 0) known++;
    this.knowledge = known / this.inputCount;
  }
}
