/**
 * VOCAL MOTOR CORTEX — Learning to make the sounds it hears
 * =========================================================
 * A map of motor units over the articulators of the vocal tract, fed by the
 * auditory cortex (the dorsal auditory–motor stream). It learns, from its own
 * babbling, which motor pattern produces which sound — and can then run that
 * knowledge backwards to imitate a sound it hears.
 *
 * Biological basis:
 *   - Motor units have a PREFERRED articulatory posture, as motor-cortex
 *     neurons have preferred directions; a population of active units encodes
 *     the posture as their activity-weighted average (population vector;
 *     Georgopoulos et al., 1986).
 *   - BABBLING: spontaneous motor activity produces a sound; the infant hears
 *     it; the motor units that were active and the auditory neurons that answer
 *     fire together, and the synapses between them strengthen (Hebb). This
 *     builds an auditory → motor map without any teacher (Westermann & Miranda,
 *     2004; the DIVA model, Guenther, 2006).
 *   - IMITATION: a heard sound drives, through those synapses, the motor units
 *     that used to produce something like it.
 *   - While vocalizing, what arrives from the auditory cortex is the
 *     consequence of its own command: it is used to learn, never as something
 *     to imitate (efference copy; otherwise the brain would echo itself forever).
 *
 * The weights of this region ARE the auditory → motor map (motor units ×
 * auditory neurons), so they persist with the rest of the synaptic state.
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { commandFromArticulators, ONSETS, type VocalCommand } from '../../core/voice/vocal-tract.js';
import { packArray, unpackFloat32 } from '../../core/persistence/binary-protocol.js';

export interface MotorCortexConfig {
  /** Motor units per side of the (square) articulatory map. */
  mapSide: number;
  /** Neurons of the auditory cortex (dimension of the input). */
  inputCount: number;
  /** Radius (in map cells) of the bump of units active for one posture. */
  bumpRadius: number;
  /** Ticks a command is held: the utterance plus the time its sound takes to come back. */
  holdTicks: number;
  /** Hebbian learning rate (per tick of coincidence). */
  learningRate: number;
  /** LTD, relative to the learning rate, for synapses whose sound came from other units. */
  depressionRatio: number;
  /** Minimum learned drive (0–1) on the best motor unit to attempt an imitation. */
  imitationThreshold: number;
  /** Ticks of silence that end a heard sound (then the imitation is issued). */
  planGapTicks: number;
  /**
   * Heard ticks at the start of an utterance during which the onset (a murmur,
   * a burst) is what is heard — one frame: the lips learn from, and read, only those.
   */
  onsetTicks: number;
}

const DEFAULT_MOTOR_CONFIG: MotorCortexConfig = {
  mapSide: 16,
  inputCount: 1000,
  bumpRadius: 1.6,
  holdTicks: 50,
  learningRate: 0.05,
  depressionRatio: 0.5,
  imitationThreshold: 0.25,
  planGapTicks: 10,
  onsetTicks: 2,
};

/** A command issued by the motor cortex, with why it was issued. */
export interface MotorOutput {
  command: VocalCommand;
  source: 'babble' | 'imitation' | 'naming' | 'call';
  /** For an imitation or a naming: how strongly the sound drove the map (0–1). */
  confidence: number;
}

export class MotorCortex extends BrainRegion {
  private readonly cfg: MotorCortexConfig;

  /** Motor pattern being executed (activity per unit), or `null` when not vocalizing. */
  private held: Float32Array | null = null;
  private heldTicks = 0;
  /**
   * The lips: a few units for how an utterance begins (open, closed with a
   * nasal murmur, closed and released). They learn the auditory → motor map
   * as the posture units do, so that a heard /m/ or /p/ drives the closure
   * that produced it. Kept apart from the posture map so that a saved map
   * of an older brain still loads.
   */
  private readonly onsetWeights: Float32Array;
  private heldOnset = 0;
  /** Ticks with something heard since the held command began: its first ones carry the onset. */
  private heldHeardTicks = 0;
  private readonly onsetPlan = new Float32Array(ONSETS.length);
  /** Share of babbles that begin with the lips closed (canonical babbling: "ma", "pa"). */
  private static readonly ONSET_BABBLE_SHARE = 0.5;

  /** Motor activity accumulated while an EXTERNAL sound is being heard. */
  private readonly plan: Float32Array;
  private planTicks = 0;
  private planSilentTicks = 0;
  private planPeakDrive = 0;

  /** Imitation ready to be executed (taken by the brain with `takeCommand`). */
  private pending: MotorOutput | null = null;

  /** Babbles produced so far (for monitoring). */
  babbleCount = 0;
  /**
   * Commands learned from, and what the map knows after the last one: the
   * fraction of its units that have produced a sound and learned what it
   * sounds like (0..1). Its growth per command is learning progress; it
   * saturates as the postures repeat — mastery.
   */
  learnings = 0;
  knowledge = 0;
  private heldTicksLearned = 0;
  /** Weight from which a (unit, heard unit) pair counts as known. */
  private static readonly KNOWN_WEIGHT = 0.1;

  /** Whether a heard sound that the map knows is repeated aloud. */
  imitate = false;
  /** What the last heard sound did to the map (for the dashboard and tests): peak drive and onset drives. */
  lastListening: { peak: number; onsets: number[]; ticks: number } | null = null;

  constructor(config: Partial<MotorCortexConfig> = {}) {
    const cfg = { ...DEFAULT_MOTOR_CONFIG, ...config };
    super('motorCortex', 'Corteza Motora Vocal', cfg.mapSide * cfg.mapSide, cfg.inputCount);
    this.cfg = cfg;
    this.plan = new Float32Array(this.neuronCount);
    this.onsetWeights = new Float32Array(ONSETS.length * cfg.inputCount);
    // The map starts blank: nothing is known about what any command sounds like.
    this.weights.fill(0);
  }

  override serializeExtra(): unknown {
    return { onsetWeights: packArray(this.onsetWeights) };
  }

  override deserializeExtra(data: unknown): void {
    if (typeof data !== 'object' || data === null) return;
    const w = unpackFloat32((data as Record<string, unknown>).onsetWeights, this.onsetWeights.length);
    if (w) this.onsetWeights.set(w);
  }

  /** The auditory → motor map is learned from scratch; no random synapses. */
  protected override initializeWeights(): void {
    // Intentionally empty (see constructor).
  }

  // ----------------------------------------------------------------
  // Motor patterns ↔ articulator postures
  // ----------------------------------------------------------------

  /** Bump of active units around a posture (x, y ∈ [0, 1]). */
  private bumpAt(x: number, y: number): Float32Array {
    const side = this.cfg.mapSide;
    const pattern = new Float32Array(this.neuronCount);
    const cx = x * (side - 1);
    const cy = y * (side - 1);
    const r2 = this.cfg.bumpRadius * this.cfg.bumpRadius;
    for (let row = 0; row < side; row++) {
      for (let col = 0; col < side; col++) {
        const d2 = (col - cx) ** 2 + (row - cy) ** 2;
        if (d2 <= r2 * 4) pattern[row * side + col] = Math.exp(-d2 / (2 * r2));
      }
    }
    return pattern;
  }

  /** Population vector: the posture encoded by a pattern of motor activity. */
  private decode(activity: Float32Array, onset: number = 0): VocalCommand | null {
    const side = this.cfg.mapSide;
    let total = 0;
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < activity.length; i++) {
      const a = activity[i];
      if (a <= 0) continue;
      total += a;
      sx += a * ((i % side) / (side - 1));
      sy += a * (Math.floor(i / side) / (side - 1));
    }
    return total > 0 ? commandFromArticulators(sx / total, sy / total, 0.9, ONSETS[onset] ?? null) : null;
  }

  // ----------------------------------------------------------------
  // Vocalizing
  // ----------------------------------------------------------------

  /** Whether a command is being executed (own voice expected from the auditory cortex). */
  get vocalizing(): boolean {
    return this.held !== null;
  }

  /**
   * Spontaneous motor activity: a random posture is tried out. Its sound will
   * come back through the auditory cortex while the pattern is still held,
   * which is when the map learns.
   */
  babble(): MotorOutput {
    this.babbleCount++;
    const r = Math.random();
    const onset = r < MotorCortex.ONSET_BABBLE_SHARE ? 1 + Math.floor((r / MotorCortex.ONSET_BABBLE_SHARE) * (ONSETS.length - 1)) : 0;
    return this.execute(this.bumpAt(Math.random(), Math.random()), 'babble', 0, onset);
  }

  private execute(pattern: Float32Array, source: MotorOutput['source'], confidence: number, onset: number = 0): MotorOutput {
    this.held = pattern;
    this.heldOnset = onset;
    this.heldHeardTicks = 0;
    this.heldTicks = this.cfg.holdTicks;
    this.resetPlan();
    return { command: this.decode(pattern, onset)!, source, confidence };
  }

  /** The onset the heard sound drives most (0 = none), from the map of the lips. */
  private onsetDrive(heard: ArrayLike<number>): Float32Array {
    const drive = new Float32Array(ONSETS.length);
    for (let o = 0; o < ONSETS.length; o++) {
      const offset = o * this.inputCount;
      let sum = 0;
      for (let h = 0; h < heard.length; h++) sum += this.onsetWeights[offset + heard[h]];
      drive[o] = heard.length > 0 ? sum / heard.length : 0;
    }
    return drive;
  }

  private static bestOnset(plan: Float32Array): number {
    let best = 0;
    for (let o = 1; o < plan.length; o++) if (plan[o] > plan[best]) best = o;
    return plan[best] > 0 ? best : 0;
  }

  /**
   * Says a sound that is only IMAGINED — an auditory engram reinstated from
   * memory (the sound that goes with what is being seen), not one arriving
   * through the ear. It drives the map exactly as a heard sound would.
   *
   * @param auditoryUnits - Neurons of the auditory cortex that stand for the sound
   * @returns The command being executed, or `null` if the voice is off, busy,
   *   or the map does not know how to make that sound yet
   */
  sayImagined(auditoryUnits: ArrayLike<number>): MotorOutput | null {
    if (!this.imitate || this.held || auditoryUnits.length === 0) return null;

    const drive = new Float32Array(this.neuronCount);
    let peak = 0;
    for (let m = 0; m < this.neuronCount; m++) {
      const offset = m * this.inputCount;
      let sum = 0;
      for (let h = 0; h < auditoryUnits.length; h++) sum += this.weights[offset + auditoryUnits[h]];
      drive[m] = sum / auditoryUnits.length;
      if (drive[m] > peak) peak = drive[m];
    }
    if (peak < this.cfg.imitationThreshold) return null;

    for (let m = 0; m < this.neuronCount; m++) if (drive[m] < peak * 0.5) drive[m] = 0;
    return this.execute(this.normalized(drive), 'naming', Math.min(1, peak), MotorCortex.bestOnset(this.onsetDrive(auditoryUnits)));
  }

  /** Takes the imitation that became ready on this tick, if any, and starts executing it. */
  takeCommand(): MotorOutput | null {
    const output = this.pending;
    this.pending = null;
    return output;
  }

  // ----------------------------------------------------------------
  // Main processing
  // ----------------------------------------------------------------

  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    const heard: number[] = [];
    for (let i = 0; i < spikes.length; i++) if (spikes[i] > 0) heard.push(i);

    // --- Vocalizing: what is heard is the consequence of the held command ---
    if (this.held) {
      if (heard.length > 0) this.learn(this.held, heard, modulationEffects.learningRateMultiplier ?? 1, this.heldHeardTicks++ < this.cfg.onsetTicks);
      const output = new Float32Array(this.held.length);
      for (let i = 0; i < output.length; i++) output[i] = this.held[i] > 0.5 ? 1 : 0;
      if (--this.heldTicks <= 0) {
        // The command is over: one report of how much the map moved for it.
        this.held = null;
        if (this.heldTicksLearned > 0) {
          this.learnings++;
          this.knowledge = this.measureKnowledge();
        }
        this.heldTicksLearned = 0;
      }
      this.spikes.set(output);
      return output;
    }

    // --- Listening: a heard sound drives the units that used to produce it ---
    const output = new Float32Array(this.neuronCount);
    if (heard.length > 0) {
      let peak = 0;
      const drive = new Float32Array(this.neuronCount);
      for (let m = 0; m < this.neuronCount; m++) {
        const offset = m * this.inputCount;
        let sum = 0;
        for (let h = 0; h < heard.length; h++) sum += this.weights[offset + heard[h]];
        drive[m] = sum / heard.length;
        if (drive[m] > peak) peak = drive[m];
      }
      if (peak > 0) {
        // Units driven at least half as much as the best one fire.
        for (let m = 0; m < this.neuronCount; m++) {
          if (drive[m] >= peak * 0.5) {
            output[m] = 1;
            this.plan[m] += drive[m];
          }
        }
      }
      // The onset is what is heard at the start: the lips read only those ticks.
      if (this.planTicks < this.cfg.onsetTicks) {
        const onsetDrive = this.onsetDrive(heard);
        for (let o = 0; o < onsetDrive.length; o++) this.onsetPlan[o] += onsetDrive[o];
      }
      this.planTicks++;
      this.planSilentTicks = 0;
      if (peak > this.planPeakDrive) this.planPeakDrive = peak;
    } else if (this.planTicks > 0 && ++this.planSilentTicks >= this.cfg.planGapTicks) {
      // The sound is over: if the map knew it well enough, imitate it.
      this.lastListening = { peak: this.planPeakDrive, onsets: Array.from(this.onsetPlan, (v) => v / Math.max(1, Math.min(this.planTicks, this.cfg.onsetTicks))), ticks: this.planTicks };
      if (this.imitate && this.planPeakDrive >= this.cfg.imitationThreshold) {
        const pattern = this.normalized(this.plan);
        const onset = MotorCortex.bestOnset(this.onsetPlan);
        this.pending = this.execute(pattern, 'imitation', Math.min(1, this.planPeakDrive), onset);
      }
      this.resetPlan();
    }

    this.spikes.set(output);
    return output;
  }

  private resetPlan(): void {
    this.plan.fill(0);
    this.onsetPlan.fill(0);
    this.planTicks = 0;
    this.planSilentTicks = 0;
    this.planPeakDrive = 0;
  }

  private normalized(activity: Float32Array): Float32Array {
    let peak = 0;
    for (let i = 0; i < activity.length; i++) if (activity[i] > peak) peak = activity[i];
    const out = new Float32Array(activity.length);
    if (peak > 0) for (let i = 0; i < activity.length; i++) out[i] = activity[i] / peak;
    return out;
  }

  /**
   * Hebbian learning of the auditory → motor map while vocalizing.
   *   - LTP: synapses from the auditory neurons that are firing onto the motor
   *     units that are active move toward the unit's activity.
   *   - LTD: the same auditory neurons onto units that are NOT active weaken —
   *     if that sound is now being produced from here, it was not (only)
   *     produced from there.
   */
  private learn(pattern: Float32Array, heard: number[], gain: number, onsetPhase: boolean = true): void {
    const lr = Math.min(1, this.cfg.learningRate * Math.max(0, gain));
    const ltd = lr * this.cfg.depressionRatio;
    for (let m = 0; m < this.neuronCount; m++) {
      const offset = m * this.inputCount;
      const activity = pattern[m];
      for (let h = 0; h < heard.length; h++) {
        const idx = offset + heard[h];
        const w = this.weights[idx];
        if (activity > 0.05) this.weights[idx] = w + lr * (activity - w);
        else if (w > 0) this.weights[idx] = w * (1 - ltd);
      }
    }
    // The lips learn the same way — from the start of the utterance, when the
    // onset is what comes back: the onset unit that is closed (or open) now
    // binds to what is heard; the others let go of it.
    for (let o = 0; onsetPhase && o < ONSETS.length; o++) {
      const offset = o * this.inputCount;
      const activity = o === this.heldOnset ? 1 : 0;
      for (let h = 0; h < heard.length; h++) {
        const idx = offset + heard[h];
        const w = this.onsetWeights[idx];
        if (activity > 0) this.onsetWeights[idx] = w + lr * (activity - w);
        else if (w > 0) this.onsetWeights[idx] = w * (1 - ltd);
      }
    }
    if (heard.length > 0) this.heldTicksLearned++;
  }

  /** Whether some map unit knows this heard (auditory) unit: the map runs on it. */
  knowsHeard(heard: number): boolean {
    if (heard < 0 || heard >= this.inputCount) return false;
    for (let m = 0; m < this.neuronCount; m++) if (this.weights[m * this.inputCount + heard] >= MotorCortex.KNOWN_WEIGHT) return true;
    return false;
  }

  /** Fraction of map units that know what they sound like (some weight to some heard unit). */
  private measureKnowledge(): number {
    let known = 0;
    for (let m = 0; m < this.neuronCount; m++) {
      const offset = m * this.inputCount;
      for (let h = 0; h < this.inputCount; h++) {
        if (this.weights[offset + h] >= MotorCortex.KNOWN_WEIGHT) { known++; break; }
      }
    }
    return known / this.neuronCount;
  }
}
