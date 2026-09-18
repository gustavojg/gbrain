/**
 * PROSODY — the innate reading of a voice's tone
 * ===========================================================================
 * What a newborn brings to the world is not the meaning of words but a
 * sensitivity to HOW they are said. Fernald (1993) showed that five-month-olds
 * respond correctly to approval and prohibition spoken in languages they have
 * never heard: approval is slow, high, with a smoothly rising or bell-shaped
 * pitch contour; prohibition is short, low, loud and staccato. Roughness —
 * fast amplitude modulation, the texture of screams and harsh noise — alarms
 * every mammal without learning (Arnal et al., 2015).
 *
 * This module is that innate layer: a fixed mapping from the envelope and
 * pitch track of an utterance to a valence/arousal appraisal. It lives on the
 * fast thalamus → amygdala route, before any cortex has classified the sound.
 * Nothing here is learned; nothing here knows a word.
 */

/** The envelope and pitch track of one utterance, as the ear delivers them. */
export interface VoiceContour {
  /** RMS level of each frame, 0..1 (silence ≈ 0). */
  rms: number[];
  /** Fundamental frequency of each frame in Hz; 0 where unvoiced. */
  f0: number[];
  /** Duration of one frame in ms. */
  frameMs: number;
}

/** What the innate detectors extracted from the contour (all 0..1 unless noted). */
export interface ProsodyFeatures {
  /** Mean level over the active frames. */
  loudness: number;
  /** How abruptly the sound starts: level reached in the first ~100 ms relative to its peak. */
  abruptness: number;
  /** Frame-to-frame level fluctuation: staccato, harshness. */
  roughness: number;
  /** Mean pitch in semitones above the speaker's usual pitch (−12..12, not normalized). */
  pitchHeight: number;
  /** Pitch movement in semitones: rising (>0), falling (<0). */
  pitchSlope: number;
  /** Bell-shaped contour: rises then falls (0..1). */
  bell: number;
  /** Pitch variability in semitones (std). */
  pitchRange: number;
  /** Duration of the utterance in ms. */
  durationMs: number;
  /** Fraction of active frames that are voiced. */
  voiced: number;
}

export interface ProsodyAppraisal {
  /** −1 (harsh, prohibitive) .. 1 (warm, approving). */
  valence: number;
  /** 0 (calm) .. 1 (alarming). */
  arousal: number;
  /** A sudden loud onset: the acoustic startle reflex fires. */
  startle: boolean;
  features: ProsodyFeatures;
}

/** Level below which a frame is silence. */
const ACTIVE_RMS = 0.015;
/** Level that counts as loud (RMS of speech close to a microphone ≈ 0.1–0.3). */
const LOUD_RMS = 0.25;
/** A start this abrupt AND this loud is a startle. */
const STARTLE_ABRUPTNESS = 0.6;
const STARTLE_LOUDNESS = 0.6;
/** Shortest utterance (ms) that can be appraised at all. */
const MIN_DURATION_MS = 80;

const semitones = (hz: number, ref: number): number => 12 * Math.log2(hz / ref);
const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));

/**
 * Extracts the innate prosodic features of a contour.
 *
 * @param contour - Envelope and pitch track of the utterance
 * @param speakerPitchHz - The speaker's usual pitch (adapts over time; ~150 Hz to start)
 */
export function extractProsody(contour: VoiceContour, speakerPitchHz: number): ProsodyFeatures | null {
  const frameMs = clamp(contour.frameMs, 5, 200);
  const n = Math.min(contour.rms.length, contour.f0.length);
  const active: number[] = [];
  for (let i = 0; i < n; i++) if (contour.rms[i] >= ACTIVE_RMS) active.push(i);
  if (active.length === 0) return null;
  const first = active[0];
  const last = active[active.length - 1];
  const durationMs = (last - first + 1) * frameMs;
  if (durationMs < MIN_DURATION_MS) return null;

  let peak = 0;
  let sum = 0;
  for (let i = first; i <= last; i++) {
    peak = Math.max(peak, contour.rms[i]);
    sum += contour.rms[i];
  }
  const loudness = clamp(sum / (last - first + 1) / LOUD_RMS, 0, 1);

  // Abruptness: how much of the peak is reached within the first ~100 ms.
  const onsetFrames = Math.max(1, Math.round(100 / frameMs));
  let onsetPeak = 0;
  for (let i = first; i < Math.min(last + 1, first + onsetFrames); i++) onsetPeak = Math.max(onsetPeak, contour.rms[i]);
  const abruptness = peak > 0 ? clamp(onsetPeak / peak, 0, 1) : 0;

  // Roughness: mean absolute frame-to-frame change relative to the mean level.
  let fluctuation = 0;
  for (let i = first + 1; i <= last; i++) fluctuation += Math.abs(contour.rms[i] - contour.rms[i - 1]);
  const mean = sum / (last - first + 1);
  const roughness = last > first && mean > 0 ? clamp(fluctuation / (last - first) / mean / 0.6, 0, 1) : 0;

  // Pitch, in semitones relative to the speaker's usual pitch.
  const pitches: Array<[number, number]> = []; // [frame, semitones]
  for (let i = first; i <= last; i++) {
    const hz = contour.f0[i];
    if (hz >= 50 && hz <= 800 && contour.rms[i] >= ACTIVE_RMS) pitches.push([i, semitones(hz, speakerPitchHz)]);
  }
  const voiced = pitches.length / (last - first + 1);
  let pitchHeight = 0;
  let pitchSlope = 0;
  let bell = 0;
  let pitchRange = 0;
  if (pitches.length >= 2) {
    const values = pitches.map((p) => p[1]);
    pitchHeight = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, v) => a + (v - pitchHeight) ** 2, 0) / values.length;
    pitchRange = Math.sqrt(variance);
    // Slope: least squares over time, in semitones over the whole utterance.
    const t0 = pitches[0][0];
    const tMean = pitches.reduce((a, p) => a + (p[0] - t0), 0) / pitches.length;
    let num = 0;
    let den = 0;
    for (const [t, v] of pitches) {
      num += (t - t0 - tMean) * (v - pitchHeight);
      den += (t - t0 - tMean) ** 2;
    }
    pitchSlope = den > 0 ? (num / den) * (pitches[pitches.length - 1][0] - t0) : 0;
    // Bell: the maximum lies inside and both ends are clearly below it.
    let maxIdx = 0;
    for (let k = 1; k < values.length; k++) if (values[k] > values[maxIdx]) maxIdx = k;
    const rise = values[maxIdx] - values[0];
    const fall = values[maxIdx] - values[values.length - 1];
    if (maxIdx > 0 && maxIdx < values.length - 1) bell = clamp(Math.min(rise, fall) / 3, 0, 1);
  }

  return { loudness, abruptness, roughness, pitchHeight, pitchSlope, bell, pitchRange, durationMs, voiced };
}

/**
 * The innate appraisal of a voice: warm or harsh, calm or alarming, and
 * whether it startles. A fixed mapping; the weights are the "wiring".
 */
export function appraiseProsody(contour: VoiceContour, speakerPitchHz: number = 150): ProsodyAppraisal | null {
  const f = extractProsody(contour, speakerPitchHz);
  if (!f) return null;

  const high = clamp(f.pitchHeight / 6, -1, 1); // ±6 semitones from usual saturates
  const rising = clamp(f.pitchSlope / 4, -1, 1);
  const longUtterance = clamp((f.durationMs - 300) / 900, 0, 1); // 300 ms → 0, 1.2 s → 1
  const shortAndLoud = (1 - longUtterance) * f.loudness;
  const lively = clamp(f.pitchRange / 4, 0, 1);
  const softness = clamp((0.4 - f.loudness) / 0.4, 0, 1); // clearly quiet, not merely not loud

  // Warm: high, rising or bell-shaped, melodious, unhurried, soft. A flat,
  // medium voice scores nothing here — neutrality is the absence of cues.
  const warmth =
    0.35 * Math.max(0, high) +
    0.30 * Math.max(0, rising) +
    0.30 * f.bell +
    0.20 * lively +
    0.10 * longUtterance +
    0.20 * softness * (1 - f.roughness);
  // Harsh: loud, abrupt (when loud), rough, low, falling, short and loud.
  const harshness =
    0.45 * f.loudness +
    0.35 * f.abruptness * f.loudness +
    0.45 * f.roughness +
    0.30 * Math.max(0, -high) +
    0.20 * Math.max(0, -rising) +
    0.30 * shortAndLoud;

  const valence = clamp(warmth - harshness, -1, 1);
  const arousal = clamp(0.15 + 0.45 * f.loudness + 0.25 * f.abruptness * f.loudness + 0.3 * f.roughness + 0.15 * lively, 0, 1);
  const startle = f.abruptness >= STARTLE_ABRUPTNESS && f.loudness >= STARTLE_LOUDNESS;
  return { valence, arousal, startle, features: f };
}
