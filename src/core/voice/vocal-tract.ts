/**
 * VOCAL TRACT — What a motor command sounds like
 * ==============================================
 * A deliberately small articulatory model: a voiced source shaped by two
 * formants. The motor cortex does not know any of this; it only finds out what
 * its commands sound like by producing them and listening (babbling).
 *
 * Two consumers share this definition so that they agree on what a command is:
 *   - the brain itself, to hear its own voice internally (the acoustic
 *     consequence of a command, as a frame of FFT magnitudes), and
 *   - the dashboard, which renders the same command audibly with WebAudio.
 *
 * Biology: the first two formants (F1 ~ jaw opening / tongue height, F2 ~
 * tongue advancement) are enough to tell vowels apart (Peterson & Barney,
 * 1952), and they are what infants explore when they babble vowel-like sounds.
 */

/**
 * How an utterance begins: with the lips closed and the velum open (a nasal
 * murmur, /m/), with the lips closed and released (a stop burst, /p/), or
 * open from the start (a bare vowel). The first consonants of babbling.
 */
export type Onset = 'nasal' | 'stop';

/** The onsets a command can have, in the order the motor cortex numbers them (0 = none). */
export const ONSETS: ReadonlyArray<Onset | null> = [null, 'nasal', 'stop'];

/** Articulatory command, in acoustic terms. */
export interface VocalCommand {
  /** First formant (Hz). */
  f1: number;
  /** Second formant (Hz). */
  f2: number;
  /** Loudness, 0–1. */
  amplitude: number;
  /** Lip closure before the vowel, if any. */
  onset?: Onset;
}

/** Duration of one frame of an utterance (the onset lasts one; the vowel follows). */
export const UTTERANCE_FRAME_MS = 200;

/** Range of the articulators: the vowel space the tract can produce. */
export const VOCAL_RANGE = {
  f1: [250, 900],
  f2: [650, 2600],
} as const;

/** Bandwidths (standard deviation, Hz) of the two formant peaks. */
const FORMANT_WIDTH = { f1: 90, f2: 120 } as const;

/** Relative level of the second formant. */
const F2_LEVEL = 0.78;

/** Maps normalized articulator positions (0–1, 0–1) to a command. */
export function commandFromArticulators(x: number, y: number, amplitude: number = 0.9, onset: Onset | null = null): VocalCommand {
  const clamp = (v: number): number => Math.max(0, Math.min(1, v));
  const command: VocalCommand = {
    f1: VOCAL_RANGE.f1[0] + clamp(x) * (VOCAL_RANGE.f1[1] - VOCAL_RANGE.f1[0]),
    f2: VOCAL_RANGE.f2[0] + clamp(y) * (VOCAL_RANGE.f2[1] - VOCAL_RANGE.f2[0]),
    amplitude: clamp(amplitude),
  };
  if (onset) command.onset = onset;
  return command;
}

/**
 * The sound of a command, as one frame of linear FFT magnitudes in [0, 1] —
 * the same format a microphone frame has (`DigitalBrain.hearFrame`).
 *
 * @param command - Articulatory command
 * @param bins - Number of FFT bins (frame length)
 * @param sampleRate - Sample rate the bins refer to (Hz)
 */
export function synthesizeSpectrum(command: VocalCommand, bins: number = 512, sampleRate: number = 48000): Float32Array {
  const spectrum = new Float32Array(bins);
  const binHz = sampleRate / 2 / bins;
  const peak = (hz: number, centre: number, width: number): number =>
    Math.exp(-((hz - centre) ** 2) / (2 * width * width));
  for (let bin = 0; bin < bins; bin++) {
    const hz = bin * binHz;
    const level = peak(hz, command.f1, FORMANT_WIDTH.f1) + F2_LEVEL * peak(hz, command.f2, FORMANT_WIDTH.f2);
    spectrum[bin] = Math.min(1, command.amplitude * level);
  }
  return spectrum;
}

/**
 * The sound of a command over time: one frame per `UTTERANCE_FRAME_MS`. A
 * bare vowel is one frame. A nasal onset is a murmur first — the lips
 * closed, the sound leaving through the nose: a low nasal formant and
 * little else — then the vowel. A stop onset is a release burst first —
 * brief, broadband, aperiodic — then the vowel. (Closure itself is silence
 * and needs no frame.)
 */
export function synthesizeFrames(command: VocalCommand, bins: number = 512, sampleRate: number = 48000): Float32Array[] {
  const vowel = synthesizeSpectrum(command, bins, sampleRate);
  if (!command.onset) return [vowel];
  const onset = new Float32Array(bins);
  const binHz = sampleRate / 2 / bins;
  for (let bin = 0; bin < bins; bin++) {
    const hz = bin * binHz;
    if (command.onset === 'nasal') {
      // The nasal murmur: a low nasal formant (~300 Hz) and the weak, damped
      // resonances of the nasal tract around 1 and 2.3 kHz (Fujimura 1962).
      const murmur =
        Math.exp(-((hz - 300) ** 2) / (2 * 90 * 90)) +
        0.4 * Math.exp(-((hz - 1000) ** 2) / (2 * 200 * 200)) +
        0.25 * Math.exp(-((hz - 2300) ** 2) / (2 * 250 * 250));
      onset[bin] = Math.min(1, 0.6 * command.amplitude * murmur);
    } else {
      // A burst: flat, slightly rising with frequency, from ~500 Hz up.
      onset[bin] = hz < 500 ? 0 : Math.min(1, 0.35 * command.amplitude * (0.6 + 0.4 * Math.min(1, hz / 4000)));
    }
  }
  return [onset, vowel];
}

/** Perceptual distance between two commands: mean formant error in Hz. */
export function formantError(a: VocalCommand, b: VocalCommand): number {
  return (Math.abs(a.f1 - b.f1) + Math.abs(a.f2 - b.f2)) / 2;
}
