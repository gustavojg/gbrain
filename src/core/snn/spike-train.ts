/**
 * SPIKE ENCODING — Utilities
 * ====================================
 * Converts continuous signals into spike trains and vice versa.
 *
 * The biological brain does not work with continuous numbers — it works with
 * discrete pulses (spikes). These methods convert between the two domains.
 *
 * Two main schemes:
 * - Rate Coding: Intensity is encoded as the FREQUENCY of spikes
 * - Temporal Coding: Intensity is encoded as the TIMING of the spike
 */

/**
 * Rate Coding: Converts a continuous value into a spike probability.
 * Higher values → more likely to fire in this timestep.
 *
 * Biology: Sensory neurons encode stimulus intensity
 * as a firing rate (e.g. more pressure → more spikes per second).
 *
 * @param value - Continuous value to encode (0-1 normalized)
 * @param maxRate - Maximum firing rate (Hz)
 * @param dt - Current time step (ms)
 * @returns 1 if it generates a spike, 0 otherwise
 */
export function rateCoding(value: number, maxRate: number = 100, dt: number = 1.0): number {
  // Spike probability in this timestep
  // P(spike) = rate * dt / 1000, where rate = value * maxRate
  const rate = Math.max(0, Math.min(1, value)) * maxRate;
  const probability = rate * dt / 1000;
  return Math.random() < probability ? 1.0 : 0.0;
}

/**
 * Temporal Coding: Converts a value into a spike time.
 * Higher values → spike EARLIER in the window.
 *
 * Biology: Some neurons encode information in the EXACT MOMENT
 * of the spike relative to a reference oscillation. This is more efficient
 * than rate coding for a single presentation of the stimulus.
 *
 * @param value - Continuous value (0-1 normalized)
 * @param maxValue - Maximum possible value
 * @param windowMs - Temporal window for the spike (ms)
 * @returns Time of the spike within the window (ms), -1 if it does not fire
 */
export function temporalCoding(value: number, maxValue: number = 1.0, windowMs: number = 20): number {
  const normalized = Math.max(0, Math.min(1, value / maxValue));
  if (normalized < 0.05) return -1; // Below threshold, does not fire
  // High value → early spike (low time)
  return windowMs * (1 - normalized);
}

/**
 * Encodes a vector of continuous values as a spike vector (0/1).
 * Uses rate coding for each component.
 *
 * @param values - Vector of continuous values (Float32Array)
 * @param dt - Time step (ms)
 * @param maxRate - Maximum firing rate (Hz)
 * @returns Spike vector (Float32Array of 0s and 1s)
 */
export function encodeSpikeVector(
  values: Float32Array, 
  dt: number = 1.0, 
  maxRate: number = 100
): Float32Array {
  const spikes = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    spikes[i] = rateCoding(values[i], maxRate, dt);
  }
  return spikes;
}

/**
 * Decodes a spike train into a firing rate (continuous value).
 *
 * @param spikeTrain - Array of 0s and 1s representing spikes in successive timesteps
 * @param windowMs - Temporal window over which to average (ms)
 * @param dt - Time step between samples (ms)
 * @returns Estimated firing rate (0-1 normalized)
 */
export function decodeSpikeRate(
  spikeTrain: Float32Array, 
  windowMs: number = 50, 
  dt: number = 1.0
): number {
  const windowSamples = Math.floor(windowMs / dt);
  const recentSpikes = spikeTrain.slice(-windowSamples);
  
  let count = 0;
  for (let i = 0; i < recentSpikes.length; i++) {
    if (recentSpikes[i] > 0.5) count++;
  }
  
  return count / recentSpikes.length;
}

/**
 * Encodes an image (array of pixels) as a spike vector
 * using rate coding with normalization.
 *
 * @param pixels - Pixel values (0-255 or 0-1)
 * @param normalize - Whether to normalize from 0-255 to 0-1
 * @returns Spike vector
 */
export function encodeImageToSpikes(
  pixels: Float32Array | number[],
  normalize: boolean = true,
  dt: number = 1.0
): Float32Array {
  const values = new Float32Array(pixels.length);

  for (let i = 0; i < pixels.length; i++) {
    values[i] = normalize ? pixels[i] / 255 : pixels[i];
  }

  return encodeSpikeVector(values, dt, 200); // Higher rate for visual
}

/**
 * Encodes an audio spectrogram as a spike vector.
 * Uses a logarithmic scale to mimic the cochlear response.
 *
 * @param spectrogram - Flat array of the spectrogram [time][freq]
 * @param numBands - Number of frequency bands
 * @param numFrames - Number of temporal frames
 * @returns Spike vector
 */
export function encodeAudioToSpikes(
  spectrogram: Float32Array | number[],
  numBands: number = 40,
  numFrames: number = 20,
  dt: number = 1.0
): Float32Array {
  const values = new Float32Array(spectrogram.length);

  for (let i = 0; i < spectrogram.length; i++) {
    // Logarithmic scale (mimics the cochlear response)
    values[i] = Math.log1p(Math.abs(spectrogram[i] as number)) / Math.log1p(1);
    values[i] = Math.min(1, values[i]);
  }

  return encodeSpikeVector(values, dt, 150); // Moderate rate for audio
}

/**
 * Encodes text as a spike vector using a distributed hash.
 * Each character is mapped to a sparse pattern of activations.
 *
 * Inspired by "random projections" — a real technique used in
 * computational neuroscience for distributed representations.
 *
 * @param text - Text to encode
 * @param vectorSize - Size of the output vector
 * @returns Spike vector
 */
export function encodeTextToSpikes(
  text: string,
  vectorSize: number = 5000,
  dt: number = 1.0
): Float32Array {
  const values = new Float32Array(vectorSize);
  
  // For each character, activate a deterministic subset of positions
  const chars = text.toLowerCase().split('');

  for (let ci = 0; ci < chars.length; ci++) {
    const charCode = chars[ci].charCodeAt(0);
    const position = ci; // Position in the sequence

    // Deterministic hash: each (character, position) activates ~5% of the vector
    // We use a seeded pseudo-random generator
    let seed = charCode * 31 + position * 7919;
    const activations = Math.floor(vectorSize * 0.05);

    for (let a = 0; a < activations; a++) {
      // Simple hash-based pseudo-random
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      const idx = seed % vectorSize;
      values[idx] = Math.min(1.0, values[idx] + 0.3);
    }
  }

  // Normalize
  let max = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }
  if (max > 0) {
    for (let i = 0; i < values.length; i++) {
      values[i] /= max;
    }
  }
  
  return encodeSpikeVector(values, dt, 100);
}
