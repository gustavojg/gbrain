/**
 * AUDITORY ENCODER — Audio to spikes encoding
 * ==================================================
 * Transforms audio signals (microphone, files) into spike trains
 * for the auditory cortex.
 *
 * Bio-inspired pipeline (artificial cochlea):
 * 1. FFT: Decomposition into frequencies (analogous to the basilar membrane)
 * 2. Cochlear bands: Logarithmic grouping of frequencies
 * 3. Sliding window: Temporal spectrogram
 * 4. Rate coding: Conversion to spike frequency
 *
 * Refactored from 08_microphone_interaction/client.ts and 07_audio_cortex/server.ts
 */

import { encodeSpikeVector } from '../core/snn/spike-train.js';

/** Auditory encoder configuration */
export interface AudioEncoderConfig {
  /** Audio sample rate (Hz) */
  sampleRate: number;
  /** FFT size (power of 2) */
  fftSize: number;
  /** Number of cochlear bands */
  numBands: number;
  /** Number of temporal frames in the spectrogram */
  numFrames: number;
  /** Minimum frequency of the auditory range (Hz) */
  minFrequency: number;
  /** Maximum frequency of the auditory range (Hz) */
  maxFrequency: number;
  /** Whether to apply a Hanning window */
  useHanningWindow: boolean;
}

const DEFAULT_AUDIO_CONFIG: AudioEncoderConfig = {
  sampleRate: 16000,
  fftSize: 2048,
  numBands: 40,
  numFrames: 20,
  minFrequency: 100,
  maxFrequency: 6000,
  useHanningWindow: true,
};

/**
 * Auditory Encoder — Simulates the cochlea and the cochlear nucleus.
 * Converts audio into spike patterns for the auditory cortex.
 */
export class AudioEncoder {
  private config: AudioEncoderConfig;
  /** Output vector size (numBands × numFrames) */
  public outputSize: number;
  /** Frequency edges of the cochlear bands (Hz) */
  private bandEdges: number[];
  /** FFT indices corresponding to the frequency edges */
  private bandBins: number[];
  /** Precomputed Hanning window */
  private hanningWindow: Float32Array;
  /** Spectrogram frame buffer (sliding window) */
  private spectrogramBuffer: Float32Array[];
  
  constructor(config: Partial<AudioEncoderConfig> = {}) {
    this.config = { ...DEFAULT_AUDIO_CONFIG, ...config };
    this.outputSize = this.config.numBands * this.config.numFrames;
    
    // Precompute cochlear band edges (logarithmic scale)
    this.bandEdges = this.calculateCochlearBands();
    this.bandBins = this.frequencyToBins(this.bandEdges);

    // Precompute Hanning window
    this.hanningWindow = new Float32Array(this.config.fftSize);
    if (this.config.useHanningWindow) {
      for (let i = 0; i < this.config.fftSize; i++) {
        this.hanningWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.config.fftSize - 1)));
      }
    } else {
      this.hanningWindow.fill(1.0);
    }
    
    // Empty spectrogram buffer
    this.spectrogramBuffer = [];
  }

  /**
   * Computes the cochlear band edges on a logarithmic scale.
   * Mimics the tonotopic distribution of the basilar membrane,
   * where low frequencies have higher resolution.
   */
  private calculateCochlearBands(): number[] {
    const { numBands, minFrequency, maxFrequency } = this.config;
    const edges: number[] = [];
    
    // Mel scale (logarithmic) — mimics human auditory perception
    const melMin = 2595 * Math.log10(1 + minFrequency / 700);
    const melMax = 2595 * Math.log10(1 + maxFrequency / 700);
    
    for (let i = 0; i <= numBands; i++) {
      const mel = melMin + (melMax - melMin) * (i / numBands);
      const freq = 700 * (Math.pow(10, mel / 2595) - 1);
      edges.push(freq);
    }
    
    return edges;
  }

  /**
   * Converts frequencies in Hz to FFT bin indices.
   */
  private frequencyToBins(frequencies: number[]): number[] {
    return frequencies.map(f => 
      Math.round(f * this.config.fftSize / this.config.sampleRate)
    );
  }

  /**
   * Encodes a chunk of PCM audio as a spike vector.
   *
   * @param audioSamples - PCM samples (Float32 or normalized Int16)
   * @param dt - Time step
   * @returns Spike vector of the full spectrogram
   */
  encode(audioSamples: Float32Array | number[], dt: number = 1.0): Float32Array {
    // 1. Apply window and compute FFT
    const frame = this.computeFrame(audioSamples);

    // 2. Group into cochlear bands
    const bands = this.applyBands(frame);

    // 3. Add to the spectrogram buffer (sliding window)
    this.spectrogramBuffer.push(bands);
    if (this.spectrogramBuffer.length > this.config.numFrames) {
      this.spectrogramBuffer.shift();
    }

    // 4. Build flat spectrogram
    const spectrogram = this.getSpectrogram();

    // 5. Convert to spikes
    return encodeSpikeVector(spectrogram, dt, 150);
  }

  /**
   * Encodes an already-computed spectrogram (e.g. received from 08_microphone_interaction).
   *
   * @param spectrogram - Flat array [time0_band0..band39, time1_band0..band39, ...]
   * @param dt - Time step
   */
  encodeSpectrogram(spectrogram: number[] | Float32Array, dt: number = 1.0): Float32Array {
    const values = new Float32Array(spectrogram.length);

    for (let i = 0; i < spectrogram.length; i++) {
      // Logarithmic scale to mimic cochlear response
      values[i] = Math.log1p(Math.abs(spectrogram[i] as number));
      values[i] = Math.min(1, values[i]);
    }

    return encodeSpikeVector(values, dt, 150);
  }

  /**
   * Computes an FFT frame with a Hanning window.
   * Simplified FFT (DFT) — in production fft.js would be used
   */
  private computeFrame(samples: Float32Array | number[]): Float32Array {
    const N = this.config.fftSize;
    const magnitudes = new Float32Array(N / 2);

    // Apply window
    const windowed = new Float32Array(N);
    for (let i = 0; i < Math.min(samples.length, N); i++) {
      windowed[i] = (samples[i] as number) * this.hanningWindow[i];
    }

    // Simplified DFT (magnitudes only, no phase)
    // For real performance, use FFT (O(N log N)); here a simplified O(N²)
    // We only compute the relevant frequencies (up to maxFrequency)
    const maxBin = Math.min(
      N / 2,
      Math.ceil(this.config.maxFrequency * N / this.config.sampleRate) + 1
    );
    
    for (let k = 0; k < maxBin; k++) {
      let real = 0;
      let imag = 0;
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        real += windowed[n] * Math.cos(angle);
        imag += windowed[n] * Math.sin(angle);
      }
      magnitudes[k] = Math.sqrt(real * real + imag * imag) / N;
    }
    
    return magnitudes;
  }

  /**
   * Groups the FFT magnitudes into cochlear bands.
   * Each band sums the energies of the corresponding bins.
   */
  private applyBands(magnitudes: Float32Array): Float32Array {
    const bands = new Float32Array(this.config.numBands);
    
    for (let b = 0; b < this.config.numBands; b++) {
      const startBin = this.bandBins[b];
      const endBin = this.bandBins[b + 1];
      
      let energy = 0;
      let count = 0;
      for (let i = startBin; i < endBin && i < magnitudes.length; i++) {
        energy += magnitudes[i] * magnitudes[i]; // Energy = magnitude²
        count++;
      }

      bands[b] = count > 0 ? Math.sqrt(energy / count) : 0; // RMS
    }

    // Normalize
    let max = 0;
    for (let i = 0; i < bands.length; i++) {
      if (bands[i] > max) max = bands[i];
    }
    if (max > 0) {
      for (let i = 0; i < bands.length; i++) {
        bands[i] /= max;
      }
    }
    
    return bands;
  }

  /**
   * Builds the current flat spectrogram from the buffer.
   */
  getSpectrogram(): Float32Array {
    const result = new Float32Array(this.outputSize);

    // Fill with available frames (pad with zeros if there aren't enough)
    const startIdx = Math.max(0, this.config.numFrames - this.spectrogramBuffer.length);
    
    for (let t = 0; t < this.spectrogramBuffer.length; t++) {
      const frame = this.spectrogramBuffer[t];
      const offset = (startIdx + t) * this.config.numBands;
      for (let b = 0; b < this.config.numBands; b++) {
        result[offset + b] = frame[b];
      }
    }
    
    return result;
  }

  /**
   * Detects whether the audio contains human voice.
   * Refactored from 07_audio_cortex/server.ts
   */
  detectVoice(spectrogram: Float32Array): { hasVoice: boolean; energy: number; centroid: number } {
    const lastFrame = spectrogram.slice(-this.config.numBands);

    // Total energy
    let energy = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      energy += lastFrame[i];
    }

    // Spectral centroid
    let weightedSum = 0;
    let sum = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      weightedSum += i * lastFrame[i];
      sum += lastFrame[i];
    }
    const centroid = sum > 0 ? weightedSum / sum : 0;

    // Voice: energy concentrated in low-mid frequencies (bands 4-24 of 40)
    let lowMidEnergy = 0;
    for (let i = 4; i < 25 && i < lastFrame.length; i++) {
      lowMidEnergy += lastFrame[i];
    }

    // High-frequency noise
    let highEnergy = 0;
    for (let i = 30; i < lastFrame.length; i++) {
      highEnergy += lastFrame[i];
    }
    
    const hasVoice = energy > 0.5 &&
      lowMidEnergy > energy * 0.25 &&
      highEnergy < energy * 0.6;
    
    return { hasVoice, energy, centroid };
  }

  /** Clears the spectrogram buffer */
  reset(): void {
    this.spectrogramBuffer = [];
  }
}
