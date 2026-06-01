/**
 * ENCODER AUDITIVO — Codificación de audio a spikes
 * ==================================================
 * Transforma señales de audio (micrófono, archivos) en trenes de spikes
 * para la corteza auditiva.
 * 
 * Pipeline bio-inspirado (cóclea artificial):
 * 1. FFT: Descomposición en frecuencias (análogo a la membrana basilar)
 * 2. Bandas cocleares: Agrupación logarítmica de frecuencias
 * 3. Sliding window: Espectrograma temporal
 * 4. Rate coding: Conversión a frecuencia de spikes
 * 
 * Refactorizado de 08_microphone_interaction/client.ts y 07_audio_cortex/server.ts
 */

import { encodeSpikeVector } from '../core/snn/spike-train.js';

/** Configuración del encoder auditivo */
export interface AudioEncoderConfig {
  /** Frecuencia de muestreo del audio (Hz) */
  sampleRate: number;
  /** Tamaño de la FFT (potencia de 2) */
  fftSize: number;
  /** Número de bandas cocleares */
  numBands: number;
  /** Número de frames temporales en el espectrograma */
  numFrames: number;
  /** Frecuencia mínima del rango auditivo (Hz) */
  minFrequency: number;
  /** Frecuencia máxima del rango auditivo (Hz) */
  maxFrequency: number;
  /** Si aplicar ventana Hanning */
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
 * Encoder Auditivo — Simula la cóclea y el núcleo coclear.
 * Convierte audio en patrones de spikes para la corteza auditiva.
 */
export class AudioEncoder {
  private config: AudioEncoderConfig;
  /** Tamaño del vector de output (numBands × numFrames) */
  public outputSize: number;
  /** Bordes de frecuencia de las bandas cocleares (Hz) */
  private bandEdges: number[];
  /** Índices FFT correspondientes a los bordes de frecuencia */
  private bandBins: number[];
  /** Ventana Hanning precalculada */
  private hanningWindow: Float32Array;
  /** Buffer de frames del espectrograma (sliding window) */
  private spectrogramBuffer: Float32Array[];
  
  constructor(config: Partial<AudioEncoderConfig> = {}) {
    this.config = { ...DEFAULT_AUDIO_CONFIG, ...config };
    this.outputSize = this.config.numBands * this.config.numFrames;
    
    // Precalcular bordes de bandas cocleares (escala logarítmica)
    this.bandEdges = this.calculateCochlearBands();
    this.bandBins = this.frequencyToBins(this.bandEdges);
    
    // Precalcular ventana Hanning
    this.hanningWindow = new Float32Array(this.config.fftSize);
    if (this.config.useHanningWindow) {
      for (let i = 0; i < this.config.fftSize; i++) {
        this.hanningWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.config.fftSize - 1)));
      }
    } else {
      this.hanningWindow.fill(1.0);
    }
    
    // Buffer de espectrograma vacío
    this.spectrogramBuffer = [];
  }

  /**
   * Calcula los bordes de las bandas cocleares en escala logarítmica.
   * Imita la distribución tonotópica de la membrana basilar,
   * donde las frecuencias bajas tienen mayor resolución.
   */
  private calculateCochlearBands(): number[] {
    const { numBands, minFrequency, maxFrequency } = this.config;
    const edges: number[] = [];
    
    // Escala Mel (logarítmica) — imita la percepción auditiva humana
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
   * Convierte frecuencias en Hz a índices del bin FFT.
   */
  private frequencyToBins(frequencies: number[]): number[] {
    return frequencies.map(f => 
      Math.round(f * this.config.fftSize / this.config.sampleRate)
    );
  }

  /**
   * Codifica un chunk de audio PCM como vector de spikes.
   * 
   * @param audioSamples - Muestras PCM (Float32 o Int16 normalizado)
   * @param dt - Paso temporal
   * @returns Vector de spikes del espectrograma completo
   */
  encode(audioSamples: Float32Array | number[], dt: number = 1.0): Float32Array {
    // 1. Aplicar ventana y calcular FFT
    const frame = this.computeFrame(audioSamples);
    
    // 2. Agrupar en bandas cocleares
    const bands = this.applyBands(frame);
    
    // 3. Añadir al buffer de espectrograma (sliding window)
    this.spectrogramBuffer.push(bands);
    if (this.spectrogramBuffer.length > this.config.numFrames) {
      this.spectrogramBuffer.shift();
    }
    
    // 4. Construir espectrograma flat
    const spectrogram = this.getSpectrogram();
    
    // 5. Convertir a spikes
    return encodeSpikeVector(spectrogram, dt, 150);
  }

  /**
   * Codifica un espectrograma ya calculado (ej: recibido de 08_microphone_interaction).
   * 
   * @param spectrogram - Flat array [time0_band0..band39, time1_band0..band39, ...]
   * @param dt - Paso temporal
   */
  encodeSpectrogram(spectrogram: number[] | Float32Array, dt: number = 1.0): Float32Array {
    const values = new Float32Array(spectrogram.length);
    
    for (let i = 0; i < spectrogram.length; i++) {
      // Escala logarítmica para imitar respuesta coclear
      values[i] = Math.log1p(Math.abs(spectrogram[i] as number));
      values[i] = Math.min(1, values[i]);
    }
    
    return encodeSpikeVector(values, dt, 150);
  }

  /**
   * Calcula un frame FFT con ventana Hanning.
   * FFT simplificada (DFT) — en producción se usaría fft.js
   */
  private computeFrame(samples: Float32Array | number[]): Float32Array {
    const N = this.config.fftSize;
    const magnitudes = new Float32Array(N / 2);
    
    // Aplicar ventana
    const windowed = new Float32Array(N);
    for (let i = 0; i < Math.min(samples.length, N); i++) {
      windowed[i] = (samples[i] as number) * this.hanningWindow[i];
    }
    
    // DFT simplificada (solo magnitudes, no fase)
    // Para rendimiento real, usar FFT (O(N log N)), aquí O(N²) simplificado
    // Solo calculamos las frecuencias relevantes (hasta maxFrequency)
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
   * Agrupa las magnitudes FFT en bandas cocleares.
   * Cada banda suma las energías de los bins correspondientes.
   */
  private applyBands(magnitudes: Float32Array): Float32Array {
    const bands = new Float32Array(this.config.numBands);
    
    for (let b = 0; b < this.config.numBands; b++) {
      const startBin = this.bandBins[b];
      const endBin = this.bandBins[b + 1];
      
      let energy = 0;
      let count = 0;
      for (let i = startBin; i < endBin && i < magnitudes.length; i++) {
        energy += magnitudes[i] * magnitudes[i]; // Energía = magnitud²
        count++;
      }
      
      bands[b] = count > 0 ? Math.sqrt(energy / count) : 0; // RMS
    }
    
    // Normalizar
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
   * Construye el espectrograma flat actual a partir del buffer.
   */
  getSpectrogram(): Float32Array {
    const result = new Float32Array(this.outputSize);
    
    // Rellenar con frames disponibles (pad con ceros si no hay suficientes)
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
   * Detecta si el audio contiene voz humana.
   * Refactorizado de 07_audio_cortex/server.ts
   */
  detectVoice(spectrogram: Float32Array): { hasVoice: boolean; energy: number; centroid: number } {
    const lastFrame = spectrogram.slice(-this.config.numBands);
    
    // Energía total
    let energy = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      energy += lastFrame[i];
    }
    
    // Centroide espectral
    let weightedSum = 0;
    let sum = 0;
    for (let i = 0; i < lastFrame.length; i++) {
      weightedSum += i * lastFrame[i];
      sum += lastFrame[i];
    }
    const centroid = sum > 0 ? weightedSum / sum : 0;
    
    // Voz: energía concentrada en frecuencias low-mid (bandas 4-24 de 40)
    let lowMidEnergy = 0;
    for (let i = 4; i < 25 && i < lastFrame.length; i++) {
      lowMidEnergy += lastFrame[i];
    }
    
    // Ruido de alta frecuencia
    let highEnergy = 0;
    for (let i = 30; i < lastFrame.length; i++) {
      highEnergy += lastFrame[i];
    }
    
    const hasVoice = energy > 0.5 &&
      lowMidEnergy > energy * 0.25 &&
      highEnergy < energy * 0.6;
    
    return { hasVoice, energy, centroid };
  }

  /** Limpia el buffer del espectrograma */
  reset(): void {
    this.spectrogramBuffer = [];
  }
}
