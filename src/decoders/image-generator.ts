/**
 * IMAGE GENERATOR — Inverse visual cortex
 * ================================================
 * Reconstructs visual representations from neuronal
 * activation patterns of the visual cortex.
 *
 * Biology: When we "imagine" something, the visual cortex activates
 * in the reverse direction — the higher layers (objects) activate
 * the lower ones (edges, pixels). This is called "imagery".
 *
 * This generator takes a cortical activation pattern and produces
 * a reconstructed pixelated image (like a "visual dream").
 */

/** Image generator configuration */
export interface ImageGeneratorConfig {
  /** Width of the output image */
  outputWidth: number;
  /** Height of the output image */
  outputHeight: number;
  /** Number of neurons in the visual cortex */
  cortexNeurons: number;
}

const DEFAULT_IMAGE_CONFIG: ImageGeneratorConfig = {
  outputWidth: 32,
  outputHeight: 32,
  cortexNeurons: 10000,
};

/**
 * Image generator — Decodes visual activity into pixels.
 */
export class ImageGenerator {
  private config: ImageGeneratorConfig;
  /** Decoding matrix: maps neurons → pixels */
  private decoderMatrix: Float32Array;
  /** Output size (width × height) */
  public outputSize: number;

  constructor(config: Partial<ImageGeneratorConfig> = {}) {
    this.config = { ...DEFAULT_IMAGE_CONFIG, ...config };
    this.outputSize = this.config.outputWidth * this.config.outputHeight;

    // Initialize the decoding matrix with random weights
    // In a trained system, this is learned together with the encoding
    this.decoderMatrix = new Float32Array(this.config.cortexNeurons * this.outputSize);
    this.initializeDecoder();
  }

  /**
   * Initializes the decoding matrix with random sparse weights.
   * Each cortical neuron "lights up" a subset of pixels.
   */
  private initializeDecoder(): void {
    const pixelsPerNeuron = Math.max(3, Math.floor(this.outputSize * 0.02)); // ~2% coverage

    for (let n = 0; n < this.config.cortexNeurons; n++) {
      // Each neuron has a "receptive field"
      // — the area of the image it is sensitive to
      const centerX = Math.random() * this.config.outputWidth;
      const centerY = Math.random() * this.config.outputHeight;
      const radius = 2 + Math.random() * 4;

      for (let p = 0; p < pixelsPerNeuron; p++) {
        // Random position near the center
        const px = Math.round(centerX + (Math.random() - 0.5) * radius * 2);
        const py = Math.round(centerY + (Math.random() - 0.5) * radius * 2);
        
        if (px >= 0 && px < this.config.outputWidth && py >= 0 && py < this.config.outputHeight) {
          const pixelIdx = py * this.config.outputWidth + px;
          const weight = 0.1 + Math.random() * 0.3;
          this.decoderMatrix[n * this.outputSize + pixelIdx] = weight;
        }
      }
    }
  }

  /**
   * Generates an image from the cortical activation.
   *
   * @param corticalActivity - Activation pattern of the visual cortex (Float32Array)
   * @returns Reconstructed image as a 2D array of values 0-255
   */
  generate(corticalActivity: Float32Array): {
    pixels: Float32Array;
    width: number;
    height: number;
    base64?: string;
  } {
    const pixels = new Float32Array(this.outputSize);

    // Multiply cortical activity × decoding matrix
    const activityLen = Math.min(corticalActivity.length, this.config.cortexNeurons);

    for (let n = 0; n < activityLen; n++) {
      if (corticalActivity[n] > 0.1) { // Only significantly active neurons
        const offset = n * this.outputSize;
        const activation = corticalActivity[n];
        
        for (let p = 0; p < this.outputSize; p++) {
          pixels[p] += this.decoderMatrix[offset + p] * activation;
        }
      }
    }
    
    // Normalize to range 0-255
    let max = 0;
    for (let i = 0; i < pixels.length; i++) {
      if (pixels[i] > max) max = pixels[i];
    }
    if (max > 0) {
      for (let i = 0; i < pixels.length; i++) {
        pixels[i] = (pixels[i] / max) * 255;
      }
    }
    
    return {
      pixels,
      width: this.config.outputWidth,
      height: this.config.outputHeight,
    };
  }

  /**
   * Learns to decode: adjusts the decoding matrix
   * so the reconstruction more closely resembles the original input.
   *
   * @param corticalActivity - Cortical activation during encoding
   * @param originalPixels - Original image (normalized 0-1)
   * @param learningRate - Learning rate
   */
  learn(
    corticalActivity: Float32Array,
    originalPixels: Float32Array,
    learningRate: number = 0.01
  ): void {
    // Generate the current reconstruction
    const { pixels: reconstructed } = this.generate(corticalActivity);

    // Compute the per-pixel error
    const activityLen = Math.min(corticalActivity.length, this.config.cortexNeurons);
    
    for (let n = 0; n < activityLen; n++) {
      if (corticalActivity[n] > 0.1) {
        const offset = n * this.outputSize;
        
        for (let p = 0; p < this.outputSize; p++) {
          const error = (originalPixels[p] * 255) - reconstructed[p];
          this.decoderMatrix[offset + p] += learningRate * error * corticalActivity[n];
          
          // Clamp
          if (this.decoderMatrix[offset + p] < 0) this.decoderMatrix[offset + p] = 0;
          if (this.decoderMatrix[offset + p] > 1) this.decoderMatrix[offset + p] = 1;
        }
      }
    }
  }

  /**
   * Generates an ASCII art representation of the image (for the console).
   */
  toAscii(pixels: Float32Array, width: number, height: number): string {
    const chars = ' .:-=+*#%@';
    const lines: string[] = [];
    
    for (let y = 0; y < height; y++) {
      let line = '';
      for (let x = 0; x < width; x++) {
        const val = pixels[y * width + x];
        const idx = Math.floor((val / 255) * (chars.length - 1));
        line += chars[Math.min(idx, chars.length - 1)];
      }
      lines.push(line);
    }
    
    return lines.join('\n');
  }
}
