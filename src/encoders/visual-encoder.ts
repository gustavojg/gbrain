/**
 * VISUAL ENCODER — Image to spikes encoding
 * ===================================================
 * Transforms visual inputs (webcam, static images) into spike
 * trains for the visual cortex.
 *
 * Bio-inspired pipeline:
 * 1. Retina: Captures and preprocesses the image
 * 2. Ganglion cells: Edge/contrast detection (simple Gabor filters)
 * 3. Foveation: Auto-centering of the relevant pattern
 * 4. Rate coding: Conversion to spike frequency
 *
 * Based on the existing visual pipeline from 06_visual_interface
 */

import { encodeSpikeVector } from '../core/snn/spike-train.js';

/** Visual encoder configuration */
export interface VisualEncoderConfig {
  /** Input resolution (width × height) */
  inputWidth: number;
  inputHeight: number;
  /** Internal processing resolution */
  processWidth: number;
  processHeight: number;
  /** Whether to apply foveation (auto-centering) */
  foveation: boolean;
  /** Whether to apply edge detection */
  edgeDetection: boolean;
  /** Number of orientations for Gabor filters */
  gaborOrientations: number;
}

const DEFAULT_VISUAL_CONFIG: VisualEncoderConfig = {
  inputWidth: 64,
  inputHeight: 64,
  processWidth: 32,
  processHeight: 32,
  foveation: true,
  edgeDetection: true,
  gaborOrientations: 4,  // 0°, 45°, 90°, 135°
};

/**
 * Visual Encoder — Simulates the retina and the lateral geniculate nucleus.
 * Converts images into spike patterns for the visual cortex.
 */
export class VisualEncoder {
  private config: VisualEncoderConfig;
  /** Output vector size (spikes) */
  public outputSize: number;

  constructor(config: Partial<VisualEncoderConfig> = {}) {
    this.config = { ...DEFAULT_VISUAL_CONFIG, ...config };
    
    // Output size: processed image + edge maps per orientation
    const baseSize = this.config.processWidth * this.config.processHeight;
    const edgeChannels = this.config.edgeDetection ? this.config.gaborOrientations : 0;
    this.outputSize = baseSize * (1 + edgeChannels);
  }

  /**
   * Encodes an image as a spike vector.
   *
   * @param pixels - Image data (grayscale, 0-255)
   * @param width - Width of the input image
   * @param height - Height of the input image
   * @param dt - Time step
   * @returns Spike vector (Float32Array)
   */
  encode(pixels: number[] | Float32Array | Uint8Array, width: number, height: number, dt: number = 1.0): Float32Array {
    // 1. Convert to normalized Float32Array (0-1)
    let normalized = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) {
      normalized[i] = (pixels[i] as number) / 255;
    }

    // 2. Resize to processing resolution
    let processed = this.resize(normalized, width, height, this.config.processWidth, this.config.processHeight);

    // 3. Foveation (center the content)
    if (this.config.foveation) {
      processed = this.foveate(processed, this.config.processWidth, this.config.processHeight);
    }

    // 4. Edge detection (simplified Gabor filters)
    let output: Float32Array;
    if (this.config.edgeDetection) {
      const edges = this.detectEdges(processed, this.config.processWidth, this.config.processHeight);
      // Concatenate processed image + edge maps
      output = new Float32Array(processed.length + edges.length);
      output.set(processed, 0);
      output.set(edges, processed.length);
    } else {
      output = processed;
    }

    // 5. Convert to spikes via rate coding
    return encodeSpikeVector(output, dt, 200);
  }

  /**
   * Encodes a drawing grid (like the one in project 06) to spikes.
   * Accepts a 2D array of 0/1 values directly.
   */
  encodeGrid(grid: number[][], dt: number = 1.0): Float32Array {
    const h = grid.length;
    const w = grid[0].length;
    const flat = new Float32Array(w * h);
    
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        flat[y * w + x] = grid[y][x] * 255;
      }
    }
    
    return this.encode(flat, w, h, dt);
  }

  /**
   * Resizes an image using bilinear interpolation.
   */
  private resize(src: Float32Array, srcW: number, srcH: number, dstW: number, dstH: number): Float32Array {
    const dst = new Float32Array(dstW * dstH);
    const xRatio = srcW / dstW;
    const yRatio = srcH / dstH;

    for (let y = 0; y < dstH; y++) {
      for (let x = 0; x < dstW; x++) {
        const srcX = x * xRatio;
        const srcY = y * yRatio;
        const x0 = Math.floor(srcX);
        const y0 = Math.floor(srcY);
        const x1 = Math.min(x0 + 1, srcW - 1);
        const y1 = Math.min(y0 + 1, srcH - 1);
        const xFrac = srcX - x0;
        const yFrac = srcY - y0;

        // Bilinear interpolation
        const v00 = src[y0 * srcW + x0];
        const v10 = src[y0 * srcW + x1];
        const v01 = src[y1 * srcW + x0];
        const v11 = src[y1 * srcW + x1];

        dst[y * dstW + x] = 
          v00 * (1 - xFrac) * (1 - yFrac) +
          v10 * xFrac * (1 - yFrac) +
          v01 * (1 - xFrac) * yFrac +
          v11 * xFrac * yFrac;
      }
    }

    return dst;
  }

  /**
   * Foveation: centers the content of the pattern.
   * Inspired by eye saccades — the eye automatically centers
   * the object of interest on the fovea.
   *
   * Refactored from 06_visual_interface/server.ts
   */
  private foveate(img: Float32Array, w: number, h: number): Float32Array {
    // Compute the center of mass of the content
    let totalMass = 0;
    let cx = 0;
    let cy = 0;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const val = img[y * w + x];
        totalMass += val;
        cx += x * val;
        cy += y * val;
      }
    }

    if (totalMass < 0.01) return img; // No content, don't center

    cx /= totalMass;
    cy /= totalMass;

    // Shift to center
    const dx = Math.round(w / 2 - cx);
    const dy = Math.round(h / 2 - cy);

    if (dx === 0 && dy === 0) return img;

    const centered = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const srcX = x - dx;
        const srcY = y - dy;
        if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h) {
          centered[y * w + x] = img[srcY * w + srcX];
        }
      }
    }

    return centered;
  }

  /**
   * Edge detection with simplified Gabor filters.
   *
   * In V1 of the real brain, neurons respond selectively to edges
   * at specific orientations (discovery by Hubel & Wiesel, Nobel 1981).
   *
   * Here we use simple Sobel-type convolutions for 4 orientations.
   */
  private detectEdges(img: Float32Array, w: number, h: number): Float32Array {
    const orientations = this.config.gaborOrientations;
    const result = new Float32Array(w * h * orientations);

    // Kernels for different orientations
    const kernels = [
      // 0° — Horizontal
      [-1, -2, -1, 0, 0, 0, 1, 2, 1],
      // 90° — Vertical
      [-1, 0, 1, -2, 0, 2, -1, 0, 1],
      // 45° — Diagonal ↗
      [0, -1, -2, 1, 0, -1, 2, 1, 0],
      // 135° — Diagonal ↘
      [-2, -1, 0, -1, 0, 1, 0, 1, 2],
    ];

    for (let o = 0; o < Math.min(orientations, kernels.length); o++) {
      const kernel = kernels[o];
      const offset = o * w * h;

      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          let sum = 0;
          let ki = 0;
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              sum += img[(y + ky) * w + (x + kx)] * kernel[ki];
              ki++;
            }
          }
          // Normalize to 0-1 and take the absolute value
          result[offset + y * w + x] = Math.min(1, Math.abs(sum) / 4);
        }
      }
    }

    return result;
  }
}
