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
    // 5. Convert to spikes via rate coding (one Bernoulli sample per channel)
    return encodeSpikeVector(this.encodeRates(pixels, width, height), dt, 200);
  }

  /**
   * Encodes an image as graded firing RATES (0–1): the retinal features
   * themselves, before any spike sampling. This is what a sustained
   * presentation needs: the visual cortex draws fresh Poisson spikes from the
   * rates on every tick, so repeated presentations of one image share their
   * statistics instead of being two unrelated single samples.
   *
   * @param pixels - Image data (grayscale, 0-255)
   * @param width - Width of the input image
   * @param height - Height of the input image
   * @returns Rate vector (intensity map + edge maps), length `outputSize`
   */
  encodeRates(pixels: number[] | Float32Array | Uint8Array, width: number, height: number): Float32Array {
    // 1. Convert to normalized Float32Array (0-1)
    let normalized: Float32Array = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) {
      normalized[i] = (pixels[i] as number) / 255;
    }

    // 1b. Nonlinear denoising in the outer retina: a 3×3 median removes
    // isolated specks (salt-and-pepper) and fills pinholes in strokes before
    // the cells pool — the photoreceptor-bipolar-amacrine circuitry is not a
    // linear averager (Baccus & Meister 2002).
    if (width >= 3 * this.config.processWidth && height >= 3 * this.config.processHeight) {
      normalized = this.median3(normalized, width, height);
    }

    // 1c. Something small is looked at more closely: the eye brings a small
    // object into the fovea and the cortex sees it at the same scale as a
    // large one (size invariance at the front end; the innate detectors keep
    // the raw view, see encodeIntensity).
    normalized = this.zoomIfSmall(normalized, width, height);

    // 2. Resize to processing resolution
    let processed = this.resize(normalized, width, height, this.config.processWidth, this.config.processHeight);

    // 2b. Receptor gain: a cell half covered by a stroke responds more than
    // half (photoreceptor responses are compressive, Naka-Rushton). Without
    // it, a stroke that straddles two cells read as two faint cells while the
    // same stroke inside one cell read as bright — a square's sides differed
    // in brightness by the accident of where they fell on the retina.
    for (let i = 0; i < processed.length; i++) processed[i] = Math.min(1, processed[i] * VisualEncoder.RECEPTOR_GAIN);

    // 3. Foveation (center the content)
    if (this.config.foveation) {
      processed = this.foveate(processed, this.config.processWidth, this.config.processHeight);
    }

    // 4. Edge detection (simplified Gabor filters)
    let output: Float32Array;
    if (this.config.edgeDetection) {
      const edges = this.detectEdges(processed, this.config.processWidth, this.config.processHeight);
      // Concatenate the contrast image + edge maps
      output = new Float32Array(processed.length + edges.length);
      output.set(this.contrast(processed, this.config.processWidth, this.config.processHeight), 0);
      output.set(edges, processed.length);
    } else {
      output = this.contrast(processed, this.config.processWidth, this.config.processHeight);
    }

    return output;
  }

  /**
   * The retinal image itself — resized and centred, luminance in 0..1 — as the
   * innate detectors read it (a face, something looming). What the cortex
   * receives is its contrast (see `encodeRates`).
   */
  encodeIntensity(pixels: number[] | Float32Array | Uint8Array, width: number, height: number): Float32Array {
    const normalized = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) normalized[i] = (pixels[i] as number) / 255;
    let processed = this.resize(normalized, width, height, this.config.processWidth, this.config.processHeight);
    if (this.config.foveation) processed = this.foveate(processed, this.config.processWidth, this.config.processHeight);
    return processed;
  }

  /**
   * Centre–surround contrast, what retinal ganglion cells report: each cell's
   * luminance minus the mean of its 3×3 neighbourhood, rectified, with a
   * response floor below which the faint contrast of scattered noise is not
   * reported. A thin stroke stays bright; the inside of a filled patch goes
   * dark and only its border remains — a filled rectangle and an outlined
   * car are no longer the same thing to the cortex, and absolute brightness
   * no longer dominates what a shape is (Kuffler 1953).
   */
  private static readonly CONTRAST_FLOOR = 0.15;
  private static readonly RECEPTOR_GAIN = 1.5;

  private contrast(img: Float32Array, w: number, h: number): Float32Array {
    const out = new Float32Array(w * h);
    let peak = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
            sum += img[yy * w + xx];
            n++;
          }
        }
        const c = img[y * w + x] - sum / n;
        out[y * w + x] = c > VisualEncoder.CONTRAST_FLOOR ? c : 0;
        if (out[y * w + x] > peak) peak = out[y * w + x];
      }
    }
    // A stroke loses part of its luminance to its own surround (its neighbours
    // along the stroke are lit too): a bounded rescale keeps strokes near the
    // level they had without amplifying faint noise.
    if (peak > 0) {
      const gain = Math.min(1 / peak, 1.25);
      for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i] * gain);
    }
    return out;
  }

  /** Content smaller than this fraction of the image is zoomed to ZOOM_TARGET of it. */
  private static readonly ZOOM_BELOW = 0.5;
  private static readonly ZOOM_TARGET = 0.75;

  /** Crops the content's bounding box (with a margin) and enlarges it to fill ZOOM_TARGET of the image. */
  private zoomIfSmall(img: Float32Array, w: number, h: number): Float32Array {
    let background = Infinity;
    for (let i = 0; i < img.length; i++) if (img[i] < background) background = img[i];
    const lit = background + 0.2;
    let minX = w, maxX = -1, minY = h, maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (img[y * w + x] < lit) continue;
        if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
    if (maxX < 0) return img;
    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    const extent = Math.max(bw, bh);
    if (extent < 4 || extent >= VisualEncoder.ZOOM_BELOW * Math.min(w, h)) return img;
    const scale = (VisualEncoder.ZOOM_TARGET * Math.min(w, h)) / extent;
    const outW = Math.min(w, Math.round(bw * scale)), outH = Math.min(h, Math.round(bh * scale));
    // Crop and enlarge (bilinear), then paste centred on the background.
    const crop = new Float32Array(bw * bh);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) crop[y * bw + x] = img[(minY + y) * w + minX + x];
    const enlarged = this.resize(crop, bw, bh, outW, outH);
    const out = new Float32Array(w * h).fill(background);
    const ox = Math.floor((w - outW) / 2), oy = Math.floor((h - outH) / 2);
    for (let y = 0; y < outH; y++) for (let x = 0; x < outW; x++) out[(oy + y) * w + ox + x] = enlarged[y * outW + x];
    return out;
  }

  /** Cells of the coarse grid on which objects are told apart, along the image's shorter side. */
  private static readonly SEGMENT_CELLS = 16;
  /** Fraction of a cell's pixels that must be lit for the cell to count (specks of noise do not). */
  private static readonly SEGMENT_OCCUPANCY = 0.2;
  /** Smallest object worth a fixation, in coarse cells — and relative to the largest object. */
  private static readonly SEGMENT_MIN_CELLS = 4;
  private static readonly SEGMENT_MIN_SHARE = 0.15;
  /** Margin around an object when the eye fixates it (fraction of its extent). */
  private static readonly FIXATION_MARGIN = 0.15;

  /**
   * The objects in a scene: connected regions of content (what stands out
   * from the background), found on a coarse grid so that a stroke's gaps do
   * not split an object and two objects a few cells apart stay two. This is
   * what the superior colliculus gets from the retina to point the eye at:
   * where things are, not what they are. Ordered by size (bottom-up
   * saliency), largest first.
   *
   * @returns Bounding boxes in pixel coordinates; empty with fewer than two objects
   */
  segment(pixels: number[] | Float32Array | Uint8Array, width: number, height: number): Array<{ x: number; y: number; w: number; h: number }> {
    let img: Float32Array = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) img[i] = (pixels[i] as number) / 255;
    if (width >= 3 * this.config.processWidth && height >= 3 * this.config.processHeight) img = this.median3(img, width, height);
    let background = Infinity;
    for (let i = 0; i < img.length; i++) if (img[i] < background) background = img[i];
    const lit = background + 0.2;
    // Coarse occupancy grid: a cell is on if enough of its pixels are lit —
    // a stroke lights it, a speck of noise does not.
    const cell = Math.max(2, Math.round(Math.min(width, height) / VisualEncoder.SEGMENT_CELLS));
    const gw = Math.ceil(width / cell), gh = Math.ceil(height / cell);
    const counts = new Int32Array(gw * gh);
    const sizes = new Int32Array(gw * gh);
    for (let y = 0; y < height; y++) {
      const gy = Math.floor(y / cell);
      for (let x = 0; x < width; x++) {
        const c = gy * gw + Math.floor(x / cell);
        sizes[c]++;
        if (img[y * width + x] >= lit) counts[c]++;
      }
    }
    const grid = new Uint8Array(gw * gh);
    for (let c = 0; c < grid.length; c++) if (counts[c] >= VisualEncoder.SEGMENT_OCCUPANCY * sizes[c]) grid[c] = 1;
    // Connected components (8-neighbourhood) by flood fill.
    const seen = new Uint8Array(gw * gh);
    const boxes: Array<{ x: number; y: number; w: number; h: number; cells: number }> = [];
    const stack: number[] = [];
    for (let start = 0; start < grid.length; start++) {
      if (grid[start] === 0 || seen[start] === 1) continue;
      let minX = gw, maxX = -1, minY = gh, maxY = -1, cells = 0;
      stack.push(start);
      seen[start] = 1;
      while (stack.length > 0) {
        const c = stack.pop() as number;
        const cx = c % gw, cy = Math.floor(c / gw);
        cells++;
        if (cx < minX) minX = cx; if (cx > maxX) maxX = cx; if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx, ny = cy + dy;
          if (nx < 0 || nx >= gw || ny < 0 || ny >= gh) continue;
          const n = ny * gw + nx;
          if (grid[n] === 1 && seen[n] === 0) { seen[n] = 1; stack.push(n); }
        }
      }
      if (cells < VisualEncoder.SEGMENT_MIN_CELLS) continue;
      // Back to pixels.
      const x0 = Math.floor((minX * width) / gw), x1 = Math.ceil(((maxX + 1) * width) / gw);
      const y0 = Math.floor((minY * height) / gh), y1 = Math.ceil(((maxY + 1) * height) / gh);
      boxes.push({ x: x0, y: y0, w: x1 - x0, h: y1 - y0, cells });
    }
    if (boxes.length < 2) return [];
    // What is tiny next to the largest thing is not another thing (a speck, a crumb of the stroke).
    const largest = Math.max(...boxes.map((b) => b.cells));
    const objects = boxes.filter((b) => b.cells >= VisualEncoder.SEGMENT_MIN_SHARE * largest);
    if (objects.length < 2) return [];
    return objects.sort((a, b) => b.cells - a.cells || a.x - b.x).map(({ x, y, w, h }) => ({ x, y, w, h }));
  }

  /**
   * What the eye sees when it fixates an object: the object's box with a
   * margin, cut out of the scene (the rest falls outside the fovea). The
   * crop goes through the ordinary retina — a small object is then zoomed
   * as any small thing is.
   */
  static fixate<T extends number[] | Float32Array | Uint8Array>(
    pixels: T, width: number, height: number, box: { x: number; y: number; w: number; h: number }, channels: number = 1,
  ): { pixels: number[]; width: number; height: number } {
    const margin = Math.max(2, Math.round(VisualEncoder.FIXATION_MARGIN * Math.max(box.w, box.h)));
    const x0 = Math.max(0, box.x - margin), y0 = Math.max(0, box.y - margin);
    const x1 = Math.min(width, box.x + box.w + margin), y1 = Math.min(height, box.y + box.h + margin);
    const cw = x1 - x0, ch = y1 - y0;
    const out = new Array<number>(cw * ch * channels);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) for (let c = 0; c < channels; c++) {
      out[(y * cw + x) * channels + c] = pixels[((y0 + y) * width + x0 + x) * channels + c] as number;
    }
    return { pixels: out, width: cw, height: ch };
  }

  /** 3×3 median filter. */
  private median3(img: Float32Array, w: number, h: number): Float32Array {
    const out = new Float32Array(w * h);
    const window = new Float32Array(9);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
            window[n++] = img[yy * w + xx];
          }
        }
        const values = Array.from(window.subarray(0, n)).sort((a, b) => a - b);
        out[y * w + x] = values[n >> 1];
      }
    }
    return out;
  }

  /**
   * The colour of what is in view, as the colour cortex receives it: a
   * histogram of hue × saturation over the chromatic pixels, plus white and
   * grey for the achromatic light ones, scaled so the dominant colour is 1.
   * Dark pixels are the background and do not count. Position, size and
   * shape play no part: the blue card and the blue car give the same code.
   *
   * Layout (26 channels): 12 hue bins × {vivid, pale}, then white, then grey.
   *
   * @param rgb - Interleaved r, g, b bytes (0–255), width × height × 3 values
   */
  static encodeColor(rgb: ArrayLike<number>, width: number, height: number): Float32Array {
    const code = new Float32Array(26);
    const n = Math.min(width * height, Math.floor(rgb.length / 3));
    for (let p = 0; p < n; p++) {
      const r = rgb[p * 3] / 255, g = rgb[p * 3 + 1] / 255, b = rgb[p * 3 + 2] / 255;
      const max = Math.max(r, g, b), min = Math.min(r, g, b);
      const light = max;
      if (light < 0.25) continue; // background / dark
      const sat = max > 0 ? (max - min) / max : 0;
      if (sat < 0.25) {
        code[light >= 0.6 ? 24 : 25]++;
        continue;
      }
      const d = max - min;
      let hue = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
      if (hue < 0) hue += 6;
      const bin = Math.min(11, Math.floor(hue * 2)); // 12 bins of 30°
      code[bin * 2 + (sat >= 0.6 ? 0 : 1)]++;
    }
    let peak = 0;
    for (let i = 0; i < code.length; i++) if (code[i] > peak) peak = code[i];
    if (peak > 0) for (let i = 0; i < code.length; i++) code[i] /= peak;
    return code;
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

    // Area pooling when shrinking: each retinal cell reports the mean of the
    // patch of image it covers, as a photoreceptor pools the light over its
    // receptive field (so isolated noisy pixels barely register). Point sampling (the previous bilinear lookup at
    // one position per cell) simply missed thin strokes that fell between two
    // sample points — parts of a drawing vanished depending on where it was.
    if (srcW >= dstW && srcH >= dstH) {
      for (let y = 0; y < dstH; y++) {
        const y0 = Math.floor((y * srcH) / dstH);
        const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * srcH) / dstH));
        for (let x = 0; x < dstW; x++) {
          const x0 = Math.floor((x * srcW) / dstW);
          const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * srcW) / dstW));
          let sum = 0;
          for (let sy = y0; sy < y1; sy++) {
            for (let sx = x0; sx < x1; sx++) sum += src[sy * srcW + sx];
          }
          dst[y * dstW + x] = sum / ((y1 - y0) * (x1 - x0));
        }
      }
      return dst;
    }

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
    // Compute the center of mass of the CONTENT: what stands out from the
    // background level. (Counting the background itself — a uniformly grey
    // canvas weighs more than the strokes drawn on it — dragged the centre of
    // mass toward the middle of the image and the content was left off-centre.)
    let background = Infinity;
    for (let i = 0; i < img.length; i++) if (img[i] < background) background = img[i];

    let totalMass = 0;
    let cx = 0;
    let cy = 0;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const val = img[y * w + x] - background;
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
