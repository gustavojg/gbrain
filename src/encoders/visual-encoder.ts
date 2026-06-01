/**
 * ENCODER VISUAL — Codificación de imágenes a spikes
 * ===================================================
 * Transforma inputs visuales (webcam, imágenes estáticas) en trenes
 * de spikes para la corteza visual.
 * 
 * Pipeline bio-inspirado:
 * 1. Retina: Captura y preprocesa la imagen
 * 2. Células ganglionares: Detección de bordes/contraste (filtros Gabor simples)
 * 3. Foveación: Auto-centrado del patrón relevante
 * 4. Rate coding: Conversión a frecuencia de spikes
 * 
 * Basado en el pipeline visual existente de 06_visual_interface
 */

import { encodeSpikeVector } from '../core/snn/spike-train.js';

/** Configuración del encoder visual */
export interface VisualEncoderConfig {
  /** Resolución de entrada (ancho × alto) */
  inputWidth: number;
  inputHeight: number;
  /** Resolución de procesamiento interna */
  processWidth: number;
  processHeight: number;
  /** Si aplicar foveación (auto-centrado) */
  foveation: boolean;
  /** Si aplicar detección de bordes */
  edgeDetection: boolean;
  /** Número de orientaciones para filtros Gabor */
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
 * Encoder Visual — Simula la retina y el núcleo geniculado lateral.
 * Convierte imágenes en patrones de spikes para la corteza visual.
 */
export class VisualEncoder {
  private config: VisualEncoderConfig;
  /** Tamaño del vector de output (spikes) */
  public outputSize: number;

  constructor(config: Partial<VisualEncoderConfig> = {}) {
    this.config = { ...DEFAULT_VISUAL_CONFIG, ...config };
    
    // Output size: processed image + edge maps per orientation
    const baseSize = this.config.processWidth * this.config.processHeight;
    const edgeChannels = this.config.edgeDetection ? this.config.gaborOrientations : 0;
    this.outputSize = baseSize * (1 + edgeChannels);
  }

  /**
   * Codifica una imagen como vector de spikes.
   * 
   * @param pixels - Datos de imagen (escala de grises, 0-255)
   * @param width - Ancho de la imagen de entrada
   * @param height - Alto de la imagen de entrada
   * @param dt - Paso temporal
   * @returns Vector de spikes (Float32Array)
   */
  encode(pixels: number[] | Float32Array | Uint8Array, width: number, height: number, dt: number = 1.0): Float32Array {
    // 1. Convertir a Float32Array normalizado (0-1)
    let normalized = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) {
      normalized[i] = (pixels[i] as number) / 255;
    }

    // 2. Redimensionar a resolución de procesamiento
    let processed = this.resize(normalized, width, height, this.config.processWidth, this.config.processHeight);

    // 3. Foveación (centrar el contenido)
    if (this.config.foveation) {
      processed = this.foveate(processed, this.config.processWidth, this.config.processHeight);
    }

    // 4. Detección de bordes (filtros Gabor simplificados)
    let output: Float32Array;
    if (this.config.edgeDetection) {
      const edges = this.detectEdges(processed, this.config.processWidth, this.config.processHeight);
      // Concatenar imagen procesada + mapas de bordes
      output = new Float32Array(processed.length + edges.length);
      output.set(processed, 0);
      output.set(edges, processed.length);
    } else {
      output = processed;
    }

    // 5. Convertir a spikes via rate coding
    return encodeSpikeVector(output, dt, 200);
  }

  /**
   * Codifica un grid de dibujo (como el de proyecto 06) a spikes.
   * Acepta directamente un array 2D de valores 0/1.
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
   * Redimensiona una imagen usando interpolación bilineal.
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

        // Interpolación bilineal
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
   * Foveación: centra el contenido del patrón.
   * Inspirado en las sacadas oculares — el ojo centra automáticamente
   * el objeto de interés en la fóvea.
   * 
   * Refactorizado de 06_visual_interface/server.ts
   */
  private foveate(img: Float32Array, w: number, h: number): Float32Array {
    // Calcular centro de masa del contenido
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

    if (totalMass < 0.01) return img; // Sin contenido, no centrar

    cx /= totalMass;
    cy /= totalMass;

    // Desplazar para centrar
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
   * Detección de bordes con filtros Gabor simplificados.
   * 
   * En V1 del cerebro real, las neuronas responden selectivamente a bordes
   * en orientaciones específicas (descubrimiento de Hubel & Wiesel, Nobel 1981).
   * 
   * Aquí usamos convoluciones simples tipo Sobel para 4 orientaciones.
   */
  private detectEdges(img: Float32Array, w: number, h: number): Float32Array {
    const orientations = this.config.gaborOrientations;
    const result = new Float32Array(w * h * orientations);

    // Kernels para diferentes orientaciones
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
          // Normalizar a 0-1 y tomar valor absoluto
          result[offset + y * w + x] = Math.min(1, Math.abs(sum) / 4);
        }
      }
    }

    return result;
  }
}
