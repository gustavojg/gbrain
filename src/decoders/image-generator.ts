/**
 * GENERADOR DE IMÁGENES — Corteza visual inversa
 * ================================================
 * Reconstruye representaciones visuales a partir de patrones
 * de activación neuronal de la corteza visual.
 * 
 * Biología: Cuando "imaginamos" algo, la corteza visual se activa
 * en dirección inversa — las capas superiores (objetos) activan
 * las inferiores (bordes, píxeles). Esto se llama "imagery".
 * 
 * Este generador toma un patrón de activación cortical y produce
 * una imagen pixelada reconstruida (como un "sueño visual").
 */

/** Configuración del generador de imágenes */
export interface ImageGeneratorConfig {
  /** Ancho de la imagen de salida */
  outputWidth: number;
  /** Alto de la imagen de salida */
  outputHeight: number;
  /** Número de neuronas en la corteza visual */
  cortexNeurons: number;
}

const DEFAULT_IMAGE_CONFIG: ImageGeneratorConfig = {
  outputWidth: 32,
  outputHeight: 32,
  cortexNeurons: 10000,
};

/**
 * Generador de imágenes — Decodifica actividad visual a píxeles.
 */
export class ImageGenerator {
  private config: ImageGeneratorConfig;
  /** Matriz de decodificación: mapea neuronas → píxeles */
  private decoderMatrix: Float32Array;
  /** Tamaño del output (ancho × alto) */
  public outputSize: number;

  constructor(config: Partial<ImageGeneratorConfig> = {}) {
    this.config = { ...DEFAULT_IMAGE_CONFIG, ...config };
    this.outputSize = this.config.outputWidth * this.config.outputHeight;
    
    // Inicializar matriz de decodificación con pesos aleatorios
    // En un sistema entrenado, esta se aprende junto con la codificación
    this.decoderMatrix = new Float32Array(this.config.cortexNeurons * this.outputSize);
    this.initializeDecoder();
  }

  /**
   * Inicializa la matriz de decodificación con pesos sparse aleatorios.
   * Cada neurona cortical "ilumina" un subconjunto de píxeles.
   */
  private initializeDecoder(): void {
    const pixelsPerNeuron = Math.max(3, Math.floor(this.outputSize * 0.02)); // ~2% cobertura
    
    for (let n = 0; n < this.config.cortexNeurons; n++) {
      // Cada neurona tiene un "campo receptivo" (receptive field)
      // — la zona de la imagen a la que es sensible
      const centerX = Math.random() * this.config.outputWidth;
      const centerY = Math.random() * this.config.outputHeight;
      const radius = 2 + Math.random() * 4;
      
      for (let p = 0; p < pixelsPerNeuron; p++) {
        // Posición aleatoria cerca del centro
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
   * Genera una imagen a partir de la activación cortical.
   * 
   * @param corticalActivity - Patrón de activación de la corteza visual (Float32Array)
   * @returns Imagen reconstruida como array 2D de valores 0-255
   */
  generate(corticalActivity: Float32Array): {
    pixels: Float32Array;
    width: number;
    height: number;
    base64?: string;
  } {
    const pixels = new Float32Array(this.outputSize);
    
    // Multiplicar actividad cortical × matriz de decodificación
    const activityLen = Math.min(corticalActivity.length, this.config.cortexNeurons);
    
    for (let n = 0; n < activityLen; n++) {
      if (corticalActivity[n] > 0.1) { // Solo neuronas significativamente activas
        const offset = n * this.outputSize;
        const activation = corticalActivity[n];
        
        for (let p = 0; p < this.outputSize; p++) {
          pixels[p] += this.decoderMatrix[offset + p] * activation;
        }
      }
    }
    
    // Normalizar a rango 0-255
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
   * Aprende a decodificar: ajusta la matriz de decodificación
   * para que la reconstrucción se parezca más al input original.
   * 
   * @param corticalActivity - Activación cortical durante la codificación
   * @param originalPixels - Imagen original (normalizada 0-1)
   * @param learningRate - Tasa de aprendizaje
   */
  learn(
    corticalActivity: Float32Array,
    originalPixels: Float32Array,
    learningRate: number = 0.01
  ): void {
    // Generar reconstrucción actual
    const { pixels: reconstructed } = this.generate(corticalActivity);
    
    // Calcular error por pixel
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
   * Genera una representación ASCII art de la imagen (para consola).
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
