/**
 * CODIFICACIÓN DE SPIKES — Utilidades
 * ====================================
 * Convierte señales continuas en trenes de spikes y viceversa.
 * 
 * El cerebro biológico no trabaja con números continuos — trabaja con
 * pulsos discretos (spikes). Estos métodos convierten entre ambos dominios.
 * 
 * Dos esquemas principales:
 * - Rate Coding: La intensidad se codifica como FRECUENCIA de spikes
 * - Temporal Coding: La intensidad se codifica como TIEMPO del spike
 */

/**
 * Rate Coding: Convierte un valor continuo en probabilidad de spike.
 * Valores más altos → más probable que dispare en este timestep.
 * 
 * Biología: Las neuronas sensoriales codifican la intensidad del estímulo
 * como frecuencia de disparo (ej: más presión → más spikes por segundo).
 * 
 * @param value - Valor continuo a codificar (0-1 normalizado)
 * @param maxRate - Frecuencia máxima de disparo (Hz)
 * @param dt - Paso temporal actual (ms)
 * @returns 1 si genera spike, 0 si no
 */
export function rateCoding(value: number, maxRate: number = 100, dt: number = 1.0): number {
  // Probabilidad de spike en este timestep
  // P(spike) = rate * dt / 1000, donde rate = value * maxRate
  const rate = Math.max(0, Math.min(1, value)) * maxRate;
  const probability = rate * dt / 1000;
  return Math.random() < probability ? 1.0 : 0.0;
}

/**
 * Temporal Coding: Convierte un valor a un tiempo de spike.
 * Valores más altos → spike más TEMPRANO en la ventana.
 * 
 * Biología: Algunas neuronas codifican información en el MOMENTO EXACTO
 * del spike relativo a una oscilación de referencia. Esto es más eficiente
 * que rate coding para una sola presentación del estímulo.
 * 
 * @param value - Valor continuo (0-1 normalizado)
 * @param maxValue - Valor máximo posible
 * @param windowMs - Ventana temporal para el spike (ms)
 * @returns Tiempo del spike dentro de la ventana (ms), -1 si no dispara
 */
export function temporalCoding(value: number, maxValue: number = 1.0, windowMs: number = 20): number {
  const normalized = Math.max(0, Math.min(1, value / maxValue));
  if (normalized < 0.05) return -1; // Debajo del umbral, no dispara
  // Valor alto → spike temprano (tiempo bajo)
  return windowMs * (1 - normalized);
}

/**
 * Codifica un vector de valores continuos como vector de spikes (0/1).
 * Usa rate coding para cada componente.
 * 
 * @param values - Vector de valores continuos (Float32Array)
 * @param dt - Paso temporal (ms)
 * @param maxRate - Frecuencia máxima de disparo (Hz)
 * @returns Vector de spikes (Float32Array de 0s y 1s)
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
 * Decodifica un tren de spikes a una tasa de disparo (valor continuo).
 * 
 * @param spikeTrain - Array de 0s y 1s representando spikes en timesteps sucesivos
 * @param windowMs - Ventana temporal sobre la que promediar (ms)
 * @param dt - Paso temporal entre muestras (ms)
 * @returns Tasa de disparo estimada (0-1 normalizada)
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
 * Codifica una imagen (array de píxeles) como vector de spikes
 * usando rate coding con normalización.
 * 
 * @param pixels - Valores de píxeles (0-255 o 0-1)
 * @param normalize - Si normalizar de 0-255 a 0-1
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
  
  return encodeSpikeVector(values, dt, 200); // Rate más alto para visual
}

/**
 * Codifica un espectrograma de audio como vector de spikes.
 * Usa escala logarítmica para imitar la respuesta coclear.
 * 
 * @param spectrogram - Flat array del espectrograma [time][freq]
 * @param numBands - Número de bandas de frecuencia
 * @param numFrames - Número de frames temporales
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
    // Escala logarítmica (imita respuesta coclear)
    values[i] = Math.log1p(Math.abs(spectrogram[i] as number)) / Math.log1p(1);
    values[i] = Math.min(1, values[i]);
  }
  
  return encodeSpikeVector(values, dt, 150); // Rate moderado para audio
}

/**
 * Codifica texto como vector de spikes usando hash distribuido.
 * Cada carácter se mapea a un patrón sparse de activaciones.
 * 
 * Inspirado en "random projections" — técnica real usada en
 * neurociencia computacional para representaciones distribuidas.
 * 
 * @param text - Texto a codificar
 * @param vectorSize - Tamaño del vector de output
 * @returns Spike vector
 */
export function encodeTextToSpikes(
  text: string,
  vectorSize: number = 5000,
  dt: number = 1.0
): Float32Array {
  const values = new Float32Array(vectorSize);
  
  // Para cada carácter, activar un subconjunto determinista de posiciones
  const chars = text.toLowerCase().split('');
  
  for (let ci = 0; ci < chars.length; ci++) {
    const charCode = chars[ci].charCodeAt(0);
    const position = ci; // Posición en la secuencia
    
    // Hash determinista: cada (carácter, posición) activa ~5% del vector
    // Usamos un generador pseudo-aleatorio seeded
    let seed = charCode * 31 + position * 7919;
    const activations = Math.floor(vectorSize * 0.05);
    
    for (let a = 0; a < activations; a++) {
      // Simple hash-based pseudo-random
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      const idx = seed % vectorSize;
      values[idx] = Math.min(1.0, values[idx] + 0.3);
    }
  }
  
  // Normalizar
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
