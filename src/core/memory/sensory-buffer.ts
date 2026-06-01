/**
 * Buffer de Memoria Sensorial (~250ms)
 * =====================================
 * Modela la memoria icónica (visual) y ecoica (auditiva) del cerebro humano.
 *
 * Biología: La memoria sensorial retiene una copia casi exacta del estímulo
 * por ~250ms (icónica) o ~3-4s (ecoica). Se implementa como un buffer circular
 * (ring buffer) de alta velocidad usando Float32Array para almacenar vectores
 * de activación sensorial con sus marcas temporales.
 *
 * El buffer circular evita asignaciones de memoria al reutilizar posiciones,
 * crítico para operaciones en tiempo real a escala de 50K+ neuronas.
 */

/**
 * Entrada individual en el buffer sensorial.
 * Contiene el vector de activación y su marca temporal.
 */
export interface SensoryEntry {
  /** Vector de activación sensorial (patrón espacial del estímulo) */
  data: Float32Array;
  /** Marca temporal en milisegundos del momento de captura */
  timestamp: number;
}

/**
 * Buffer circular de memoria sensorial.
 *
 * Implementa un anillo (ring buffer) de tamaño fijo que almacena vectores
 * sensoriales con marcas temporales. Cuando el buffer se llena, las entradas
 * más antiguas se sobrescriben automáticamente, imitando el decaimiento
 * natural de la traza sensorial en el cerebro.
 */
export class SensoryBuffer {
  /** Almacenamiento contiguo para todos los vectores sensoriales */
  private readonly storage: Float32Array;
  /** Marcas temporales correspondientes a cada posición del anillo */
  private readonly timestamps: Float64Array;
  /** Capacidad máxima del buffer (número de entradas) */
  private readonly capacity: number;
  /** Dimensión de cada vector sensorial */
  private readonly vectorSize: number;
  /** Índice de escritura actual (cabeza del anillo) */
  private head: number = 0;
  /** Número de entradas válidas almacenadas */
  private count: number = 0;

  /**
   * Crea un nuevo buffer sensorial circular.
   *
   * @param capacity - Número máximo de entradas que puede almacenar el buffer
   * @param vectorSize - Dimensión de cada vector sensorial (ej: 128 para espectrograma)
   */
  constructor(capacity: number, vectorSize: number) {
    if (capacity <= 0 || vectorSize <= 0) {
      throw new Error(
        `[SensoryBuffer] Capacidad (${capacity}) y tamaño de vector (${vectorSize}) deben ser > 0`
      );
    }
    this.capacity = capacity;
    this.vectorSize = vectorSize;
    // Almacenamiento plano: capacity * vectorSize floats contiguos en memoria
    this.storage = new Float32Array(capacity * vectorSize);
    this.timestamps = new Float64Array(capacity);
  }

  /**
   * Inserta un nuevo vector sensorial en el buffer.
   *
   * Biología: Equivale a la llegada de un nuevo fotograma retinal o
   * segmento coclear al registro sensorial.
   *
   * @param data - Vector de activación sensorial a almacenar
   * @param timestamp - Marca temporal en ms del estímulo
   */
  push(data: Float32Array, timestamp: number): void {
    if (data.length !== this.vectorSize) {
      throw new Error(
        `[SensoryBuffer] Tamaño de datos (${data.length}) no coincide con vectorSize (${this.vectorSize})`
      );
    }

    // Calcular offset en el almacenamiento plano
    const offset = this.head * this.vectorSize;
    // Copiar datos al anillo (operación O(vectorSize), muy rápida con typed arrays)
    this.storage.set(data, offset);
    this.timestamps[this.head] = timestamp;

    // Avanzar cabeza del anillo
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  /**
   * Recupera todas las entradas dentro de una ventana temporal reciente.
   *
   * Biología: Simula el acceso a la traza sensorial que aún persiste
   * dentro de la ventana de memoria icónica/ecoica.
   *
   * @param durationMs - Duración de la ventana hacia atrás desde la entrada más reciente (ms)
   * @returns Array de entradas sensoriales dentro de la ventana, ordenadas cronológicamente
   */
  getRecent(durationMs: number): SensoryEntry[] {
    if (this.count === 0) return [];

    // Encontrar la marca temporal más reciente
    const lastIndex = (this.head - 1 + this.capacity) % this.capacity;
    const latestTimestamp = this.timestamps[lastIndex];
    const cutoff = latestTimestamp - durationMs;

    const results: SensoryEntry[] = [];

    // Recorrer el anillo desde la entrada más antigua a la más reciente
    for (let i = 0; i < this.count; i++) {
      const idx = (this.head - this.count + i + this.capacity) % this.capacity;
      const ts = this.timestamps[idx];

      if (ts >= cutoff) {
        // Extraer copia del vector desde el almacenamiento plano
        const offset = idx * this.vectorSize;
        const data = new Float32Array(this.vectorSize);
        data.set(this.storage.subarray(offset, offset + this.vectorSize));
        results.push({ data, timestamp: ts });
      }
    }

    return results;
  }

  /**
   * Limpia completamente el buffer sensorial.
   *
   * Biología: Equivale a un reset atencional abrupto, como el efecto
   * de un parpadeo o cambio de fijación visual (saccade).
   */
  clear(): void {
    this.storage.fill(0);
    this.timestamps.fill(0);
    this.head = 0;
    this.count = 0;
  }

  /**
   * Indica si el buffer ha alcanzado su capacidad máxima.
   * Una vez lleno, las nuevas entradas sobrescriben las más antiguas.
   */
  isFull(): boolean {
    return this.count >= this.capacity;
  }

  /**
   * Retorna el número de entradas válidas actualmente almacenadas.
   */
  get size(): number {
    return this.count;
  }

  /**
   * Retorna la capacidad máxima del buffer.
   */
  get maxCapacity(): number {
    return this.capacity;
  }

  /**
   * Retorna la dimensión de cada vector sensorial.
   */
  get dimensions(): number {
    return this.vectorSize;
  }
}
