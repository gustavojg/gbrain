/**
 * Memoria de Trabajo (7±2 slots)
 * ================================
 * Modela la memoria de trabajo (working memory) según el modelo de Baddeley & Hitch.
 *
 * Biología: La corteza prefrontal mantiene ~7±2 elementos activos simultáneamente
 * (Miller, 1956). Cada "slot" es un patrón de activación neural sostenido por
 * actividad recurrente en circuitos prefrontales-parietales. Sin "rehearsal"
 * (refresco atencional), los patrones decaen en ~15-30 segundos.
 *
 * Implementación: Cada slot almacena un patrón (Float32Array), una etiqueta,
 * su marca temporal y un contador de refrescos. Los patrones no refrescados
 * pierden fuerza y son desalojados cuando su intensidad cae bajo un umbral.
 */

/**
 * Slot individual de memoria de trabajo.
 * Representa un ítem activamente mantenido en la conciencia.
 */
export interface WorkingMemorySlot {
  /** Patrón de activación neural representando el contenido */
  pattern: Float32Array;
  /** Etiqueta semántica del contenido (ej: "rostro_conocido", "palabra_escuchada") */
  label: string;
  /** Marca temporal del último acceso/refresco (ms) */
  timestamp: number;
  /** Número de veces que este patrón ha sido refrescado (rehearsal) */
  refreshCount: number;
  /** Fuerza actual del patrón (1.0 = máxima, decae sin refresco) */
  strength: number;
}

/**
 * Sistema de memoria de trabajo con capacidad limitada.
 *
 * Implementa el modelo clásico de 7±2 slots con decaimiento temporal.
 * Los patrones que no son atendidos (refrescados) decaen exponencialmente
 * y son eventualmente desalojados, liberando espacio para nueva información.
 */
export class WorkingMemory {
  /** Slots activos de memoria de trabajo */
  private slots: WorkingMemorySlot[] = [];
  /** Número máximo de slots disponibles (7±2) */
  public readonly maxSlots: number;
  /** Umbral mínimo de fuerza; bajo este valor el slot se desaloja */
  private readonly evictionThreshold: number = 0.15;
  /** Constante de decaimiento temporal (ms). Tiempo de vida media ~20s */
  private readonly decayHalfLife: number = 20_000;

  /**
   * Crea un nuevo sistema de memoria de trabajo.
   *
   * @param maxSlots - Capacidad máxima (default 7, según Miller 1956)
   */
  constructor(maxSlots: number = 7) {
    if (maxSlots <= 0) {
      throw new Error(`[WorkingMemory] maxSlots debe ser > 0, recibido: ${maxSlots}`);
    }
    this.maxSlots = maxSlots;
  }

  /**
   * Atiende (agrega o refresca) un patrón en la memoria de trabajo.
   *
   * Biología: Equivale al proceso de "rehearsal" prefrontal. Si el patrón
   * ya existe (por similitud coseno con un slot existente), se refresca
   * su fuerza y timestamp. Si es nuevo y hay espacio, se agrega.
   * Si no hay espacio, se desaloja el slot más débil.
   *
   * @param pattern - Vector de activación neural del patrón a mantener
   * @param label - Etiqueta semántica descriptiva
   * @returns true si se añadió/refrescó exitosamente
   */
  attend(pattern: Float32Array, label: string): boolean {
    const now = Date.now();

    // Buscar slot existente con la misma etiqueta para refrescar
    const existingIndex = this.slots.findIndex(s => s.label === label);

    if (existingIndex >= 0) {
      // Rehearsal: refrescar el patrón existente
      const slot = this.slots[existingIndex];
      slot.pattern.set(pattern);
      slot.timestamp = now;
      slot.refreshCount++;
      slot.strength = Math.min(1.0, slot.strength + 0.3);
      return true;
    }

    // Crear nuevo slot
    const newSlot: WorkingMemorySlot = {
      pattern: new Float32Array(pattern),
      label,
      timestamp: now,
      refreshCount: 0,
      strength: 1.0,
    };

    if (this.slots.length < this.maxSlots) {
      // Hay espacio disponible
      this.slots.push(newSlot);
    } else {
      // Desalojar el slot más débil (menor strength)
      let weakestIndex = 0;
      let weakestStrength = this.slots[0].strength;

      for (let i = 1; i < this.slots.length; i++) {
        if (this.slots[i].strength < weakestStrength) {
          weakestStrength = this.slots[i].strength;
          weakestIndex = i;
        }
      }

      this.slots[weakestIndex] = newSlot;
    }

    return true;
  }

  /**
   * Aplica decaimiento temporal a todos los slots.
   *
   * Biología: Sin actividad recurrente prefrontal que mantenga los patrones
   * activos, la representación neural se desvanece por interferencia y
   * ruido sináptico. La constante de decaimiento modela esto como una
   * exponencial con vida media de ~20 segundos.
   *
   * @param dt - Tiempo transcurrido desde el último tick (ms)
   */
  decay(dt: number): void {
    // Factor de decaimiento exponencial: strength *= e^(-dt * ln(2) / halfLife)
    const decayFactor = Math.exp((-dt * Math.LN2) / this.decayHalfLife);

    // Iterar en reversa para poder eliminar elementos de forma segura
    for (let i = this.slots.length - 1; i >= 0; i--) {
      this.slots[i].strength *= decayFactor;

      // Desalojar si cayó bajo el umbral
      if (this.slots[i].strength < this.evictionThreshold) {
        this.slots.splice(i, 1);
      }
    }
  }

  /**
   * Retorna los patrones actualmente mantenidos en memoria de trabajo.
   *
   * @returns Copia de los slots activos, ordenados por fuerza descendente
   */
  getActive(): ReadonlyArray<WorkingMemorySlot> {
    // Retornar copia ordenada por fuerza (más fuerte primero)
    return [...this.slots].sort((a, b) => b.strength - a.strength);
  }

  /**
   * Busca un slot por su etiqueta.
   *
   * @param label - Etiqueta del slot a buscar
   * @returns El slot si existe, undefined si no
   */
  find(label: string): WorkingMemorySlot | undefined {
    return this.slots.find(s => s.label === label);
  }

  /**
   * Indica si la memoria de trabajo está llena.
   * Cuando está llena, nuevos patrones desalojan al más débil.
   */
  isFull(): boolean {
    return this.slots.length >= this.maxSlots;
  }

  /**
   * Limpia completamente la memoria de trabajo.
   *
   * Biología: Equivale a una distracción abrupta o cambio de tarea
   * que libera todos los recursos prefrontales.
   */
  clear(): void {
    this.slots = [];
  }

  /**
   * Retorna el número de slots actualmente ocupados.
   */
  get activeCount(): number {
    return this.slots.length;
  }
}
