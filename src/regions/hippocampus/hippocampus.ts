/**
 * HIPOCAMPO — Centro de Memoria Episódica (núcleo biológico real)
 * ================================================================
 * Modela la formación hipocampal del lóbulo temporal medial. A diferencia
 * de la corteza visual (mapa feedforward Izhikevich+STDP), el cómputo
 * natural del hipocampo es una MEMORIA AUTOASOCIATIVA RECURRENTE: una red
 * de atractores que reconstruye un patrón completo a partir de una pista
 * parcial o degradada (pattern completion).
 *
 * Circuito trisináptico modelado:
 *   Corteza entorrinal → DG (separación) → CA3 (autoasociación) → salida
 *
 *   - Giro dentado (DG): separación de patrones. Proyección sparse FIJA
 *     (fibras musgosas, no plásticas) + competencia k-WTA que ortogonaliza
 *     entradas similares en códigos ultra-sparse (~2% activo). La proyección
 *     es DETERMINISTA (sembrada con una semilla constante) para que el mismo
 *     estímulo genere SIEMPRE el mismo código, también tras recargar pesos.
 *
 *   - CA3: completación de patrones. Red autoasociativa con conexiones
 *     recurrentes plásticas (matriz N×N = this.weights). Cada episodio se
 *     graba por aprendizaje Hebbiano (regla de coincidencia, producto
 *     externo del código sparse) con cota suave. El recuerdo NO es un
 *     lookup sobre patrones guardados: emerge de la DINÁMICA DE ATRACTORES
 *     (iteración recurrente h = W·s + k-WTA hasta converger).
 *
 *   Consecuencia clave: la memoria vive en las SINAPSIS (this.weights), así
 *   que el protocolo de persistencia binaria la serializa automáticamente.
 *   El aprendizaje sobrevive a recargas y redeploys.
 *
 * Referencias: O'Reilly & McClelland (1994), "Hippocampal conjunctive
 *   encoding, storage, and recall"; Treves & Rolls (1994), atractores CA3;
 *   Marr (1971), teoría del archicórtex.
 */

import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';

// ==================================================================
// Interfaces (API pública estable)
// ==================================================================

/** Contexto asociado a una memoria episódica. */
export interface EpisodicContext {
  /** Marca temporal del momento de codificación (ms) */
  timestamp: number;
  /** Valencia emocional del evento (-1 negativo … +1 positivo). */
  emotionalValence: number;
  /** Región cerebral de origen del patrón (ej: 'visualCortex') */
  sourceRegion: string;
}

/**
 * Índice episódico: metadato ligero de un evento codificado.
 * El PATRÓN aquí es el código sparse del DG (la "llave" del atractor); la
 * reconstrucción real la produce CA3 desde los pesos, no este registro.
 */
export interface EpisodicMemory {
  /** Código sparse del DG asociado al evento (engrama-índice). */
  pattern: Float32Array;
  /** Contexto asociado al evento (cuándo, valencia, fuente). */
  context: EpisodicContext;
  /** Fuerza del trazo (0–1); decae con el olvido, sube con el replay. */
  strength: number;
}

/** Resultado de una operación de recuerdo (recall). */
export interface RecallResult {
  /** Memoria recuperada (metadato + contexto). */
  memory: EpisodicMemory;
  /** Solapamiento [0,1] entre el atractor recuperado y el código del evento. */
  similarity: number;
}

/** Configuración del hipocampo. */
export interface HippocampusConfig {
  /** Sparsity del código CA3/DG (fracción de unidades activas). */
  sparsity: number;
  /** Conexiones de entrada por unidad DG (fan-in de fibras musgosas). */
  dgFanIn: number;
  /** Tasa de aprendizaje Hebbiano (escala de Δw por coincidencia). */
  learnRate: number;
  /** Techo de peso recurrente (cota suave LTP). */
  maxWeight: number;
  /** Iteraciones de la dinámica de atractores en el completado. */
  attractorIterations: number;
  /** Energía mínima de entrada para considerar que hay estímulo. */
  inputEnergyThreshold: number;
  /**
   * Umbral de novedad: si el código DG del input solapa con el último
   * codificado por encima de esto, NO se vuelve a grabar (codificación
   * dirigida por eventos, no por tick → evita saturar y floodear el índice).
   */
  noveltyOverlapThreshold: number;
  /** Semilla determinista de la proyección DG (conectividad fija). */
  dgSeed: number;
}

const DEFAULT_HIPPO_CONFIG: HippocampusConfig = {
  sparsity: 0.02,
  dgFanIn: 32,
  learnRate: 0.5,
  maxWeight: 1.0,
  attractorIterations: 6,
  inputEnergyThreshold: 1.0,
  noveltyOverlapThreshold: 0.9,
  dgSeed: 0x1d0c_a3e5,
};

// ==================================================================
// PRNG determinista (mulberry32) — proyección DG reproducible
// ==================================================================

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ==================================================================
// Clase Hippocampus — CA3 autoasociativo
// ==================================================================

export class Hippocampus extends BrainRegion {
  private readonly cfg: HippocampusConfig;

  /** Unidades activas por código (k del k-WTA). */
  private readonly kActive: number;

  /** Capacidad máxima del índice episódico (metadatos). */
  private readonly maxCapacity: number;

  /** Índice episódico (metadatos de contexto; la memoria real está en pesos). */
  private episodicMemories: EpisodicMemory[] = [];

  /** Proyección DG determinista: índices de entrada por unidad (N×fanIn). */
  private readonly dgIdx: Int32Array;
  /** Proyección DG determinista: signo ±1 por conexión (N×fanIn). */
  private readonly dgSign: Float32Array;

  /** Último código DG grabado (para la codificación por novedad). */
  private lastStoredCode: Float32Array;

  // --- Buffers reutilizados (evitan asignaciones por tick) ---
  private readonly actBuf: Float32Array;
  private readonly stateBuf: Float32Array;
  private readonly nextBuf: Float32Array;
  private readonly sortIdx: Int32Array;

  // --- Métricas de aprendizaje en vivo (paridad con la corteza visual) ---
  private prevEngram: Float32Array;
  private lastEngram: Float32Array;
  private weightChangeEMA = 0;
  private engramStabilityEMA = 0;
  private activityEMA = 0;
  private cumWeightChange = 0;

  /**
   * @param neuronCount - Unidades CA3 (default: 1000)
   * @param inputCount - Dimensión del input cortical (default: 1000)
   * @param maxCapacity - Capacidad del índice episódico (default: 10000)
   * @param config - Sobre-escrituras de configuración
   */
  constructor(
    neuronCount: number = 1000,
    inputCount: number = 1000,
    maxCapacity: number = 10000,
    config: Partial<HippocampusConfig> = {},
  ) {
    super('hippocampus', 'Hipocampo — Memoria Episódica', neuronCount, inputCount);

    this.cfg = { ...DEFAULT_HIPPO_CONFIG, ...config };
    this.maxCapacity = maxCapacity;
    this.kActive = Math.max(1, Math.floor(neuronCount * this.cfg.sparsity));

    // CA3 empieza en blanco: la base inicializó pesos sparse aleatorios, los
    // ponemos a cero para que el aprendizaje Hebbiano sea la única fuente de
    // las conexiones recurrentes (memoria limpia, sin atractores espurios).
    this.weights.fill(0);

    // Proyección DG determinista (conectividad fija de fibras musgosas).
    const fanIn = this.cfg.dgFanIn;
    this.dgIdx = new Int32Array(neuronCount * fanIn);
    this.dgSign = new Float32Array(neuronCount * fanIn);
    const rng = mulberry32(this.cfg.dgSeed);
    for (let u = 0; u < neuronCount; u++) {
      const base = u * fanIn;
      for (let f = 0; f < fanIn; f++) {
        this.dgIdx[base + f] = Math.floor(rng() * inputCount);
        this.dgSign[base + f] = rng() < 0.5 ? -1 : 1;
      }
    }

    this.actBuf = new Float32Array(neuronCount);
    this.stateBuf = new Float32Array(neuronCount);
    this.nextBuf = new Float32Array(neuronCount);
    this.sortIdx = new Int32Array(neuronCount);
    this.lastStoredCode = new Float32Array(neuronCount);
    this.prevEngram = new Float32Array(neuronCount);
    this.lastEngram = new Float32Array(neuronCount);
  }

  // ----------------------------------------------------------------
  // Giro Dentado: separación de patrones (determinista)
  // ----------------------------------------------------------------

  /**
   * Separación de patrones (DG): proyección sparse fija + ReLU + k-WTA.
   * Ortogonaliza entradas similares en códigos ultra-sparse reproducibles.
   *
   * @param input - Patrón de entrada cortical (Float32Array, dim inputCount)
   * @returns Código sparse binario (Float32Array dim neuronCount; 1 = activo)
   */
  patternSeparation(input: Float32Array): Float32Array {
    const out = new Float32Array(this.neuronCount);
    this.dgEncodeInto(input, out);
    return out;
  }

  /** Codifica DG sobre un buffer destino (sin asignar). */
  private dgEncodeInto(input: Float32Array, out: Float32Array): void {
    const n = this.neuronCount;
    const fanIn = this.cfg.dgFanIn;
    const act = this.actBuf;

    // Proyección aleatoria fija + ReLU.
    for (let u = 0; u < n; u++) {
      const base = u * fanIn;
      let sum = 0;
      for (let f = 0; f < fanIn; f++) {
        const idx = this.dgIdx[base + f];
        const v = idx < input.length ? input[idx] : 0;
        sum += this.dgSign[base + f] * v;
      }
      act[u] = sum > 0 ? sum : 0;
    }

    // k-WTA: solo las kActive unidades más excitadas sobreviven (→ binario).
    out.fill(0);
    const winners = this.topKIndices(act, this.kActive);
    for (let i = 0; i < winners.length; i++) out[winners[i]] = 1;
  }

  // ----------------------------------------------------------------
  // CA3: almacenamiento autoasociativo (Hebbian) y completado (atractor)
  // ----------------------------------------------------------------

  /**
   * Graba un episodio en CA3 por aprendizaje Hebbiano (producto externo del
   * código sparse) con cota suave. La memoria queda en los pesos recurrentes.
   *
   * @param pattern - Patrón cortical a codificar
   * @param context - Contexto asociado (timestamp, valencia, fuente)
   * @returns Σ|Δw| aplicado (energía de aprendizaje de este evento)
   */
  store(pattern: Float32Array, context: Partial<EpisodicContext> = {}): number {
    const code = this.patternSeparation(pattern);
    const dw = this.imprint(code, this.cfg.learnRate);

    const fullContext: EpisodicContext = {
      timestamp: context.timestamp ?? this.currentTime,
      emotionalValence: context.emotionalValence ?? 0,
      sourceRegion: context.sourceRegion ?? 'unknown',
    };
    const memory: EpisodicMemory = { pattern: code, context: fullContext, strength: 1.0 };

    if (this.episodicMemories.length >= this.maxCapacity) {
      // Reemplazar el trazo más débil (interferencia/olvido por competencia).
      let weakestIdx = 0;
      for (let i = 1; i < this.episodicMemories.length; i++) {
        if (this.episodicMemories[i].strength < this.episodicMemories[weakestIdx].strength) {
          weakestIdx = i;
        }
      }
      this.episodicMemories[weakestIdx] = memory;
    } else {
      this.episodicMemories.push(memory);
    }

    this.lastStoredCode.set(code);
    return dw;
  }

  /**
   * Producto externo Hebbiano del código sparse sobre los pesos recurrentes.
   * Solo recorre pares activos (kActive²), así que es barato pese a N×N.
   * Cota suave: Δw ∝ (maxW − w) → punto fijo estable, sin saturación dura.
   */
  private imprint(code: Float32Array, lr: number): number {
    const n = this.neuronCount;
    const active: number[] = [];
    for (let i = 0; i < n; i++) if (code[i] > 0) active.push(i);

    const maxW = this.cfg.maxWeight;
    let dwTotal = 0;
    for (let a = 0; a < active.length; a++) {
      const i = active[a];
      const row = i * n;
      for (let b = 0; b < active.length; b++) {
        if (a === b) continue; // sin auto-conexión
        const j = active[b];
        const idx = row + j;
        const dw = lr * (maxW - this.weights[idx]);
        this.weights[idx] += dw;
        dwTotal += Math.abs(dw);
      }
    }
    return dwTotal;
  }

  /**
   * Completación de patrones (CA3): dinámica de atractores recurrente.
   * Parte de la pista degradada (vía DG), itera h = W·s seguido de k-WTA, y
   * converge al atractor (episodio) más cercano en la cuenca de la pista.
   *
   * @param partialInput - Patrón parcial o degradado (dim inputCount)
   * @returns Código sparse reconstruido (Float32Array dim neuronCount)
   */
  patternCompletion(partialInput: Float32Array): Float32Array {
    const n = this.neuronCount;
    let state = this.stateBuf;
    let next = this.nextBuf;

    // Estado inicial = código DG de la pista (puede estar incompleto/ruidoso).
    this.dgEncodeInto(partialInput, state);

    for (let iter = 0; iter < this.cfg.attractorIterations; iter++) {
      // h[i] = Σ_j W[i,j] · state[j], recorriendo solo j activos (sparse).
      const activeJ: number[] = [];
      for (let j = 0; j < n; j++) if (state[j] > 0) activeJ.push(j);

      // Sin aprendizaje todavía → devolver el propio código DG (graceful).
      if (activeJ.length === 0) break;

      for (let i = 0; i < n; i++) {
        const row = i * n;
        let h = 0;
        for (let a = 0; a < activeJ.length; a++) h += this.weights[row + activeJ[a]];
        this.actBuf[i] = h;
      }

      next.fill(0);
      const winners = this.topKIndices(this.actBuf, this.kActive);
      // Si los pesos no producen señal (todo 0), preservar el estado actual.
      let anySignal = false;
      for (let i = 0; i < winners.length; i++) {
        if (this.actBuf[winners[i]] > 0) {
          next[winners[i]] = 1;
          anySignal = true;
        }
      }
      if (!anySignal) break;

      // ¿Convergió? (estado estable → atractor alcanzado)
      if (Hippocampus.overlapBinary(state, next) >= 0.999) {
        const tmp = state;
        state = next;
        next = tmp;
        break;
      }
      const tmp = state;
      state = next;
      next = tmp;
    }

    return new Float32Array(state);
  }

  // ----------------------------------------------------------------
  // Recuerdo, replay y olvido (sobre el índice episódico)
  // ----------------------------------------------------------------

  /**
   * Recupera las K memorias cuyo código solapa más con el atractor evocado
   * por la pista. El recuerdo se fundamenta en la dinámica de CA3 (pesos),
   * no en comparar la pista cruda con patrones guardados.
   */
  recall(cue: Float32Array, topK: number = 5): RecallResult[] {
    if (this.episodicMemories.length === 0) return [];
    const completed = this.patternCompletion(cue);

    const results: RecallResult[] = [];
    for (const memory of this.episodicMemories) {
      const sim = Hippocampus.overlapBinary(completed, memory.pattern);
      const adjusted = sim * (0.5 + 0.5 * memory.strength);
      results.push({ memory, similarity: adjusted });
    }
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }

  /**
   * Replay hipocampal: reactiva (y refuerza) los episodios más recientes y
   * fuertes. Cada reactivación re-graba el código en CA3 (consolidación).
   */
  replay(count: number = 10): EpisodicMemory[] {
    if (this.episodicMemories.length === 0) return [];
    const candidates = [...this.episodicMemories]
      .sort((a, b) => {
        const sa = a.context.timestamp * 0.0001 + a.strength;
        const sb = b.context.timestamp * 0.0001 + b.strength;
        return sb - sa;
      })
      .slice(0, count);

    for (const memory of candidates) {
      memory.strength = Math.min(1.0, memory.strength + 0.05);
      this.imprint(memory.pattern, this.cfg.learnRate); // refuerzo del engrama
    }
    return candidates;
  }

  /**
   * Olvido gradual del índice episódico. Las memorias emocionales decaen más
   * lento (modulación β-adrenérgica de la consolidación, McGaugh 2004).
   * No borra pesos recurrentes: el atractor persiste aunque el índice se pierda.
   */
  forget(decayFactor: number): void {
    for (let i = this.episodicMemories.length - 1; i >= 0; i--) {
      const memory = this.episodicMemories[i];
      const emotionalProtection = 1.0 - Math.abs(memory.context.emotionalValence) * 0.3;
      memory.strength *= decayFactor * emotionalProtection;
      if (memory.strength < 0.01) this.episodicMemories.splice(i, 1);
    }
  }

  // ----------------------------------------------------------------
  // Procesamiento principal (un tick del cerebro)
  // ----------------------------------------------------------------

  /**
   * Procesa spikes corticales: codifica el episodio (si es novedoso),
   * completa el patrón por atractor y emite el engrama reconstruido.
   */
  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    let inputEnergy = 0;
    for (let i = 0; i < spikes.length; i++) inputEnergy += spikes[i];

    if (inputEnergy < this.cfg.inputEnergyThreshold) {
      this.spikes.fill(0);
      return new Float32Array(this.neuronCount);
    }

    const input = this.adaptInput(spikes);
    const code = this.patternSeparation(input);

    // Codificación dirigida por novedad: solo grabar eventos distintos del
    // último (evita imprimir el mismo patrón en cada tick y floodear el índice).
    const novelty = 1 - Hippocampus.overlapBinary(code, this.lastStoredCode);
    let dw = 0;
    if (novelty > 1 - this.cfg.noveltyOverlapThreshold) {
      const lrMul = modulationEffects.learningRateMultiplier ?? 1.0;
      dw = this.storeWithGain(code, lrMul, {
        timestamp: this.currentTime,
        emotionalValence: 0,
        sourceRegion: 'sensory',
      });
    }

    // Completar por dinámica de atractores → engrama reconstruido.
    const completed = this.patternCompletion(input);

    const gain = modulationEffects.spikeGainMultiplier ?? 1.0;
    const out = new Float32Array(this.neuronCount);
    for (let i = 0; i < this.neuronCount; i++) {
      const v = completed[i] > 0 ? gain : 0;
      out[i] = v;
      this.spikes[i] = completed[i] > 0 ? 1 : 0;
    }

    this.updateLearningMetrics(dw, completed);
    return out;
  }

  /** Variante de store que escala la tasa Hebbiana por neuromodulación. */
  private storeWithGain(
    code: Float32Array,
    lrMul: number,
    context: EpisodicContext,
  ): number {
    const dw = this.imprint(code, this.cfg.learnRate * lrMul);

    const memory: EpisodicMemory = { pattern: code, context, strength: 1.0 };
    if (this.episodicMemories.length >= this.maxCapacity) {
      let weakestIdx = 0;
      for (let i = 1; i < this.episodicMemories.length; i++) {
        if (this.episodicMemories[i].strength < this.episodicMemories[weakestIdx].strength) {
          weakestIdx = i;
        }
      }
      this.episodicMemories[weakestIdx] = memory;
    } else {
      this.episodicMemories.push(memory);
    }
    this.lastStoredCode.set(code);
    return dw;
  }

  // ----------------------------------------------------------------
  // Métricas de aprendizaje en vivo (dashboard)
  // ----------------------------------------------------------------

  private updateLearningMetrics(dw: number, engram: Float32Array): void {
    this.prevEngram = this.lastEngram;
    this.lastEngram = engram;

    const stability = Hippocampus.overlapBinary(this.prevEngram, this.lastEngram);
    this.engramStabilityEMA = 0.9 * this.engramStabilityEMA + 0.1 * stability;
    this.weightChangeEMA = 0.95 * this.weightChangeEMA + 0.05 * dw;
    this.cumWeightChange += dw;

    let active = 0;
    for (let i = 0; i < engram.length; i++) if (engram[i] > 0) active++;
    this.activityEMA = 0.95 * this.activityEMA + 0.05 * (active / this.neuronCount);
  }

  /** Métricas de aprendizaje (misma forma que la corteza visual). */
  getLearningMetrics(): {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
    memoryCount: number;
  } {
    const engram: number[] = [];
    for (let i = 0; i < this.lastEngram.length; i++) {
      if (this.lastEngram[i] > 0) engram.push(i);
    }
    return {
      engram,
      engramSize: engram.length,
      stability: this.engramStabilityEMA,
      weightChange: this.weightChangeEMA,
      cumWeightChange: this.cumWeightChange,
      activity: this.activityEMA,
      neuronCount: this.neuronCount,
      memoryCount: this.episodicMemories.length,
    };
  }

  // ----------------------------------------------------------------
  // Utilidades internas
  // ----------------------------------------------------------------

  /** Adapta un vector de entrada a inputCount (trunca o rellena con ceros). */
  private adaptInput(input: Float32Array): Float32Array {
    if (input.length === this.inputCount) return input;
    const adapted = new Float32Array(this.inputCount);
    adapted.set(input.subarray(0, Math.min(input.length, this.inputCount)));
    return adapted;
  }

  /** Índices de los k valores mayores (> 0) de un array. */
  private topKIndices(arr: Float32Array, k: number): Int32Array {
    const n = arr.length;
    for (let i = 0; i < n; i++) this.sortIdx[i] = i;
    this.sortIdx.sort((a, b) => arr[b] - arr[a]);
    const out: number[] = [];
    for (let i = 0; i < k && i < n; i++) {
      if (arr[this.sortIdx[i]] > 0) out.push(this.sortIdx[i]);
    }
    return Int32Array.from(out);
  }

  /** Solapamiento de Jaccard entre dos códigos sparse binarios (>0 = activo). */
  static overlapBinary(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let inter = 0;
    let ca = 0;
    let cb = 0;
    for (let i = 0; i < len; i++) {
      const ai = a[i] > 0 ? 1 : 0;
      const bi = b[i] > 0 ? 1 : 0;
      ca += ai;
      cb += bi;
      inter += ai & bi;
    }
    if (ca === 0 && cb === 0) return 1;
    const union = ca + cb - inter;
    return union > 0 ? inter / union : 0;
  }

  // ----------------------------------------------------------------
  // Propiedades públicas
  // ----------------------------------------------------------------

  /** Número actual de episodios indexados. */
  get memoryCount(): number {
    return this.episodicMemories.length;
  }

  /** Capacidad máxima del índice episódico. */
  get capacity(): number {
    return this.maxCapacity;
  }

  /** Ocupación [0,1] del índice episódico. */
  get occupancy(): number {
    return this.episodicMemories.length / this.maxCapacity;
  }
}
