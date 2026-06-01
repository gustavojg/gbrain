/**
 * CORTEZA VISUAL — Procesamiento de patrones visuales (núcleo biológico real)
 * ===========================================================================
 * Región de referencia del Camino A: dinámica de membrana de Izhikevich
 * ejecutada tick a tick, inhibición lateral k-WTA, período refractario,
 * corriente de fondo, homeostasis de excitabilidad y plasticidad STDP
 * basada en trazas pre/post.
 *
 * Base biológica:
 *   - Las neuronas de V1 (Hubel & Wiesel, 1962) responden a bordes/orientaciones.
 *   - Codificación sparse por inhibición lateral GABAérgica (k-WTA).
 *   - STDP (Bi & Poo, 1998): la pre que dispara ANTES que la post potencia
 *     la sinapsis (LTP); la pre que dispara DESPUÉS deprime (LTD).
 *   - Homeostasis (Turrigiano, 1998/2008): escalado sináptico + ajuste de
 *     excitabilidad intrínseca para mantener una tasa de disparo objetivo.
 *
 * Implementación de STDP:
 *   Se usa la formulación por TRAZAS (pair-based, nearest-neighbour), más
 *   estable numéricamente que comparar lastSpikeTime crudos cuando pre y post
 *   disparan en el mismo tick. Cada entrada mantiene una traza pre que decae
 *   con τ+, y cada neurona una traza post que decae con τ-:
 *     - Al disparar la post:  Δw_i = +A+ · preTrace_i      (LTP)
 *     - Al disparar la pre :  Δw_i = -A- · postTrace_n     (LTD)
 */

import type { SpikePacket } from '../../core/bus/spike-bus.js';
import { SpikeBus } from '../../core/bus/spike-bus.js';
import { BrainRegion } from '../../core/brain-region.js';
import type { ModulationEffects } from '../../core/neuromodulators/modulator-system.js';
import { SpikingNeuron, createNeuronPopulation } from '../../core/snn/neuron.js';
import type { NeuronTypeName } from '../../core/snn/neuron.js';
import { rateCoding } from '../../core/snn/spike-train.js';

// ====================================================================
// Tipos de la Corteza Visual
// ====================================================================

/** Memoria visual: patrón de neuronas activas asociado a una etiqueta. */
export interface VisualMemory {
  pattern: Int32Array;
  label: string;
  strength: number;
  createdAt: number;
}

/** Resultado del procesamiento visual. */
export interface VisualProcessingResult {
  winners: Int32Array;
  potentials: Float32Array;
  predictions: string[];
  activity: number;
}

/** Resultado de presentar un estímulo durante una ventana temporal. */
export interface PresentationResult {
  /** Engrama: índices de neuronas que más dispararon en la ventana. */
  engram: Int32Array;
  /** Conteo de spikes por neurona durante la ventana. */
  spikeCounts: Int32Array;
  /** Magnitud total del cambio de pesos en la ventana (Σ|Δw|). */
  weightChange: number;
  /** Fracción media de actividad cortical durante la ventana. */
  activity: number;
}

/** Configuración de la corteza visual. */
export interface VisualCortexConfig {
  neuronCount: number;
  inputCount: number;
  /** Número de neuronas ganadoras en k-WTA (tamaño del engrama). */
  kWinners: number;
  /** Factor de fatiga homeostática (penalización por wins acumulados). */
  fatigueFactor: number;
  /** Tasa de aprendizaje base (escala global de Δw). */
  learningRate: number;
  /** Amplitud de potenciación STDP (LTP). */
  aPlus: number;
  /** Amplitud de depresión STDP (LTD). */
  aMinus: number;
  /** Constante de tiempo de la traza LTP (ms). */
  tauPlus: number;
  /** Constante de tiempo de la traza LTD (ms). */
  tauMinus: number;
  /** Presupuesto máximo de peso sináptico por neurona (normalización). */
  maxWeightBudget: number;
  /** Rango de pesos permitidos [min, max]. */
  weightRange: [number, number];
  /** Ganancia de la corriente sináptica de entrada. */
  inputGain: number;
  /** Corriente de fondo (mantiene a las neuronas cerca del umbral). */
  backgroundCurrent: number;
  /** Techo de corriente de membrana (evita inestabilidad numérica de Izhikevich). */
  maxCurrent: number;
  /** Amplitud del ruido sináptico de fondo. */
  noiseAmplitude: number;
  /** Período refractario absoluto (ms). */
  refractoryMs: number;
  /** Ticks de simulación por presentación de un estímulo. */
  presentationTicks: number;
  /** Tasa de disparo objetivo por neurona (fracción, para homeostasis). */
  targetRate: number;
  /** Velocidad de la homeostasis de excitabilidad intrínseca. */
  homeostasisRate: number;
  /** Umbral de overlap para reconocimiento (de k neuronas). */
  recognitionThreshold: number;
  /** Tipo de neurona cortical. */
  neuronType: NeuronTypeName;
}

const DEFAULT_VISUAL_CONFIG: VisualCortexConfig = {
  neuronCount: 2000,
  inputCount: 1000,
  kWinners: 20,
  fatigueFactor: 0.05,
  learningRate: 1.0,
  aPlus: 0.02,
  aMinus: 0.012,
  tauPlus: 20.0,
  tauMinus: 20.0,
  maxWeightBudget: 40.0,
  weightRange: [-1.0, 4.0],
  inputGain: 8.0,
  backgroundCurrent: 2.0,
  maxCurrent: 20.0,
  noiseAmplitude: 0.6,
  refractoryMs: 2.0,
  presentationTicks: 60,
  targetRate: 0.02,
  homeostasisRate: 0.01,
  recognitionThreshold: 6,
  neuronType: 'RegularSpiking',
};

// ====================================================================
// Clase VisualCortex
// ====================================================================

export class VisualCortex extends BrainRegion {
  private config: VisualCortexConfig;

  /** Neuronas spiking locales (modelo Izhikevich con dinámica real). */
  private localNeurons: SpikingNeuron[];

  /** Traza presináptica por entrada (decae con τ+). */
  private preTrace: Float32Array;
  /** Traza postsináptica por neurona (decae con τ-). */
  private postTrace: Float32Array;
  /** Bias de excitabilidad intrínseca por neurona (homeostasis). */
  private homeostaticBias: Float32Array;
  /** Media móvil de actividad por neurona (para homeostasis). */
  private avgActivity: Float32Array;
  /** Traza de fatiga por neurona (decae cada tick; adaptación transitoria). */
  private winCounts: Float32Array;
  /** Excitación sináptica continua por neurona (Σ w·rate), reusado. */
  private excBuf: Float32Array;
  /** Buffer de corriente sináptica por neurona (reusado). */
  private currentBuf: Float32Array;
  /** Índices para el k-WTA (reusado). */
  private sortIdx: Int32Array;

  /** Memorias visuales almacenadas (engramas etiquetados). */
  private memories: VisualMemory[] = [];

  /** Referencia al spike bus. */
  private spikeBus: SpikeBus | null = null;

  /** Últimos ganadores (para consultas externas). */
  private lastWinners: Int32Array = new Int32Array(0);

  /** dt del último step (capturado desde la clase base). */
  private _dt: number = 1.0;

  // --- Métricas de aprendizaje en vivo (para el dashboard) ---
  /** Conteo de spikes con fuga: define un engrama estable pese al ruido por tick. */
  private recentSpikeCounts: Float32Array;
  /** Engrama en vivo del tick anterior (para medir estabilidad). */
  private prevLiveEngram: Int32Array = new Int32Array(0);
  /** Engrama en vivo actual (top-k de recentSpikeCounts). */
  private liveEngram: Int32Array = new Int32Array(0);
  /** EMA de Σ|Δw| por tick → decae al converger el aprendizaje. */
  private weightChangeEMA = 0;
  /** EMA del overlap engrama_t vs engrama_{t-1} → sube al consolidarse. */
  private engramStabilityEMA = 0;
  /** EMA de la fracción de neuronas activas por tick. */
  private liveActivityEMA = 0;
  /** Σ|Δw| acumulado desde el arranque (energía total de aprendizaje). */
  private cumWeightChange = 0;

  constructor(config: Partial<VisualCortexConfig> = {}) {
    const cfg = { ...DEFAULT_VISUAL_CONFIG, ...config };
    super('visualCortex', 'Corteza Visual', cfg.neuronCount, cfg.inputCount);

    this.config = cfg;
    this.localNeurons = createNeuronPopulation(cfg.neuronCount, cfg.neuronType);

    this.preTrace = new Float32Array(cfg.inputCount);
    this.postTrace = new Float32Array(cfg.neuronCount);
    this.homeostaticBias = new Float32Array(cfg.neuronCount);
    this.avgActivity = new Float32Array(cfg.neuronCount);
    this.winCounts = new Float32Array(cfg.neuronCount);
    this.excBuf = new Float32Array(cfg.neuronCount);
    this.currentBuf = new Float32Array(cfg.neuronCount);
    this.sortIdx = new Int32Array(cfg.neuronCount);
    this.recentSpikeCounts = new Float32Array(cfg.neuronCount);

    this.initializeVisualWeights();
  }

  /** Inicializa pesos sinápticos pequeños y aleatorios (sinaptogénesis). */
  private initializeVisualWeights(): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = Math.random() * 0.1;
    }
  }

  connectBus(bus: SpikeBus): void {
    this.spikeBus = bus;
    bus.register(this.id);
  }

  /** Captura dt y delega en la lógica de la clase base. */
  override step(dt: number, modulationEffects: ModulationEffects) {
    this._dt = dt;
    return super.step(dt, modulationEffects);
  }

  /**
   * Un tick de dinámica cortical: integra corrientes, ejecuta Izhikevich,
   * aplica inhibición lateral k-WTA y (si learn) plasticidad STDP por trazas.
   *
   * @param rates - Vector de tasas de entrada (0-1); se muestrea Poisson por tick.
   * @param dt - Paso temporal (ms).
   * @param t - Tiempo de simulación (ms).
   * @param learn - Si aplicar STDP en este tick.
   * @param lrMul - Multiplicador de tasa de aprendizaje (neuromodulación).
   * @param gain - Multiplicador de ganancia de spikes (neuromodulación).
   * @returns Δw total aplicado en este tick (Σ|Δw|).
   */
  private dynamicsTick(
    rates: Float32Array,
    dt: number,
    t: number,
    learn: boolean,
    lrMul: number,
    gain: number,
  ): number {
    const n = this.neuronCount;
    const m = this.inputCount;
    const cfg = this.config;

    // --- 0. Índices de entrada activos + decaer traza pre + muestrear Poisson ---
    const decayPre = Math.exp(-dt / cfg.tauPlus);
    for (let i = 0; i < m; i++) this.preTrace[i] *= decayPre;

    const activeInputs: number[] = [];
    for (let i = 0; i < m; i++) if (rates[i] > 0) activeInputs.push(i);

    // Spikes de entrada de este tick (rate coding Poisson) → realismo temporal STDP.
    const firedInputs: number[] = [];
    for (let a = 0; a < activeInputs.length; a++) {
      const i = activeInputs[a];
      if (rateCoding(rates[i], 200, dt) > 0.5) {
        firedInputs.push(i);
        this.preTrace[i] = 1.0; // nearest-neighbour: resetea la traza al spike
      }
    }

    // --- 1. Excitación CONTINUA (Σ w·rate) y SCORE de coincidencia coseno ---
    // excBuf = w·rate (magnitud, para la corriente de membrana).
    // scoreBuf = (w·rate)/‖w‖ → similitud coseno con el patrón: mide cuán bien
    // los pesos de la neurona "apuntan" al estímulo actual, no su magnitud bruta.
    // Sin esta normalización, unas pocas sinapsis compartidas saturadas a maxW
    // harían que el engrama de A se filtrara a B (falsa coincidencia).
    const score = this.currentBuf;
    for (let nn = 0; nn < n; nn++) {
      const offset = nn * m;
      let exc = 0;
      let norm2 = 0;
      for (let i = 0; i < m; i++) {
        const w = this.weights[offset + i];
        norm2 += w * w;
        if (rates[i] > 0) exc += w * rates[i];
      }
      this.excBuf[nn] = exc;
      score[nn] = exc / (Math.sqrt(norm2) + 1e-6);
    }

    // --- 2. Inhibición lateral k-WTA por score coseno (+ sesgo homeostático) ---
    // El bias entra con un peso PEQUEÑO (BIAS_GAIN): es un empujón de equidad
    // para que neuronas crónicamente silentes ganen desempates, NO un término
    // que pueda anular la coincidencia con el estímulo (eso colapsaría todos los
    // patrones al mismo engrama de "neuronas menos activas").
    const BIAS_GAIN = 0.05;
    for (let i = 0; i < n; i++) this.sortIdx[i] = i;
    const bias = this.homeostaticBias;
    this.sortIdx.sort((a, b) => score[b] + BIAS_GAIN * bias[b] - (score[a] + BIAS_GAIN * bias[a]));
    const k = cfg.kWinners;
    const isWinner = new Uint8Array(n);
    for (let i = 0; i < k && i < n; i++) isWinner[this.sortIdx[i]] = 1;

    // --- 3. Dinámica de membrana de Izhikevich (real) ---
    // Ganadores: corriente supra-umbral sostenida (fondo + ganancia·exc).
    // No-ganadores: solo fondo sub-umbral → su membrana evoluciona pero
    // (salvo empuje homeostático) no cruza el umbral. Modela la inhibición
    // lateral GABAérgica sin resetear la membrana de los competidores.
    const decayPost = Math.exp(-dt / cfg.tauMinus);
    const winnersFired: number[] = [];

    for (let nn = 0; nn < n; nn++) {
      this.postTrace[nn] *= decayPost;

      // Refractario absoluto
      if (t - this.localNeurons[nn].lastSpikeTime < cfg.refractoryMs) {
        this.localNeurons[nn].fired = false;
        this.spikes[nn] = 0;
        continue;
      }

      // El fondo es SUB-umbral para todos → un no-ganador nunca dispara solo por
      // fondo. El disparo queda estrictamente regido por el k-WTA: el sesgo
      // homeostático NO entra en la corriente de los no-ganadores (si lo hiciera,
      // un bias alto los haría disparar ignorando el estímulo y el engrama se
      // volvería independiente del patrón). El bias solo influye en QUIÉN gana
      // (vía BIAS_GAIN en el WTA) y modula levemente a los ganadores.
      const noise = (Math.random() - 0.5) * 2 * cfg.noiseAmplitude;
      let I = cfg.backgroundCurrent + noise;
      if (isWinner[nn]) {
        const fatigue = Math.min(this.winCounts[nn] * cfg.fatigueFactor, 3.0);
        I += this.homeostaticBias[nn] + cfg.inputGain * this.excBuf[nn] * gain - fatigue;
      }
      // Techo de corriente: la excitación de un engrama entrenado (w→maxW) puede
      // alcanzar cientos; con dt=1 ms el término 0.04v² de Izhikevich se vuelve
      // numéricamente inestable y u explota, haciendo que la neurona dispare en
      // bucle ignorando el estímulo. El clamp mantiene la integración estable.
      if (I > cfg.maxCurrent) I = cfg.maxCurrent;
      const fired = this.localNeurons[nn].step(I, dt, t);

      this.spikes[nn] = fired ? 1 : 0;
      if (fired) {
        this.postTrace[nn] = 1.0; // traza post para LTD futura
        if (isWinner[nn]) winnersFired.push(nn);
      }
    }

    // --- 4. Plasticidad STDP por trazas (solo si learn) ---
    let weightChange = 0;
    if (learn) {
      const lr = cfg.learningRate * lrMul;
      const [minW, maxW] = cfg.weightRange;

      // LTP: la post disparó → potenciar sinapsis con traza pre activa.
      // Cota SUAVE: Δw ∝ (maxW − w) → la potenciación se frena al acercarse al
      // techo, llevando la sinapsis a un punto fijo estable (convergencia de Δw).
      for (let w = 0; w < winnersFired.length; w++) {
        const nn = winnersFired[w];
        this.winCounts[nn]++;
        const offset = nn * m;
        for (let i = 0; i < m; i++) {
          const pre = this.preTrace[i];
          if (pre > 1e-4) {
            const dw = lr * cfg.aPlus * pre * (maxW - this.weights[offset + i]);
            this.weights[offset + i] += dw;
            weightChange += Math.abs(dw);
          }
        }
      }

      // LTD: la pre disparó → deprimir sinapsis hacia neuronas que dispararon
      // ANTES (traza post activa) pero NO ahora ni recientemente. Se excluyen:
      //   - los ganadores (isWinner): protegidos, su sinapsis activa es causal;
      //   - los que disparan en este tick (spikes===0 ya lo garantiza);
      //   - los refractarios (acaban de disparar → el silencio es artefacto).
      // Así la LTD decorrela a los perdedores sin deshacer la LTP de los engramas.
      for (let f = 0; f < firedInputs.length; f++) {
        const i = firedInputs[f];
        for (let nn = 0; nn < n; nn++) {
          if (isWinner[nn] || this.spikes[nn] !== 0) continue;
          if (t - this.localNeurons[nn].lastSpikeTime < cfg.refractoryMs) continue;
          const post = this.postTrace[nn];
          if (post > 1e-4) {
            const offset = nn * m;
            // Cota suave simétrica: Δw ∝ (w − minW) → la depresión se frena
            // cerca del suelo, evitando oscilaciones y dando punto fijo estable.
            const dw = lr * cfg.aMinus * post * (this.weights[offset + i] - minW);
            this.weights[offset + i] -= dw;
            weightChange += Math.abs(dw);
          }
        }
      }
    }

    // --- 5. Homeostasis de excitabilidad intrínseca (solo durante aprendizaje) ---
    // Ajusta el bias para acercar la tasa media de cada neurona a targetRate y
    // decae la fatiga (adaptación transitoria). El bias se acota a [-2, 2] para
    // que nunca apague a un ganador supra-umbral (estabilidad del engrama).
    // Las sondas (learn=false) NO mutan el estado homeostático → medición limpia.
    if (learn) {
      const a = cfg.homeostasisRate;
      for (let nn = 0; nn < n; nn++) {
        this.avgActivity[nn] = (1 - a) * this.avgActivity[nn] + a * this.spikes[nn];
        let b = this.homeostaticBias[nn] + a * (cfg.targetRate - this.avgActivity[nn]);
        if (b > 1) b = 1;
        else if (b < -1) b = -1;
        this.homeostaticBias[nn] = b;
        this.winCounts[nn] *= 0.98; // fatiga leaky
      }
    }

    return weightChange;
  }

  /**
   * Procesa un input visual: ejecuta UN tick de dinámica con aprendizaje vivo.
   * Llamado por la clase base en cada tick del cerebro.
   *
   * @param spikes - Vector de entrada (tratado como tasas 0-1).
   * @param modulationEffects - Efectos de neuromodulación.
   * @returns Vector de spikes de salida (1.0 = disparó, 0.0 = silente).
   */
  processInput(spikes: Float32Array, modulationEffects: ModulationEffects): Float32Array {
    const lrMul = modulationEffects.learningRateMultiplier ?? 1.0;
    const gain = modulationEffects.spikeGainMultiplier ?? 1.0;
    const dw = this.dynamicsTick(spikes, this._dt, this.currentTime, true, lrMul, gain);

    // Engrama instantáneo: neuronas que dispararon este tick
    const winners: number[] = [];
    const out = new Float32Array(this.neuronCount);
    for (let nn = 0; nn < this.neuronCount; nn++) {
      if (this.spikes[nn] > 0) {
        out[nn] = 1.0;
        winners.push(nn);
      }
    }
    this.lastWinners = Int32Array.from(winners);
    this.updateLearningMetrics(dw, winners.length);
    return out;
  }

  /**
   * Actualiza las métricas de aprendizaje en vivo tras un tick.
   * El engrama instantáneo (qué disparó ESTE tick) es ruidoso por el muestreo
   * Poisson y el refractario, así que mantenemos un conteo de spikes con fuga
   * (recentSpikeCounts) cuyo top-k define un engrama estable. Sobre él medimos:
   *   - estabilidad: overlap con el engrama del tick previo (sube al consolidar);
   *   - cambio de peso (EMA de Σ|Δw|): baja al converger el aprendizaje;
   *   - actividad: fracción de neuronas activas.
   */
  private updateLearningMetrics(dw: number, firedCount: number): void {
    const n = this.neuronCount;
    const decay = 0.85; // fuga del acumulador → ventana efectiva ~6-7 ticks
    for (let nn = 0; nn < n; nn++) {
      this.recentSpikeCounts[nn] = this.recentSpikeCounts[nn] * decay + this.spikes[nn];
    }

    // Engrama estable = top-kWinners de recentSpikeCounts (solo cuentas > 0).
    for (let i = 0; i < n; i++) this.sortIdx[i] = i;
    const counts = this.recentSpikeCounts;
    this.sortIdx.sort((a, b) => counts[b] - counts[a]);
    const k = this.config.kWinners;
    const engram: number[] = [];
    for (let i = 0; i < k && i < n; i++) {
      if (counts[this.sortIdx[i]] > 0.05) engram.push(this.sortIdx[i]);
    }
    this.prevLiveEngram = this.liveEngram;
    this.liveEngram = Int32Array.from(engram.sort((a, b) => a - b));

    const stability = VisualCortex.engramOverlap(this.prevLiveEngram, this.liveEngram);
    this.engramStabilityEMA = 0.9 * this.engramStabilityEMA + 0.1 * stability;
    this.weightChangeEMA = 0.95 * this.weightChangeEMA + 0.05 * dw;
    this.cumWeightChange += dw;
    this.liveActivityEMA = 0.95 * this.liveActivityEMA + 0.05 * (firedCount / n);
  }

  /**
   * Métricas de aprendizaje en vivo para monitorización externa (dashboard).
   * No muta estado: es seguro llamarla en cada broadcast.
   */
  getLearningMetrics(): {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
  } {
    return {
      engram: Array.from(this.liveEngram),
      engramSize: this.liveEngram.length,
      stability: this.engramStabilityEMA,
      weightChange: this.weightChangeEMA,
      cumWeightChange: this.cumWeightChange,
      activity: this.liveActivityEMA,
      neuronCount: this.neuronCount,
    };
  }

  /**
   * Presenta un estímulo durante una ventana temporal y devuelve el engrama
   * resultante (las neuronas que más dispararon). Es la API de referencia
   * para experimentos deterministas de aprendizaje.
   *
   * @param rates - Vector de tasas de entrada (0-1).
   * @param opts - Opciones: ticks, learn, modulación.
   */
  present(
    rates: Float32Array,
    opts: { ticks?: number; learn?: boolean; lrMul?: number; gain?: number } = {},
  ): PresentationResult {
    const ticks = opts.ticks ?? this.config.presentationTicks;
    const learn = opts.learn ?? true;
    const lrMul = opts.lrMul ?? 1.0;
    const gain = opts.gain ?? 1.0;

    const spikeCounts = new Int32Array(this.neuronCount);
    let totalWeightChange = 0;
    let totalActive = 0;

    for (let tick = 0; tick < ticks; tick++) {
      this.currentTime += this._dt;
      const dw = this.dynamicsTick(rates, this._dt, this.currentTime, learn, lrMul, gain);
      totalWeightChange += dw;
      for (let nn = 0; nn < this.neuronCount; nn++) {
        if (this.spikes[nn] > 0) {
          spikeCounts[nn]++;
          totalActive++;
        }
      }
    }

    // Engrama = top-kWinners por conteo de spikes en la ventana
    for (let i = 0; i < this.neuronCount; i++) this.sortIdx[i] = i;
    const counts = spikeCounts;
    this.sortIdx.sort((a, b) => counts[b] - counts[a]);
    const k = this.config.kWinners;
    const engram: number[] = [];
    for (let i = 0; i < k && i < this.neuronCount; i++) {
      if (counts[this.sortIdx[i]] > 0) engram.push(this.sortIdx[i]);
    }
    const engramArr = Int32Array.from(engram.sort((a, b) => a - b));
    this.lastWinners = engramArr;

    return {
      engram: engramArr,
      spikeCounts,
      weightChange: totalWeightChange,
      activity: ticks > 0 ? totalActive / (ticks * this.neuronCount) : 0,
    };
  }

  /**
   * Overlap (índice de Jaccard) entre dos engramas. 1.0 = idénticos, 0 = disjuntos.
   */
  static engramOverlap(a: Int32Array, b: Int32Array): number {
    if (a.length === 0 && b.length === 0) return 1;
    if (a.length === 0 || b.length === 0) return 0;
    const setA = new Set(a);
    let inter = 0;
    for (let i = 0; i < b.length; i++) if (setA.has(b[i])) inter++;
    const union = a.length + b.length - inter;
    return union > 0 ? inter / union : 0;
  }

  /**
   * Aprende un patrón asociándolo a una etiqueta (envoltorio sobre present()).
   * Mantiene compatibilidad con el flujo etiquetado previo.
   */
  learn(
    input: Float32Array,
    label: string,
    _dt: number,
    timestamp: number,
    modulationEffects?: ModulationEffects,
  ): VisualProcessingResult {
    const lrMul = modulationEffects?.learningRateMultiplier ?? 1.0;
    const gain = modulationEffects?.spikeGainMultiplier ?? 1.0;
    const result = this.present(input, { learn: true, lrMul, gain });

    this.memories.push({
      pattern: new Int32Array(result.engram),
      label,
      strength: 1,
      createdAt: timestamp,
    });

    return {
      winners: result.engram,
      potentials: new Float32Array(this.neuronCount),
      predictions: this.compareWithMemories(result.engram),
      activity: result.activity,
    };
  }

  /** Predice sin aprender (solo inferencia). */
  predict(input: Float32Array, _dt: number, _timestamp: number): VisualProcessingResult {
    const result = this.present(input, { learn: false });
    return {
      winners: result.engram,
      potentials: new Float32Array(this.neuronCount),
      predictions: this.compareWithMemories(result.engram),
      activity: result.activity,
    };
  }

  /** Compara un patrón de activación con las memorias almacenadas. */
  compareWithMemories(currentWinners: Int32Array): string[] {
    const matches: string[] = [];
    const seenLabels = new Set<string>();
    const current = new Set(currentWinners);

    for (const mem of this.memories) {
      let overlap = 0;
      for (let m = 0; m < mem.pattern.length; m++) {
        if (current.has(mem.pattern[m])) overlap++;
      }
      if (overlap >= this.config.recognitionThreshold && !seenLabels.has(mem.label)) {
        matches.push(mem.label);
        seenLabels.add(mem.label);
      }
    }
    return matches;
  }

  /** Envía la representación cortical actual al spike bus. */
  emitToDownstream(timestamp: number, targets: string[] = ['hippocampus', 'amygdala']): void {
    if (!this.spikeBus) return;
    const outSpikes = new Float32Array(this.neuronCount);
    for (let i = 0; i < this.lastWinners.length; i++) {
      outSpikes[this.lastWinners[i]] = 1.0;
    }
    this.spikeBus.send({
      source: this.id,
      targets,
      spikes: outSpikes,
      timestamp,
      metadata: { winnerCount: this.lastWinners.length, activity: this.getLocalActivity() },
    });
  }

  get memoryCount(): number {
    return this.memories.length;
  }

  getLastWinners(): Int32Array {
    return this.lastWinners;
  }

  /** Fracción de neuronas que dispararon en el último tick. */
  getLocalActivity(): number {
    let active = 0;
    for (let i = 0; i < this.neuronCount; i++) if (this.spikes[i] > 0) active++;
    return active / this.neuronCount;
  }

  getStats(): {
    memoryCount: number;
    averageWeight: number;
    maxFatigue: number;
    activity: number;
    meanBias: number;
  } {
    let weightSum = 0;
    for (let i = 0; i < this.weights.length; i++) weightSum += this.weights[i];
    let maxFatigue = 0;
    for (let i = 0; i < this.winCounts.length; i++) {
      if (this.winCounts[i] > maxFatigue) maxFatigue = this.winCounts[i];
    }
    let biasSum = 0;
    for (let i = 0; i < this.homeostaticBias.length; i++) biasSum += this.homeostaticBias[i];
    return {
      memoryCount: this.memories.length,
      averageWeight: weightSum / this.weights.length,
      maxFatigue,
      activity: this.getLocalActivity(),
      meanBias: biasSum / this.homeostaticBias.length,
    };
  }

  /** Resetea la fatiga homeostática (efecto del sueño). */
  resetFatigue(): void {
    this.winCounts.fill(0);
  }
}
