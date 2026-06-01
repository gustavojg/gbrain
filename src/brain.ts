/**
 * CEREBRO DIGITAL — Orquestador Principal
 * ========================================
 * Clase principal que unifica todas las regiones cerebrales,
 * el bus de spikes, los neuromoduladores, y el sistema de memoria
 * en un único sistema coherente.
 * 
 * El cerebro funciona con un loop principal de percepción:
 * 1. PERCIBIR: Recibir inputs sensoriales (audio, imagen, texto)
 * 2. FILTRAR: El tálamo selecciona los inputs más relevantes
 * 3. PROCESAR: Las cortezas sensoriales extraen features
 * 4. EVALUAR: La amígdala asigna valencia emocional
 * 5. RECORDAR: El hipocampo almacena/recupera memorias
 * 6. DECIDIR: La corteza prefrontal integra y decide
 * 7. RESPONDER: Broca/Wernicke genera lenguaje
 * 8. MODULAR: Los neuromoduladores ajustan todo el sistema
 * 
 * Biología: Este loop imita el ciclo percepción-acción del cerebro real,
 * donde la información fluye desde las cortezas sensoriales primarias
 * hacia las áreas de asociación y el lóbulo frontal en ~300-500ms.
 */

import { type BrainConfiguration, DEFAULT_BRAIN_CONFIG, getTotalNeurons, estimateMemoryUsage } from './brain.config.js';
import { SpikeBus, type SpikePacket } from './core/bus/spike-bus.js';
import { Connectome } from './core/bus/connectome.js';
import { NeuromodulatorSystem, ModulatorType, type ModulationEffects } from './core/neuromodulators/modulator-system.js';
import { type BrainRegion, type RegionActivity } from './core/brain-region.js';
import { ConsolidationEngine } from './core/memory/consolidation.js';
import { BrainPersistence } from './core/persistence/binary-protocol.js';
import { VisualEncoder } from './encoders/visual-encoder.js';
import { AudioEncoder } from './encoders/audio-encoder.js';
import { TextEncoder as BrainTextEncoder } from './encoders/text-encoder.js';
import { TextDecoder as BrainTextDecoder } from './decoders/text-decoder.js';
import { EmotionDecoder, type EmotionalState, type ModulatorLevels } from './decoders/emotion-decoder.js';
import { SpeechSynthesizer } from './decoders/speech-synthesizer.js';
import { ImageGenerator } from './decoders/image-generator.js';

// --- Regiones cerebrales ---
import { Thalamus } from './regions/thalamus/thalamus.js';
import { VisualCortex } from './regions/visual-cortex/visual-cortex.js';
import { AuditoryCortex } from './regions/auditory-cortex/auditory-cortex.js';
import { Hippocampus } from './regions/hippocampus/hippocampus.js';
import { Amygdala } from './regions/amygdala/amygdala.js';
import { PrefrontalCortex } from './regions/prefrontal-cortex/prefrontal-cortex.js';
import { BrocaArea, type LanguageResponse } from './regions/broca-wernicke/broca.js';
import { WernickeArea } from './regions/broca-wernicke/wernicke.js';
import { Lexicon } from './regions/broca-wernicke/lexicon.js';
import { seedSpanishLexicon, encodeSentenceToLexiconSpace } from './regions/broca-wernicke/spanish-lexicon.js';

// ================================================================
// TIPOS
// ================================================================

/** Estado completo del cerebro en un momento dado */
export interface BrainState {
  /** Tiempo de simulación actual (ms) */
  time: number;
  /** Actividad de cada región */
  regions: Record<string, RegionActivity>;
  /** Niveles de neuromoduladores */
  modulators: ModulatorLevels;
  /** Estado emocional actual */
  emotion: EmotionalState;
  /** Número de memorias en el hipocampo */
  memoriesCount: number;
  /** Tráfico del bus de spikes */
  busTraffic: Record<string, { sent: number; received: number }>;
  /** Ticks totales procesados */
  tickCount: number;
  /** Última respuesta de Broca */
  broca?: { lastResponse: string; words: string[]; confidence: number };
  /** Tamaño del vocabulario */
  vocabCount: number;
  /** Métricas de aprendizaje de la corteza visual (engrama, estabilidad, convergencia). */
  learning?: {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
  };
  /** Métricas de aprendizaje del hipocampo CA3 (engrama, estabilidad, episodios). */
  learningHippocampus?: {
    engram: number[];
    engramSize: number;
    stability: number;
    weightChange: number;
    cumWeightChange: number;
    activity: number;
    neuronCount: number;
    memoryCount: number;
  };
}

/** Resultado de una percepción */
export interface PerceptionResult {
  /** Tipo de input procesado */
  inputType: 'visual' | 'auditory' | 'text' | 'image';
  /** Estado emocional tras procesar */
  emotion: EmotionalState;
  /** Respuesta textual generada (si aplica) */
  textResponse?: string;
  /** Parámetros de voz (si aplica) */
  speechParams?: { text: string; rate: number; pitch: number; volume: number };
  /** Regiones que se activaron */
  activeRegions: string[];
  /** Tiempo de procesamiento (ms simulados) */
  processingTime: number;
}

/** Evento emitido por el cerebro */
export interface BrainEvent {
  type: 'spike' | 'emotion' | 'memory' | 'consolidation' | 'response';
  timestamp: number;
  data: Record<string, unknown>;
}

// ================================================================
// CLASE PRINCIPAL
// ================================================================

/**
 * Cerebro Digital — Sistema principal.
 * Orquesta 7 regiones cerebrales con SNNs, neuromodulación y memoria jerárquica.
 */
export class DigitalBrain {
  // --- Configuración ---
  private config: BrainConfiguration;

  // --- Infraestructura core ---
  private bus: SpikeBus;
  private connectome: Connectome;
  private modulators: NeuromodulatorSystem;
  private consolidationEngine: ConsolidationEngine;
  private persistence: BrainPersistence = new BrainPersistence();

  // --- Regiones cerebrales ---
  private regions: Map<string, BrainRegion> = new Map();

  // --- Encoders (inputs) ---
  private visualEncoder: VisualEncoder;
  private audioEncoder: AudioEncoder;
  private textEncoder: BrainTextEncoder;

  /**
   * Intención lingüística limpia del último `read()`, en el espacio del léxico.
   * Permite que `speak()` regenere la respuesta de Broca de forma determinista
   * (reproducible y discriminativa), sin que la dilución del bucle recurrente
   * la sobrescriba con ruido de fondo.
   */
  private lastLinguisticIntention: Float32Array | null = null;

  // --- Decoders (outputs) ---
  private textDecoder: BrainTextDecoder;
  private emotionDecoder: EmotionDecoder;
  private speechSynthesizer: SpeechSynthesizer;
  private imageGenerator: ImageGenerator;

  // --- Estado ---
  private currentTime: number = 0;
  private tickCount: number = 0;
  private isRunning: boolean = false;
  private eventListeners: Map<string, Array<(event: BrainEvent) => void>> = new Map();

  // --- Consolidación automática ---
  private lastConsolidation: number = 0;

  constructor(config: Partial<BrainConfiguration> = {}) {
    this.config = { ...DEFAULT_BRAIN_CONFIG, ...config };

    console.log(`\n🧠 ═══════════════════════════════════════════`);
    console.log(`   CEREBRO DIGITAL — Inicializando...`);
    console.log(`   Neuronas totales: ${getTotalNeurons(this.config).toLocaleString()}`);
    console.log(`   Memoria estimada: ${estimateMemoryUsage(this.config).toFixed(1)} MB`);
    console.log(`   Regiones: ${Object.keys(this.config.regions).length}`);
    console.log(`═══════════════════════════════════════════════\n`);

    // 1. Inicializar bus de spikes
    this.bus = new SpikeBus();

    // 2. Inicializar conectoma
    this.connectome = new Connectome(this.config.connectome);

    // 3. Inicializar neuromoduladores
    this.modulators = new NeuromodulatorSystem();

    // 4. Inicializar consolidación
    this.consolidationEngine = new ConsolidationEngine();

    // 5. Inicializar encoders
    this.visualEncoder = new VisualEncoder({
      processWidth: 32,
      processHeight: 32,
      foveation: true,
      edgeDetection: true,
    });

    this.audioEncoder = new AudioEncoder({
      sampleRate: 16000,
      fftSize: 2048,
      numBands: 40,
      numFrames: 20,
    });

    this.textEncoder = new BrainTextEncoder({
      vectorSize: 5000,
      tokenization: 'word',
    });

    // 6. Inicializar decoders
    this.textDecoder = new BrainTextDecoder(0.3);
    this.emotionDecoder = new EmotionDecoder();
    this.speechSynthesizer = new SpeechSynthesizer({ lang: 'es-ES' });
    this.imageGenerator = new ImageGenerator({ outputWidth: 32, outputHeight: 32 });

    // 7. Registrar regiones en el bus
    this.initializeRegions();

    // 8. Configurar conectoma en el bus
    this.setupConnectome();

    console.log(`✅ Cerebro inicializado correctamente.`);
    console.log(`   Tick rate: ${this.config.tickRate} Hz`);
    console.log(`   Consolidación cada: ${this.config.memory.consolidationIntervalMs / 1000}s\n`);
  }

  /**
   * Inicializa las regiones cerebrales y las registra en el bus.
   * Usa import dinámico para cargar las implementaciones concretas.
   */
  /** Léxico compartido Broca↔Wernicke */
  private lexicon!: Lexicon;

  private initializeRegions(): void {
    // Registrar IDs del config en el bus (para el conectoma)
    for (const regionId of Object.keys(this.config.regions)) {
      this.bus.register(regionId);
    }

    // Registrar Broca y Wernicke como regiones individuales
    this.bus.register('broca');
    this.bus.register('wernicke');

    // Crear léxico compartido con vocabulario español
    // Pattern size reducido a 1000 para eficiencia (matching con inputCount de Broca/Wernicke)
    this.lexicon = new Lexicon(1000);
    seedSpanishLexicon(this.lexicon);
    console.log(`  📚 Léxico inicializado: ${this.lexicon.size} palabras`);

    // ── Instanciar regiones con tamaños reducidos ──
    // Total: ~10K neuronas (vs 50K antes) → rendimiento fluido
    this.addRegion(new Thalamus({ neuronCount: 500, totalInputSize: 500, bottleneckSize: 100 }));
    this.addRegion(new VisualCortex({ neuronCount: 2000, inputCount: 1000 }));
    this.addRegion(new AuditoryCortex({ neuronCount: 1000, inputCount: 400 }));
    this.addRegion(new Hippocampus(1000, 1000));
    this.addRegion(new Amygdala(500, 500));
    this.addRegion(new PrefrontalCortex(3000, 1000));
    this.addRegion(new BrocaArea(this.lexicon, 1000, 1000));
    this.addRegion(new WernickeArea(this.lexicon, 1000, 1000));

    // Conectar Broca/Wernicke al bus como alias de 'brocaWernicke'
    // para recibir paquetes del conectoma existente
    this.bus.onReceive('brocaWernicke', (packet: SpikePacket) => {
      const broca = this.regions.get('broca');
      const wernicke = this.regions.get('wernicke');
      if (wernicke) wernicke.feedInput(packet.spikes, packet.timestamp);
      if (broca) broca.feedInput(packet.spikes, packet.timestamp);
    });

    // Calcular total real de neuronas
    let totalNeurons = 0;
    for (const [, region] of this.regions) {
      totalNeurons += region.neurons;
    }
    console.log(`  🧩 ${this.regions.size} regiones instanciadas (${totalNeurons.toLocaleString()} neuronas reales)`);
  }

  /**
   * Añade una región implementada al cerebro.
   */
  addRegion(region: BrainRegion): void {
    this.regions.set(region.id, region);
    if (!this.bus.isRegistered(region.id)) {
      this.bus.register(region.id);
    }

    // Suscribir la región al bus para recibir spikes
    this.bus.onReceive(region.id, (packet: SpikePacket) => {
      region.feedInput(packet.spikes, packet.timestamp);
    });

    console.log(`  🧩 Región añadida: ${region.id} (${region.name})`);
  }

  /**
   * Configura los retardos y pesos del conectoma en el bus de spikes.
   */
  private setupConnectome(): void {
    const connections = this.connectome.getAllConnections();
    for (const conn of connections) {
      this.bus.setDelay(conn.from, conn.to, conn.delay);
      this.bus.setWeight(conn.from, conn.to, conn.weight);
    }
    console.log(`  🔗 Conectoma configurado: ${connections.length} conexiones`);
  }

  // ================================================================
  // API DE ALTO NIVEL — INPUTS
  // ================================================================

  /**
   * El cerebro "ve" una imagen.
   * 
   * @param pixels - Datos de imagen (escala de grises, 0-255)
   * @param width - Ancho
   * @param height - Alto
   */
  see(pixels: number[] | Float32Array | Uint8Array, width: number, height: number): PerceptionResult {
    console.log(`👁️  Percibiendo imagen (${width}×${height})...`);

    // Codificar a spikes
    const spikes = this.visualEncoder.encode(pixels, width, height, this.config.snn.dt);

    // Enviar al tálamo
    this.injectSensoryInput('visual', spikes);

    // Procesar varios ticks para propagar por el cerebro
    return this.processPerception('visual');
  }

  /**
   * El cerebro "escucha" audio.
   * 
   * @param audioSamples - Muestras PCM
   */
  hear(audioSamples: Float32Array | number[]): PerceptionResult {
    // Codificar a spikes via espectrograma
    const spikes = this.audioEncoder.encode(audioSamples, this.config.snn.dt);

    // Enviar al tálamo
    this.injectSensoryInput('auditory', spikes);

    return this.processPerception('auditory');
  }

  /**
   * El cerebro "escucha" un espectrograma ya calculado (de 08_microphone_interaction).
   */
  hearSpectrogram(spectrogram: number[] | Float32Array): PerceptionResult {
    const spikes = this.audioEncoder.encodeSpectrogram(spectrogram, this.config.snn.dt);
    this.injectSensoryInput('auditory', spikes);
    return this.processPerception('auditory');
  }

  /**
   * El cerebro "lee" texto.
   * 
   * @param text - Texto a procesar
   */
  /**
   * Mapa de palabras emocionales → efectos neuromoduladores.
   * Biología: el área de Wernicke reconoce palabras con carga emocional
   * y activa la amígdala, que a su vez libera neuromoduladores.
   */
  private static readonly EMOTIONAL_WORDS: Record<string, { modulator: ModulatorType; amount: number }[]> = {
    // Positivas → dopamina + serotonina
    'feliz': [{ modulator: ModulatorType.Dopamine, amount: 0.15 }, { modulator: ModulatorType.Serotonin, amount: 0.1 }],
    'alegria': [{ modulator: ModulatorType.Dopamine, amount: 0.2 }],
    'amor': [{ modulator: ModulatorType.Oxytocin, amount: 0.2 }, { modulator: ModulatorType.Dopamine, amount: 0.1 }],
    'carino': [{ modulator: ModulatorType.Oxytocin, amount: 0.15 }],
    'bien': [{ modulator: ModulatorType.Serotonin, amount: 0.1 }],
    'gracias': [{ modulator: ModulatorType.Oxytocin, amount: 0.1 }, { modulator: ModulatorType.Serotonin, amount: 0.05 }],
    'hola': [{ modulator: ModulatorType.Dopamine, amount: 0.05 }, { modulator: ModulatorType.Oxytocin, amount: 0.05 }],
    'bonito': [{ modulator: ModulatorType.Dopamine, amount: 0.1 }],
    'entusiasmo': [{ modulator: ModulatorType.Dopamine, amount: 0.2 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'esperanza': [{ modulator: ModulatorType.Serotonin, amount: 0.15 }],
    'paz': [{ modulator: ModulatorType.Serotonin, amount: 0.2 }],
    'calma': [{ modulator: ModulatorType.Serotonin, amount: 0.15 }],
    'curiosidad': [{ modulator: ModulatorType.Dopamine, amount: 0.1 }, { modulator: ModulatorType.Acetylcholine, amount: 0.1 }],
    'genial': [{ modulator: ModulatorType.Dopamine, amount: 0.15 }, { modulator: ModulatorType.Serotonin, amount: 0.1 }],
    'contento': [{ modulator: ModulatorType.Dopamine, amount: 0.1 }, { modulator: ModulatorType.Serotonin, amount: 0.1 }],
    // Negativas → cortisol + norepinefrina
    'triste': [{ modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'tristeza': [{ modulator: ModulatorType.Cortisol, amount: 0.15 }],
    'miedo': [{ modulator: ModulatorType.Cortisol, amount: 0.2 }, { modulator: ModulatorType.Norepinephrine, amount: 0.15 }],
    'odio': [{ modulator: ModulatorType.Cortisol, amount: 0.15 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'ansiedad': [{ modulator: ModulatorType.Cortisol, amount: 0.2 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'estres': [{ modulator: ModulatorType.Cortisol, amount: 0.25 }],
    'enojo': [{ modulator: ModulatorType.Norepinephrine, amount: 0.2 }, { modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'mal': [{ modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'feo': [{ modulator: ModulatorType.Cortisol, amount: 0.05 }],
    'soledad': [{ modulator: ModulatorType.Cortisol, amount: 0.1 }],
    'frustracion': [{ modulator: ModulatorType.Cortisol, amount: 0.15 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    // Activación → acetilcolina + norepinefrina
    'pensar': [{ modulator: ModulatorType.Acetylcholine, amount: 0.1 }],
    'aprender': [{ modulator: ModulatorType.Acetylcholine, amount: 0.15 }, { modulator: ModulatorType.Dopamine, amount: 0.05 }],
    'recordar': [{ modulator: ModulatorType.Acetylcholine, amount: 0.1 }],
    'atencion': [{ modulator: ModulatorType.Acetylcholine, amount: 0.15 }, { modulator: ModulatorType.Norepinephrine, amount: 0.1 }],
    'dormir': [{ modulator: ModulatorType.Serotonin, amount: 0.2 }],
    'sonar': [{ modulator: ModulatorType.Serotonin, amount: 0.1 }, { modulator: ModulatorType.Dopamine, amount: 0.05 }],
  };

  read(text: string): PerceptionResult {
    console.log(`📖 Leyendo: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);

    // Codificar el texto en el ESPACIO DEL LÉXICO (no con el hash genérico del
    // TextEncoder, que produce vectores ortogonales a los engramas y deja a
    // Wernicke sin reconocer nada). Así Wernicke recupera las palabras reales
    // y Broca puede producir una respuesta asociativa reproducible.
    const spikes = encodeSentenceToLexiconSpace(text, this.lexicon.dimensions);

    // Guardar la intención limpia para que speak() genere una respuesta
    // reproducible (texto→Wernicke→Broca) sin dilución del bucle recurrente.
    this.lastLinguisticIntention = spikes;

    // Enviar al tálamo (ruta lingüística)
    this.injectSensoryInput('linguistic', spikes);

    // ── Detección emocional de palabras ──
    // Biología: Wernicke reconoce palabras emocionales → activa amígdala
    // → amígdala libera neuromoduladores según la valencia
    const words = text.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^\w\sáéíóúüñ]/g, '').split(/\s+/).filter(w => w.length > 0);
    let emotionalHits = 0;
    for (const word of words) {
      // También comparar sin acentos
      const effects = DigitalBrain.EMOTIONAL_WORDS[word];
      if (effects) {
        for (const effect of effects) {
          this.modulators.release(effect.modulator, effect.amount);
        }
        emotionalHits++;
      }
    }

    if (emotionalHits > 0) {
      console.log(`  💭 ${emotionalHits} palabras emocionales detectadas`);
    }

    // Novedad → norepinefrina
    this.modulators.release(ModulatorType.Norepinephrine, 0.05);

    return this.processPerception('text');
  }

  // ================================================================
  // API DE ALTO NIVEL — OUTPUTS
  // ================================================================

  /**
   * El cerebro "habla" — genera una respuesta textual.
   */
  speak(): { text: string; speech?: { rate: number; pitch: number; volume: number } } {
    const emotion = this.feel();

    // Obtener la respuesta de Broca
    const brocaRegion = this.regions.get('broca') as BrocaArea | undefined;
    if (!brocaRegion) {
      return { text: '...' };
    }

    let text: string;

    // Ruta determinista del lenguaje (texto→Wernicke→Broca):
    // si hubo una lectura reciente, regeneramos la respuesta desde la intención
    // LIMPIA en el espacio del léxico. Esto la hace reproducible para el mismo
    // texto y discriminativa entre textos distintos, sin que el ruido recurrente
    // del bucle la sobrescriba.
    if (this.lastLinguisticIntention) {
      const wernicke = this.regions.get('wernicke') as WernickeArea | undefined;
      // Construir la intención semántica desde lo que Wernicke comprende.
      const intention = new Float32Array(this.lexicon.dimensions);
      let understoodAny = false;
      if (wernicke) {
        const understood = wernicke.comprehend(this.lastLinguisticIntention);
        for (const m of understood) {
          const p = this.lexicon.lookup(m.word);
          if (p) {
            for (let i = 0; i < intention.length; i++) intention[i] += p[i] * m.similarity;
            understoodAny = true;
          }
        }
      }
      // Si no comprendió nada, usar la codificación limpia directamente.
      const semantic = understoodAny ? intention : this.lastLinguisticIntention;
      const response = brocaRegion.generateResponse(semantic, {
        valence: emotion.valence,
        arousal: emotion.arousal,
      });
      text = response.words.length > 0 ? response.words.join(' ') : '...';
      const speechParams = this.speechSynthesizer.synthesize(text, emotion);
      return { text, speech: speechParams };
    }

    // Verificar si Broca generó una respuesta
    const lastResponse = brocaRegion.getLastResponse();

    if (lastResponse && lastResponse.words.length > 0) {
      text = lastResponse.words.join(' ');
    } else {
      // Forzar generación: crear intención desde la actividad del prefrontal
      const pfcRegion = this.regions.get('prefrontalCortex');
      const pfcActivity = pfcRegion?.getActivity();
      if (pfcActivity && pfcActivity.outputSpikes) {
        const response = brocaRegion.generateResponse(
          pfcActivity.outputSpikes,
          { valence: emotion.valence, arousal: emotion.arousal }
        );
        text = response.words.length > 0 ? response.words.join(' ') : '...';
      } else {
        text = '...';
      }
    }

    // Sintetizar parámetros de voz
    const speechParams = this.speechSynthesizer.synthesize(text, emotion);

    return { text, speech: speechParams };
  }

  /**
   * El cerebro "siente" — retorna el estado emocional actual.
   */
  feel(): EmotionalState {
    const levels: ModulatorLevels = {
      dopamine: this.modulators.getLevel(ModulatorType.Dopamine),
      serotonin: this.modulators.getLevel(ModulatorType.Serotonin),
      norepinephrine: this.modulators.getLevel(ModulatorType.Norepinephrine),
      cortisol: this.modulators.getLevel(ModulatorType.Cortisol),
      acetylcholine: this.modulators.getLevel(ModulatorType.Acetylcholine),
      oxytocin: this.modulators.getLevel(ModulatorType.Oxytocin),
    };

    return this.emotionDecoder.decode(levels);
  }

  /**
   * El cerebro "imagina" — genera una imagen desde la corteza visual.
   */
  imagine(): { pixels: Float32Array; width: number; height: number; ascii: string } {
    const visualRegion = this.regions.get('visualCortex');
    if (!visualRegion) {
      const emptyPixels = new Float32Array(32 * 32);
      return { pixels: emptyPixels, width: 32, height: 32, ascii: '' };
    }

    const activity = visualRegion.getActivity();
    const image = this.imageGenerator.generate(activity.outputSpikes);
    const ascii = this.imageGenerator.toAscii(image.pixels, image.width, image.height);

    return { ...image, ascii };
  }

  // ================================================================
  // PROCESAMIENTO INTERNO
  // ================================================================

  /**
   * Inyecta input sensorial en el tálamo.
   */
  private injectSensoryInput(type: 'visual' | 'auditory' | 'linguistic', spikes: Float32Array): void {
    const thalamus = this.regions.get('thalamus');
    if (thalamus) {
      thalamus.feedInput(spikes, this.currentTime);
    }

    // También enviar directamente a la corteza apropiada via bus
    const targets = type === 'visual' ? ['visualCortex'] :
                    type === 'auditory' ? ['auditoryCortex'] :
                    ['wernicke', 'broca'];

    this.bus.send({
      source: 'thalamus',
      targets,
      spikes,
      timestamp: this.currentTime,
      metadata: { inputType: type },
    });
  }

  /**
   * Procesa una percepción: ejecuta N ticks para propagar señales por el cerebro.
   */
  private processPerception(inputType: 'visual' | 'auditory' | 'text' | 'image'): PerceptionResult {
    const startTime = this.currentTime;
    const processingTicks = 50; // ~50ms de procesamiento

    // Ejecutar ticks
    for (let i = 0; i < processingTicks; i++) {
      this.tick();
    }

    // Obtener estado emocional resultante
    const emotion = this.feel();

    // Identificar regiones activas
    const activeRegions: string[] = [];
    for (const [id, region] of this.regions) {
      const activity = region.getActivity();
      if (activity.firingRate > 0.01) {
        activeRegions.push(id);
      }
    }

    return {
      inputType,
      emotion,
      activeRegions,
      processingTime: this.currentTime - startTime,
    };
  }

  /**
   * Ejecuta un tick del cerebro.
   * Este es el loop principal de simulación.
   */
  tick(): void {
    const dt = this.config.snn.dt;
    this.currentTime += dt;
    this.tickCount++;

    // 1. Obtener efectos de neuromodulación
    const effects = this.modulators.getEffects();

    // 2. Procesar cada región
    for (const [regionId, region] of this.regions) {
      const activity = region.step(dt, effects);

      // 3. Enviar spikes de salida al bus
      if (activity.activeNeurons.length > 0) {
        const outgoing = this.connectome.getOutgoing(regionId);
        if (outgoing.length > 0) {
          this.bus.send({
            source: regionId,
            targets: outgoing.map(c => c.to),
            spikes: activity.outputSpikes,
            timestamp: this.currentTime,
          });
        }
      }
    }

    // 4. Despachar paquetes del bus (entrega diferida por retardos axonales)
    this.bus.tick(this.currentTime);

    // 5. Decaimiento de neuromoduladores
    this.modulators.decay(dt);

    // 6. Consolidación periódica ("sueño")
    if (this.currentTime - this.lastConsolidation > this.config.memory.consolidationIntervalMs) {
      this.sleep();
    }
  }

  /**
   * Proceso de consolidación ("sueño").
   * Replay de memorias del hipocampo para fortalecer cortezas.
   */
  sleep(): void {
    console.log(`💤 Consolidación iniciada (t=${this.currentTime.toFixed(0)}ms)...`);
    this.lastConsolidation = this.currentTime;

    // La consolidación se delegaría al hippocampus
    const hippocampus = this.regions.get('hippocampus');
    if (hippocampus) {
      const stats = this.consolidationEngine.consolidate([], this.regions);
      console.log(`   Memorias replayed: ${stats.memoriesReplayed}`);
      console.log(`   Sinapsis fortalecidas: ${stats.synapsesStrengthened}`);
    }

    // Emitir evento
    this.emitEvent({
      type: 'consolidation',
      timestamp: this.currentTime,
      data: { consolidated: true },
    });
  }

  // ================================================================
  // ESTADO Y MONITORIZACIÓN
  // ================================================================

  /**
   * Retorna el estado completo del cerebro.
   */
  getState(): BrainState {
    const regionsActivity: Record<string, RegionActivity> = {};
    for (const [id, region] of this.regions) {
      regionsActivity[id] = region.getActivity();
    }

    const busTraffic: Record<string, { sent: number; received: number }> = {};
    for (const [id, stats] of this.bus.getTraffic()) {
      busTraffic[id] = { sent: stats.packetsSent, received: stats.packetsReceived };
    }

    // Broca response data
    const brocaRegion = this.regions.get('broca') as BrocaArea | undefined;
    const brocaResponse = brocaRegion?.getLastResponse();

    // Métricas de aprendizaje de la corteza visual
    const visualRegion = this.regions.get('visualCortex') as VisualCortex | undefined;
    const learning = visualRegion?.getLearningMetrics();

    // Métricas de aprendizaje del hipocampo CA3
    const hippoRegion = this.regions.get('hippocampus') as Hippocampus | undefined;
    const learningHippocampus = hippoRegion?.getLearningMetrics();

    return {
      time: this.currentTime,
      regions: regionsActivity,
      modulators: {
        dopamine: this.modulators.getLevel(ModulatorType.Dopamine),
        serotonin: this.modulators.getLevel(ModulatorType.Serotonin),
        norepinephrine: this.modulators.getLevel(ModulatorType.Norepinephrine),
        cortisol: this.modulators.getLevel(ModulatorType.Cortisol),
        acetylcholine: this.modulators.getLevel(ModulatorType.Acetylcholine),
        oxytocin: this.modulators.getLevel(ModulatorType.Oxytocin),
      },
      emotion: this.feel(),
      memoriesCount: learningHippocampus?.memoryCount ?? 0,
      busTraffic,
      tickCount: this.tickCount,
      broca: brocaResponse ? {
        lastResponse: brocaResponse.words.join(' '),
        words: brocaResponse.words,
        confidence: brocaResponse.confidence,
      } : undefined,
      vocabCount: this.lexicon?.size ?? 0,
      learning,
      learningHippocampus,
    };
  }

  // ================================================================
  // PERSISTENCIA (aprendizaje entre sesiones)
  // ================================================================

  /**
   * Guarda el estado sináptico de todas las regiones + neuromoduladores a disco.
   * Es lo que hace que el aprendizaje sobreviva a reinicios del proceso.
   */
  saveState(filePath: string): void {
    this.persistence.save(filePath, this.regions, this.modulators);
  }

  /**
   * Restaura el estado desde un archivo previamente guardado. Aplica los pesos
   * a cada región por id; salta (sin abortar) las regiones cuyas dimensiones no
   * coincidan con el archivo —p. ej. si cambió neuronCount entre versiones—.
   *
   * @returns Qué regiones se restauraron y cuáles se saltaron.
   */
  loadState(filePath: string): { loaded: string[]; skipped: string[] } {
    const data = this.persistence.load(filePath);
    const loaded: string[] = [];
    const skipped: string[] = [];
    for (const [id, region] of this.regions) {
      const rd = data.regions.get(id);
      if (rd && rd.weights.length === region.getNetworkConfig().weights.length) {
        region.loadWeights(rd.weights);
        loaded.push(id);
      } else {
        skipped.push(id);
      }
    }
    if (data.modulatorState) {
      this.modulators.deserialize(data.modulatorState);
    }
    return { loaded, skipped };
  }

  /**
   * Obtiene el bus de spikes (para monitorización externa).
   */
  getBus(): SpikeBus {
    return this.bus;
  }

  /**
   * Obtiene el sistema de neuromoduladores.
   */
  getModulators(): NeuromodulatorSystem {
    return this.modulators;
  }

  /**
   * Obtiene una región por ID.
   */
  getRegion(id: string): BrainRegion | undefined {
    return this.regions.get(id);
  }

  /**
   * Obtiene todas las regiones.
   */
  getRegions(): Map<string, BrainRegion> {
    return this.regions;
  }

  // ================================================================
  // EVENTOS
  // ================================================================

  /**
   * Registra un listener para eventos del cerebro.
   */
  on(eventType: string, callback: (event: BrainEvent) => void): void {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, []);
    }
    this.eventListeners.get(eventType)!.push(callback);
  }

  /**
   * Emite un evento a todos los listeners registrados.
   */
  private emitEvent(event: BrainEvent): void {
    const listeners = this.eventListeners.get(event.type) || [];
    for (const listener of listeners) {
      listener(event);
    }
    // También emitir a listeners de '*'
    const allListeners = this.eventListeners.get('*') || [];
    for (const listener of allListeners) {
      listener(event);
    }
  }

  // ================================================================
  // CONFIGURACIÓN
  // ================================================================

  /**
   * Retorna la configuración actual del cerebro.
   */
  getConfig(): BrainConfiguration {
    return { ...this.config };
  }

  /**
   * Tiempo actual de simulación.
   */
  get time(): number {
    return this.currentTime;
  }

  /**
   * Número total de ticks procesados.
   */
  get ticks(): number {
    return this.tickCount;
  }
}
