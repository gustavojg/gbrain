/**
 * Conectoma Inter-Regional
 * =========================
 * Define el mapa de conexiones entre regiones cerebrales del cerebro digital.
 *
 * Biología: El conectoma es el mapa completo de conexiones neurales del cerebro.
 * A nivel macro, las regiones cerebrales se conectan mediante tractos de sustancia
 * blanca (axones mielinizados) con retardos de conducción proporcionales a la
 * distancia física. Cada conexión tiene un peso (fuerza sináptica media) y un
 * retardo axonal (delay) que refleja el tiempo de propagación del potencial de acción.
 *
 * Regiones modeladas:
 * - thalamus: Filtro atencional y relé sensorial
 * - visualCortex: Procesamiento visual (V1-V4)
 * - auditoryCortex: Procesamiento auditivo (A1-A2)
 * - hippocampus: Formación de memorias y navegación espacial
 * - amygdala: Procesamiento emocional y valencia afectiva
 * - prefrontalCortex: Control ejecutivo y toma de decisiones
 * - brocaWernicke: Producción y comprensión del lenguaje
 */

/**
 * Conexión dirigida entre dos regiones cerebrales.
 * Modela un tracto de sustancia blanca con propiedades fisiológicas.
 */
export interface Connection {
  /** Región de origen (emisora de spikes) */
  from: string;
  /** Región destino (receptora de spikes) */
  to: string;
  /** Peso sináptico medio de la conexión (0.0 - 1.0) */
  weight: number;
  /** Retardo axonal de conducción en milisegundos */
  delay: number;
  /** Si la conexión puede ser modulada por neuromoduladores */
  modulatable: boolean;
}

/**
 * Conectoma por defecto con conexiones biológicamente inspiradas.
 *
 * Organización:
 * 1. Vía ventral (Tálamo → Corteza Visual → Hipocampo): procesamiento del "qué"
 * 2. Vía dorsal (Tálamo → Corteza Auditiva → Hipocampo): procesamiento del "qué" auditivo
 * 3. Circuito límbico (Amígdala ↔ Prefrontal): regulación emocional
 * 4. Circuito del lenguaje (Broca-Wernicke ↔ Prefrontal): producción verbal
 * 5. Feedback top-down (Prefrontal → Tálamo, Visual): control atencional
 */
export const DEFAULT_CONNECTOME: Connection[] = [
  // === Vías feedforward sensoriales (tálamo → cortezas) ===
  {
    from: 'thalamus',
    to: 'visualCortex',
    weight: 1.0,
    delay: 5,
    modulatable: true,
  },
  {
    from: 'thalamus',
    to: 'auditoryCortex',
    weight: 1.0,
    delay: 5,
    modulatable: true,
  },
  {
    from: 'thalamus',
    to: 'brocaWernicke',
    weight: 0.8,
    delay: 8,
    modulatable: true,
  },

  // === Vías corteza sensorial → memoria/emoción ===
  {
    from: 'visualCortex',
    to: 'hippocampus',
    weight: 0.9,
    delay: 10,
    modulatable: true,
  },
  {
    from: 'visualCortex',
    to: 'amygdala',
    weight: 0.7,
    delay: 8,
    modulatable: true,
  },
  {
    from: 'auditoryCortex',
    to: 'hippocampus',
    weight: 0.9,
    delay: 10,
    modulatable: true,
  },
  {
    from: 'auditoryCortex',
    to: 'amygdala',
    weight: 0.7,
    delay: 8,
    modulatable: true,
  },

  // === Vías asociativas → control ejecutivo ===
  {
    from: 'hippocampus',
    to: 'prefrontalCortex',
    weight: 0.8,
    delay: 15,
    modulatable: true,
  },
  {
    from: 'amygdala',
    to: 'prefrontalCortex',
    weight: 0.9,
    delay: 10,
    modulatable: true,
  },

  // === Circuito del lenguaje ===
  {
    from: 'prefrontalCortex',
    to: 'brocaWernicke',
    weight: 1.0,
    delay: 12,
    modulatable: true,
  },
  {
    from: 'brocaWernicke',
    to: 'hippocampus',
    weight: 0.6,
    delay: 10,
    modulatable: true,
  },

  // === Conexiones feedback (top-down) ===
  {
    from: 'prefrontalCortex',
    to: 'thalamus',
    weight: 0.5,
    delay: 15,
    modulatable: true,
  },
  {
    from: 'prefrontalCortex',
    to: 'visualCortex',
    weight: 0.4,
    delay: 12,
    modulatable: true,
  },
  {
    from: 'amygdala',
    to: 'thalamus',
    weight: 0.6,
    delay: 8,
    modulatable: true,
  },
];

/**
 * Gestiona el conectoma inter-regional del cerebro digital.
 *
 * Permite consultar conexiones, obtener rutas de entrada/salida para
 * cada región, y modular dinámicamente los pesos sinápticos en respuesta
 * a neuromoduladores o plasticidad a largo plazo.
 */
export class Connectome {
  /** Mapa de conexiones indexado por región de origen */
  private readonly outgoing: Map<string, Connection[]> = new Map();
  /** Mapa de conexiones indexado por región destino */
  private readonly incoming: Map<string, Connection[]> = new Map();
  /** Lista completa de conexiones */
  private readonly connections: Connection[];
  /** Set de todas las regiones registradas en el conectoma */
  private readonly regionIds: Set<string> = new Set();

  /**
   * Crea un nuevo conectoma a partir de una lista de conexiones.
   *
   * @param connections - Lista de conexiones inter-regionales (default: DEFAULT_CONNECTOME)
   */
  constructor(connections: Connection[] = DEFAULT_CONNECTOME) {
    // Clonar conexiones para permitir mutación segura
    this.connections = connections.map(c => ({ ...c }));
    this.buildIndices();
  }

  /**
   * Construye los índices de conexiones entrantes y salientes.
   */
  private buildIndices(): void {
    this.outgoing.clear();
    this.incoming.clear();
    this.regionIds.clear();

    for (const conn of this.connections) {
      this.regionIds.add(conn.from);
      this.regionIds.add(conn.to);

      // Índice saliente
      if (!this.outgoing.has(conn.from)) {
        this.outgoing.set(conn.from, []);
      }
      this.outgoing.get(conn.from)!.push(conn);

      // Índice entrante
      if (!this.incoming.has(conn.to)) {
        this.incoming.set(conn.to, []);
      }
      this.incoming.get(conn.to)!.push(conn);
    }
  }

  /**
   * Obtiene todas las conexiones salientes de una región.
   *
   * Biología: Equivale a los tractos axonales que parten de esta región
   * hacia otras áreas corticales/subcorticales.
   *
   * @param regionId - Identificador de la región de origen
   * @returns Lista de conexiones salientes (vacía si la región no tiene salidas)
   */
  getOutgoing(regionId: string): ReadonlyArray<Connection> {
    return this.outgoing.get(regionId) ?? [];
  }

  /**
   * Obtiene todas las conexiones entrantes a una región.
   *
   * Biología: Equivale a las aferencias (axones que llegan) a esta región
   * desde otras áreas del cerebro.
   *
   * @param regionId - Identificador de la región destino
   * @returns Lista de conexiones entrantes
   */
  getIncoming(regionId: string): ReadonlyArray<Connection> {
    return this.incoming.get(regionId) ?? [];
  }

  /**
   * Obtiene la conexión directa entre dos regiones, si existe.
   *
   * @param from - Región de origen
   * @param to - Región destino
   * @returns La conexión si existe, undefined si no hay conexión directa
   */
  getConnection(from: string, to: string): Connection | undefined {
    const outConns = this.outgoing.get(from);
    if (!outConns) return undefined;
    return outConns.find(c => c.to === to);
  }

  /**
   * Modula el peso de una conexión específica.
   *
   * Biología: Los neuromoduladores (dopamina, serotonina, etc.) pueden
   * fortalecer o debilitar tractos completos de sustancia blanca mediante
   * cambios en la probabilidad de liberación de neurotransmisores.
   *
   * @param from - Región de origen
   * @param to - Región destino
   * @param factor - Factor multiplicativo para el peso (ej: 1.2 = +20%)
   * @returns true si la conexión fue modulada exitosamente
   */
  modulateWeight(from: string, to: string, factor: number): boolean {
    const conn = this.getConnection(from, to);
    if (!conn || !conn.modulatable) return false;

    conn.weight = Math.max(0, Math.min(2.0, conn.weight * factor));
    return true;
  }

  /**
   * Establece directamente el peso de una conexión.
   *
   * @param from - Región de origen
   * @param to - Región destino
   * @param weight - Nuevo peso sináptico (clamped a [0, 2.0])
   * @returns true si se actualizó exitosamente
   */
  setWeight(from: string, to: string, weight: number): boolean {
    const conn = this.getConnection(from, to);
    if (!conn) return false;

    conn.weight = Math.max(0, Math.min(2.0, weight));
    return true;
  }

  /**
   * Retorna todas las regiones registradas en el conectoma.
   */
  getRegions(): ReadonlySet<string> {
    return this.regionIds;
  }

  /**
   * Retorna todas las conexiones del conectoma.
   */
  getAllConnections(): ReadonlyArray<Connection> {
    return this.connections;
  }

  /**
   * Retorna el número total de conexiones.
   */
  get connectionCount(): number {
    return this.connections.length;
  }
}
