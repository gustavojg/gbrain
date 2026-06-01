/**
 * Bus Central de Comunicación de Spikes
 * =======================================
 * Sistema de mensajería inter-regional para el cerebro digital.
 *
 * Biología: En el cerebro real, las señales entre regiones viajan como
 * potenciales de acción (spikes) a través de axones mielinizados con
 * velocidades de conducción de 1-100 m/s. El retardo axonal depende de
 * la distancia y el grado de mielinización del tracto. Este bus modela
 * esa latencia usando una cola de prioridad ordenada por tiempo de entrega.
 *
 * Arquitectura: Basado en EventEmitter de Node.js para notificación
 * asíncrona a regiones suscritas. Los paquetes de spikes se encolan
 * con un tiempo de entrega futuro (timestamp + delay) y se despachan
 * en cada tick del simulador cuando su tiempo ha llegado.
 */

import { EventEmitter } from 'events';

/**
 * Paquete de spikes transmitido entre regiones cerebrales.
 * Modela un volley de potenciales de acción viajando por un tracto axonal.
 */
export interface SpikePacket {
  /** Identificador de la región emisora */
  source: string;
  /** Identificadores de las regiones destino */
  targets: string[];
  /** Vector de spikes (0.0 = silencio, 1.0 = spike) */
  spikes: Float32Array;
  /** Marca temporal de emisión (ms del reloj de simulación) */
  timestamp: number;
  /** Metadatos opcionales (ej: tipo de señal, modulación, prioridad) */
  metadata?: Record<string, unknown>;
}

/**
 * Paquete en tránsito con tiempo de entrega calculado.
 * Incluye el retardo axonal y peso sináptico de la conexión.
 */
interface DelayedPacket {
  /** Paquete original de spikes */
  packet: SpikePacket;
  /** Región destino específica (un paquete puede tener múltiples targets) */
  target: string;
  /** Tiempo de entrega = timestamp + delay axonal */
  deliveryTime: number;
  /** Peso de la conexión para escalar los spikes */
  connectionWeight: number;
}

/**
 * Estadísticas de tráfico por región.
 */
export interface RegionTrafficStats {
  /** Número de paquetes enviados desde esta región */
  packetsSent: number;
  /** Número de paquetes recibidos por esta región */
  packetsReceived: number;
  /** Número total de spikes (sum de spikes > 0) enviados */
  totalSpikesEmitted: number;
  /** Último timestamp de actividad */
  lastActivity: number;
}

/**
 * Bus central de comunicación de spikes entre regiones cerebrales.
 *
 * Gestiona el enrutamiento de señales neurales con retardos axonales
 * realistas. Las regiones se registran, envían paquetes de spikes,
 * y reciben entregas diferidas según el conectoma.
 *
 * Uso:
 * ```typescript
 * const bus = new SpikeBus();
 * bus.register('visualCortex');
 * bus.register('hippocampus');
 * bus.onReceive('hippocampus', (packet) => { ... });
 * bus.send({ source: 'visualCortex', targets: ['hippocampus'], spikes, timestamp: 100 });
 * bus.tick(110); // Entrega paquetes cuyo deliveryTime <= 110
 * ```
 */
export class SpikeBus extends EventEmitter {
  /** Regiones registradas con sus stats de tráfico */
  private readonly regions: Map<string, RegionTrafficStats> = new Map();

  /**
   * Cola de prioridad de paquetes retrasados.
   * Ordenada por deliveryTime ascendente para despacho eficiente.
   * Se usa inserción binaria para mantener el orden.
   */
  private readonly delayQueue: DelayedPacket[] = [];

  /** Retardos axonales por defecto por par de regiones (ms) */
  private readonly defaultDelay: number = 10;

  /** Pesos de conexión por defecto */
  private readonly defaultWeight: number = 1.0;

  /** Mapa de retardos personalizados: "from->to" → delay */
  private readonly customDelays: Map<string, number> = new Map();

  /** Mapa de pesos personalizados: "from->to" → weight */
  private readonly customWeights: Map<string, number> = new Map();

  /** Contador total de paquetes procesados (para estadísticas) */
  private totalPacketsProcessed: number = 0;

  constructor() {
    super();
    // Aumentar límite de listeners para muchas regiones
    this.setMaxListeners(100);
  }

  /**
   * Registra una región cerebral en el bus de comunicación.
   *
   * @param regionId - Identificador único de la región
   */
  register(regionId: string): void {
    if (this.regions.has(regionId)) return;

    this.regions.set(regionId, {
      packetsSent: 0,
      packetsReceived: 0,
      totalSpikesEmitted: 0,
      lastActivity: 0,
    });
  }

  /**
   * Configura el retardo axonal para una conexión específica.
   *
   * @param from - Región de origen
   * @param to - Región destino
   * @param delay - Retardo en milisegundos
   */
  setDelay(from: string, to: string, delay: number): void {
    this.customDelays.set(`${from}->${to}`, delay);
  }

  /**
   * Configura el peso sináptico para una conexión específica.
   *
   * @param from - Región de origen
   * @param to - Región destino
   * @param weight - Peso sináptico (0.0 - 2.0)
   */
  setWeight(from: string, to: string, weight: number): void {
    this.customWeights.set(`${from}->${to}`, weight);
  }

  /**
   * Obtiene el retardo configurado para una conexión.
   */
  private getDelay(from: string, to: string): number {
    return this.customDelays.get(`${from}->${to}`) ?? this.defaultDelay;
  }

  /**
   * Obtiene el peso configurado para una conexión.
   */
  private getWeight(from: string, to: string): number {
    return this.customWeights.get(`${from}->${to}`) ?? this.defaultWeight;
  }

  /**
   * Envía un paquete de spikes desde una región a sus targets.
   *
   * Biología: Modela la propagación de un volley de potenciales de acción
   * desde una región cortical hacia sus proyecciones axonales. El paquete
   * se encola con un retardo calculado según el conectoma.
   *
   * @param packet - Paquete de spikes a transmitir
   */
  send(packet: SpikePacket): void {
    const sourceStats = this.regions.get(packet.source);
    if (!sourceStats) {
      throw new Error(
        `[SpikeBus] Región origen '${packet.source}' no registrada. Registrar con bus.register().`
      );
    }

    // Contar spikes activos
    let spikeCount = 0;
    for (let i = 0; i < packet.spikes.length; i++) {
      if (packet.spikes[i] > 0) spikeCount++;
    }

    sourceStats.packetsSent++;
    sourceStats.totalSpikesEmitted += spikeCount;
    sourceStats.lastActivity = packet.timestamp;

    // Encolar paquete para cada target con su retardo específico
    for (const target of packet.targets) {
      if (!this.regions.has(target)) continue; // Ignorar targets no registrados

      const delay = this.getDelay(packet.source, target);
      const weight = this.getWeight(packet.source, target);
      const deliveryTime = packet.timestamp + delay;

      const delayed: DelayedPacket = {
        packet,
        target,
        deliveryTime,
        connectionWeight: weight,
      };

      // Inserción binaria para mantener orden por deliveryTime
      this.insertSorted(delayed);
    }
  }

  /**
   * Inserta un paquete en la cola manteniendo orden ascendente por deliveryTime.
   * Usa búsqueda binaria para eficiencia O(log n).
   */
  private insertSorted(item: DelayedPacket): void {
    let lo = 0;
    let hi = this.delayQueue.length;

    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.delayQueue[mid].deliveryTime <= item.deliveryTime) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    this.delayQueue.splice(lo, 0, item);
  }

  /**
   * Suscribe una región para recibir paquetes de spikes.
   *
   * @param regionId - Identificador de la región receptora
   * @param callback - Función a invocar cuando llega un paquete
   */
  onReceive(
    regionId: string,
    callback: (packet: SpikePacket) => void
  ): void {
    this.on(`spike:${regionId}`, callback);
  }

  /**
   * Despacha todos los paquetes cuyo tiempo de entrega ha llegado.
   *
   * Biología: Simula el paso del tiempo en el simulador. Los paquetes
   * que han completado su viaje axonal se entregan a las regiones destino.
   * El peso de la conexión escala los spikes, modelando la eficacia
   * sináptica del tracto.
   *
   * @param currentTime - Tiempo actual de simulación (ms)
   * @returns Número de paquetes entregados en este tick
   */
  tick(currentTime: number): number {
    let delivered = 0;

    // La cola está ordenada, así que procesamos desde el inicio
    while (
      this.delayQueue.length > 0 &&
      this.delayQueue[0].deliveryTime <= currentTime
    ) {
      const delayed = this.delayQueue.shift()!;
      const targetStats = this.regions.get(delayed.target);

      if (targetStats) {
        targetStats.packetsReceived++;
        targetStats.lastActivity = currentTime;
      }

      // Escalar spikes por peso de conexión si es distinto de 1.0
      let deliveredSpikes = delayed.packet.spikes;
      if (delayed.connectionWeight !== 1.0) {
        deliveredSpikes = new Float32Array(delayed.packet.spikes.length);
        for (let i = 0; i < deliveredSpikes.length; i++) {
          deliveredSpikes[i] =
            delayed.packet.spikes[i] * delayed.connectionWeight;
        }
      }

      // Emitir evento con paquete ajustado
      const deliveredPacket: SpikePacket = {
        source: delayed.packet.source,
        targets: [delayed.target],
        spikes: deliveredSpikes,
        timestamp: delayed.packet.timestamp,
        metadata: delayed.packet.metadata,
      };

      this.emit(`spike:${delayed.target}`, deliveredPacket);
      delivered++;
      this.totalPacketsProcessed++;
    }

    return delivered;
  }

  /**
   * Retorna estadísticas de tráfico actuales por región.
   *
   * @returns Mapa de regionId → estadísticas de tráfico
   */
  getTraffic(): Map<string, RegionTrafficStats> {
    const copy = new Map<string, RegionTrafficStats>();
    for (const [id, stats] of this.regions) {
      copy.set(id, { ...stats });
    }
    return copy;
  }

  /**
   * Retorna el número de paquetes pendientes en la cola de retardo.
   */
  get pendingCount(): number {
    return this.delayQueue.length;
  }

  /**
   * Retorna el número total de paquetes procesados desde la creación del bus.
   */
  get processedCount(): number {
    return this.totalPacketsProcessed;
  }

  /**
   * Verifica si una región está registrada.
   */
  isRegistered(regionId: string): boolean {
    return this.regions.has(regionId);
  }

  /**
   * Limpia la cola de paquetes pendientes.
   */
  clearQueue(): void {
    this.delayQueue.length = 0;
  }

  /**
   * Resetea todas las estadísticas de tráfico.
   */
  resetStats(): void {
    for (const [, stats] of this.regions) {
      stats.packetsSent = 0;
      stats.packetsReceived = 0;
      stats.totalSpikesEmitted = 0;
      stats.lastActivity = 0;
    }
    this.totalPacketsProcessed = 0;
  }
}
