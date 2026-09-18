/**
 * Central Spike Communication Bus
 * =======================================
 * Inter-regional messaging system for the digital brain.
 *
 * Biology: In the real brain, signals between regions travel as
 * action potentials (spikes) through myelinated axons with
 * conduction velocities of 1-100 m/s. The axonal delay depends on
 * the distance and the degree of myelination of the tract. This bus models
 * that latency using a priority queue ordered by delivery time.
 *
 * Architecture: Based on Node.js EventEmitter for asynchronous
 * notification to subscribed regions. Spike packets are enqueued
 * with a future delivery time (timestamp + delay) and are dispatched
 * on each simulator tick when their time has arrived.
 */

import { EventEmitter } from 'events';

/**
 * Spike packet transmitted between brain regions.
 * Models a volley of action potentials traveling through an axonal tract.
 */
export interface SpikePacket {
  /** Identifier of the emitting region */
  source: string;
  /** Identifiers of the target regions */
  targets: string[];
  /** Spike vector (0.0 = silence, 1.0 = spike) */
  spikes: Float32Array;
  /** Emission timestamp (ms of the simulation clock) */
  timestamp: number;
  /** Optional metadata (e.g. signal type, modulation, priority) */
  metadata?: Record<string, unknown>;
}

/**
 * Packet in transit with computed delivery time.
 * Includes the axonal delay and synaptic weight of the connection.
 */
interface DelayedPacket {
  /** Original spike packet */
  packet: SpikePacket;
  /** Specific target region (a packet may have multiple targets) */
  target: string;
  /** Delivery time = timestamp + axonal delay */
  deliveryTime: number;
  /** Connection weight to scale the spikes */
  connectionWeight: number;
}

/**
 * Traffic statistics per region.
 */
export interface RegionTrafficStats {
  /** Number of packets sent from this region */
  packetsSent: number;
  /** Number of packets received by this region */
  packetsReceived: number;
  /** Total number of spikes (sum of spikes > 0) sent */
  totalSpikesEmitted: number;
  /** Last activity timestamp */
  lastActivity: number;
}

/**
 * Central spike communication bus between brain regions.
 *
 * Manages the routing of neural signals with realistic axonal
 * delays. Regions register, send spike packets,
 * and receive deferred deliveries according to the connectome.
 *
 * Usage:
 * ```typescript
 * const bus = new SpikeBus();
 * bus.register('visualCortex');
 * bus.register('hippocampus');
 * bus.onReceive('hippocampus', (packet) => { ... });
 * bus.send({ source: 'visualCortex', targets: ['hippocampus'], spikes, timestamp: 100 });
 * bus.tick(110); // Delivers packets whose deliveryTime <= 110
 * ```
 */
export class SpikeBus extends EventEmitter {
  /** Registered regions with their traffic stats */
  private readonly regions: Map<string, RegionTrafficStats> = new Map();

  /**
   * Priority queue of delayed packets.
   * Ordered by ascending deliveryTime for efficient dispatch.
   * Binary insertion is used to maintain the order.
   */
  private readonly delayQueue: DelayedPacket[] = [];

  /** Default axonal delays per region pair (ms) */
  private readonly defaultDelay: number = 10;

  /** Default connection weights */
  private readonly defaultWeight: number = 1.0;

  /** Map of custom delays: "from->to" → delay */
  private readonly customDelays: Map<string, number> = new Map();

  /** Map of custom weights: "from->to" → weight */
  private readonly customWeights: Map<string, number> = new Map();

  /** Total counter of processed packets (for statistics) */
  private totalPacketsProcessed: number = 0;

  /**
   * Short-term synaptic depression state per connection: "from->to" → state.
   *
   * Biology: every volley consumes part of a projection's readily releasable
   * vesicle pool, which recovers over hundreds of ms (Tsodyks & Markram, 1997).
   * A pathway driven continuously runs out of resources and stops transmitting
   * until it has recovered. This is what makes reverberating activity in the
   * recurrent loops of the connectome (PFC → thalamus → cortex → PFC, …) fade
   * out after a stimulus instead of echoing forever: every region's k-WTA
   * re-normalizes its output to k spikes, so without depression the loop gain
   * never drops below 1.
   */
  private readonly depression: Map<string, { resources: number; lastTime: number; failed: boolean }> = new Map();

  /** Fraction of the available resources consumed by one volley. */
  private static readonly DEPRESSION_USE = 0.05;
  /** Recovery time constant of the resources (ms). */
  private static readonly DEPRESSION_RECOVERY_MS = 200;
  /** Below this level the pathway fails (stops transmitting)… */
  private static readonly DEPRESSION_FAIL_BELOW = 0.2;
  /** …and it only resumes once recovered above this level (hysteresis). */
  private static readonly DEPRESSION_RESUME_ABOVE = 0.5;

  constructor() {
    super();
    // Increase the listener limit for many regions
    this.setMaxListeners(100);
  }

  /**
   * Registers a brain region on the communication bus.
   *
   * @param regionId - Unique identifier of the region
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
   * Configures the axonal delay for a specific connection.
   *
   * @param from - Source region
   * @param to - Target region
   * @param delay - Delay in milliseconds
   */
  setDelay(from: string, to: string, delay: number): void {
    this.customDelays.set(`${from}->${to}`, delay);
  }

  /**
   * Configures the synaptic weight for a specific connection.
   *
   * @param from - Source region
   * @param to - Target region
   * @param weight - Synaptic weight (0.0 - 2.0)
   */
  setWeight(from: string, to: string, weight: number): void {
    this.customWeights.set(`${from}->${to}`, weight);
  }

  /**
   * Gets the configured delay for a connection.
   */
  private getDelay(from: string, to: string): number {
    return this.customDelays.get(`${from}->${to}`) ?? this.defaultDelay;
  }

  /**
   * Gets the configured weight for a connection.
   */
  private getWeight(from: string, to: string): number {
    return this.customWeights.get(`${from}->${to}`) ?? this.defaultWeight;
  }

  /**
   * Sends a spike packet from a region to its targets.
   *
   * Biology: Models the propagation of a volley of action potentials
   * from a cortical region toward its axonal projections. The packet
   * is enqueued with a delay computed according to the connectome.
   *
   * @param packet - Spike packet to transmit
   */
  send(packet: SpikePacket): void {
    const sourceStats = this.regions.get(packet.source);
    if (!sourceStats) {
      throw new Error(
        `[SpikeBus] Región origen '${packet.source}' no registrada. Registrar con bus.register().`
      );
    }

    // Count active spikes
    let spikeCount = 0;
    for (let i = 0; i < packet.spikes.length; i++) {
      if (packet.spikes[i] > 0) spikeCount++;
    }

    sourceStats.packetsSent++;
    sourceStats.totalSpikesEmitted += spikeCount;
    sourceStats.lastActivity = packet.timestamp;

    // Enqueue a packet for each target with its specific delay
    for (const target of packet.targets) {
      if (!this.regions.has(target)) continue; // Ignore unregistered targets

      // Short-term depression: a silent volley costs nothing; an exhausted
      // pathway transmits nothing until it has recovered.
      if (spikeCount > 0 && !this.transmits(packet.source, target, packet.timestamp)) continue;

      const delay = this.getDelay(packet.source, target);
      const weight = this.getWeight(packet.source, target);
      const deliveryTime = packet.timestamp + delay;

      const delayed: DelayedPacket = {
        packet,
        target,
        deliveryTime,
        connectionWeight: weight,
      };

      // Binary insertion to keep the order by deliveryTime
      this.insertSorted(delayed);
    }
  }

  /**
   * Updates the short-term depression of a connection for a volley sent at
   * `time` and tells whether the volley gets through.
   */
  private transmits(from: string, to: string, time: number): boolean {
    const key = `${from}->${to}`;
    let state = this.depression.get(key);
    if (!state) {
      state = { resources: 1, lastTime: time, failed: false };
      this.depression.set(key, state);
    }

    // Exponential recovery toward 1 since the last volley
    const elapsed = Math.max(0, time - state.lastTime);
    state.resources = 1 - (1 - state.resources) * Math.exp(-elapsed / SpikeBus.DEPRESSION_RECOVERY_MS);
    state.lastTime = time;

    if (state.failed) {
      if (state.resources < SpikeBus.DEPRESSION_RESUME_ABOVE) return false;
      state.failed = false;
    }

    state.resources *= 1 - SpikeBus.DEPRESSION_USE;
    if (state.resources < SpikeBus.DEPRESSION_FAIL_BELOW) state.failed = true;
    return true;
  }

  /**
   * Available synaptic resources (0–1) of a connection, for monitoring.
   */
  getSynapticResources(from: string, to: string): number {
    return this.depression.get(`${from}->${to}`)?.resources ?? 1;
  }

  /**
   * Inserts a packet into the queue keeping ascending order by deliveryTime.
   * Uses binary search for O(log n) efficiency.
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
   * Subscribes a region to receive spike packets.
   *
   * @param regionId - Identifier of the receiving region
   * @param callback - Function to invoke when a packet arrives
   */
  onReceive(
    regionId: string,
    callback: (packet: SpikePacket) => void
  ): void {
    this.on(`spike:${regionId}`, callback);
  }

  /**
   * Dispatches all packets whose delivery time has arrived.
   *
   * Biology: Simulates the passage of time in the simulator. Packets
   * that have completed their axonal journey are delivered to the target regions.
   * The connection weight scales the spikes, modeling the synaptic
   * efficacy of the tract.
   *
   * @param currentTime - Current simulation time (ms)
   * @returns Number of packets delivered in this tick
   */
  tick(currentTime: number): number {
    let delivered = 0;

    // The queue is ordered, so we process from the start
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

      // Scale spikes by connection weight if it differs from 1.0
      let deliveredSpikes = delayed.packet.spikes;
      if (delayed.connectionWeight !== 1.0) {
        deliveredSpikes = new Float32Array(delayed.packet.spikes.length);
        for (let i = 0; i < deliveredSpikes.length; i++) {
          deliveredSpikes[i] =
            delayed.packet.spikes[i] * delayed.connectionWeight;
        }
      }

      // Emit event with adjusted packet
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
   * Returns current traffic statistics per region.
   *
   * @returns Map of regionId → traffic statistics
   */
  getTraffic(): Map<string, RegionTrafficStats> {
    const copy = new Map<string, RegionTrafficStats>();
    for (const [id, stats] of this.regions) {
      copy.set(id, { ...stats });
    }
    return copy;
  }

  /**
   * Returns the number of pending packets in the delay queue.
   */
  get pendingCount(): number {
    return this.delayQueue.length;
  }

  /**
   * Returns the total number of packets processed since the bus was created.
   */
  get processedCount(): number {
    return this.totalPacketsProcessed;
  }

  /**
   * Checks whether a region is registered.
   */
  isRegistered(regionId: string): boolean {
    return this.regions.has(regionId);
  }

  /**
   * Clears the queue of pending packets.
   */
  clearQueue(): void {
    this.delayQueue.length = 0;
  }

  /**
   * Resets all traffic statistics.
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
