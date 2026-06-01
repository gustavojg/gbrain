/**
 * Inter-Regional Connectome
 * =========================
 * Defines the map of connections between brain regions of the digital brain.
 *
 * Biology: The connectome is the complete map of the brain's neural connections.
 * At the macro level, brain regions connect via white matter tracts
 * (myelinated axons) with conduction delays proportional to physical
 * distance. Each connection has a weight (mean synaptic strength) and an
 * axonal delay that reflects the action potential propagation time.
 *
 * Modeled regions:
 * - thalamus: Attentional filter and sensory relay
 * - visualCortex: Visual processing (V1-V4)
 * - auditoryCortex: Auditory processing (A1-A2)
 * - hippocampus: Memory formation and spatial navigation
 * - amygdala: Emotional processing and affective valence
 * - prefrontalCortex: Executive control and decision making
 * - brocaWernicke: Language production and comprehension
 */

/**
 * Directed connection between two brain regions.
 * Models a white matter tract with physiological properties.
 */
export interface Connection {
  /** Source region (spike emitter) */
  from: string;
  /** Target region (spike receiver) */
  to: string;
  /** Mean synaptic weight of the connection (0.0 - 1.0) */
  weight: number;
  /** Axonal conduction delay in milliseconds */
  delay: number;
  /** Whether the connection can be modulated by neuromodulators */
  modulatable: boolean;
}

/**
 * Default connectome with biologically inspired connections.
 *
 * Organization:
 * 1. Ventral pathway (Thalamus → Visual Cortex → Hippocampus): "what" processing
 * 2. Dorsal pathway (Thalamus → Auditory Cortex → Hippocampus): auditory "what" processing
 * 3. Limbic circuit (Amygdala ↔ Prefrontal): emotional regulation
 * 4. Language circuit (Broca-Wernicke ↔ Prefrontal): verbal production
 * 5. Top-down feedback (Prefrontal → Thalamus, Visual): attentional control
 */
export const DEFAULT_CONNECTOME: Connection[] = [
  // === Sensory feedforward pathways (thalamus → cortices) ===
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

  // === Sensory cortex → memory/emotion pathways ===
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

  // === Associative pathways → executive control ===
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

  // === Language circuit ===
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

  // === Feedback connections (top-down) ===
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
 * Manages the inter-regional connectome of the digital brain.
 *
 * Allows querying connections, obtaining input/output routes for
 * each region, and dynamically modulating synaptic weights in response
 * to neuromodulators or long-term plasticity.
 */
export class Connectome {
  /** Connection map indexed by source region */
  private readonly outgoing: Map<string, Connection[]> = new Map();
  /** Connection map indexed by target region */
  private readonly incoming: Map<string, Connection[]> = new Map();
  /** Complete list of connections */
  private readonly connections: Connection[];
  /** Set of all regions registered in the connectome */
  private readonly regionIds: Set<string> = new Set();

  /**
   * Creates a new connectome from a list of connections.
   *
   * @param connections - List of inter-regional connections (default: DEFAULT_CONNECTOME)
   */
  constructor(connections: Connection[] = DEFAULT_CONNECTOME) {
    // Clone connections to allow safe mutation
    this.connections = connections.map(c => ({ ...c }));
    this.buildIndices();
  }

  /**
   * Builds the incoming and outgoing connection indices.
   */
  private buildIndices(): void {
    this.outgoing.clear();
    this.incoming.clear();
    this.regionIds.clear();

    for (const conn of this.connections) {
      this.regionIds.add(conn.from);
      this.regionIds.add(conn.to);

      // Outgoing index
      if (!this.outgoing.has(conn.from)) {
        this.outgoing.set(conn.from, []);
      }
      this.outgoing.get(conn.from)!.push(conn);

      // Incoming index
      if (!this.incoming.has(conn.to)) {
        this.incoming.set(conn.to, []);
      }
      this.incoming.get(conn.to)!.push(conn);
    }
  }

  /**
   * Gets all outgoing connections from a region.
   *
   * Biology: Equivalent to the axonal tracts that depart from this region
   * toward other cortical/subcortical areas.
   *
   * @param regionId - Identifier of the source region
   * @returns List of outgoing connections (empty if the region has no outputs)
   */
  getOutgoing(regionId: string): ReadonlyArray<Connection> {
    return this.outgoing.get(regionId) ?? [];
  }

  /**
   * Gets all incoming connections to a region.
   *
   * Biology: Equivalent to the afferents (incoming axons) to this region
   * from other areas of the brain.
   *
   * @param regionId - Identifier of the target region
   * @returns List of incoming connections
   */
  getIncoming(regionId: string): ReadonlyArray<Connection> {
    return this.incoming.get(regionId) ?? [];
  }

  /**
   * Gets the direct connection between two regions, if it exists.
   *
   * @param from - Source region
   * @param to - Target region
   * @returns The connection if it exists, undefined if there is no direct connection
   */
  getConnection(from: string, to: string): Connection | undefined {
    const outConns = this.outgoing.get(from);
    if (!outConns) return undefined;
    return outConns.find(c => c.to === to);
  }

  /**
   * Modulates the weight of a specific connection.
   *
   * Biology: Neuromodulators (dopamine, serotonin, etc.) can
   * strengthen or weaken entire white matter tracts through
   * changes in the release probability of neurotransmitters.
   *
   * @param from - Source region
   * @param to - Target region
   * @param factor - Multiplicative factor for the weight (e.g. 1.2 = +20%)
   * @returns true if the connection was modulated successfully
   */
  modulateWeight(from: string, to: string, factor: number): boolean {
    const conn = this.getConnection(from, to);
    if (!conn || !conn.modulatable) return false;

    conn.weight = Math.max(0, Math.min(2.0, conn.weight * factor));
    return true;
  }

  /**
   * Directly sets the weight of a connection.
   *
   * @param from - Source region
   * @param to - Target region
   * @param weight - New synaptic weight (clamped to [0, 2.0])
   * @returns true if it was updated successfully
   */
  setWeight(from: string, to: string, weight: number): boolean {
    const conn = this.getConnection(from, to);
    if (!conn) return false;

    conn.weight = Math.max(0, Math.min(2.0, weight));
    return true;
  }

  /**
   * Returns all regions registered in the connectome.
   */
  getRegions(): ReadonlySet<string> {
    return this.regionIds;
  }

  /**
   * Returns all connections in the connectome.
   */
  getAllConnections(): ReadonlyArray<Connection> {
    return this.connections;
  }

  /**
   * Returns the total number of connections.
   */
  get connectionCount(): number {
    return this.connections.length;
  }
}
