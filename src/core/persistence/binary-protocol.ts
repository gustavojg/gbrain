/**
 * Multi-Region Binary Protocol for Brain Persistence
 * ===========================================================
 * Extends the original binary protocol (04_binary_optimization) to support
 * multiple brain regions and neuromodulation state.
 *
 * Biology: The human brain requires ~20W of energy and stores ~2.5 PB
 * of information in synapses. To persist our digital brain with
 * minimal latency and maximum fidelity, we use a compact binary protocol
 * that avoids JSON serialization overhead.
 *
 * Binary file format:
 * ┌─────────────────────────────────────────────────────────────┐
 * │ HEADER                                                      │
 * │  magic (4 bytes) = 0xBRA1001 ("BRAIN001")                  │
 * │  version (4 bytes) = 3  (1 = no CRC; 2 = CRC, no extras)    │
 * │  numRegions (4 bytes)                                       │
 * │  modulatorBlockOffset (4 bytes)                             │
 * │  modulatorBlockSize (4 bytes)                               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ REGION OFFSET TABLE (numRegions × 12 bytes)                │
 * │  per entry: regionIdHash(4) + offset(4) + size(4)          │
 * ├─────────────────────────────────────────────────────────────┤
 * │ REGION DATA BLOCKS (variable size per region)              │
 * │  per region:                                                │
 * │    idLength(4) + id(string) + neuronCount(4) + inputCount(4)│
 * │    + weights(Float32Array: neuronCount × inputCount × 4)    │
 * ├─────────────────────────────────────────────────────────────┤
 * │ MODULATOR BLOCK                                             │
 * │  numModulators(4) + per modulator:                          │
 * │    nameLength(4) + name(string) + level(4) + baseline(4)   │
 * │    + decayRate(4) + lastUpdate(8)                           │
 * ├─────────────────────────────────────────────────────────────┤
 * │ EXTRAS BLOCK (version ≥ 3)                                  │
 * │  length(4) + UTF-8 JSON: everything learned that is not a   │
 * │  weight matrix (episodic index, homeostatic state, clock…)  │
 * ├─────────────────────────────────────────────────────────────┤
 * │ TRAILER (version ≥ 2)                                       │
 * │  crc32 (4 bytes) of every preceding byte                    │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Durability: files are written to `<path>.tmp`, fsynced and renamed over the
 * target, keeping the previous snapshot as `<path>.bak`. A crash mid-write can
 * therefore never leave a truncated state file as the only copy, and the CRC
 * lets `load()` reject a corrupted file instead of restoring garbage weights.
 */

import * as fs from 'fs';
import * as zlib from 'zlib';
import type { BrainRegion } from '../brain-region.js';
import type {
  NeuromodulatorSystem,
  NeuromodulatorSnapshot,
  ModulatorState,
  ModulatorType,
} from '../neuromodulators/modulator-system.js';

/** Magic number to identify digital brain files: "BRA1N001" encoded */
export const MAGIC_NUMBER = 0xb4a10001;

/** Current version of the binary protocol (2 added the CRC-32 trailer, 3 the extras block) */
export const PROTOCOL_VERSION = 3;

/** Oldest version `load()` still understands (no CRC trailer). */
const LEGACY_PROTOCOL_VERSION = 1;

/** Size of the CRC-32 trailer in bytes. */
const CRC_SIZE = 4;

/** Suffix of the previous snapshot kept next to the state file. */
export const BACKUP_SUFFIX = '.bak';

let crcTable: Uint32Array | null = null;

/**
 * CRC-32 (IEEE) of a buffer. Uses the native `zlib.crc32` when the runtime
 * has it (Node ≥ 20.15 / 22.2), otherwise a table-driven fallback.
 */
export function crc32(data: Buffer): number {
  const native = (zlib as { crc32?: (data: Buffer) => number }).crc32;
  if (typeof native === 'function') return native(data) >>> 0;

  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = crcTable[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Writes a file atomically: temp file + fsync + rename. When `keepBackup` is
 * set, the previous version survives as `<path>.bak`.
 */
export function writeFileAtomic(filePath: string, data: Buffer | string, keepBackup = false): void {
  const tmpPath = `${filePath}.tmp`;
  const fd = fs.openSync(tmpPath, 'w');
  try {
    fs.writeFileSync(fd, data);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  if (keepBackup && fs.existsSync(filePath)) {
    fs.renameSync(filePath, `${filePath}${BACKUP_SUFFIX}`);
  }
  fs.renameSync(tmpPath, filePath);
}

/**
 * Size of the main header in bytes:
 * magic(4) + version(4) + numRegions(4) + modulatorBlockOffset(4) + modulatorBlockSize(4)
 */
const HEADER_SIZE = 20;

/** Size of each entry in the offset table: hash(4) + offset(4) + size(4) */
const OFFSET_ENTRY_SIZE = 12;

/**
 * Data of a brain region extracted from the binary file.
 */
export interface LoadedRegionData {
  /** Identifier of the region */
  id: string;
  /** Number of neurons */
  neuronCount: number;
  /** Number of inputs */
  inputCount: number;
  /** Restored synaptic weights */
  weights: Float32Array;
}

/**
 * Complete result of loading a brain file.
 */
export interface LoadedBrainData {
  /** Data of each region, indexed by id */
  regions: Map<string, LoadedRegionData>;
  /** State of the neuromodulators (if it was saved) */
  modulatorState: NeuromodulatorSnapshot | null;
  /** Protocol version the file was written with */
  version: number;
  /** Non-weight learned state, keyed by region id (plus 'brain'); empty before v3 */
  extras: Record<string, unknown>;
}

/** Packs a typed array as base64 (compact and bit-exact inside the JSON extras). */
export function packArray(array: Float32Array | Int32Array): string {
  return Buffer.from(array.buffer, array.byteOffset, array.byteLength).toString('base64');
}

/** Unpacks a base64 Float32Array; `null` unless it has exactly `length` finite values. */
export function unpackFloat32(data: unknown, length: number): Float32Array | null {
  if (typeof data !== 'string') return null;
  const bytes = Buffer.from(data, 'base64');
  if (bytes.length !== length * 4) return null;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = bytes.readFloatLE(i * 4);
  return out.every(Number.isFinite) ? out : null;
}

/** Unpacks a base64 Int32Array; `null` unless it has exactly `length` values. */
export function unpackInt32(data: unknown, length: number): Int32Array | null {
  if (typeof data !== 'string') return null;
  const bytes = Buffer.from(data, 'base64');
  if (bytes.length !== length * 4) return null;
  const out = new Int32Array(length);
  for (let i = 0; i < length; i++) out[i] = bytes.readInt32LE(i * 4);
  return out;
}

/**
 * Computes a simple 32-bit hash for a string.
 *
 * Used for the binary protocol's offset table.
 * 32-bit FNV-1a algorithm for good distribution.
 *
 * @param str - String to hash
 * @returns 32-bit hash as an unsigned number
 */
export function hashRegionId(str: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0; // FNV prime, unsigned
  }
  return hash >>> 0;
}

/**
 * Serializes the data block of a brain region.
 *
 * Format: idLength(4) + id(UTF-8) + neuronCount(4) + inputCount(4) + weights(Float32Array)
 *
 * @param region - Brain region to serialize
 * @returns Buffer with the binary data of the region
 */
export function serializeRegion(region: BrainRegion): Buffer {
  const config = region.getNetworkConfig();
  const idBytes = Buffer.from(region.id, 'utf-8');

  // Compute total size of the block
  const idLenSize = 4;
  const metaSize = 8; // neuronCount(4) + inputCount(4)
  const weightsSize = config.weights.byteLength;
  const totalSize = idLenSize + idBytes.length + metaSize + weightsSize;

  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  // Write the id length
  buffer.writeUInt32LE(idBytes.length, offset);
  offset += 4;

  // Write the id
  idBytes.copy(buffer, offset);
  offset += idBytes.length;

  // Write neuronCount
  buffer.writeUInt32LE(config.neuronCount, offset);
  offset += 4;

  // Write inputCount
  buffer.writeUInt32LE(config.inputCount, offset);
  offset += 4;

  // Write weights (copy Float32Array directly)
  const weightsBuffer = Buffer.from(
    config.weights.buffer,
    config.weights.byteOffset,
    config.weights.byteLength
  );
  weightsBuffer.copy(buffer, offset);

  return buffer;
}

/**
 * Serializes the neuromodulation state.
 *
 * Format:
 *   numModulators(4)
 *   per modulator:
 *     nameLength(4) + name(UTF-8) + level(Float32) + baseline(Float32)
 *     + decayRate(Float32) + lastUpdate(Float64)
 *
 * @param system - Neuromodulation system to serialize
 * @returns Buffer with binary data of the modulation state
 */
function serializeModulators(system: NeuromodulatorSystem): Buffer {
  const snapshot = system.serialize();
  const entries = Object.entries(snapshot.modulators);

  // Compute total size
  let totalSize = 4; // numModulators
  for (const [name] of entries) {
    const nameBytes = Buffer.from(name, 'utf-8');
    totalSize += 4 + nameBytes.length + 4 + 4 + 4 + 8;
    // nameLen + name + level + baseline + decayRate + lastUpdate
  }

  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  // Number of modulators
  buffer.writeUInt32LE(entries.length, offset);
  offset += 4;

  for (const [name, state] of entries) {
    const nameBytes = Buffer.from(name, 'utf-8');

    // Name length
    buffer.writeUInt32LE(nameBytes.length, offset);
    offset += 4;

    // Name
    nameBytes.copy(buffer, offset);
    offset += nameBytes.length;

    // level
    buffer.writeFloatLE(state.level, offset);
    offset += 4;

    // baseline
    buffer.writeFloatLE(state.baseline, offset);
    offset += 4;

    // decayRate
    buffer.writeFloatLE(state.decayRate, offset);
    offset += 4;

    // lastUpdate
    buffer.writeDoubleLE(state.lastUpdate, offset);
    offset += 8;
  }

  return buffer;
}

/**
 * Deserializes the modulator block from a buffer.
 *
 * @param buffer - Buffer containing modulation data
 * @param startOffset - Initial offset in the buffer
 * @returns Restored neuromodulation snapshot
 */
function deserializeModulators(
  buffer: Buffer,
  startOffset: number
): NeuromodulatorSnapshot {
  let offset = startOffset;

  const numModulators = buffer.readUInt32LE(offset);
  offset += 4;

  const modulators = {} as Record<string, ModulatorState>;

  for (let i = 0; i < numModulators; i++) {
    // Read name
    const nameLen = buffer.readUInt32LE(offset);
    offset += 4;
    const name = buffer.subarray(offset, offset + nameLen).toString('utf-8');
    offset += nameLen;

    // Read state
    const level = buffer.readFloatLE(offset);
    offset += 4;
    const baseline = buffer.readFloatLE(offset);
    offset += 4;
    const decayRate = buffer.readFloatLE(offset);
    offset += 4;
    const lastUpdate = buffer.readDoubleLE(offset);
    offset += 8;

    modulators[name] = { level, baseline, decayRate, lastUpdate };
  }

  return {
    modulators: modulators as Record<ModulatorType, ModulatorState>,
    timestamp: Date.now(),
  };
}

/**
 * Binary persistence system for the multi-region digital brain.
 *
 * Serializes and deserializes the complete state of the brain (all regions
 * and neuromodulators) in a compact binary format optimized for
 * fast reads/writes of Float32Array.
 */
export class BrainPersistence {
  /**
   * Saves the complete state of the brain to a binary file.
   *
   * @param filePath - Path of the destination file
   * @param regions - Map of brain regions to save
   * @param modulators - Neuromodulation system to save
   */
  save(
    filePath: string,
    regions: Map<string, BrainRegion>,
    modulators: NeuromodulatorSystem,
    extras: Record<string, unknown> = {}
  ): void {
    const numRegions = regions.size;
    const extrasJson = Buffer.from(JSON.stringify(extras), 'utf-8');

    // 1. Serialize each region
    const regionBuffers: { id: string; buffer: Buffer }[] = [];
    for (const [id, region] of regions) {
      regionBuffers.push({ id, buffer: serializeRegion(region) });
    }

    // 2. Serialize modulators
    const modulatorBuffer = serializeModulators(modulators);

    // 3. Compute offsets
    const offsetTableSize = numRegions * OFFSET_ENTRY_SIZE;
    let currentOffset = HEADER_SIZE + offsetTableSize;

    const offsetEntries: { hash: number; offset: number; size: number }[] = [];
    for (const rb of regionBuffers) {
      offsetEntries.push({
        hash: hashRegionId(rb.id),
        offset: currentOffset,
        size: rb.buffer.length,
      });
      currentOffset += rb.buffer.length;
    }

    const modulatorBlockOffset = currentOffset;
    const modulatorBlockSize = modulatorBuffer.length;

    // 4. Build the complete file
    const totalSize =
      HEADER_SIZE + offsetTableSize +
      regionBuffers.reduce((sum, rb) => sum + rb.buffer.length, 0) +
      modulatorBlockSize +
      4 + extrasJson.length +
      CRC_SIZE;

    const fileBuffer = Buffer.alloc(totalSize);
    let writeOffset = 0;

    // Header
    fileBuffer.writeUInt32LE(MAGIC_NUMBER, writeOffset);
    writeOffset += 4;
    fileBuffer.writeUInt32LE(PROTOCOL_VERSION, writeOffset);
    writeOffset += 4;
    fileBuffer.writeUInt32LE(numRegions, writeOffset);
    writeOffset += 4;
    fileBuffer.writeUInt32LE(modulatorBlockOffset, writeOffset);
    writeOffset += 4;
    fileBuffer.writeUInt32LE(modulatorBlockSize, writeOffset);
    writeOffset += 4;

    // Offset table
    for (const entry of offsetEntries) {
      fileBuffer.writeUInt32LE(entry.hash, writeOffset);
      writeOffset += 4;
      fileBuffer.writeUInt32LE(entry.offset, writeOffset);
      writeOffset += 4;
      fileBuffer.writeUInt32LE(entry.size, writeOffset);
      writeOffset += 4;
    }

    // Region data blocks
    for (const rb of regionBuffers) {
      rb.buffer.copy(fileBuffer, writeOffset);
      writeOffset += rb.buffer.length;
    }

    // Modulator block
    modulatorBuffer.copy(fileBuffer, writeOffset);
    writeOffset += modulatorBuffer.length;

    // Extras block
    fileBuffer.writeUInt32LE(extrasJson.length, writeOffset);
    writeOffset += 4;
    extrasJson.copy(fileBuffer, writeOffset);
    writeOffset += extrasJson.length;

    // Trailer: CRC-32 of everything before it
    fileBuffer.writeUInt32LE(crc32(fileBuffer.subarray(0, writeOffset)), writeOffset);

    // 5. Write to disk (atomically, keeping the previous snapshot as .bak)
    writeFileAtomic(filePath, fileBuffer, true);
  }

  /**
   * Loads the brain state from a binary file.
   *
   * @param filePath - Path of the file to load
   * @returns Loaded data of regions and modulators
   * @throws Error if the file does not exist or has an invalid format
   */
  load(filePath: string): LoadedBrainData {
    if (!fs.existsSync(filePath)) {
      throw new Error(`[BrainPersistence] Archivo no encontrado: ${filePath}`);
    }

    const fileBuffer = fs.readFileSync(filePath);
    let readOffset = 0;

    if (fileBuffer.length < HEADER_SIZE) {
      throw new Error(`[BrainPersistence] Archivo truncado (${fileBuffer.length} bytes)`);
    }

    // Read header
    const magic = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;
    if (magic !== MAGIC_NUMBER) {
      throw new Error(
        `[BrainPersistence] Número mágico inválido: 0x${magic.toString(16)}. ` +
          `Esperado: 0x${MAGIC_NUMBER.toString(16)}`
      );
    }

    const version = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;
    if (version < LEGACY_PROTOCOL_VERSION || version > PROTOCOL_VERSION) {
      throw new Error(
        `[BrainPersistence] Versión de protocolo incompatible: ${version}. ` +
          `Esperada: ${PROTOCOL_VERSION}`
      );
    }

    // Verify integrity before trusting any offset in the file.
    if (version >= 2) {
      const bodyLength = fileBuffer.length - CRC_SIZE;
      const stored = fileBuffer.readUInt32LE(bodyLength);
      const actual = crc32(fileBuffer.subarray(0, bodyLength));
      if (stored !== actual) {
        throw new Error(
          `[BrainPersistence] CRC inválido (archivo corrupto): ` +
            `0x${actual.toString(16)} ≠ 0x${stored.toString(16)}`
        );
      }
    }

    const numRegions = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;
    const modulatorBlockOffset = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;
    const modulatorBlockSize = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;

    // Read offset table
    const offsetTable: { hash: number; offset: number; size: number }[] = [];
    for (let i = 0; i < numRegions; i++) {
      const hash = fileBuffer.readUInt32LE(readOffset);
      readOffset += 4;
      const offset = fileBuffer.readUInt32LE(readOffset);
      readOffset += 4;
      const size = fileBuffer.readUInt32LE(readOffset);
      readOffset += 4;
      offsetTable.push({ hash, offset, size });
    }

    // Read region blocks
    const regions = new Map<string, LoadedRegionData>();
    for (const entry of offsetTable) {
      let regionOffset = entry.offset;

      // Read id
      const idLen = fileBuffer.readUInt32LE(regionOffset);
      regionOffset += 4;
      const id = fileBuffer
        .subarray(regionOffset, regionOffset + idLen)
        .toString('utf-8');
      regionOffset += idLen;

      // Read neuronCount and inputCount
      const neuronCount = fileBuffer.readUInt32LE(regionOffset);
      regionOffset += 4;
      const inputCount = fileBuffer.readUInt32LE(regionOffset);
      regionOffset += 4;

      // Read weights
      const weightsLength = neuronCount * inputCount;
      const weights = new Float32Array(weightsLength);
      for (let i = 0; i < weightsLength; i++) {
        weights[i] = fileBuffer.readFloatLE(regionOffset);
        regionOffset += 4;
      }

      regions.set(id, { id, neuronCount, inputCount, weights });
    }

    // Read modulators
    let modulatorState: NeuromodulatorSnapshot | null = null;
    if (modulatorBlockSize > 0) {
      modulatorState = deserializeModulators(
        fileBuffer,
        modulatorBlockOffset
      );
    }

    // Read extras (v3+): right after the modulator block
    let extras: Record<string, unknown> = {};
    if (version >= 3) {
      const extrasOffset = modulatorBlockOffset + modulatorBlockSize;
      const extrasLength = fileBuffer.readUInt32LE(extrasOffset);
      const json = fileBuffer.subarray(extrasOffset + 4, extrasOffset + 4 + extrasLength).toString('utf-8');
      const parsed: unknown = JSON.parse(json);
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        extras = parsed as Record<string, unknown>;
      }
    }

    return { regions, modulatorState, version, extras };
  }

  /**
   * Serializes a single brain region as an independent Buffer.
   *
   * Useful for network transmission or incremental storage.
   *
   * @param region - Brain region to serialize
   * @returns Buffer with the binary data of the region
   */
  saveRegion(region: BrainRegion): Buffer {
    return serializeRegion(region);
  }

  /**
   * Loads a single region from an independent Buffer.
   *
   * @param buffer - Buffer with data of a serialized region
   * @returns Data of the loaded region
   */
  loadRegion(buffer: Buffer): LoadedRegionData {
    let offset = 0;

    // Read id
    const idLen = buffer.readUInt32LE(offset);
    offset += 4;
    const id = buffer.subarray(offset, offset + idLen).toString('utf-8');
    offset += idLen;

    // Read dimensions
    const neuronCount = buffer.readUInt32LE(offset);
    offset += 4;
    const inputCount = buffer.readUInt32LE(offset);
    offset += 4;

    // Read weights
    const weightsLength = neuronCount * inputCount;
    const weights = new Float32Array(weightsLength);
    for (let i = 0; i < weightsLength; i++) {
      weights[i] = buffer.readFloatLE(offset);
      offset += 4;
    }

    return { id, neuronCount, inputCount, weights };
  }
}
