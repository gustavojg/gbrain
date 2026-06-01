/**
 * Protocolo Binario Multi-Región para Persistencia Cerebral
 * ===========================================================
 * Extiende el protocolo binario original (04_binary_optimization) para soportar
 * múltiples regiones cerebrales y estado de neuromodulación.
 *
 * Biología: El cerebro humano requiere ~20W de energía y almacena ~2.5 PB
 * de información en sinapsis. Para persistir nuestro cerebro digital con
 * mínima latencia y máxima fidelidad, usamos un protocolo binario compacto
 * que evita overhead de serialización JSON.
 *
 * Formato del archivo binario:
 * ┌─────────────────────────────────────────────────────────────┐
 * │ HEADER                                                      │
 * │  magic (4 bytes) = 0xBRA1001 ("BRAIN001")                  │
 * │  version (4 bytes) = 1                                      │
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
 * └─────────────────────────────────────────────────────────────┘
 */

import * as fs from 'fs';
import type { BrainRegion } from '../brain-region.js';
import type {
  NeuromodulatorSystem,
  NeuromodulatorSnapshot,
  ModulatorState,
  ModulatorType,
} from '../neuromodulators/modulator-system.js';

/** Número mágico para identificar archivos de cerebro digital: "BRA1N001" codificado */
export const MAGIC_NUMBER = 0xb4a10001;

/** Versión actual del protocolo binario */
export const PROTOCOL_VERSION = 1;

/**
 * Tamaño del header principal en bytes:
 * magic(4) + version(4) + numRegions(4) + modulatorBlockOffset(4) + modulatorBlockSize(4)
 */
const HEADER_SIZE = 20;

/** Tamaño de cada entrada en la tabla de offsets: hash(4) + offset(4) + size(4) */
const OFFSET_ENTRY_SIZE = 12;

/**
 * Datos de una región cerebral extraídos del archivo binario.
 */
export interface LoadedRegionData {
  /** Identificador de la región */
  id: string;
  /** Número de neuronas */
  neuronCount: number;
  /** Número de entradas */
  inputCount: number;
  /** Pesos sinápticos restaurados */
  weights: Float32Array;
}

/**
 * Resultado completo de la carga de un archivo de cerebro.
 */
export interface LoadedBrainData {
  /** Datos de cada región, indexados por id */
  regions: Map<string, LoadedRegionData>;
  /** Estado de los neuromoduladores (si fue guardado) */
  modulatorState: NeuromodulatorSnapshot | null;
}

/**
 * Calcula un hash simple de 32 bits para un string.
 *
 * Usado para la tabla de offsets del protocolo binario.
 * Algoritmo FNV-1a de 32 bits para buena distribución.
 *
 * @param str - String a hashear
 * @returns Hash de 32 bits como número unsigned
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
 * Serializa el bloque de datos de una región cerebral.
 *
 * Formato: idLength(4) + id(UTF-8) + neuronCount(4) + inputCount(4) + weights(Float32Array)
 *
 * @param region - Región cerebral a serializar
 * @returns Buffer con los datos binarios de la región
 */
export function serializeRegion(region: BrainRegion): Buffer {
  const config = region.getNetworkConfig();
  const idBytes = Buffer.from(region.id, 'utf-8');

  // Calcular tamaño total del bloque
  const idLenSize = 4;
  const metaSize = 8; // neuronCount(4) + inputCount(4)
  const weightsSize = config.weights.byteLength;
  const totalSize = idLenSize + idBytes.length + metaSize + weightsSize;

  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  // Escribir longitud del id
  buffer.writeUInt32LE(idBytes.length, offset);
  offset += 4;

  // Escribir id
  idBytes.copy(buffer, offset);
  offset += idBytes.length;

  // Escribir neuronCount
  buffer.writeUInt32LE(config.neuronCount, offset);
  offset += 4;

  // Escribir inputCount
  buffer.writeUInt32LE(config.inputCount, offset);
  offset += 4;

  // Escribir pesos (copiar Float32Array directamente)
  const weightsBuffer = Buffer.from(
    config.weights.buffer,
    config.weights.byteOffset,
    config.weights.byteLength
  );
  weightsBuffer.copy(buffer, offset);

  return buffer;
}

/**
 * Serializa el estado de neuromodulación.
 *
 * Formato:
 *   numModulators(4)
 *   per modulator:
 *     nameLength(4) + name(UTF-8) + level(Float32) + baseline(Float32)
 *     + decayRate(Float32) + lastUpdate(Float64)
 *
 * @param system - Sistema de neuromodulación a serializar
 * @returns Buffer con datos binarios del estado de modulación
 */
function serializeModulators(system: NeuromodulatorSystem): Buffer {
  const snapshot = system.serialize();
  const entries = Object.entries(snapshot.modulators);

  // Calcular tamaño total
  let totalSize = 4; // numModulators
  for (const [name] of entries) {
    const nameBytes = Buffer.from(name, 'utf-8');
    totalSize += 4 + nameBytes.length + 4 + 4 + 4 + 8;
    // nameLen + name + level + baseline + decayRate + lastUpdate
  }

  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  // Número de moduladores
  buffer.writeUInt32LE(entries.length, offset);
  offset += 4;

  for (const [name, state] of entries) {
    const nameBytes = Buffer.from(name, 'utf-8');

    // Longitud del nombre
    buffer.writeUInt32LE(nameBytes.length, offset);
    offset += 4;

    // Nombre
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
 * Deserializa el bloque de moduladores desde un buffer.
 *
 * @param buffer - Buffer conteniendo datos de modulación
 * @param startOffset - Offset inicial en el buffer
 * @returns Snapshot de neuromodulación restaurado
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
    // Leer nombre
    const nameLen = buffer.readUInt32LE(offset);
    offset += 4;
    const name = buffer.subarray(offset, offset + nameLen).toString('utf-8');
    offset += nameLen;

    // Leer state
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
 * Sistema de persistencia binaria para el cerebro digital multi-región.
 *
 * Serializa y deserializa el estado completo del cerebro (todas las regiones
 * y neuromoduladores) en un formato binario compacto optimizado para
 * lecturas/escrituras rápidas de Float32Array.
 */
export class BrainPersistence {
  /**
   * Guarda el estado completo del cerebro en un archivo binario.
   *
   * @param filePath - Ruta del archivo de destino
   * @param regions - Mapa de regiones cerebrales a guardar
   * @param modulators - Sistema de neuromodulación a guardar
   */
  save(
    filePath: string,
    regions: Map<string, BrainRegion>,
    modulators: NeuromodulatorSystem
  ): void {
    const numRegions = regions.size;

    // 1. Serializar cada región
    const regionBuffers: { id: string; buffer: Buffer }[] = [];
    for (const [id, region] of regions) {
      regionBuffers.push({ id, buffer: serializeRegion(region) });
    }

    // 2. Serializar moduladores
    const modulatorBuffer = serializeModulators(modulators);

    // 3. Calcular offsets
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

    // 4. Construir el archivo completo
    const totalSize =
      HEADER_SIZE + offsetTableSize +
      regionBuffers.reduce((sum, rb) => sum + rb.buffer.length, 0) +
      modulatorBlockSize;

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

    // 5. Escribir a disco
    fs.writeFileSync(filePath, fileBuffer);
  }

  /**
   * Carga el estado del cerebro desde un archivo binario.
   *
   * @param filePath - Ruta del archivo a cargar
   * @returns Datos cargados de regiones y moduladores
   * @throws Error si el archivo no existe o tiene formato inválido
   */
  load(filePath: string): LoadedBrainData {
    if (!fs.existsSync(filePath)) {
      throw new Error(`[BrainPersistence] Archivo no encontrado: ${filePath}`);
    }

    const fileBuffer = fs.readFileSync(filePath);
    let readOffset = 0;

    // Leer header
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
    if (version !== PROTOCOL_VERSION) {
      throw new Error(
        `[BrainPersistence] Versión de protocolo incompatible: ${version}. ` +
          `Esperada: ${PROTOCOL_VERSION}`
      );
    }

    const numRegions = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;
    const modulatorBlockOffset = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;
    const modulatorBlockSize = fileBuffer.readUInt32LE(readOffset);
    readOffset += 4;

    // Leer offset table
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

    // Leer bloques de regiones
    const regions = new Map<string, LoadedRegionData>();
    for (const entry of offsetTable) {
      let regionOffset = entry.offset;

      // Leer id
      const idLen = fileBuffer.readUInt32LE(regionOffset);
      regionOffset += 4;
      const id = fileBuffer
        .subarray(regionOffset, regionOffset + idLen)
        .toString('utf-8');
      regionOffset += idLen;

      // Leer neuronCount e inputCount
      const neuronCount = fileBuffer.readUInt32LE(regionOffset);
      regionOffset += 4;
      const inputCount = fileBuffer.readUInt32LE(regionOffset);
      regionOffset += 4;

      // Leer pesos
      const weightsLength = neuronCount * inputCount;
      const weights = new Float32Array(weightsLength);
      for (let i = 0; i < weightsLength; i++) {
        weights[i] = fileBuffer.readFloatLE(regionOffset);
        regionOffset += 4;
      }

      regions.set(id, { id, neuronCount, inputCount, weights });
    }

    // Leer moduladores
    let modulatorState: NeuromodulatorSnapshot | null = null;
    if (modulatorBlockSize > 0) {
      modulatorState = deserializeModulators(
        fileBuffer,
        modulatorBlockOffset
      );
    }

    return { regions, modulatorState };
  }

  /**
   * Serializa una única región cerebral como Buffer independiente.
   *
   * Útil para transmisión por red o almacenamiento incremental.
   *
   * @param region - Región cerebral a serializar
   * @returns Buffer con los datos binarios de la región
   */
  saveRegion(region: BrainRegion): Buffer {
    return serializeRegion(region);
  }

  /**
   * Carga una única región desde un Buffer independiente.
   *
   * @param buffer - Buffer con datos de una región serializada
   * @returns Datos de la región cargada
   */
  loadRegion(buffer: Buffer): LoadedRegionData {
    let offset = 0;

    // Leer id
    const idLen = buffer.readUInt32LE(offset);
    offset += 4;
    const id = buffer.subarray(offset, offset + idLen).toString('utf-8');
    offset += idLen;

    // Leer dimensiones
    const neuronCount = buffer.readUInt32LE(offset);
    offset += 4;
    const inputCount = buffer.readUInt32LE(offset);
    offset += 4;

    // Leer pesos
    const weightsLength = neuronCount * inputCount;
    const weights = new Float32Array(weightsLength);
    for (let i = 0; i < weightsLength; i++) {
      weights[i] = buffer.readFloatLE(offset);
      offset += 4;
    }

    return { id, neuronCount, inputCount, weights };
  }
}
