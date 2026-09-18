/**
 * SERVER GUARDS — Input validation, rate limiting and origin checks
 * ==================================================================
 * Everything the server receives from the network is untrusted. This module
 * turns raw JSON payloads into validated, bounded values (or rejects them)
 * before they reach the brain, so that a single malformed message cannot
 * poison the simulation (NaN neuromodulators), exhaust memory (huge arrays)
 * or monopolize the CPU (unthrottled perception).
 *
 * Kept free of side effects (no server, no brain) so it is unit-testable.
 */

import { timingSafeEqual } from 'crypto';
import { ModulatorType } from './core/neuromodulators/modulator-system.js';

// ================================================================
// LIMITS
// ================================================================

/** Max HTTP request body (a 128×128 JSON image is ~65 KB). */
export const MAX_BODY_BYTES = 128 * 1024;
/** Max WebSocket message size. */
export const MAX_WS_PAYLOAD_BYTES = 128 * 1024;
/** Max characters of text per `read`. */
export const MAX_TEXT_LENGTH = 500;
/** Max image side (pixels). */
export const MAX_IMAGE_SIDE = 128;
/** Max spectrogram length (values). */
export const MAX_SPECTROGRAM_LENGTH = 4096;
/** Max |amount| of a single manual neuromodulator injection. */
export const MAX_MODULATOR_AMOUNT = 1.0;

/** Error carrying an HTTP status, thrown by the validators/handlers. */
export class HttpError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

// ================================================================
// PAYLOAD VALIDATION
// ================================================================

const MODULATOR_TYPES: ReadonlySet<string> = new Set(Object.values(ModulatorType));

function asRecord(data: unknown): Record<string, unknown> {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new HttpError(400, 'Expected a JSON object');
  }
  return data as Record<string, unknown>;
}

/** Validates `{ type, amount }` for a manual neuromodulator injection. */
export function parseModulatorInput(data: unknown): { type: ModulatorType; amount: number } {
  const { type, amount } = asRecord(data);
  if (typeof type !== 'string' || !MODULATOR_TYPES.has(type)) {
    throw new HttpError(400, `Unknown modulator type (expected one of: ${[...MODULATOR_TYPES].join(', ')})`);
  }
  if (typeof amount !== 'number' || !Number.isFinite(amount)) {
    throw new HttpError(400, 'Modulator amount must be a finite number');
  }
  const clamped = Math.max(-MAX_MODULATOR_AMOUNT, Math.min(MAX_MODULATOR_AMOUNT, amount));
  return { type: type as ModulatorType, amount: clamped };
}

/** Validates `{ text }`: non-empty string, truncated to `MAX_TEXT_LENGTH`. */
export function parseTextInput(data: unknown): string {
  const { text } = asRecord(data);
  if (typeof text !== 'string') {
    throw new HttpError(400, 'text must be a string');
  }
  const trimmed = text.trim().slice(0, MAX_TEXT_LENGTH);
  if (trimmed.length === 0) {
    throw new HttpError(400, 'text must not be empty');
  }
  return trimmed;
}

/** Validates `{ pixels, width, height }` and returns grayscale bytes. */
export function parseImageInput(data: unknown): { pixels: Uint8Array; width: number; height: number } {
  const { pixels, width, height } = asRecord(data);
  if (
    typeof width !== 'number' || typeof height !== 'number' ||
    !Number.isInteger(width) || !Number.isInteger(height) ||
    width < 1 || height < 1 || width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE
  ) {
    throw new HttpError(400, `width/height must be integers in [1, ${MAX_IMAGE_SIDE}]`);
  }
  if (!Array.isArray(pixels) || pixels.length !== width * height) {
    throw new HttpError(400, 'pixels must be an array of length width × height');
  }
  const out = new Uint8Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    const v = pixels[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new HttpError(400, 'pixels must contain only finite numbers');
    }
    out[i] = Math.max(0, Math.min(255, Math.round(v)));
  }
  return { pixels: out, width, height };
}

/** Validates `{ spectrogram }` and returns magnitudes clamped to [0, 1]. */
export function parseSpectrogramInput(data: unknown): Float32Array {
  const { spectrogram } = asRecord(data);
  if (!Array.isArray(spectrogram) || spectrogram.length === 0 || spectrogram.length > MAX_SPECTROGRAM_LENGTH) {
    throw new HttpError(400, `spectrogram must be an array of 1..${MAX_SPECTROGRAM_LENGTH} numbers`);
  }
  const out = new Float32Array(spectrogram.length);
  for (let i = 0; i < spectrogram.length; i++) {
    const v = spectrogram[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new HttpError(400, 'spectrogram must contain only finite numbers');
    }
    out[i] = Math.max(0, Math.min(1, v));
  }
  return out;
}

/** Validates `{ babble?, imitate? }`: which parts of the voice to switch. */
export function parseVoiceInput(data: unknown): { babble?: boolean; imitate?: boolean } {
  const { babble, imitate } = asRecord(data);
  if ((babble !== undefined && typeof babble !== 'boolean') || (imitate !== undefined && typeof imitate !== 'boolean')) {
    throw new HttpError(400, 'babble and imitate must be booleans');
  }
  if (babble === undefined && imitate === undefined) {
    throw new HttpError(400, 'Expected babble and/or imitate');
  }
  return { babble: babble as boolean | undefined, imitate: imitate as boolean | undefined };
}

/** Validates `{ scribble?, copy? }`: which parts of the hand to switch. */
export function parseHandInput(data: unknown): { scribble?: boolean; copy?: boolean } {
  const { scribble, copy } = asRecord(data);
  if ((scribble !== undefined && typeof scribble !== 'boolean') || (copy !== undefined && typeof copy !== 'boolean')) {
    throw new HttpError(400, 'scribble and copy must be booleans');
  }
  if (scribble === undefined && copy === undefined) {
    throw new HttpError(400, 'Expected scribble and/or copy');
  }
  return { scribble: scribble as boolean | undefined, copy: copy as boolean | undefined };
}

/** Validates `{ positive }`: the teacher's verdict on what the brain has just recalled. */
export function parseFeedbackInput(data: unknown): boolean {
  const { positive } = asRecord(data);
  if (typeof positive !== 'boolean') throw new HttpError(400, 'positive must be a boolean');
  return positive;
}

/** Validates the optional `sampleRate` of an audio frame (Hz); `undefined` if absent. */
export function parseSampleRate(data: unknown): number | undefined {
  const { sampleRate } = asRecord(data);
  if (sampleRate === undefined) return undefined;
  if (typeof sampleRate !== 'number' || !Number.isFinite(sampleRate) || sampleRate < 8000 || sampleRate > 192000) {
    throw new HttpError(400, 'sampleRate must be a number between 8000 and 192000');
  }
  return sampleRate;
}

// ================================================================
// RATE LIMITING
// ================================================================

/**
 * Classic token bucket: `capacity` burst, refilled at `refillPerSec`.
 * `now` is injectable for deterministic tests.
 */
export class TokenBucket {
  private tokens: number;
  private last: number;

  constructor(
    private readonly capacity: number,
    private readonly refillPerSec: number,
    now: number = Date.now(),
  ) {
    this.tokens = capacity;
    this.last = now;
  }

  /** Takes one token if available. */
  tryTake(now: number = Date.now()): boolean {
    const elapsed = Math.max(0, now - this.last) / 1000;
    this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.refillPerSec);
    this.last = now;
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }
}

/** Per-kind limits for one client: `[burst, refill per second]`. */
export const CLIENT_LIMITS = {
  text: [5, 2],
  image: [1, 0.5],
  audio: [2, 1],
  modulator: [10, 5],
  tick: [10, 10],
} as const satisfies Record<string, readonly [number, number]>;

export type LimitedKind = keyof typeof CLIENT_LIMITS;

/** Set of token buckets for a single client (WS connection or HTTP address). */
export class ClientLimiter {
  private readonly buckets = new Map<LimitedKind, TokenBucket>();
  /** Last time this limiter was used (for eviction of idle HTTP clients). */
  lastSeen: number = Date.now();

  allow(kind: LimitedKind, now: number = Date.now()): boolean {
    this.lastSeen = now;
    let bucket = this.buckets.get(kind);
    if (!bucket) {
      const [capacity, refill] = CLIENT_LIMITS[kind];
      bucket = new TokenBucket(capacity, refill, now);
      this.buckets.set(kind, bucket);
    }
    return bucket.tryTake(now);
  }
}

// ================================================================
// ORIGIN / ADMIN CHECKS
// ================================================================

/**
 * Decides whether a browser `Origin` may drive the brain.
 *
 * - No `Origin` header → non-browser client (curl, scripts): allowed.
 * - Same host as one of `hosts` (the request's `Host`, or `X-Forwarded-Host`
 *   when behind a proxy) → the dashboard itself: allowed.
 * - Listed in `allowlist` (exact origin match): allowed.
 * - Anything else (a third-party page scripting `ws://localhost:3000`): denied.
 */
export function isOriginAllowed(
  origin: string | undefined,
  hosts: ReadonlyArray<string | undefined>,
  allowlist: readonly string[],
): boolean {
  if (!origin) return true;
  if (allowlist.includes(origin)) return true;
  try {
    const originHost = new URL(origin).host;
    return hosts.some((h) => h !== undefined && h === originHost);
  } catch {
    return false;
  }
}

/** Parses a comma-separated `ALLOWED_ORIGINS` value. */
export function parseAllowlist(value: string | undefined): string[] {
  return (value ?? '').split(',').map((s) => s.trim()).filter((s) => s.length > 0);
}

/**
 * Authorizes an admin-only endpoint (`/api/save`, `/api/tick`).
 *
 * With `BRAIN_ADMIN_TOKEN` configured, requires `Authorization: Bearer <token>`.
 * Without it, only direct loopback connections are accepted, so a public
 * deployment is closed by default while local development keeps working.
 */
export function isAdminAuthorized(
  authorization: string | undefined,
  adminToken: string | undefined,
  isLoopback: boolean,
): boolean {
  if (!adminToken) return isLoopback;
  const expected = Buffer.from(`Bearer ${adminToken}`);
  const given = Buffer.from(authorization ?? '');
  return given.length === expected.length && timingSafeEqual(given, expected);
}
