/**
 * HTTP + WebSocket SERVER — Digital brain interface
 * =========================================================
 * Server that exposes the digital brain via HTTP API and WebSocket
 * for real-time communication with the 3D dashboard.
 *
 * Endpoints:
 * - POST /api/input/text    → The brain reads text
 * - POST /api/input/image   → The brain sees an image
 * - POST /api/input/audio   → The brain hears one microphone frame ({ spectrogram: FFT magnitudes, sampleRate })
 * - GET  /api/state         → Complete brain state
 * - GET  /api/feel          → Emotional state
 * - GET  /api/speak         → The brain speaks
 * - GET  /api/imagine       → The brain imagines
 * - POST /api/modulator     → Inject a neuromodulator manually
 * - POST /api/voice         → Switch babbling / vocal imitation ({ babble?, imitate? })
 * - POST /api/hand          → Switch scribbling / drawing ({ scribble?, copy? })
 * - POST /api/feedback      → Teacher's verdict on the last recall ({ positive })
 * - POST /api/lesson        → Teach: show something with its name / sound, N times
 * - POST /api/practice      → Let it babble / scribble N times (learns its motor maps)
 * - WS   /ws                → Real-time stream
 *
 * The WebSocket streams state updates at 2 Hz.
 *
 * Everything received from the network is untrusted: payloads are validated
 * and bounded (server-guards.ts), inputs are rate-limited per client, and
 * perceptions are propagated in slices (perception-scheduler.ts) so no client
 * can block the event loop.
 *
 * Environment:
 * - PORT                 HTTP port (default 3000)
 * - BRAIN_STATE_PATH     Where the learning is persisted
 * - BRAIN_TICK_HZ        Ticks per second the brain is driven at (default 10, up to 100)
 * - BRAIN_SPEED          Simulation speed, ticks per timer interval (default 1; see below)
 * - BRAIN_ADMIN_TOKEN    Bearer token for /api/save and /api/tick. Without it
 *                        those endpoints only accept loopback connections.
 * - ALLOWED_ORIGINS      Comma-separated extra origins allowed to drive the
 *                        brain from a browser (the dashboard's own origin is
 *                        always allowed).
 */

import http from 'http';
import { readFileSync, existsSync, statSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer, type WebSocket } from 'ws';
import { DigitalBrain, type BrainState, type PerceptionResult } from './brain.js';
import { DEFAULT_BRAIN_CONFIG } from './brain.config.js';
import { BACKUP_SUFFIX } from './core/persistence/binary-protocol.js';
import { synthesizeSpectrum } from './core/voice/vocal-tract.js';
import { PerceptionScheduler, SchedulerBusyError } from './perception-scheduler.js';
import {
  ClientLimiter,
  HttpError,
  MAX_BODY_BYTES,
  MAX_WS_PAYLOAD_BYTES,
  TokenBucket,
  isAdminAuthorized,
  isOriginAllowed,
  parseAllowlist,
  parseImageInput,
  parseFeedbackInput,
  parseHandInput,
  parseLessonInput,
  parsePracticeInput,
  type LessonInput,
  parseModulatorInput,
  parseSampleRate,
  parseSpectrogramInput,
  parseTextInput,
  parseVoiceInput,
  type LimitedKind,
  parseVoiceContour,
} from './server-guards.js';

// ================================================================
// CONFIGURATION
// ================================================================

const PORT = parseInt(process.env.PORT || '3000', 10);
// import.meta.dirname only exists in Node ≥20.11; deriving it from import.meta.url
// also makes it robust on Node 16/18 (otherwise DASHBOARD_DIR falls back to cwd and 404s).
const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url));
const DASHBOARD_DIR = path.resolve(SERVER_DIR, 'dashboard');
/**
 * Tick rate (BRAIN_TICK_HZ, 1–100; default 10). Everything on a human timescale
 * (how long a stimulus stays in view, how long a percept waits to be bound with
 * the next, the pause between babbles) is defined in real milliseconds inside
 * the brain and converted to ticks with this rate, so raising it does not
 * change what the brain does per second of wall clock — only how much neural
 * time (dt per tick) it lives through in that second.
 */
const BRAIN_TICK_HZ = Math.max(1, Math.min(100, Math.round(Number(process.env.BRAIN_TICK_HZ) || 10)));
const TICK_INTERVAL_MS = Math.max(10, Math.round(1000 / BRAIN_TICK_HZ));
/**
 * Simulation speed: brain ticks per timer interval (BRAIN_SPEED, 1–20). Above 1
 * the brain lives faster than the clock on the wall — babbling and scribbling
 * sessions finish in minutes instead of hours — but everything measured in
 * ticks (how long a percept stays in mind to be bound with the next one, ~30 s
 * at speed 1) shrinks accordingly, so keep 1 for interactive teaching.
 */
const BRAIN_SPEED = Math.max(1, Math.min(20, Math.round(Number(process.env.BRAIN_SPEED) || 1)));
const BROADCAST_INTERVAL_MS = 500; // 2 Hz dashboard update (lighter)
const THOUGHT_INTERVAL_MS = 1200; // ~0.8 Hz live "thought" stream
const AUTOSAVE_INTERVAL_MS = 5 * 60_000; // Save the learning every 5 min
const SLEEP_INTERVAL_MS = 5 * 60_000; // Consolidate ("sleep") every 5 min of real time
const MAX_WS_CLIENTS = 100;
const HTTP_LIMITER_IDLE_MS = 10 * 60_000; // Forget idle HTTP clients after 10 min

const ADMIN_TOKEN = process.env.BRAIN_ADMIN_TOKEN || undefined;
const ALLOWED_ORIGINS = parseAllowlist(process.env.ALLOWED_ORIGINS);

// Path of the persisted state. On Railway the FS is ephemeral unless there is a
// mounted volume (RAILWAY_VOLUME_MOUNT_PATH); use it if it exists so that
// learning survives redeploys. Explicit override via BRAIN_STATE_PATH.
const STATE_PATH =
  process.env.BRAIN_STATE_PATH ??
  (process.env.RAILWAY_VOLUME_MOUNT_PATH
    ? path.join(process.env.RAILWAY_VOLUME_MOUNT_PATH, 'brain_state.bin')
    : path.resolve(process.cwd(), 'brain_state.bin'));

// MIME types
const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
};

// ================================================================
// INITIALIZATION
// ================================================================

console.log(`\n🌐 Starting Digital Brain server (${(1000 / TICK_INTERVAL_MS).toFixed(0)} Hz × speed ${BRAIN_SPEED})...\n`);

// Create the brain. Simulated time advances `dt` ms per tick and the server
// ticks every TICK_INTERVAL_MS, so the consolidation interval (simulated ms)
// is scaled to make the brain sleep every SLEEP_INTERVAL_MS of REAL time —
// unscaled, the default "5 minutes" would come around every ~8 hours.
const brain = new DigitalBrain({
  tickRate: 1000 / TICK_INTERVAL_MS,
  memory: {
    ...DEFAULT_BRAIN_CONFIG.memory,
    consolidationIntervalMs: (SLEEP_INTERVAL_MS / TICK_INTERVAL_MS) * BRAIN_SPEED * DEFAULT_BRAIN_CONFIG.snn.dt,
  },
});

// Perceptions are propagated in slices so the event loop is never blocked.
const scheduler = new PerceptionScheduler(() => brain.tick(), {
  ticksPerJob: brain.perceptionTicks,
});

// Restore previous learning if it exists (or its backup, if a save was interrupted)
if (existsSync(STATE_PATH) || existsSync(`${STATE_PATH}${BACKUP_SUFFIX}`)) {
  try {
    const { loaded, skipped } = brain.loadState(STATE_PATH);
    console.log(
      `💾 State restored from ${STATE_PATH} — regions: ${loaded.join(', ') || 'none'}` +
        (skipped.length ? ` | skipped (incompatible dims): ${skipped.join(', ')}` : ''),
    );
  } catch (err) {
    console.error(`⚠️  Could not restore state (${(err as Error).message}); starting fresh.`);
  }
} else {
  console.log(`💾 No previous state at ${STATE_PATH}; starting fresh.`);
}

// Warn if storage is ephemeral in production
if (process.env.RAILWAY_ENVIRONMENT && !process.env.RAILWAY_VOLUME_MOUNT_PATH) {
  console.warn(
    '⚠️  On Railway without a mounted volume: state will be lost on the next redeploy. ' +
      'Mount a volume (Railway → Service → Volumes) so learning persists.',
  );
}

/** Saves the state in a reentrant-safe way. */
let saving = false;
function persist(reason: string): void {
  if (saving) return;
  saving = true;
  try {
    brain.saveState(STATE_PATH);
    console.log(`💾 State saved (${reason}) → ${STATE_PATH}`);
  } catch (err) {
    console.error(`⚠️  Error saving state: ${(err as Error).message}`);
  } finally {
    saving = false;
  }
}

// ================================================================
// HTTP SERVER
// ================================================================

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://localhost:${PORT}`);
  const origin = req.headers.origin;
  const originAllowed = isOriginAllowed(origin, requestHosts(req), ALLOWED_ORIGINS);

  // CORS: read-only GETs are public; anything that drives the brain must come
  // from the dashboard's own origin or an explicitly allowed one.
  if (origin && originAllowed) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  } else if (req.method === 'GET') {
    res.setHeader('Access-Control-Allow-Origin', '*');
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('X-Content-Type-Options', 'nosniff');

  if (req.method === 'OPTIONS') {
    res.writeHead(originAllowed ? 204 : 403);
    res.end();
    return;
  }

  try {
    // --- API ROUTES ---
    if (url.pathname.startsWith('/api/')) {
      if (req.method !== 'GET' && !originAllowed) {
        throw new HttpError(403, 'Origin not allowed');
      }
      await handleApiRoute(url, req, res);
      return;
    }

    // --- STATIC FILES (Dashboard) ---
    const filePath = url.pathname === '/' ? '/index.html' : url.pathname;
    const fullPath = path.join(DASHBOARD_DIR, filePath);
    const insideDashboard = fullPath.startsWith(DASHBOARD_DIR + path.sep);

    if (insideDashboard && existsSync(fullPath) && statSync(fullPath).isFile()) {
      const ext = path.extname(fullPath);
      const mime = MIME_TYPES[ext] || 'application/octet-stream';
      const content = readFileSync(fullPath);
      res.writeHead(200, { 'Content-Type': mime });
      res.end(content);
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Not found', path: url.pathname }));
    }
  } catch (err) {
    const status =
      err instanceof HttpError ? err.status :
      err instanceof SchedulerBusyError ? 503 :
      500;
    if (status === 500) console.error('❌ Error:', err);
    if (status === 413) {
      // The rest of the oversized body is never read: close the connection.
      res.setHeader('Connection', 'close');
      res.on('finish', () => req.destroy());
    }
    if (!res.headersSent) {
      res.writeHead(status, { 'Content-Type': 'application/json' });
    }
    res.end(JSON.stringify({ error: status === 500 ? 'Internal server error' : (err as Error).message }));
  }
});

// ----------------------------------------------------------------
// Per-client rate limiting (HTTP clients are keyed by address)
// ----------------------------------------------------------------

const httpLimiters: Map<string, ClientLimiter> = new Map();
/** `/api/sleep` is public (dashboard button) but global: one every 10 s. */
const sleepBucket = new TokenBucket(1, 0.1);

/** Hosts this request was addressed to (a proxy may move the public one to X-Forwarded-Host). */
function requestHosts(req: http.IncomingMessage): Array<string | undefined> {
  const forwardedHost = req.headers['x-forwarded-host'];
  return [req.headers.host, typeof forwardedHost === 'string' ? forwardedHost : undefined];
}

function clientAddress(req: http.IncomingMessage): string {
  // Behind Railway's proxy the socket address is the proxy's; the client is
  // the first hop of X-Forwarded-For. Only trusted when actually deployed there.
  const forwarded = req.headers['x-forwarded-for'];
  if (process.env.RAILWAY_ENVIRONMENT && typeof forwarded === 'string') {
    return forwarded.split(',')[0].trim();
  }
  return req.socket.remoteAddress ?? 'unknown';
}

function isLoopback(req: http.IncomingMessage): boolean {
  if (req.headers['x-forwarded-for']) return false; // proxied → not local
  const addr = req.socket.remoteAddress ?? '';
  return addr === '127.0.0.1' || addr === '::1' || addr === '::ffff:127.0.0.1';
}

function enforceHttpLimit(req: http.IncomingMessage, kind: LimitedKind): void {
  const key = clientAddress(req);
  let limiter = httpLimiters.get(key);
  if (!limiter) {
    limiter = new ClientLimiter();
    httpLimiters.set(key, limiter);
  }
  if (!limiter.allow(kind)) {
    throw new HttpError(429, 'Too many requests');
  }
}

function requireAdmin(req: http.IncomingMessage): void {
  if (!isAdminAuthorized(req.headers.authorization, ADMIN_TOKEN, isLoopback(req))) {
    throw new HttpError(401, 'Admin token required');
  }
}

/** Parses a JSON body, mapping syntax errors to 400. */
async function parseJsonBody(req: http.IncomingMessage): Promise<unknown> {
  const body = await parseBody(req);
  try {
    return JSON.parse(body);
  } catch {
    throw new HttpError(400, 'Invalid JSON');
  }
}

/**
 * Queues a perception and waits for it. `client` scopes the coalescing of
 * streaming inputs, so one client's frames never replace another's.
 */
function perceive(
  inputType: PerceptionResult['inputType'],
  inject: () => void,
  coalesceKey?: string,
): Promise<PerceptionResult | null> {
  let startTime = 0;
  return scheduler.submit({
    coalesceKey,
    inject: () => {
      startTime = brain.time;
      inject();
    },
    finish: () => brain.describePerception(inputType, startTime),
  });
}

/** Sends a message to every connected dashboard. */
function broadcast(type: string, data: unknown): void {
  if (clients.size === 0) return;
  const msg = JSON.stringify({ type, data }, replacer);
  for (const client of clients) {
    if (client.readyState === 1) client.send(msg);
  }
}

// ----------------------------------------------------------------
// Teaching: lessons and practice
// ----------------------------------------------------------------
// A lesson shows something together with its name and/or its sound, a few
// times, exactly as one would teach an infant — and reports, after every
// repetition, how much the brain already brought back from memory before that
// repetition: the learning curve. Practice lets the brain babble or scribble
// many times in a row; both run through the perception scheduler, so they are
// fast (ticks in slices) and never block the event loop.

/** Formants of the vowels a lesson can pair with (same vocal tract as the brain's). */
const LESSON_VOWELS: Record<NonNullable<LessonInput['vowel']>, [number, number]> = {
  a: [700, 1200], e: [500, 1900], i: [300, 2300], o: [500, 900], u: [350, 800],
};
/** Between the parts of one repetition (the first part is complete, and still in mind). */
const LESSON_STEP_TICKS = brain.ticksFor(5000);
/** After a repetition, for the wave to end. */
const LESSON_REST_TICKS = brain.ticksFor(20_000);
/** Per babble / scribble during practice (the utterance and its way back). */
const PRACTICE_ROUND_TICKS = brain.ticksFor(6000);

let teaching: string | null = null;

/** Runs `inject`, then `ticks` ticks, through the scheduler. */
const run = (inject: () => void, ticks: number): Promise<unknown> =>
  scheduler.submit({ inject, ticks, finish: () => null });

async function runLesson(lesson: LessonInput): Promise<void> {
  const label = [lesson.image ? 'drawing' : null, lesson.text ? `"${lesson.text}"` : null, lesson.vowel ? `/${lesson.vowel}/` : null]
    .filter(Boolean).join(' + ');
  teaching = `lesson: ${label}`;
  console.log(`🎓 Lesson started: ${label} × ${lesson.repetitions}`);
  try {
    for (let rep = 1; rep <= lesson.repetitions; rep++) {
      if (lesson.image) {
        const { pixels, width, height } = lesson.image;
        await run(() => brain.see(pixels, width, height, { propagate: false }), LESSON_STEP_TICKS);
      }
      if (lesson.vowel) {
        const [f1, f2] = LESSON_VOWELS[lesson.vowel];
        await run(() => brain.hearFrame(synthesizeSpectrum({ f1, f2, amplitude: 0.9 }), 48000, { propagate: false }), LESSON_STEP_TICKS);
      }
      if (lesson.text) {
        const text = lesson.text;
        await run(() => brain.read(text, { propagate: false }), LESSON_STEP_TICKS);
      }
      // What the LAST part brought back from memory — learned from the
      // repetitions before this one.
      const recall = brain.getLastRecall();
      broadcast('lesson', {
        label,
        repetition: rep,
        of: lesson.repetitions,
        confidence: recall?.confidence ?? 0,
        confident: recall?.confident ?? false,
        recalled: recall
          ? { words: recall.words.map((w) => w.word), visual: recall.visual?.label ?? null, auditory: recall.auditory?.label ?? null }
          : null,
        bindings: brain.getState().association?.bindings ?? 0,
      });
      await run(() => {}, LESSON_REST_TICKS);
    }
  } finally {
    teaching = null;
    broadcast('lesson', { label, done: true, bindings: brain.getState().association?.bindings ?? 0 });
    console.log(`🎓 Lesson finished: ${label}`);
  }
}

async function runPractice(rounds: { voice: number; hand: number }): Promise<void> {
  teaching = 'practice';
  try {
    for (const [kind, total] of [['voice', rounds.voice], ['hand', rounds.hand]] as const) {
      for (let i = 1; i <= total; i++) {
        await run(() => { if (kind === 'voice') brain.babbleOnce(); else brain.scribbleOnce(); }, PRACTICE_ROUND_TICKS);
        if (i % 10 === 0 || i === total) broadcast('practice', { kind, done: i, of: total });
      }
    }
  } finally {
    teaching = null;
    broadcast('practice', { done: true, voice: brain.getState().voice?.babbles ?? 0, hand: brain.getState().hand?.scribbles ?? 0 });
  }
}

/** Starts a lesson or a practice session, one at a time. */
function startTeaching(job: () => Promise<void>): void {
  if (teaching) throw new HttpError(409, `Busy: ${teaching}`);
  job().catch((err) => console.error('❌ Teaching failed:', err));
}

/**
 * Handles API routes.
 */
async function handleApiRoute(url: URL, req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const sendJSON = (data: unknown, status: number = 200) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data, replacer));
  };

  // GET /api/state — Complete brain state
  if (url.pathname === '/api/state' && req.method === 'GET') {
    sendJSON(brain.getState());
    return;
  }

  // GET /api/feel — Emotional state
  if (url.pathname === '/api/feel' && req.method === 'GET') {
    sendJSON(brain.feel());
    return;
  }

  // GET /api/speak — The brain speaks
  if (url.pathname === '/api/speak' && req.method === 'GET') {
    sendJSON(brain.speak());
    return;
  }

  // GET /api/imagine — The brain imagines
  if (url.pathname === '/api/imagine' && req.method === 'GET') {
    const image = brain.imagine();
    sendJSON({
      width: image.width,
      height: image.height,
      ascii: image.ascii,
      pixels: Array.from(image.pixels),
    });
    return;
  }

  // POST /api/input/text — Read text
  if (url.pathname === '/api/input/text' && req.method === 'POST') {
    enforceHttpLimit(req, 'text');
    const text = parseTextInput(await parseJsonBody(req));
    sendJSON(await perceive('text', () => brain.read(text, { propagate: false })));
    return;
  }

  // POST /api/input/image — See image
  if (url.pathname === '/api/input/image' && req.method === 'POST') {
    enforceHttpLimit(req, 'image');
    const { pixels, width, height } = parseImageInput(await parseJsonBody(req));
    sendJSON(await perceive('visual', () => brain.see(pixels, width, height, { propagate: false })));
    return;
  }

  // POST /api/input/audio — Hear audio (spectrogram)
  if (url.pathname === '/api/input/audio' && req.method === 'POST') {
    enforceHttpLimit(req, 'audio');
    // One frame of linear FFT magnitudes (+ the source's sample rate)
    const body = await parseJsonBody(req);
    const frame = parseSpectrogramInput(body);
    const sampleRate = parseSampleRate(body);
    sendJSON(await perceive('auditory', () => brain.hearFrame(frame, sampleRate, { propagate: false })));
    return;
  }

  // POST /api/input/voice — Hear the TONE of a voice (envelope + pitch track of one utterance)
  if (url.pathname === '/api/input/voice' && req.method === 'POST') {
    enforceHttpLimit(req, 'audio');
    const contour = parseVoiceContour(await parseJsonBody(req));
    const appraisal = brain.hearVoice(contour);
    sendJSON({ ok: true, heard: appraisal !== null, voice: brain.getInnate().lastVoice, emotion: brain.feel() });
    return;
  }

  // POST /api/modulator — Inject a neuromodulator
  if (url.pathname === '/api/modulator' && req.method === 'POST') {
    enforceHttpLimit(req, 'modulator');
    const { type, amount } = parseModulatorInput(await parseJsonBody(req));
    brain.getModulators().release(type, amount);
    sendJSON({ ok: true, emotion: brain.feel() });
    return;
  }

  // POST /api/voice — Switch babbling / vocal imitation
  if (url.pathname === '/api/voice' && req.method === 'POST') {
    enforceHttpLimit(req, 'modulator');
    brain.setVoice(parseVoiceInput(await parseJsonBody(req)));
    sendJSON({ ok: true, voice: brain.getState().voice });
    return;
  }

  // POST /api/hand — Switch scribbling / drawing
  if (url.pathname === '/api/hand' && req.method === 'POST') {
    enforceHttpLimit(req, 'modulator');
    brain.setHand(parseHandInput(await parseJsonBody(req)));
    sendJSON({ ok: true, hand: brain.getState().hand });
    return;
  }

  // POST /api/feedback — "yes, that's it" / "no, that's not it"
  if (url.pathname === '/api/feedback' && req.method === 'POST') {
    enforceHttpLimit(req, 'modulator');
    const applied = brain.giveFeedback(parseFeedbackInput(await parseJsonBody(req)));
    sendJSON({ ok: true, applied, emotion: brain.feel() });
    return;
  }

  // POST /api/lesson — Teach: show something with its name / sound, N times
  if (url.pathname === '/api/lesson' && req.method === 'POST') {
    enforceHttpLimit(req, 'text');
    const lesson = parseLessonInput(await parseJsonBody(req));
    startTeaching(() => runLesson(lesson));
    sendJSON({ ok: true, started: 'lesson', repetitions: lesson.repetitions }, 202);
    return;
  }

  // POST /api/practice — Let it babble / scribble N times
  if (url.pathname === '/api/practice' && req.method === 'POST') {
    enforceHttpLimit(req, 'text');
    const rounds = parsePracticeInput(await parseJsonBody(req));
    startTeaching(() => runPractice(rounds));
    sendJSON({ ok: true, started: 'practice', ...rounds }, 202);
    return;
  }

  // POST /api/tick — Run a manual tick (admin)
  if (url.pathname === '/api/tick' && req.method === 'POST') {
    requireAdmin(req);
    brain.tick();
    sendJSON({ ok: true, time: brain.time });
    return;
  }

  // POST /api/sleep — Manual consolidation
  if (url.pathname === '/api/sleep' && req.method === 'POST') {
    if (!sleepBucket.tryTake()) throw new HttpError(429, 'The brain slept a moment ago');
    const stats = brain.sleep();
    sendJSON({
      ok: true,
      memoriesReplayed: stats.memoriesReplayed,
      episodesConsolidated: stats.consolidatedLabels.length,
      synapsesStrengthened: stats.synapsesStrengthened,
    });
    return;
  }

  // POST /api/save — Persist the learning state on demand (admin)
  if (url.pathname === '/api/save' && req.method === 'POST') {
    requireAdmin(req);
    try {
      brain.saveState(STATE_PATH);
      sendJSON({ ok: true, path: STATE_PATH });
    } catch (err) {
      sendJSON({ ok: false, error: (err as Error).message }, 500);
    }
    return;
  }

  sendJSON({ error: 'Unknown endpoint' }, 404);
}

/**
 * Reads the body of a request, rejecting anything over `MAX_BODY_BYTES`.
 */
function parseBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        req.pause();
        reject(new HttpError(413, `Body exceeds ${MAX_BODY_BYTES} bytes`));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    req.on('error', reject);
  });
}

/**
 * The brain state as the dashboard needs it: without the per-neuron spike
 * vectors of every region (10 regions × up to 3,000 floats, 2 × per second),
 * which no panel reads — they stay in `/api/state` for programmatic use.
 */
function dashboardState(): Omit<BrainState, 'regions'> & {
  regions: Record<string, Omit<BrainState['regions'][string], 'outputSpikes' | 'activeNeurons'> & { activeCount: number }>;
} {
  const state = brain.getState();
  const regions: Record<string, Omit<BrainState['regions'][string], 'outputSpikes' | 'activeNeurons'> & { activeCount: number }> = {};
  for (const [id, activity] of Object.entries(state.regions)) {
    const { outputSpikes: _spikes, activeNeurons, ...rest } = activity;
    regions[id] = { ...rest, activeCount: activeNeurons.length };
  }
  return { ...state, regions };
}

/**
 * JSON replacer for Float32Array.
 */
function replacer(_key: string, value: unknown): unknown {
  if (value instanceof Float32Array) {
    return Array.from(value);
  }
  return value;
}

// ================================================================
// WEBSOCKET SERVER
// ================================================================

const wss = new WebSocketServer({ noServer: true, maxPayload: MAX_WS_PAYLOAD_BYTES });
const clients: Set<WebSocket> = new Set();
let nextClientId = 1;

// Handle the HTTP → WebSocket upgrade on the same port
server.on('upgrade', (request, socket, head) => {
  // Browsers always send Origin on WebSocket handshakes and do not apply CORS
  // to them, so this check is what stops a third-party page from driving the brain.
  if (!isOriginAllowed(request.headers.origin, requestHosts(request), ALLOWED_ORIGINS)) {
    socket.write('HTTP/1.1 403 Forbidden\r\n\r\n');
    socket.destroy();
    return;
  }
  if (clients.size >= MAX_WS_CLIENTS) {
    socket.write('HTTP/1.1 503 Service Unavailable\r\n\r\n');
    socket.destroy();
    return;
  }
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});

wss.on('connection', (ws: WebSocket) => {
  clients.add(ws);
  const clientId = nextClientId++;
  const limiter = new ClientLimiter();
  let lastNoticeAt = 0;
  console.log(`🔌 WebSocket client connected (total: ${clients.size})`);

  /** Tells the client (at most once per second) why an input was not processed. */
  const notify = (message: string): void => {
    const now = Date.now();
    if (now - lastNoticeAt < 1000 || ws.readyState !== 1) return;
    lastNoticeAt = now;
    ws.send(JSON.stringify({ type: 'notice', data: { message } }));
  };

  /** Rate-limits, then queues a perception; failures are reported, never thrown. */
  const submit = (
    kind: LimitedKind,
    inputType: PerceptionResult['inputType'],
    inject: () => void,
    coalesce: boolean,
  ): void => {
    if (!limiter.allow(kind)) {
      // Streaming inputs (webcam, mic) are expected to overshoot: drop quietly.
      if (!coalesce) notify('Too many inputs — slow down');
      return;
    }
    perceive(inputType, inject, coalesce ? `${kind}:${clientId}` : undefined).catch((err: Error) => {
      if (err instanceof SchedulerBusyError) notify('The brain is busy — input dropped');
      else console.error('❌ Perception failed:', err);
    });
  };

  // Send initial state
  ws.send(JSON.stringify({ type: 'init', data: dashboardState() }, replacer));

  // Handle messages from the client
  ws.on('message', (message: Buffer) => {
    try {
      const msg = JSON.parse(message.toString()) as { type?: unknown; data?: unknown };

      switch (msg.type) {
        case 'input:text': {
          const text = parseTextInput(msg.data);
          submit('text', 'text', () => brain.read(text, { propagate: false }), false);
          break;
        }
        case 'input:image': {
          const { pixels, width, height } = parseImageInput(msg.data);
          submit('image', 'visual', () => brain.see(pixels, width, height, { propagate: false }), true);
          break;
        }
        case 'input:audio': {
          const frame = parseSpectrogramInput(msg.data);
          const sampleRate = parseSampleRate(msg.data);
          submit('audio', 'auditory', () => brain.hearFrame(frame, sampleRate, { propagate: false }), true);
          break;
        }
        case 'input:voice': {
          const contour = parseVoiceContour(msg.data);
          if (limiter.allow('audio')) brain.hearVoice(contour);
          else notify('Too many voice frames — slow down');
          break;
        }
        case 'modulator': {
          const { type, amount } = parseModulatorInput(msg.data);
          if (limiter.allow('modulator')) brain.getModulators().release(type, amount);
          else notify('Too many injections — slow down');
          break;
        }
        case 'voice': {
          const voice = parseVoiceInput(msg.data);
          if (limiter.allow('modulator')) brain.setVoice(voice);
          else notify('Too many changes — slow down');
          break;
        }
        case 'hand': {
          const hand = parseHandInput(msg.data);
          if (limiter.allow('modulator')) brain.setHand(hand);
          else notify('Too many changes — slow down');
          break;
        }
        case 'feedback': {
          const positive = parseFeedbackInput(msg.data);
          if (limiter.allow('modulator')) brain.giveFeedback(positive);
          else notify('Too much feedback — slow down');
          break;
        }
        case 'lesson': {
          const lesson = parseLessonInput(msg.data);
          if (!limiter.allow('text')) { notify('Too many lessons — slow down'); break; }
          try { startTeaching(() => runLesson(lesson)); } catch (err) { notify((err as Error).message); }
          break;
        }
        case 'practice': {
          const rounds = parsePracticeInput(msg.data);
          if (!limiter.allow('text')) { notify('Too many requests — slow down'); break; }
          try { startTeaching(() => runPractice(rounds)); } catch (err) { notify((err as Error).message); }
          break;
        }
        case 'tick':
          if (limiter.allow('tick')) brain.tick();
          break;
      }
    } catch (err) {
      if (err instanceof HttpError || err instanceof SyntaxError) {
        notify(`Invalid message: ${err.message}`);
      } else {
        console.error('❌ WS message failed:', err);
      }
    }
  });

  ws.on('error', (err) => {
    // e.g. a frame over maxPayload; ws closes the socket right after.
    console.warn(`⚠️  WebSocket error: ${err.message}`);
  });

  ws.on('close', () => {
    clients.delete(ws);
    console.log(`🔌 WebSocket client disconnected (total: ${clients.size})`);
  });
});

// ================================================================
// MAIN LOOP — State broadcast
// ================================================================

// The brain's own voice: every vocalization is pushed to the dashboards, which
// render it with their synthesizer.
// …and so is everything else it DOES: what it draws and what it writes.
const RESPONSE_KINDS = new Set(['vocalization', 'drawing', 'writing']);
brain.on('response', (event) => {
  const kind = event.data.kind;
  if (typeof kind !== 'string' || !RESPONSE_KINDS.has(kind)) return;
  broadcast(kind, event.data);
});
// …and what its innate layer reacts to: a tone of voice, a startle, a face, something looming.
brain.on('affect', (event) => broadcast('affect', event.data));

let tickTimer: ReturnType<typeof setInterval>;
let broadcastTimer: ReturnType<typeof setInterval>;
let thoughtTimer: ReturnType<typeof setInterval>;
let autosaveTimer: ReturnType<typeof setInterval>;
let limiterSweepTimer: ReturnType<typeof setInterval>;

function startBrainLoop(): void {
  // Brain tick — processes neurons (fast, no I/O)
  tickTimer = setInterval(() => {
    for (let i = 0; i < BRAIN_SPEED; i++) brain.tick();
  }, TICK_INTERVAL_MS);

  // Separate broadcast — less frequent to avoid saturation
  broadcastTimer = setInterval(() => {
    if (clients.size > 0) {
      broadcast('state', dashboardState());
    }
  }, BROADCAST_INTERVAL_MS);

  // Live "thought" stream — decodes the brain's current internal activation
  // (Wernicke + Broca + decaying input trace) into a short emotion-framed phrase.
  thoughtTimer = setInterval(() => {
    if (clients.size > 0) {
      broadcast('thought', brain.think());
    }
  }, THOUGHT_INTERVAL_MS);

  // Autosave — learning survives restarts even without a clean shutdown
  autosaveTimer = setInterval(() => persist('autosave'), AUTOSAVE_INTERVAL_MS);

  // Forget rate-limit state of HTTP clients that went away
  limiterSweepTimer = setInterval(() => {
    const cutoff = Date.now() - HTTP_LIMITER_IDLE_MS;
    for (const [key, limiter] of httpLimiters) {
      if (limiter.lastSeen < cutoff) httpLimiters.delete(key);
    }
  }, HTTP_LIMITER_IDLE_MS);
}

// ================================================================
// STARTUP
// ================================================================

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🌐 ═══════════════════════════════════════════`);
  console.log(`   HTTP+WS server:    http://0.0.0.0:${PORT}`);
  console.log(`   Dashboard:         http://localhost:${PORT}/`);
  console.log(`   API State:         http://localhost:${PORT}/api/state`);
  console.log(`═══════════════════════════════════════════════\n`);
  
  // Start the brain loop
  startBrainLoop();
});

// Graceful shutdown — saves the learning before exiting.
// Railway sends SIGTERM on every redeploy; capturing it is key to not losing it.
let shuttingDown = false;
function shutdown(signal: string): void {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`\n🛑 ${signal} — saving state and stopping brain...`);
  clearInterval(tickTimer);
  clearInterval(broadcastTimer);
  clearInterval(thoughtTimer);
  clearInterval(autosaveTimer);
  clearInterval(limiterSweepTimer);
  persist(signal);
  wss.close();
  server.close();
  process.exit(0);
}
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

export { brain, server, wss };
