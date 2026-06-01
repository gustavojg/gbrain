/**
 * HTTP + WebSocket SERVER — Digital brain interface
 * =========================================================
 * Server that exposes the digital brain via HTTP API and WebSocket
 * for real-time communication with the 3D dashboard.
 *
 * Endpoints:
 * - POST /api/input/text    → The brain reads text
 * - POST /api/input/image   → The brain sees an image
 * - POST /api/input/audio   → The brain hears audio
 * - GET  /api/state         → Complete brain state
 * - GET  /api/feel          → Emotional state
 * - GET  /api/speak         → The brain speaks
 * - GET  /api/imagine       → The brain imagines
 * - POST /api/modulator     → Inject a neuromodulator manually
 * - WS   /ws                → Real-time stream
 *
 * The WebSocket sends state updates on every brain tick.
 */

import http from 'http';
import { readFileSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer, type WebSocket } from 'ws';
import { DigitalBrain, type BrainState } from './brain.js';
import { ModulatorType } from './core/neuromodulators/modulator-system.js';

// ================================================================
// CONFIGURATION
// ================================================================

const PORT = parseInt(process.env.PORT || '3000', 10);
// import.meta.dirname only exists in Node ≥20.11; deriving it from import.meta.url
// also makes it robust on Node 16/18 (otherwise DASHBOARD_DIR falls back to cwd and 404s).
const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url));
const DASHBOARD_DIR = path.resolve(SERVER_DIR, 'dashboard');
const TICK_INTERVAL_MS = 100; // 10 Hz brain tick
const BROADCAST_INTERVAL_MS = 500; // 2 Hz dashboard update (lighter)
const THOUGHT_INTERVAL_MS = 1200; // ~0.8 Hz live "thought" stream
const IMAGE_THROTTLE_MS = 2000; // Max 1 frame every 2 seconds
const AUTOSAVE_INTERVAL_MS = 5 * 60_000; // Save the learning every 5 min

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

console.log(`\n🌐 Starting Digital Brain server...\n`);

// Create the brain
const brain = new DigitalBrain();

// Restore previous learning if it exists
if (existsSync(STATE_PATH)) {
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
  
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  try {
    // --- API ROUTES ---
    if (url.pathname.startsWith('/api/')) {
      await handleApiRoute(url, req, res);
      return;
    }

    // --- STATIC FILES (Dashboard) ---
    let filePath = url.pathname === '/' ? '/index.html' : url.pathname;
    const fullPath = path.join(DASHBOARD_DIR, filePath);
    
    if (existsSync(fullPath)) {
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
    console.error('❌ Error:', err);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Internal server error' }));
  }
});

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
    const body = await parseBody(req);
    const { text } = JSON.parse(body) as { text: string };
    const result = brain.read(text);
    sendJSON(result);
    return;
  }

  // POST /api/input/image — See image
  if (url.pathname === '/api/input/image' && req.method === 'POST') {
    const body = await parseBody(req);
    const { pixels, width, height } = JSON.parse(body) as { pixels: number[]; width: number; height: number };
    const result = brain.see(pixels, width, height);
    sendJSON(result);
    return;
  }

  // POST /api/input/audio — Hear audio (spectrogram)
  if (url.pathname === '/api/input/audio' && req.method === 'POST') {
    const body = await parseBody(req);
    const { spectrogram } = JSON.parse(body) as { spectrogram: number[] };
    const result = brain.hearSpectrogram(spectrogram);
    sendJSON(result);
    return;
  }

  // POST /api/modulator — Inject a neuromodulator
  if (url.pathname === '/api/modulator' && req.method === 'POST') {
    const body = await parseBody(req);
    const { type, amount } = JSON.parse(body) as { type: string; amount: number };
    const modulatorType = type as ModulatorType;
    brain.getModulators().release(modulatorType, amount);
    sendJSON({ ok: true, emotion: brain.feel() });
    return;
  }

  // POST /api/tick — Run a manual tick
  if (url.pathname === '/api/tick' && req.method === 'POST') {
    brain.tick();
    sendJSON({ ok: true, time: brain.time });
    return;
  }

  // POST /api/sleep — Manual consolidation
  if (url.pathname === '/api/sleep' && req.method === 'POST') {
    brain.sleep();
    sendJSON({ ok: true, memoriesReplayed: 0 });
    return;
  }

  // POST /api/save — Persist the learning state on demand
  if (url.pathname === '/api/save' && req.method === 'POST') {
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
 * Parses the body of a request.
 */
function parseBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
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

const wss = new WebSocketServer({ noServer: true });

// Handle the HTTP → WebSocket upgrade on the same port
server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});
const clients: Set<WebSocket> = new Set();

wss.on('connection', (ws: WebSocket) => {
  clients.add(ws);
  console.log(`🔌 WebSocket client connected (total: ${clients.size})`);

  // Send initial state
  ws.send(JSON.stringify({
    type: 'init',
    data: brain.getState(),
  }, replacer));

  // Handle messages from the client
  ws.on('message', (message: Buffer) => {
    try {
      const msg = JSON.parse(message.toString()) as { type: string; data?: Record<string, unknown> };

      switch (msg.type) {
        case 'input:text':
          brain.read((msg.data as { text: string }).text);
          break;
        case 'input:image':
          const now = Date.now();
          if (now - lastImageTime >= IMAGE_THROTTLE_MS) {
            lastImageTime = now;
            const imgData = msg.data as { pixels: number[]; width: number; height: number };
            brain.see(imgData.pixels, imgData.width, imgData.height);
          }
          break;
        case 'input:audio':
          const audioData = msg.data as { spectrogram: number[] };
          brain.hearSpectrogram(audioData.spectrogram);
          break;
        case 'modulator':
          const modData = msg.data as { type: string; amount: number };
          brain.getModulators().release(modData.type as ModulatorType, modData.amount);
          break;
        case 'tick':
          brain.tick();
          break;
      }
    } catch (err) {
      console.error('❌ Invalid WS message:', err);
    }
  });

  ws.on('close', () => {
    clients.delete(ws);
    console.log(`🔌 WebSocket client disconnected (total: ${clients.size})`);
  });
});

// ================================================================
// MAIN LOOP — State broadcast
// ================================================================

let tickTimer: ReturnType<typeof setInterval>;
let broadcastTimer: ReturnType<typeof setInterval>;
let thoughtTimer: ReturnType<typeof setInterval>;
let autosaveTimer: ReturnType<typeof setInterval>;
let lastImageTime = 0;

function startBrainLoop(): void {
  // Brain tick — processes neurons (fast, no I/O)
  tickTimer = setInterval(() => {
    brain.tick();
  }, TICK_INTERVAL_MS);

  // Separate broadcast — less frequent to avoid saturation
  broadcastTimer = setInterval(() => {
    if (clients.size > 0) {
      const state = brain.getState();
      const msg = JSON.stringify({ type: 'state', data: state }, replacer);
      
      for (const client of clients) {
        if (client.readyState === 1) {
          client.send(msg);
        }
      }
    }
  }, BROADCAST_INTERVAL_MS);

  // Live "thought" stream — decodes the brain's current internal activation
  // (Wernicke + Broca + decaying input trace) into a short emotion-framed phrase.
  thoughtTimer = setInterval(() => {
    if (clients.size > 0) {
      const thought = brain.think();
      const msg = JSON.stringify({ type: 'thought', data: thought }, replacer);
      for (const client of clients) {
        if (client.readyState === 1) {
          client.send(msg);
        }
      }
    }
  }, THOUGHT_INTERVAL_MS);

  // Autosave — learning survives restarts even without a clean shutdown
  autosaveTimer = setInterval(() => persist('autosave'), AUTOSAVE_INTERVAL_MS);
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
  persist(signal);
  wss.close();
  server.close();
  process.exit(0);
}
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

export { brain, server, wss };
