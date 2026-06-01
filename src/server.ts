/**
 * SERVIDOR HTTP + WebSocket — Interfaz del cerebro digital
 * =========================================================
 * Servidor que expone el cerebro digital via HTTP API y WebSocket
 * para comunicación en tiempo real con el dashboard 3D.
 * 
 * Endpoints:
 * - POST /api/input/text    → El cerebro lee texto
 * - POST /api/input/image   → El cerebro ve una imagen
 * - POST /api/input/audio   → El cerebro escucha audio
 * - GET  /api/state         → Estado completo del cerebro
 * - GET  /api/feel          → Estado emocional
 * - GET  /api/speak         → El cerebro habla
 * - GET  /api/imagine       → El cerebro imagina
 * - POST /api/modulator     → Inyectar neuromodulador manualmente
 * - WS   /ws                → Stream en tiempo real
 * 
 * El WebSocket envía updates del estado cada tick del cerebro.
 */

import http from 'http';
import { readFileSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer, type WebSocket } from 'ws';
import { DigitalBrain, type BrainState } from './brain.js';
import { ModulatorType } from './core/neuromodulators/modulator-system.js';

// ================================================================
// CONFIGURACIÓN
// ================================================================

const PORT = parseInt(process.env.PORT || '3000', 10);
// import.meta.dirname solo existe en Node ≥20.11; derivarlo de import.meta.url
// lo hace robusto también en Node 16/18 (si no, DASHBOARD_DIR cae a cwd y 404ea).
const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url));
const DASHBOARD_DIR = path.resolve(SERVER_DIR, 'dashboard');
const TICK_INTERVAL_MS = 100; // 10 Hz brain tick
const BROADCAST_INTERVAL_MS = 500; // 2 Hz dashboard update (lighter)
const IMAGE_THROTTLE_MS = 2000; // Max 1 frame cada 2 segundos
const AUTOSAVE_INTERVAL_MS = 5 * 60_000; // Guardar el aprendizaje cada 5 min

// Ruta del estado persistido. En Railway el FS es efímero salvo que haya un
// volumen montado (RAILWAY_VOLUME_MOUNT_PATH); usarlo si existe para que el
// aprendizaje sobreviva a redeploys. Override explícito vía BRAIN_STATE_PATH.
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
// INICIALIZACIÓN
// ================================================================

console.log(`\n🌐 Iniciando servidor del Cerebro Digital...\n`);

// Crear el cerebro
const brain = new DigitalBrain();

// Restaurar aprendizaje previo si existe
if (existsSync(STATE_PATH)) {
  try {
    const { loaded, skipped } = brain.loadState(STATE_PATH);
    console.log(
      `💾 Estado restaurado desde ${STATE_PATH} — regiones: ${loaded.join(', ') || 'ninguna'}` +
        (skipped.length ? ` | saltadas (dim. incompatibles): ${skipped.join(', ')}` : ''),
    );
  } catch (err) {
    console.error(`⚠️  No se pudo restaurar el estado (${(err as Error).message}); arrancando en limpio.`);
  }
} else {
  console.log(`💾 Sin estado previo en ${STATE_PATH}; arrancando en limpio.`);
}

// Aviso si el almacenamiento es efímero en producción
if (process.env.RAILWAY_ENVIRONMENT && !process.env.RAILWAY_VOLUME_MOUNT_PATH) {
  console.warn(
    '⚠️  En Railway sin volumen montado: el estado se perderá en el próximo redeploy. ' +
      'Monta un volumen (Railway → Service → Volumes) para que el aprendizaje persista.',
  );
}

/** Guarda el estado de forma reentrante-segura. */
let saving = false;
function persist(reason: string): void {
  if (saving) return;
  saving = true;
  try {
    brain.saveState(STATE_PATH);
    console.log(`💾 Estado guardado (${reason}) → ${STATE_PATH}`);
  } catch (err) {
    console.error(`⚠️  Error guardando estado: ${(err as Error).message}`);
  } finally {
    saving = false;
  }
}

// ================================================================
// SERVIDOR HTTP
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
 * Maneja rutas de la API.
 */
async function handleApiRoute(url: URL, req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  const sendJSON = (data: unknown, status: number = 200) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data, replacer));
  };

  // GET /api/state — Estado completo del cerebro
  if (url.pathname === '/api/state' && req.method === 'GET') {
    sendJSON(brain.getState());
    return;
  }

  // GET /api/feel — Estado emocional
  if (url.pathname === '/api/feel' && req.method === 'GET') {
    sendJSON(brain.feel());
    return;
  }

  // GET /api/speak — El cerebro habla
  if (url.pathname === '/api/speak' && req.method === 'GET') {
    sendJSON(brain.speak());
    return;
  }

  // GET /api/imagine — El cerebro imagina
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

  // POST /api/input/text — Leer texto
  if (url.pathname === '/api/input/text' && req.method === 'POST') {
    const body = await parseBody(req);
    const { text } = JSON.parse(body) as { text: string };
    const result = brain.read(text);
    sendJSON(result);
    return;
  }

  // POST /api/input/image — Ver imagen
  if (url.pathname === '/api/input/image' && req.method === 'POST') {
    const body = await parseBody(req);
    const { pixels, width, height } = JSON.parse(body) as { pixels: number[]; width: number; height: number };
    const result = brain.see(pixels, width, height);
    sendJSON(result);
    return;
  }

  // POST /api/input/audio — Escuchar audio (espectrograma)
  if (url.pathname === '/api/input/audio' && req.method === 'POST') {
    const body = await parseBody(req);
    const { spectrogram } = JSON.parse(body) as { spectrogram: number[] };
    const result = brain.hearSpectrogram(spectrogram);
    sendJSON(result);
    return;
  }

  // POST /api/modulator — Inyectar neuromodulador
  if (url.pathname === '/api/modulator' && req.method === 'POST') {
    const body = await parseBody(req);
    const { type, amount } = JSON.parse(body) as { type: string; amount: number };
    const modulatorType = type as ModulatorType;
    brain.getModulators().release(modulatorType, amount);
    sendJSON({ ok: true, emotion: brain.feel() });
    return;
  }

  // POST /api/tick — Ejecutar un tick manual
  if (url.pathname === '/api/tick' && req.method === 'POST') {
    brain.tick();
    sendJSON({ ok: true, time: brain.time });
    return;
  }

  // POST /api/sleep — Consolidación manual
  if (url.pathname === '/api/sleep' && req.method === 'POST') {
    brain.sleep();
    sendJSON({ ok: true, memoriesReplayed: 0 });
    return;
  }

  // POST /api/save — Persistir el estado de aprendizaje a demanda
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
 * Parsea el body de una request.
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
 * JSON replacer para Float32Array.
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

// Manejar upgrade de HTTP → WebSocket en el mismo puerto
server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});
const clients: Set<WebSocket> = new Set();

wss.on('connection', (ws: WebSocket) => {
  clients.add(ws);
  console.log(`🔌 WebSocket cliente conectado (total: ${clients.size})`);

  // Enviar estado inicial
  ws.send(JSON.stringify({
    type: 'init',
    data: brain.getState(),
  }, replacer));

  // Manejar mensajes del cliente
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
      console.error('❌ WS mensaje inválido:', err);
    }
  });

  ws.on('close', () => {
    clients.delete(ws);
    console.log(`🔌 WebSocket cliente desconectado (total: ${clients.size})`);
  });
});

// ================================================================
// LOOP PRINCIPAL — Broadcast del estado
// ================================================================

let tickTimer: ReturnType<typeof setInterval>;
let broadcastTimer: ReturnType<typeof setInterval>;
let autosaveTimer: ReturnType<typeof setInterval>;
let lastImageTime = 0;

function startBrainLoop(): void {
  // Brain tick — procesa neuronas (rápido, sin I/O)
  tickTimer = setInterval(() => {
    brain.tick();
  }, TICK_INTERVAL_MS);

  // Broadcast separado — menos frecuente para no saturar
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

  // Autosave — el aprendizaje sobrevive a reinicios incluso sin apagado limpio
  autosaveTimer = setInterval(() => persist('autosave'), AUTOSAVE_INTERVAL_MS);
}

// ================================================================
// ARRANQUE
// ================================================================

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🌐 ═══════════════════════════════════════════`);
  console.log(`   Servidor HTTP+WS:  http://0.0.0.0:${PORT}`);
  console.log(`   Dashboard:         http://localhost:${PORT}/`);
  console.log(`   API State:         http://localhost:${PORT}/api/state`);
  console.log(`═══════════════════════════════════════════════\n`);
  
  // Iniciar loop del cerebro
  startBrainLoop();
});

// Graceful shutdown — guarda el aprendizaje antes de salir.
// Railway envía SIGTERM en cada redeploy; capturarlo es clave para no perderlo.
let shuttingDown = false;
function shutdown(signal: string): void {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`\n🛑 ${signal} — guardando estado y deteniendo cerebro...`);
  clearInterval(tickTimer);
  clearInterval(broadcastTimer);
  clearInterval(autosaveTimer);
  persist(signal);
  wss.close();
  server.close();
  process.exit(0);
}
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

export { brain, server, wss };
