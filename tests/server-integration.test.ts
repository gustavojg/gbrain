/**
 * VERIFICATION TEST — The real server under hostile input
 * ===========================================================================
 * Boots the actual HTTP + WebSocket server on a throwaway port and attacks it
 * the way an anonymous visitor of the public demo could:
 *
 *   1. POISONING  — malformed neuromodulator messages (HTTP and WS) must be
 *                   rejected and must leave every level finite.
 *   2. ORIGIN     — a third-party web page must not be able to drive the brain
 *                   (cross-origin POST and cross-origin WebSocket refused).
 *   3. LIMITS     — oversized bodies → 413, floods → 429, admin endpoints → 401.
 *   4. LIVENESS   — a perception must not block the event loop: while the brain
 *                   propagates an input, the loop keeps turning in short slices.
 */

import { existsSync, rmSync } from 'fs';
import { WebSocket } from 'ws';
import { seedRandom } from './helpers/seed.js';

// Reproducible brain (seeded before the server module builds it).
seedRandom(108);

const PORT = 38000 + (process.pid % 1000);
const STATE_PATH = `/tmp/gbrain-server-test-${process.pid}.bin`;
const ADMIN_TOKEN = 'test-admin-token';
process.env.PORT = String(PORT);
process.env.BRAIN_STATE_PATH = STATE_PATH;
process.env.BRAIN_ADMIN_TOKEN = ADMIN_TOKEN;
delete process.env.ALLOWED_ORIGINS;

const BASE = `http://127.0.0.1:${PORT}`;
const log = console.log;

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean): void => {
  results.push([name, ok]);
  log(`   ${ok ? '✅' : '❌'} ${name}`);
};

function cleanup(): void {
  for (const suffix of ['', '.bak', '.tmp', '.lexicon.json']) {
    if (existsSync(STATE_PATH + suffix)) rmSync(STATE_PATH + suffix);
  }
}

const post = (path: string, body: unknown, headers: Record<string, string> = {}): Promise<Response> =>
  fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  });

async function modulatorsFinite(): Promise<boolean> {
  const state = (await (await fetch(`${BASE}/api/state`)).json()) as { modulators: Record<string, unknown> };
  return Object.values(state.modulators).every((v) => typeof v === 'number' && Number.isFinite(v));
}

function openSocket(origin?: string): Promise<WebSocket | null> {
  return new Promise((resolve) => {
    const ws = new WebSocket(`ws://127.0.0.1:${PORT}`, origin ? { origin } : {});
    ws.once('open', () => resolve(ws));
    ws.once('error', () => resolve(null));
    ws.once('unexpected-response', () => resolve(null));
  });
}

log('── Verification: real server under hostile input ──\n');
cleanup();

// Boot the real server (it starts listening on import); keep its logs quiet.
console.log = () => {};
const { server } = await import('../src/server.js');
if (!server.listening) await new Promise<void>((resolve) => server.once('listening', () => resolve()));

// ── 1. POISONING ────────────────────────────────────────────────────────────
log('1. POISONING');
{
  const bad = await Promise.all([
    post('/api/modulator', { type: 'dopamine' }),
    post('/api/modulator', { type: 'dopamine', amount: 'lots' }),
    post('/api/modulator', { type: 'dopamine', amount: null }),
    post('/api/modulator', { type: '__proto__', amount: 0.2 }),
    post('/api/modulator', '{not json'),
  ]);
  check('HTTP: malformed modulator payloads → 400', bad.every((r) => r.status === 400));

  const ws = await openSocket();
  let notice = '';
  if (ws) {
    ws.on('message', (raw) => {
      const msg = JSON.parse(raw.toString()) as { type: string; data?: { message?: string } };
      if (msg.type === 'notice') notice = msg.data?.message ?? '';
    });
    ws.send(JSON.stringify({ type: 'modulator', data: { type: 'dopamine' } }));
    ws.send(JSON.stringify({ type: 'input:audio', data: { spectrogram: { length: 1e9 } } }));
    ws.send('{broken');
    await new Promise((resolve) => setTimeout(resolve, 300));
    ws.close();
  }
  check('WS: malformed messages answered with a notice, connection survives', ws !== null && notice.startsWith('Invalid message'));
  check('every neuromodulator level is still finite', await modulatorsFinite());

  const ok = await post('/api/modulator', { type: 'dopamine', amount: 0.2 });
  check('valid injection still works', ok.status === 200);
}

// ── 2. ORIGIN ───────────────────────────────────────────────────────────────
log('\n2. ORIGIN');
{
  const evil = await post('/api/input/text', { text: 'hola' }, { Origin: 'https://evil.example' });
  check('cross-origin POST refused (403)', evil.status === 403);

  const own = await post('/api/modulator', { type: 'serotonin', amount: 0.1 }, { Origin: BASE });
  check('same-origin POST accepted', own.status === 200);

  const evilSocket = await openSocket('https://evil.example');
  check('cross-origin WebSocket refused', evilSocket === null);
  evilSocket?.close();

  const read = await fetch(`${BASE}/api/state`, { headers: { Origin: 'https://evil.example' } });
  check('read-only state stays public', read.status === 200);
}

// ── 3. LIMITS ───────────────────────────────────────────────────────────────
log('\n3. LIMITS');
{
  const huge = await post('/api/input/text', { text: 'x'.repeat(300_000) }).then((r) => r.status, () => 0);
  check('oversized body refused (413)', huge === 413);

  const noToken = await post('/api/save', {});
  const wrongToken = await post('/api/save', {}, { Authorization: 'Bearer nope' });
  check('admin endpoints refuse missing / wrong token (401)', noToken.status === 401 && wrongToken.status === 401);

  const withToken = await post('/api/save', {}, { Authorization: `Bearer ${ADMIN_TOKEN}` });
  check('admin endpoint works with the token', withToken.status === 200 && existsSync(STATE_PATH));

  const flood = await Promise.all(
    Array.from({ length: 30 }, () => post('/api/modulator', { type: 'oxytocin', amount: 0.01 })),
  );
  check('flood is rate-limited (429)', flood.some((r) => r.status === 429) && flood.some((r) => r.status === 200));
}

// ── 4. LIVENESS ─────────────────────────────────────────────────────────────
log('\n4. LIVENESS');
{
  // Measure the longest stall of the event loop while a perception propagates.
  let maxStall = 0;
  let last = performance.now();
  const probe = setInterval(() => {
    const now = performance.now();
    maxStall = Math.max(maxStall, now - last);
    last = now;
  }, 5);

  const started = performance.now();
  const response = await post('/api/input/text', { text: 'hola cerebro tengo alegria' });
  const total = performance.now() - started;
  clearInterval(probe);

  const result = (await response.json()) as { inputType?: string; emotion?: unknown; processingTime?: number };
  check('perception completes and returns its result',
    response.status === 200 && result.inputType === 'text' && result.emotion !== undefined && (result.processingTime ?? 0) >= 50);
  log(`      perception took ${total.toFixed(0)} ms; longest event-loop stall ${maxStall.toFixed(0)} ms`);
  // Inline propagation used to stall the loop for the WHOLE perception.
  // Sliced, the longest stall is one short slice plus a regular tick.
  check('event loop never stalls for more than ~1/3 of a perception', maxStall < total / 3);
}

// ── Verdict ─────────────────────────────────────────────────────────────────
server.close();
cleanup();
const failed = results.filter(([, ok]) => !ok);
log('');
if (failed.length === 0) {
  log(`✅ SERVER HARDENING VERIFIED: ${results.length}/${results.length} checks against the live server.`);
  process.exit(0);
} else {
  log(`❌ FAIL: ${failed.map(([name]) => name).join(' | ')}`);
  process.exit(1);
}
