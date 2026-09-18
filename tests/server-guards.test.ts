/**
 * VERIFICATION TEST — Server guards, scheduler and durable persistence
 * ===========================================================================
 * The brain is fed by untrusted network clients. We verify, without starting
 * the server, that:
 *
 *   1. VALIDATION   — malformed payloads are rejected; valid ones are bounded.
 *   2. NaN GUARD    — a non-finite neuromodulator amount can never poison the
 *                     brain, and a poisoned snapshot is sanitized on load.
 *   3. RATE LIMITS  — token buckets cap bursts and refill over time.
 *   4. ORIGIN/ADMIN — third-party origins and missing tokens are refused.
 *   5. SCHEDULER    — perceptions run in slices, coalesce and shed load.
 *   6. VOCABULARY   — acquisition from hostile text stays bounded.
 *   7. PERSISTENCE  — a corrupted state file is detected (CRC) and the
 *                     previous snapshot (.bak) is restored instead.
 */

import { existsSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { DigitalBrain } from '../src/brain.js';
import { ModulatorType, NeuromodulatorSystem } from '../src/core/neuromodulators/modulator-system.js';
import { PerceptionScheduler, SchedulerBusyError } from '../src/perception-scheduler.js';
import {
  ClientLimiter,
  HttpError,
  MAX_TEXT_LENGTH,
  TokenBucket,
  isAdminAuthorized,
  isOriginAllowed,
  parseImageInput,
  parseModulatorInput,
  parseSpectrogramInput,
  parseTextInput,
} from '../src/server-guards.js';
import { seedRandom } from './helpers/seed.js';

// Reproducible brain: weights, noise and spike encoding all draw from Math.random.
seedRandom(107);

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}`);
};
const rejects = (fn: () => unknown): boolean => {
  try {
    fn();
    return false;
  } catch (err) {
    return err instanceof HttpError && err.status === 400;
  }
};

console.log('── Verification: server guards, scheduler and persistence ──\n');

// ── 1. VALIDATION ───────────────────────────────────────────────────────────
console.log('1. VALIDATION');
check('modulator: NaN / string / missing amount rejected',
  rejects(() => parseModulatorInput({ type: 'dopamine', amount: NaN })) &&
  rejects(() => parseModulatorInput({ type: 'dopamine', amount: '0.2' })) &&
  rejects(() => parseModulatorInput({ type: 'dopamine' })));
check('modulator: unknown type rejected', rejects(() => parseModulatorInput({ type: 'bogus', amount: 0.1 })));
check('modulator: amount clamped to ±1', parseModulatorInput({ type: 'dopamine', amount: 1e9 }).amount === 1);
check('text: non-string / empty rejected',
  rejects(() => parseTextInput({ text: 42 })) && rejects(() => parseTextInput({ text: '   ' })) && rejects(() => parseTextInput(null)));
check('text: truncated to the limit', parseTextInput({ text: 'a'.repeat(10_000) }).length === MAX_TEXT_LENGTH);
check('image: array-like / wrong size / huge dims rejected',
  rejects(() => parseImageInput({ pixels: { length: 1e9 }, width: 1, height: 1 })) &&
  rejects(() => parseImageInput({ pixels: [1, 2, 3], width: 2, height: 2 })) &&
  rejects(() => parseImageInput({ pixels: [], width: 100_000, height: 100_000 })));
check('image: non-finite pixel rejected', rejects(() => parseImageInput({ pixels: [1, null, 3, 4], width: 2, height: 2 })));
check('image: valid pixels clamped to bytes',
  Array.from(parseImageInput({ pixels: [-5, 300, 12.4, 0], width: 2, height: 2 }).pixels).join() === '0,255,12,0');
check('spectrogram: array-like / empty / oversized rejected',
  rejects(() => parseSpectrogramInput({ spectrogram: { length: 1e9 } })) &&
  rejects(() => parseSpectrogramInput({ spectrogram: [] })) &&
  rejects(() => parseSpectrogramInput({ spectrogram: new Array(5000).fill(0) })));
check('spectrogram: values clamped to [0, 1]',
  Array.from(parseSpectrogramInput({ spectrogram: [-1, 0.5, 7] })).join() === '0,0.5,1');

// ── 2. NaN GUARD ────────────────────────────────────────────────────────────
console.log('\n2. NaN GUARD');
{
  const mods = new NeuromodulatorSystem();
  mods.release(ModulatorType.Dopamine, NaN);
  mods.release(ModulatorType.Dopamine, undefined as unknown as number);
  mods.release(ModulatorType.Dopamine, Infinity);
  check('release() ignores non-finite amounts', mods.getLevel(ModulatorType.Dopamine) === 0.4);

  const snapshot = mods.serialize();
  snapshot.modulators[ModulatorType.Dopamine].level = NaN;
  snapshot.modulators[ModulatorType.Cortisol].level = 42;
  const restored = new NeuromodulatorSystem();
  restored.deserialize(snapshot);
  check('deserialize() sanitizes a poisoned snapshot',
    restored.getLevel(ModulatorType.Dopamine) === 0.4 && restored.getLevel(ModulatorType.Cortisol) === 1);
}

// ── 3. RATE LIMITS ──────────────────────────────────────────────────────────
console.log('\n3. RATE LIMITS');
{
  const bucket = new TokenBucket(3, 1, 0);
  const burst = [bucket.tryTake(0), bucket.tryTake(0), bucket.tryTake(0), bucket.tryTake(0)];
  check('bucket allows the burst then refuses', burst.join() === 'true,true,true,false');
  check('bucket refills over time', bucket.tryTake(1000) === true && bucket.tryTake(1000) === false);

  const limiter = new ClientLimiter();
  check('image budget: 1 frame, then 1 every 2 s',
    limiter.allow('image', 0) && !limiter.allow('image', 500) && limiter.allow('image', 2100));
}

// ── 4. ORIGIN / ADMIN ───────────────────────────────────────────────────────
console.log('\n4. ORIGIN / ADMIN');
check('no Origin (curl) allowed', isOriginAllowed(undefined, ['localhost:3000'], []));
check('same-origin dashboard allowed', isOriginAllowed('http://localhost:3000', ['localhost:3000'], []));
check('proxied public host allowed', isOriginAllowed('https://gbrain.example', ['10.0.0.5:3000', 'gbrain.example'], []));
check('third-party page refused',
  !isOriginAllowed('https://evil.example', ['localhost:3000'], []) && !isOriginAllowed('null', ['localhost:3000'], []));
check('allowlisted origin allowed', isOriginAllowed('https://friend.example', ['localhost:3000'], ['https://friend.example']));
check('admin: no token configured → loopback only',
  isAdminAuthorized(undefined, undefined, true) && !isAdminAuthorized(undefined, undefined, false));
check('admin: token required when configured',
  isAdminAuthorized('Bearer s3cret', 's3cret', false) &&
  !isAdminAuthorized('Bearer wrong!', 's3cret', true) &&
  !isAdminAuthorized(undefined, 's3cret', true));

// ── 5. SCHEDULER ────────────────────────────────────────────────────────────
console.log('\n5. SCHEDULER');
{
  // Manual event loop: every deferred callback is one "yield".
  const deferred: Array<() => void> = [];
  let ticks = 0;
  let maxTicksPerTurn = 0;
  let ticksThisTurn = 0;
  const scheduler = new PerceptionScheduler(() => { ticks++; ticksThisTurn++; }, {
    ticksPerJob: 50,
    sliceTicks: 5,
    maxQueue: 2,
    defer: (fn) => deferred.push(fn),
  });
  const drain = (): void => {
    maxTicksPerTurn = Math.max(maxTicksPerTurn, ticksThisTurn);
    while (deferred.length > 0) {
      ticksThisTurn = 0;
      deferred.shift()!();
      maxTicksPerTurn = Math.max(maxTicksPerTurn, ticksThisTurn);
    }
  };

  const order: string[] = [];
  const job = (name: string, coalesceKey?: string) => ({
    coalesceKey,
    inject: () => { order.push(name); },
    finish: () => name,
  });

  const first = scheduler.submit(job('text-1'));
  const frameA = scheduler.submit(job('frame-A', 'image:1'));
  const frameB = scheduler.submit(job('frame-B', 'image:1')); // replaces frame-A
  const other = scheduler.submit(job('text-2'));
  let busy = false;
  const overflow = scheduler.submit(job('text-3')).catch((err) => { busy = err instanceof SchedulerBusyError; });

  drain();
  const settled = await Promise.all([first, frameA, frameB, other, overflow]);

  check('runs at most one slice per event-loop turn', maxTicksPerTurn === 5);
  check('runs the full propagation for each job', ticks === 150);
  check('newer streaming frame replaces the queued one', settled[1] === null && settled[2] === 'frame-B');
  check('jobs run one at a time, in order', order.join() === 'text-1,frame-B,text-2');
  check('sheds load when the queue is full', busy);
}

// Silence the brain's own logging for the remaining sections.
const log = console.log;
const warn = console.warn;
const quiet = <T>(fn: () => T): T => {
  console.log = () => {};
  console.warn = () => {};
  try {
    return fn();
  } finally {
    console.log = log;
    console.warn = warn;
  }
};

// ── 6. VOCABULARY ───────────────────────────────────────────────────────────
console.log('\n6. VOCABULARY');
{
  const brain = quiet(() => new DigitalBrain());
  const before = brain.getVocabularyStats().total;
  quiet(() => {
    // 2000 distinct alphabetic junk words, each seen twice (below the threshold).
    const alpha = (n: number): string =>
      n.toString(26).replace(/./g, (c) => String.fromCharCode(97 + parseInt(c, 26)));
    for (let rep = 0; rep < 2; rep++) {
      for (let chunk = 0; chunk < 20; chunk++) {
        const words = Array.from({ length: 100 }, (_, i) => `junk${alpha(1000 + chunk * 100 + i)}`);
        brain.read(words.join(' '), { propagate: false });
      }
    }
    // Tokens that must never be learned, however often they are read.
    for (let i = 0; i < 5; i++) {
      brain.read(`x1y2z3 ${'q'.repeat(60)} <img_src=x>`, { propagate: false });
    }
  });
  const stats = brain.getVocabularyStats();
  const state = brain.getState().vocabulary!;
  check('pending vocabulary is capped (LRU)', stats.pendingCount <= 500);
  check('non-alphabetic / oversized tokens are never learned',
    stats.total === before && !brain.knowsWord('x1y2z3') && !brain.knowsWord('q'.repeat(60)));
  check('streamed state is trimmed', state.pending.length <= 6 && state.learnedThisSession.length <= 24);
}

// ── 7. PERSISTENCE ──────────────────────────────────────────────────────────
console.log('\n7. PERSISTENCE');
{
  const statePath = `/tmp/gbrain-guards-test-${process.pid}.bin`;
  const files = [statePath, `${statePath}.bak`, `${statePath}.tmp`, `${statePath}.lexicon.json`];
  const cleanup = (): void => { for (const f of files) if (existsSync(f)) rmSync(f); };
  cleanup();

  const brain = quiet(() => new DigitalBrain());
  quiet(() => brain.saveState(statePath)); // snapshot 1
  quiet(() => brain.saveState(statePath)); // snapshot 2 → snapshot 1 becomes .bak
  check('atomic save leaves no temp file and keeps a backup',
    existsSync(statePath) && existsSync(`${statePath}.bak`) && !existsSync(`${statePath}.tmp`));

  // Flip one byte in the middle of the weights: only the CRC can notice.
  const bytes = readFileSync(statePath);
  bytes[Math.floor(bytes.length / 2)] ^= 0xff;
  writeFileSync(statePath, bytes);

  const fresh = quiet(() => new DigitalBrain());
  let restored: { loaded: string[]; skipped: string[] } | null = null;
  try {
    restored = quiet(() => fresh.loadState(statePath));
  } catch {
    restored = null;
  }
  check('corrupted file detected; previous snapshot restored', restored !== null && restored.loaded.length === 12);

  rmSync(`${statePath}.bak`);
  let threw = false;
  try { quiet(() => fresh.loadState(statePath)); } catch { threw = true; }
  check('corrupted file with no backup is refused (not loaded as garbage)', threw);

  // Truncated write (what a crash mid-save used to leave behind).
  writeFileSync(statePath, bytes.subarray(0, 1000));
  threw = false;
  try { quiet(() => fresh.loadState(statePath)); } catch { threw = true; }
  check('truncated file is refused', threw);
  cleanup();
}

// ── Verdict ─────────────────────────────────────────────────────────────────
const failed = results.filter(([, ok]) => !ok);
console.log('');
if (failed.length === 0) {
  console.log(`✅ SERVER GUARDS VERIFIED: ${results.length}/${results.length} checks — untrusted input cannot poison, exhaust or stall the brain.`);
  process.exit(0);
} else {
  console.log(`❌ FAIL: ${failed.map(([name]) => name).join(' | ')}`);
  process.exit(1);
}
