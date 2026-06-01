/**
 * VERIFICATION TEST — Real vocabulary learning
 * ===========================================================================
 * The brain learns NEW words from text it reads. We verify three properties:
 *
 *   1. THRESHOLD   — a novel word is NOT learned on the first exposures; it is
 *                    committed to the lexicon only after LEARN_THRESHOLD reps.
 *   2. COMPREHENSION — once learned, the brain recognizes the word (knowsWord)
 *                    and Broca can echo it back in a response.
 *   3. PERSISTENCE — the learned word survives a save → fresh brain → load
 *                    cycle (lexicon sidecar), so learning is durable.
 */

import { DigitalBrain } from '../src/brain.js';
import { existsSync, rmSync } from 'fs';

// Invented words guaranteed NOT to be in the seeded ES/EN lexicon.
const NOVEL = 'zarpalio';
const NOVEL_SENTENCE = `el ${NOVEL} corre feliz`; // novel word appears once per read

const STATE_PATH = `/tmp/gbrain-vocab-test-${process.pid}.bin`;
const SIDECAR = `${STATE_PATH}.lexicon.json`;

function cleanup(): void {
  for (const p of [STATE_PATH, SIDECAR]) if (existsSync(p)) rmSync(p);
}

console.log('── Verification: real vocabulary learning ──\n');
cleanup();

const brain = new DigitalBrain();
for (let i = 0; i < 10; i++) brain.tick(); // warm up

const threshold = brain.getVocabularyStats().threshold;
console.log(`LEARN_THRESHOLD = ${threshold}`);
console.log(`Initial vocabulary: ${brain.getVocabularyStats().total} words`);
console.log(`knowsWord("${NOVEL}") before: ${brain.knowsWord(NOVEL)}\n`);

// 1. THRESHOLD — read the sentence (threshold - 1) times; must NOT be learned.
for (let i = 1; i < threshold; i++) {
  brain.read(NOVEL_SENTENCE);
  console.log(`  after exposure #${i}: knows="${brain.knowsWord(NOVEL)}"`);
}
const learnedEarly = brain.knowsWord(NOVEL);

// The exposure that crosses the threshold should commit the word.
brain.read(NOVEL_SENTENCE);
const learnedAtThreshold = brain.knowsWord(NOVEL);
console.log(`  after exposure #${threshold}: knows="${learnedAtThreshold}"`);

// 2. COMPREHENSION — Broca can now produce the learned word.
brain.read(NOVEL_SENTENCE);
const said = brain.speak().text ?? '';
const echoed = said.toLowerCase().includes(NOVEL);
console.log(`\nBroca says: "${said}"`);
console.log(`Echoes learned word: ${echoed}`);

const stats = brain.getVocabularyStats();
console.log(`learnedThisSession: [${stats.learnedThisSession.join(', ')}]`);

// 3. PERSISTENCE — save, spin up a fresh brain, load, and check it still knows.
brain.saveState(STATE_PATH);
const sidecarWritten = existsSync(SIDECAR);

const brain2 = new DigitalBrain();
const knownBeforeLoad = brain2.knowsWord(NOVEL); // fresh brain → should be false
brain2.loadState(STATE_PATH);
const knownAfterLoad = brain2.knowsWord(NOVEL);
console.log(`\nFresh brain knows "${NOVEL}" before load: ${knownBeforeLoad}`);
console.log(`Fresh brain knows "${NOVEL}" after load:  ${knownAfterLoad}`);

cleanup();

// ── Criteria ──
const okThreshold = learnedEarly === false && learnedAtThreshold === true;
const okComprehension = echoed && stats.learnedThisSession.includes(NOVEL);
const okPersistence = sidecarWritten && knownBeforeLoad === false && knownAfterLoad === true;

console.log('\n── Metrics ──');
console.log(`  Threshold gating (not before, yes at #${threshold}): ${okThreshold}`);
console.log(`  Comprehension (Broca echoes learned word):    ${okComprehension}`);
console.log(`  Persistence (survives save/load):             ${okPersistence}`);

console.log('');
if (okThreshold && okComprehension && okPersistence) {
  console.log('✅ VOCABULARY LEARNING VERIFIED: learns new words, comprehends them, and persists them.');
  process.exit(0);
} else {
  console.log(`❌ FAIL: threshold=${okThreshold} comprehension=${okComprehension} persistence=${okPersistence}`);
  process.exit(1);
}
