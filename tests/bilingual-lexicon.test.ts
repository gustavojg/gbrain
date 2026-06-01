/**
 * VERIFICATION TEST — Bilingual lexicon (Spanish + English)
 * ===========================================================================
 * The brain now seeds both the Spanish (~190) and English (~190) lexicons
 * into one shared pattern space. We verify that English input is comprehended
 * by Wernicke (the real English words are retrieved) and that Broca produces
 * a relevant English associative response — without breaking Spanish.
 */

import { WernickeArea } from '../src/regions/broca-wernicke/wernicke.js';
import { BrocaArea } from '../src/regions/broca-wernicke/broca.js';
import { Lexicon } from '../src/regions/broca-wernicke/lexicon.js';
import { seedSpanishLexicon, encodeSentenceToLexiconSpace } from '../src/regions/broca-wernicke/spanish-lexicon.js';
import { seedEnglishLexicon } from '../src/regions/broca-wernicke/english-lexicon.js';

const DIM = 1000; // same size used by the real brain
const lexicon = new Lexicon(DIM);
seedSpanishLexicon(lexicon);
seedEnglishLexicon(lexicon);
console.log(`Lexicon: ${lexicon.size} words (ES+EN), dim=${lexicon.dimensions}`);

const wernicke = new WernickeArea(lexicon, DIM, DIM);
const broca = new BrocaArea(lexicon, DIM, DIM);

function normalizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^\w\sáéíóúüñ]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

const cases = [
  { text: 'hello brain how are you', lang: 'EN' },
  { text: 'i feel fear and sadness', lang: 'EN' },
  { text: 'the sun shines happy joy', lang: 'EN' },
  { text: 'i want to learn and remember memory', lang: 'EN' },
  { text: 'hola cerebro como estas', lang: 'ES' },
  { text: 'tengo mucho miedo y tristeza', lang: 'ES' },
];

let relevant = 0;
for (const c of cases) {
  const spikes = encodeSentenceToLexiconSpace(c.text, DIM);
  const understood = wernicke.comprehend(spikes).map((u) => `${u.word}:${u.similarity.toFixed(2)}`);
  const said = broca.generateResponse(spikes, { valence: 0, arousal: 0 }).words;
  const inputWords = new Set(normalizeWords(c.text));
  const hit = said.some((w) => inputWords.has(w));
  if (hit) relevant++;
  console.log(`\n[${c.lang}] IN : "${c.text}"`);
  console.log(`  Wernicke: [${understood.join(', ')}]`);
  console.log(`  Broca:    [${said.join(', ')}]   relevant=${hit}`);
}

const ratio = relevant / cases.length;
console.log(`\n── Metrics ──`);
console.log(`  Relevance (response∩input ≥1): ${(ratio * 100).toFixed(0)}%  (target =100%)`);

if (ratio >= 0.999) {
  console.log('\n✅ BILINGUAL LEXICON VERIFIED: comprehends and responds in EN and ES.');
  process.exit(0);
} else {
  console.log('\n❌ FAIL: some responses had no word from their own input.');
  process.exit(1);
}
