/**
 * PROBE: does Wernicke understand the CLEAN text (without the brain's dilution)?
 * If findClosest over the direct encoding returns the real words,
 * the problem is purely signal flow (not comprehension) and Option 2 =
 * carry that clean signal to Broca.
 */

import { WernickeArea } from '../src/regions/broca-wernicke/wernicke.js';
import { BrocaArea } from '../src/regions/broca-wernicke/broca.js';
import { Lexicon } from '../src/regions/broca-wernicke/lexicon.js';
import { seedSpanishLexicon, encodeSentenceToLexiconSpace } from '../src/regions/broca-wernicke/spanish-lexicon.js';

const DIM = 1000; // same size used by the real brain
const lexicon = new Lexicon(DIM);
seedSpanishLexicon(lexicon);
console.log(`Léxico: ${lexicon.size} palabras, dim=${lexicon.dimensions}`);

const wernicke = new WernickeArea(lexicon, DIM, DIM);
const broca = new BrocaArea(lexicon, DIM, DIM);
// Encoder in the lexicon space (the one brain.read() now uses).
const enc = { encode: (t: string, _dt: number) => encodeSentenceToLexiconSpace(t, DIM) };

// Compare the TextEncoder encoding with the lexicon pattern for the
// same word (are they in the same representational space?).
function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
for (const w of ['hola', 'miedo', 'feliz', 'cerebro']) {
  const encPat = enc.encode(w, 1);
  const lexPat = lexicon.lookup(w);
  const sim = lexPat ? cosine(encPat, lexPat) : NaN;
  console.log(`  cos(encoder["${w}"], lexico["${w}"]) = ${sim.toFixed(3)}`);
}

const texts = ['hola cerebro', 'tengo miedo', 'sol feliz alegria', 'aprender memoria'];
for (const t of texts) {
  const spikes = enc.encode(t, 1);
  const understood = wernicke.comprehend(spikes).map((c) => `${c.word}:${c.similarity.toFixed(2)}`);
  const said = broca.generateResponse(spikes, { valence: 0, arousal: 0 }).words;
  console.log(`\nIN : "${t}"`);
  console.log(`  Wernicke entiende: [${understood.join(', ')}]`);
  console.log(`  Broca diría:       [${said.join(', ')}]`);
}
