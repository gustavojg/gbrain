/**
 * SONDA: ¿comprende Wernicke el texto LIMPIO (sin dilución del cerebro)?
 * Si findClosest sobre la codificación directa devuelve las palabras reales,
 * el problema es solo de flujo de señal (no de comprensión) y la Opción 2 =
 * llevar esa señal limpia a Broca.
 */

import { WernickeArea } from '../src/regions/broca-wernicke/wernicke.js';
import { BrocaArea } from '../src/regions/broca-wernicke/broca.js';
import { Lexicon } from '../src/regions/broca-wernicke/lexicon.js';
import { seedSpanishLexicon, encodeSentenceToLexiconSpace } from '../src/regions/broca-wernicke/spanish-lexicon.js';

const DIM = 1000; // mismo tamaño que usa el cerebro real
const lexicon = new Lexicon(DIM);
seedSpanishLexicon(lexicon);
console.log(`Léxico: ${lexicon.size} palabras, dim=${lexicon.dimensions}`);

const wernicke = new WernickeArea(lexicon, DIM, DIM);
const broca = new BrocaArea(lexicon, DIM, DIM);
// Codificador en el espacio del léxico (el que ahora usa brain.read()).
const enc = { encode: (t: string, _dt: number) => encodeSentenceToLexiconSpace(t, DIM) };

// Comparar la codificación del TextEncoder con el patrón del léxico para la
// misma palabra (¿están en el mismo espacio representacional?).
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
