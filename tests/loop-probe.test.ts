/**
 * PROBE (not yet a test with criteria): what does Broca say given different
 * texts? We want to see whether the response is (a) non-empty, (b) reproducible
 * for the same input, (c) different for different inputs. This guides the design
 * of the associative loop (Option 2).
 */

import { DigitalBrain } from '../src/brain.js';

function ask(brain: DigitalBrain, text: string): string[] {
  brain.read(text);
  // read() already runs ~50 propagation ticks.
  const r = brain.speak();
  return r.text ? r.text.split(/\s+/).filter(Boolean) : [];
}

const inputs = [
  'hola cerebro como estas',
  'tengo mucho miedo y tristeza',
  'el sol brilla feliz alegria',
];

console.log('── Sonda del bucle texto → Broca ──\n');

const brain = new DigitalBrain();
// Warm up
for (let i = 0; i < 20; i++) brain.tick();

for (let round = 0; round < 2; round++) {
  console.log(`\n=== Ronda ${round + 1} ===`);
  for (const t of inputs) {
    const words = ask(brain, t);
    console.log(`  IN : "${t}"`);
    console.log(`  OUT: [${words.join(', ')}]`);
  }
}
