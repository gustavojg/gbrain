/**
 * SONDA (no es aún un test con criterios): ¿qué dice Broca ante distintos
 * textos? Queremos ver si la respuesta es (a) no vacía, (b) reproducible para
 * el mismo input, (c) distinta para inputs distintos. Esto orienta el diseño
 * del bucle asociativo (Opción 2).
 */

import { DigitalBrain } from '../src/brain.js';

function ask(brain: DigitalBrain, text: string): string[] {
  brain.read(text);
  // read() ya ejecuta ~50 ticks de propagación.
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
// Calentar
for (let i = 0; i < 20; i++) brain.tick();

for (let round = 0; round < 2; round++) {
  console.log(`\n=== Ronda ${round + 1} ===`);
  for (const t of inputs) {
    const words = ask(brain, t);
    console.log(`  IN : "${t}"`);
    console.log(`  OUT: [${words.join(', ')}]`);
  }
}
