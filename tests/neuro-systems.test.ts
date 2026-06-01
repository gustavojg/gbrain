/**
 * VERIFICATION TEST — Affect, attention & neuromodulation
 * ===========================================================================
 * The vision / memory / language capabilities already ship quantitative tests.
 * This one closes the gap for the three remaining subsystems, verifying that
 * each behaves the way its neuroscience says it should:
 *
 *   A. AMYGDALA / EMOTION  — reading POSITIVE text yields a clearly higher
 *      valence than reading NEGATIVE text (affect tracks input sign).
 *   B. THALAMUS / ATTENTION — acetylcholine WIDENS the attentional bottleneck,
 *      so more salient signals pass the relay (ACh ↑ ⇒ effective K ↑).
 *   C. NEUROMODULATION LAW — dopamine RAISES the plasticity (learning-rate)
 *      multiplier and cortisol SUPPRESSES it — the documented STDP gating.
 *
 * These are deterministic checks of the modulation pipeline (no SNN randomness
 * in the criteria), so they are stable in CI.
 */

import { DigitalBrain } from '../src/brain.js';
import { ModulatorType } from '../src/core/neuromodulators/modulator-system.js';
import { Thalamus } from '../src/regions/thalamus/thalamus.js';

console.log('── Verification: affect, attention & neuromodulation ──\n');

// ===========================================================================
// A. AMYGDALA — affect tracks the sign of the input
// ===========================================================================
// Words are the exact (unaccented) keys of the brain's emotional lexicon.
const POSITIVE = 'feliz alegria amor genial entusiasmo';
const NEGATIVE = 'miedo tristeza ansiedad odio estres';

const brainPos = new DigitalBrain();
brainPos.read(POSITIVE);
const ePos = brainPos.feel(); // feel() reads modulator levels directly (no tick needed)

const brainNeg = new DigitalBrain();
brainNeg.read(NEGATIVE);
const eNeg = brainNeg.feel();

console.log('A. AMYGDALA');
console.log(`   positive → valence=${ePos.valence.toFixed(3)} arousal=${ePos.arousal.toFixed(3)} (${ePos.primaryEmotion} ${ePos.emoji})`);
console.log(`   negative → valence=${eNeg.valence.toFixed(3)} arousal=${eNeg.arousal.toFixed(3)} (${eNeg.primaryEmotion} ${eNeg.emoji})`);

// Positive input must read clearly more pleasant than negative input.
const okAmygdala = ePos.valence > eNeg.valence + 0.25;
console.log(`   valence(pos) > valence(neg) + 0.25 : ${okAmygdala}\n`);

// ===========================================================================
// B. THALAMUS — acetylcholine widens the attentional bottleneck
// ===========================================================================
const brainAtt = new DigitalBrain();
const thalamus = brainAtt.getRegion('thalamus') as unknown as Thalamus;
const mods = brainAtt.getModulators();

// A sensory vector with 1000 supra-threshold candidates (> noiseFloor 0.1).
const input = new Float32Array(50000);
for (let i = 0; i < 1000; i++) input[i] = 0.5;

const baseEffects = mods.getEffects();
const baseOut = thalamus.processAttention(input, baseEffects);

mods.release(ModulatorType.Acetylcholine, 0.6); // basal-forebrain ACh burst
const achEffects = mods.getEffects();
const achOut = thalamus.processAttention(input, achEffects);

console.log('B. THALAMUS');
console.log(`   attentionGain : ${baseEffects.attentionGain.toFixed(3)} → ${achEffects.attentionGain.toFixed(3)} (after ACh)`);
console.log(`   effective bottleneck K : ${baseOut.effectiveBottleneck} → ${achOut.effectiveBottleneck}`);
console.log(`   signals passed : ${baseOut.salientIndices.length} → ${achOut.salientIndices.length}`);

const okThalamus =
  achEffects.attentionGain > baseEffects.attentionGain &&
  achOut.effectiveBottleneck > baseOut.effectiveBottleneck &&
  achOut.salientIndices.length > baseOut.salientIndices.length;
console.log(`   ACh widens the bottleneck (more signals pass) : ${okThalamus}\n`);

// ===========================================================================
// C. NEUROMODULATION — dopamine boosts plasticity, cortisol suppresses it
// ===========================================================================
// learningRateMultiplier = (1 + DA*0.8) * (1 - CORT*0.4)
const brainDA = new DigitalBrain();
const modsDA = brainDA.getModulators();
const lrBaseDA = modsDA.getEffects().learningRateMultiplier;
modsDA.release(ModulatorType.Dopamine, 0.4);
const lrAfterDA = modsDA.getEffects().learningRateMultiplier;

const brainCORT = new DigitalBrain();
const modsCORT = brainCORT.getModulators();
const lrBaseCORT = modsCORT.getEffects().learningRateMultiplier;
modsCORT.release(ModulatorType.Cortisol, 0.4);
const lrAfterCORT = modsCORT.getEffects().learningRateMultiplier;

console.log('C. NEUROMODULATION (plasticity gate)');
console.log(`   dopamine  : learningRate ×${lrBaseDA.toFixed(3)} → ×${lrAfterDA.toFixed(3)} (should rise)`);
console.log(`   cortisol  : learningRate ×${lrBaseCORT.toFixed(3)} → ×${lrAfterCORT.toFixed(3)} (should fall)`);

const okNeuromod = lrAfterDA > lrBaseDA + 0.05 && lrAfterCORT < lrBaseCORT - 0.05;
console.log(`   DA raises plasticity AND CORT suppresses it : ${okNeuromod}\n`);

// ===========================================================================
// Verdict
// ===========================================================================
console.log('── Metrics ──');
console.log(`  A. Amygdala affect tracks input sign     : ${okAmygdala}`);
console.log(`  B. Thalamus ACh widens attention         : ${okThalamus}`);
console.log(`  C. Neuromodulation gates plasticity      : ${okNeuromod}`);
console.log('');

if (okAmygdala && okThalamus && okNeuromod) {
  console.log('✅ AFFECT/ATTENTION/NEUROMODULATION VERIFIED: each subsystem behaves as its neuroscience predicts.');
  process.exit(0);
} else {
  console.log(`❌ FAIL: amygdala=${okAmygdala} thalamus=${okThalamus} neuromod=${okNeuromod}`);
  process.exit(1);
}
