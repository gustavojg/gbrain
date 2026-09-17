/**
 * AFFECTIVE LEXICON — Early-learned word → emotion associations (ES + EN)
 * =======================================================================
 * Seed associations that are conditioned into the amygdala at start-up, the
 * way a child has already attached affect to common words long before any
 * conversation with this brain takes place.
 *
 * Biological basis:
 *   Emotional words evoke amygdala responses through learned associations
 *   (Isenberg et al., 1999; Hamann & Mao, 2002). Affect is described on the
 *   circumplex (Russell, 1980): valence (−1 … +1) × arousal (0 … 1). The
 *   amygdala's central nucleus then turns that state into neuromodulator
 *   release (see `Amygdala.produceNeuromodulators`), so e.g. high-arousal
 *   negative words end up as cortisol + norepinephrine, and calm positive
 *   ones as serotonin + oxytocin.
 *
 * Words are stored normalized (lowercase, no diacritics) — the same form the
 * tokens of `DigitalBrain.read()` have.
 */

import type { EmotionalState } from './amygdala.js';

const JOY: EmotionalState = { valence: 0.85, arousal: 0.7 };
const CALM: EmotionalState = { valence: 0.7, arousal: 0.25 };
const BOND: EmotionalState = { valence: 0.8, arousal: 0.35 };
const MILD_POSITIVE: EmotionalState = { valence: 0.45, arousal: 0.4 };
const ENGAGED: EmotionalState = { valence: 0.25, arousal: 0.65 };
const THREAT: EmotionalState = { valence: -0.85, arousal: 0.85 };
const ANGER: EmotionalState = { valence: -0.75, arousal: 0.8 };
const SADNESS: EmotionalState = { valence: -0.7, arousal: 0.3 };
const MILD_NEGATIVE: EmotionalState = { valence: -0.45, arousal: 0.4 };

const GROUPS: Array<[EmotionalState, string]> = [
  [JOY, 'feliz felices felicidad alegria alegre entusiasmo genial contento contenta ' +
        'happy happiness joy joyful excited excitement great wonderful amazing'],
  [CALM, 'paz calma tranquilo tranquila sereno dormir descanso ' +
         'peace calm peaceful serene relax relaxed rest'],
  [BOND, 'amor carino gracias amigo amiga abrazo querer ' +
         'love loved thanks thank friend hug kind kindness'],
  [MILD_POSITIVE, 'bien bueno buena bonito bonita hola esperanza sonar ' +
                  'good nice beautiful hello hope dream'],
  [ENGAGED, 'curiosidad pensar aprender recordar atencion ' +
            'curiosity curious think learn remember attention'],
  [THREAT, 'miedo terror panico ansiedad estres peligro ' +
           'fear afraid scared terror panic anxiety stress danger'],
  [ANGER, 'odio enojo rabia ira furia ' +
          'hate hatred anger angry rage fury'],
  [SADNESS, 'triste tristeza soledad llorar dolor ' +
            'sad sadness lonely loneliness cry pain grief'],
  [MILD_NEGATIVE, 'mal malo mala feo fea frustracion ' +
                  'bad ugly awful frustration frustrated'],
];

/** Every seeded association as `[word, emotion]`. */
export const AFFECTIVE_LEXICON: ReadonlyArray<readonly [string, EmotionalState]> = GROUPS.flatMap(
  ([emotion, words]) => words.split(' ').map((word) => [word, emotion] as const),
);
