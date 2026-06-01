/**
 * SINTETIZADOR DE VOZ — Texto a voz
 * ===================================
 * Convierte el texto generado por el área de Broca en audio hablado.
 * 
 * Usa la Web Speech API (SpeechSynthesis) cuando se ejecuta en el
 * navegador. En Node.js, genera comandos para TTS externo.
 * 
 * Biología: El área de Broca controla la producción del habla.
 * La velocidad, tono y volumen se modulan por el estado emocional.
 */

import type { EmotionalState } from './emotion-decoder.js';

/** Configuración de síntesis de voz */
export interface SpeechConfig {
  /** Idioma (código BCP 47) */
  lang: string;
  /** Velocidad base de habla (0.1-10, 1 = normal) */
  rate: number;
  /** Tono base (0-2, 1 = normal) */
  pitch: number;
  /** Volumen (0-1) */
  volume: number;
}

const DEFAULT_SPEECH_CONFIG: SpeechConfig = {
  lang: 'es-ES',
  rate: 1.0,
  pitch: 1.0,
  volume: 0.8,
};

/**
 * Sintetizador de voz que modula parámetros según estado emocional.
 */
export class SpeechSynthesizer {
  private config: SpeechConfig;
  /** Historial de utterances generadas */
  private history: Array<{ text: string; timestamp: number; emotion?: string }> = [];

  constructor(config: Partial<SpeechConfig> = {}) {
    this.config = { ...DEFAULT_SPEECH_CONFIG, ...config };
  }

  /**
   * Genera parámetros de habla modulados por emoción.
   * En el navegador, esto alimentaría SpeechSynthesisUtterance.
   * 
   * @param text - Texto a hablar
   * @param emotion - Estado emocional actual (opcional)
   * @returns Parámetros de habla ajustados
   */
  synthesize(text: string, emotion?: EmotionalState): {
    text: string;
    rate: number;
    pitch: number;
    volume: number;
    lang: string;
    ssml?: string;
  } {
    let rate = this.config.rate;
    let pitch = this.config.pitch;
    let volume = this.config.volume;

    if (emotion) {
      // Modular velocidad según arousal
      // Alto arousal → habla más rápida
      rate *= (0.8 + emotion.arousal * 0.6);

      // Modular tono según valencia
      // Positivo → tono más alto, Negativo → tono más bajo
      pitch *= (0.85 + (emotion.valence + 1) * 0.15);

      // Modular volumen según intensidad
      volume = Math.min(1, volume * (0.7 + emotion.intensity * 0.5));
    }

    // Limitar rangos
    rate = Math.max(0.5, Math.min(2.0, rate));
    pitch = Math.max(0.5, Math.min(2.0, pitch));
    volume = Math.max(0.1, Math.min(1.0, volume));

    // Registrar en historial
    this.history.push({
      text,
      timestamp: Date.now(),
      emotion: emotion?.primaryEmotion,
    });

    // Generar SSML para control fino (si el TTS lo soporta)
    const ssml = this.generateSSML(text, rate, pitch, volume);

    return { text, rate, pitch, volume, lang: this.config.lang, ssml };
  }

  /**
   * Genera SSML (Speech Synthesis Markup Language) para control preciso.
   */
  private generateSSML(text: string, rate: number, pitch: number, volume: number): string {
    const ratePercent = Math.round(rate * 100);
    const pitchPercent = Math.round((pitch - 1) * 100);
    const pitchStr = pitchPercent >= 0 ? `+${pitchPercent}%` : `${pitchPercent}%`;

    return `<speak>
  <prosody rate="${ratePercent}%" pitch="${pitchStr}" volume="${Math.round(volume * 100)}">
    ${this.escapeSSML(text)}
  </prosody>
</speak>`;
  }

  /**
   * Escapa caracteres especiales para SSML.
   */
  private escapeSSML(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }

  /**
   * Genera el comando de TTS para Node.js (macOS `say` command).
   */
  generateNodeCommand(text: string, emotion?: EmotionalState): string {
    const params = this.synthesize(text, emotion);
    // macOS `say` command
    const rateWPM = Math.round(params.rate * 175); // 175 WPM es velocidad normal
    return `say -r ${rateWPM} "${text.replace(/"/g, '\\"')}"`;
  }

  /** Obtiene el historial de utterances */
  getHistory(): typeof this.history {
    return [...this.history];
  }

  /** Limpia el historial */
  clearHistory(): void {
    this.history = [];
  }
}
