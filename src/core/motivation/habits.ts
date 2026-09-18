/**
 * HABITS — when a practised response stops needing a reason
 * ===========================================================================
 * Two systems can produce the same response to a cue (Dickinson 1985; Yin &
 * Knowlton 2006; Daw, Niv & Dayan 2005):
 *
 *   - GOAL-DIRECTED (dorsomedial striatum, prefrontal): the response is made
 *     because of what it is expected to bring. It stops as soon as the
 *     outcome is devalued (a reprimand where praise was expected) or the
 *     memory behind it weakens.
 *   - HABITUAL (dorsolateral striatum): a stimulus–response link, stamped in
 *     by repetition under reinforcement. Once strong enough it takes over:
 *     the response runs off the cue itself, faster and without attention —
 *     and keeps running after the outcome is devalued, extinguishing only
 *     slowly through repeated non-reinforcement.
 *
 * Here a habit is a link from a cue (a perceptual category) to the response
 * that has been made to it. Every execution stamps it in a little; praise
 * after it more; a reprimand after it wears it down, slowly. Above the
 * threshold the habit is what answers the cue.
 */

export interface Habit {
  cue: string;
  response: string;
  /** 0..1: how stamped in the link is. */
  strength: number;
  executions: number;
}

/** Stamped in per execution, praised or not (mere repetition counts a little). */
const PRACTICE_RATE = 0.04;
/** Stamped in by praise right after the response (reinforced practice is what makes habits). */
const REWARD_RATE = 0.18;
/** Worn down by a reprimand right after the response: slower than it built up. */
const EXTINCTION_RATE = 0.09;
/** Strength from which the habit takes over the response. */
export const HABIT_THRESHOLD = 0.6;
const MAX_HABITS = 500;

export class HabitSystem {
  /** cue → response → habit */
  private links = new Map<string, Map<string, Habit>>();
  /** The last response executed (what a verdict right after it credits). */
  private last: Habit | null = null;

  /** A response to a cue was executed (by whichever system): the link is stamped in. */
  practice(cue: string, response: string): Habit {
    let byResponse = this.links.get(cue);
    if (!byResponse) {
      byResponse = new Map();
      this.links.set(cue, byResponse);
      if (this.links.size > MAX_HABITS) this.links.delete(this.links.keys().next().value as string);
    }
    let habit = byResponse.get(response);
    if (!habit) {
      habit = { cue, response, strength: 0, executions: 0 };
      byResponse.set(response, habit);
    }
    habit.executions++;
    habit.strength += PRACTICE_RATE * (1 - habit.strength);
    this.last = habit;
    return habit;
  }

  /**
   * A verdict on the last response: praise stamps the link in further, a
   * reprimand wears it down — slowly, which is what makes a habit a habit.
   */
  credit(reward: number): Habit | null {
    if (!this.last) return null;
    const habit = this.last;
    if (reward > 0) habit.strength += REWARD_RATE * reward * (1 - habit.strength);
    else if (reward < 0) habit.strength = Math.max(0, habit.strength - EXTINCTION_RATE * -reward * habit.strength);
    return habit;
  }

  /** The habitual response to a cue, if one is stamped in enough to take over. */
  habitFor(cue: string): Habit | null {
    const byResponse = this.links.get(cue);
    if (!byResponse) return null;
    let best: Habit | null = null;
    for (const habit of byResponse.values()) if (!best || habit.strength > best.strength) best = habit;
    return best && best.strength >= HABIT_THRESHOLD ? best : null;
  }

  /** Every link, strongest first. */
  list(): Habit[] {
    const all: Habit[] = [];
    for (const byResponse of this.links.values()) for (const habit of byResponse.values()) all.push(habit);
    return all.sort((a, b) => b.strength - a.strength);
  }

  /** Links strong enough to be habits. */
  get size(): number {
    return this.list().filter((h) => h.strength >= HABIT_THRESHOLD).length;
  }

  serialize(): unknown {
    return this.list().map((h) => [h.cue, h.response, h.strength, h.executions]);
  }

  deserialize(data: unknown): void {
    if (!Array.isArray(data)) return;
    this.links = new Map();
    this.last = null;
    for (const entry of data.slice(0, MAX_HABITS)) {
      if (!Array.isArray(entry) || entry.length !== 4 || typeof entry[0] !== 'string' || typeof entry[1] !== 'string') continue;
      const strength = typeof entry[2] === 'number' && Number.isFinite(entry[2]) ? Math.max(0, Math.min(1, entry[2])) : 0;
      const executions = Number.isInteger(entry[3]) && (entry[3] as number) >= 0 ? Math.min(1e9, entry[3] as number) : 0;
      const cue = (entry[0] as string).slice(0, 64), response = (entry[1] as string).slice(0, 64);
      let byResponse = this.links.get(cue);
      if (!byResponse) this.links.set(cue, (byResponse = new Map()));
      byResponse.set(response, { cue, response, strength, executions });
    }
  }
}
