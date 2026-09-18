/**
 * STROKES — how a drawing is made, not only what it looks like
 * ===========================================================================
 * The whiteboard map (HandMotorCortex) knows WHICH cells make an image. This
 * module holds HOW they are laid down: the order of the strokes, the way
 * each one runs, and how long it takes — an internal model of the movement,
 * learned by watching someone draw. In the brain that is the cerebellum's
 * job: it learns the timing and sequence of a movement from demonstration
 * and error (Wolpert & Kawato 1998; Ito 2008), and it is what turns a set of
 * marks into a fluent gesture.
 *
 * A demonstration is a list of strokes, each a path of grid cells with its
 * duration. Demonstrations of the same drawing (alike enough in their cells)
 * are pooled; the order shown most often is the one the hand follows. When
 * the hand copies a drawing it knows a gesture for, its own cells are laid
 * along that gesture: each cell joins the stroke that passes nearest to it,
 * in the order the stroke passes.
 */
import { GRID_SIDE, centred, likeness, type Drawing } from './whiteboard.js';

/** One stroke: the cells it runs through, in order, and how long it takes. */
export interface StrokePath {
  cells: number[];
  durationMs: number;
}

/** Cells alike enough for two demonstrations to be of the same drawing. */
const SAME_DRAWING = 0.5;
/** A cell joins a stroke that passes within this distance (grid cells). */
const NEAR = 1.6;
/** Demonstrations kept per drawing, and drawings kept. */
const MAX_DEMOS = 12;
const MAX_DRAWINGS = 64;
/** Duration of a stroke made up of cells no demonstrated stroke passed near. */
const LEFTOVER_MS = 200;

interface Gesture {
  /** Order signature: each stroke's first and last cell. */
  key: string;
  strokes: StrokePath[];
  count: number;
}

interface KnownDrawing {
  cells: Drawing;
  gestures: Gesture[];
  demonstrations: number;
}

/** The shift `centred` applies to a drawing (rows, cols). */
export function centringShift(cells: Drawing): { dRow: number; dCol: number } {
  if (cells.length === 0) return { dRow: 0, dCol: 0 };
  let sumRow = 0, sumCol = 0;
  for (const cell of cells) { sumRow += Math.floor(cell / GRID_SIDE); sumCol += cell % GRID_SIDE; }
  return { dRow: Math.round(GRID_SIDE / 2 - sumRow / cells.length), dCol: Math.round(GRID_SIDE / 2 - sumCol / cells.length) };
}

/** Shifts a path by (dRow, dCol), dropping what falls off the grid. */
function shifted(cells: readonly number[], dRow: number, dCol: number): number[] {
  const out: number[] = [];
  for (const cell of cells) {
    const row = Math.floor(cell / GRID_SIDE) + dRow, col = (cell % GRID_SIDE) + dCol;
    if (row >= 0 && row < GRID_SIDE && col >= 0 && col < GRID_SIDE) out.push(row * GRID_SIDE + col);
  }
  return out;
}

const keyOf = (strokes: readonly StrokePath[]): string =>
  strokes.map((s) => `${s.cells[0]}>${s.cells[s.cells.length - 1]}`).join('|');

export class StrokeMemory {
  private drawings: KnownDrawing[] = [];

  /** Drawings it knows a gesture for. */
  get size(): number {
    return this.drawings.length;
  }

  /**
   * Someone drew this, stroke by stroke. The strokes are centred as the
   * drawing is when seen, so that a copy — centred too — can follow them.
   */
  observe(strokes: readonly StrokePath[]): void {
    const paths = strokes.filter((s) => s.cells.length > 0);
    if (paths.length === 0) return;
    const union = centred([...new Set(paths.flatMap((s) => s.cells))]);
    const { dRow, dCol } = centringShift([...new Set(paths.flatMap((s) => s.cells))]);
    const demo: StrokePath[] = paths
      .map((s) => ({ cells: shifted(s.cells, dRow, dCol), durationMs: Math.max(50, Math.min(10_000, s.durationMs || LEFTOVER_MS)) }))
      .filter((s) => s.cells.length > 0);
    if (demo.length === 0) return;

    let known = this.drawings.find((d) => likeness(d.cells, union) >= SAME_DRAWING);
    if (!known) {
      known = { cells: union, gestures: [], demonstrations: 0 };
      this.drawings.push(known);
      if (this.drawings.length > MAX_DRAWINGS) this.drawings.shift();
    }
    known.demonstrations++;
    const key = keyOf(demo);
    const gesture = known.gestures.find((g) => g.key === key);
    if (gesture) {
      gesture.count++;
      // The timing follows the latest demonstrations (a running mean).
      for (let i = 0; i < gesture.strokes.length && i < demo.length; i++) {
        gesture.strokes[i].durationMs += 0.3 * (demo[i].durationMs - gesture.strokes[i].durationMs);
      }
    } else {
      known.gestures.push({ key, strokes: demo, count: 1 });
      if (known.gestures.length > MAX_DEMOS) known.gestures.sort((a, b) => b.count - a.count).pop();
    }
  }

  /** The gesture shown most often for a drawing like this one, or `null`. */
  planFor(cells: Drawing): StrokePath[] | null {
    let best: KnownDrawing | null = null;
    let bestLikeness = 0;
    for (const d of this.drawings) {
      const l = likeness(d.cells, cells);
      if (l > bestLikeness) { bestLikeness = l; best = d; }
    }
    if (!best || bestLikeness < SAME_DRAWING || best.gestures.length === 0) return null;
    return [...best.gestures].sort((a, b) => b.count - a.count)[0].strokes;
  }

  /**
   * Lays the copy's own cells along a gesture: each cell joins the stroke
   * that passes nearest to it, in the order that stroke passes; cells no
   * stroke comes near are drawn last, together.
   */
  static follow(cells: Drawing, plan: readonly StrokePath[]): StrokePath[] {
    const rc = (cell: number): [number, number] => [Math.floor(cell / GRID_SIDE), cell % GRID_SIDE];
    const buckets: Array<Array<{ cell: number; at: number }>> = plan.map(() => []);
    const leftover: number[] = [];
    for (const cell of cells) {
      const [r, c] = rc(cell);
      let bestStroke = -1, bestAt = 0, bestDist = NEAR;
      plan.forEach((stroke, s) => {
        stroke.cells.forEach((p, i) => {
          const [pr, pc] = rc(p);
          const dist = Math.hypot(pr - r, pc - c);
          if (dist < bestDist) { bestDist = dist; bestStroke = s; bestAt = i; }
        });
      });
      if (bestStroke < 0) leftover.push(cell);
      else buckets[bestStroke].push({ cell, at: bestAt });
    }
    const out: StrokePath[] = [];
    buckets.forEach((bucket, s) => {
      if (bucket.length === 0) return;
      bucket.sort((a, b) => a.at - b.at || a.cell - b.cell);
      out.push({ cells: bucket.map((b) => b.cell), durationMs: plan[s].durationMs });
    });
    if (leftover.length > 0) out.push({ cells: leftover, durationMs: LEFTOVER_MS });
    return out;
  }

  serialize(): unknown {
    return this.drawings.map((d) => [d.cells, d.demonstrations, d.gestures.map((g) => [g.count, g.strokes.map((s) => [s.cells, Math.round(s.durationMs)])])]);
  }

  deserialize(data: unknown): void {
    if (!Array.isArray(data)) return;
    const cellList = (x: unknown): number[] | null =>
      Array.isArray(x) && x.every((v) => Number.isInteger(v) && v >= 0 && v < GRID_SIDE * GRID_SIDE) ? (x as number[]).slice(0, GRID_SIDE * GRID_SIDE) : null;
    this.drawings = [];
    for (const entry of data.slice(0, MAX_DRAWINGS)) {
      if (!Array.isArray(entry) || entry.length !== 3) continue;
      const cells = cellList(entry[0]);
      if (!cells || cells.length === 0 || !Array.isArray(entry[2])) continue;
      const gestures: Gesture[] = [];
      for (const g of (entry[2] as unknown[]).slice(0, MAX_DEMOS)) {
        if (!Array.isArray(g) || g.length !== 2 || !Number.isInteger(g[0]) || !Array.isArray(g[1])) continue;
        const strokes: StrokePath[] = [];
        for (const s of (g[1] as unknown[]).slice(0, 64)) {
          if (!Array.isArray(s) || s.length !== 2) continue;
          const path = cellList(s[0]);
          if (!path || path.length === 0 || typeof s[1] !== 'number' || !Number.isFinite(s[1])) continue;
          strokes.push({ cells: path, durationMs: Math.max(50, Math.min(10_000, s[1] as number)) });
        }
        if (strokes.length > 0) gestures.push({ key: keyOf(strokes), strokes, count: Math.max(1, Math.min(1e6, g[0] as number)) });
      }
      this.drawings.push({ cells, gestures, demonstrations: Number.isInteger(entry[1]) && (entry[1] as number) > 0 ? (entry[1] as number) : gestures.length });
    }
  }
}
