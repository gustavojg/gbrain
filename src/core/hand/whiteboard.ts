/**
 * WHITEBOARD — What a drawing command looks like
 * ==============================================
 * The hand draws on a coarse grid of cells: a command is simply the set of
 * cells to ink. This module renders a command as the image the eye will see
 * (the same 64×64, dark-background, white-ink format the dashboard's
 * whiteboard produces), and measures how alike two drawings are.
 *
 * It is shared by the brain (to see what its hand has drawn) and the tests; the
 * dashboard paints the same cells on its own canvas.
 */

/** Cells per side of the drawing grid (= the retina's resolution). */
export const GRID_SIDE = 14;
/** Pixels per side of the rendered whiteboard. */
export const BOARD_SIDE = 64;
/** Grey level of the empty whiteboard (#111827) and of the ink. */
export const BOARD_BACKGROUND = 27;
export const BOARD_INK = 255;

/** Fraction of a cell that must be covered by ink for the cell to count as inked. */
const INK_COVERAGE = 0.35;

/** A drawing: the inked cells of the grid, as `row * GRID_SIDE + col`. */
export type Drawing = number[];

/** Renders a drawing as a BOARD_SIDE × BOARD_SIDE grayscale image. */
export function renderDrawing(cells: Drawing): Uint8Array {
  const pixels = new Uint8Array(BOARD_SIDE * BOARD_SIDE).fill(BOARD_BACKGROUND);
  // Cell boundaries are the ones the retina pools over (VisualEncoder.resize):
  // a cell's ink must not spill into the patch of its neighbour.
  const edge = (index: number): number => Math.floor((index * BOARD_SIDE) / GRID_SIDE);
  for (const cell of cells) {
    const row = Math.floor(cell / GRID_SIDE);
    const col = cell % GRID_SIDE;
    const x0 = edge(col);
    const x1 = edge(col + 1);
    const y0 = edge(row);
    const y1 = edge(row + 1);
    for (let y = y0; y < y1; y++) pixels.fill(BOARD_INK, y * BOARD_SIDE + x0, y * BOARD_SIDE + x1);
  }
  return pixels;
}

/**
 * The cells of the grid that an image inks, centred on the grid like the eye
 * centres what it looks at. Used to compare a copy with its model.
 */
export function inkedCells(pixels: ArrayLike<number>, width: number, height: number): Drawing {
  const cells: Drawing = [];
  // Same criterion as the eye (VisualEncoder pools the mean of each patch): a
  // cell counts as inked when ink covers a good part of it, not when a stroke
  // merely grazes it.
  const threshold = BOARD_BACKGROUND + (BOARD_INK - BOARD_BACKGROUND) * INK_COVERAGE;
  for (let row = 0; row < GRID_SIDE; row++) {
    for (let col = 0; col < GRID_SIDE; col++) {
      const x0 = Math.floor((col * width) / GRID_SIDE);
      const x1 = Math.max(x0 + 1, Math.floor(((col + 1) * width) / GRID_SIDE));
      const y0 = Math.floor((row * height) / GRID_SIDE);
      const y1 = Math.max(y0 + 1, Math.floor(((row + 1) * height) / GRID_SIDE));
      let sum = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) sum += pixels[y * width + x];
      if (sum / ((y1 - y0) * (x1 - x0)) > threshold) cells.push(row * GRID_SIDE + col);
    }
  }
  return centred(cells);
}

/** Shifts a drawing so that its centre of mass sits at the centre of the grid. */
export function centred(cells: Drawing): Drawing {
  if (cells.length === 0) return [];
  let sumRow = 0;
  let sumCol = 0;
  for (const cell of cells) {
    sumRow += Math.floor(cell / GRID_SIDE);
    sumCol += cell % GRID_SIDE;
  }
  // Same convention as the eye's foveation (VisualEncoder.foveate), so that a
  // centred drawing is seen exactly where it was drawn.
  const dRow = Math.round(GRID_SIDE / 2 - sumRow / cells.length);
  const dCol = Math.round(GRID_SIDE / 2 - sumCol / cells.length);
  const shifted: Drawing = [];
  for (const cell of cells) {
    const row = Math.floor(cell / GRID_SIDE) + dRow;
    const col = (cell % GRID_SIDE) + dCol;
    if (row >= 0 && row < GRID_SIDE && col >= 0 && col < GRID_SIDE) shifted.push(row * GRID_SIDE + col);
  }
  return shifted.sort((a, b) => a - b);
}

/**
 * Likeness of two drawings, 0–1: intersection over union of their inked cells,
 * allowing the copy to sit one cell off (where exactly a drawing is centred
 * depends on rounding). Strict otherwise — a blob that covers the model scores low.
 */
export function likeness(a: Drawing, b: Drawing): number {
  if (a.length === 0 || b.length === 0) return 0;
  const setA = new Set(a);
  let best = 0;
  for (let dRow = -1; dRow <= 1; dRow++) {
    for (let dCol = -1; dCol <= 1; dCol++) {
      let shared = 0;
      for (const cell of b) {
        const row = Math.floor(cell / GRID_SIDE) + dRow;
        const col = (cell % GRID_SIDE) + dCol;
        if (row >= 0 && row < GRID_SIDE && col >= 0 && col < GRID_SIDE && setA.has(row * GRID_SIDE + col)) shared++;
      }
      best = Math.max(best, shared / (a.length + b.length - shared));
    }
  }
  return best;
}
