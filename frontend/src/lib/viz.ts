/**
 * Chart tokens.
 *
 * Categorical slots are the validated dark-mode order, checked against this
 * app's own surface (#111820) rather than assumed:
 *
 *   Lightness band   PASS  all inside L 0.48–0.67
 *   Chroma floor     PASS  all >= 0.1
 *   CVD separation   PASS  worst adjacent ΔE 8.4 (protan)
 *   Normal vision    PASS  worst adjacent ΔE 19.8
 *   Contrast         PASS  all >= 3:1 against the surface
 *
 * Series identity is never carried by colour alone — every chart with two or
 * more series also direct-labels them.
 */

/** Fixed order. Assigned by slot, never cycled, never by rank. */
export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"] as const;

/** One hue, light → dark, for magnitude. Stops no darker than step 600 on dark. */
export const SEQUENTIAL = [
  "#9ec5f4",
  "#86b6ef",
  "#6da7ec",
  "#5598e7",
  "#3987e5",
  "#2a78d6",
  "#256abf",
  "#1c5cab",
] as const;

/** Reserved for state. Never reused as "series 5". Always paired with a label. */
export const STATUS = {
  good: "#4ade80",
  warning: "#eda100",
  critical: "#f87171",
} as const;

export const INK = {
  surface: "#111820",
  grid: "#1e2a36",
  axis: "#2a3947",
  muted: "#7d8da0",
  text: "#e6edf3",
  deemphasis: "#3d4a58",
} as const;

/** Recessive axes and grid: the data should be the only assertive thing on screen. */
export const AXIS_PROPS = {
  stroke: INK.axis,
  tick: { fill: INK.muted, fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: INK.axis },
} as const;

export const GRID_PROPS = {
  stroke: INK.grid,
  strokeDasharray: "2 4",
  vertical: false,
} as const;

/** Pick a sequential step by rank within n items. Darker = larger magnitude. */
export function sequentialStep(index: number, total: number): string {
  if (total <= 1) return SEQUENTIAL[SEQUENTIAL.length - 2];
  const position = 1 - index / (total - 1); // index 0 is the largest value
  const step = Math.round(position * (SEQUENTIAL.length - 1));
  return SEQUENTIAL[Math.min(Math.max(step, 0), SEQUENTIAL.length - 1)];
}
