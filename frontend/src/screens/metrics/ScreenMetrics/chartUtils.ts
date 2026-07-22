export const CHART_WINDOW = 60;
export const GAN_ARCH = "rrdb_esrgan";

// ── Formatting ────────────────────────────────────────────────────────────

export function fmt(n: number | null, decimals = 4): string {
  if (n == null) return "—";
  return n.toFixed(decimals);
}

export function fmtPct(n: number | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)}%`;
}

export function fmtGb(n: number | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)} GB`;
}

export function formatEta(sec: number | null): string {
  if (sec == null || sec <= 0) return "";
  if (sec < 60) return `ETA ${Math.round(sec)}s`;
  if (sec < 3600) return `ETA ${Math.round(sec / 60)}m`;
  return `ETA ${(sec / 3600).toFixed(1)}h`;
}

export function computeEtaSec(
  batch: number,
  totalBatch: number,
  epoch: number,
  totalEpochs: number,
  speed: number | null,
): number | null {
  if (!speed || speed <= 0 || totalBatch <= 0) return null;
  const rem = (totalEpochs - epoch) * totalBatch + (totalBatch - batch);
  return rem > 0 ? rem / speed : null;
}

// ── Axis ticks ────────────────────────────────────────────────────────────

function niceNumber(x: number): number {
  const exp = Math.floor(Math.log10(x));
  const frac = x / 10 ** exp;
  if (frac <= 1) return 10 ** exp;
  if (frac <= 2) return 2 * 10 ** exp;
  if (frac <= 5) return 5 * 10 ** exp;
  return 10 * 10 ** exp;
}

export function niceTicks(min: number, max: number, count: number): number[] {
  if (max - min < 1e-12) return [min];
  const range = niceNumber(max - min);
  const step = niceNumber(range / Math.max(count - 1, 1));
  const start = Math.floor(min / step) * step;
  const raw: number[] = [];
  for (let v = start; v <= max + step * 0.5; v += step) {
    if (raw.length >= count) break;
    if (v >= min - step * 0.01) raw.push(v);
  }
  // Rounding can occasionally produce two ticks that collapse to the same
  // rendered label (e.g. a near-flat series) — dedupe so labels never stack.
  const seen = new Set<string>();
  const ticks = raw.filter((t) => {
    const key = fmtAxisLoss(t);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return ticks.length < 2 && max > min ? [min, max] : ticks;
}

export function fmtAxisLoss(v: number): string {
  if (v >= 100) return v.toFixed(0);
  if (v >= 1) return v.toFixed(2);
  if (v >= 0.01) return v.toFixed(4);
  return v.toExponential(1);
}

// Simple trend indicator: compares last value against value ~5 samples back.
export function trendOf(
  series: number[],
  invert = false,
): { dir: "up" | "down" | "flat"; pct: number } | null {
  if (series.length < 2) return null;
  const back = Math.max(0, series.length - 6);
  const prev = series[back];
  const last = series[series.length - 1];
  if (prev === 0) return null;
  const pct = ((last - prev) / Math.abs(prev)) * 100;
  const dir = Math.abs(pct) < 0.5 ? "flat" : pct > 0 ? "up" : "down";
  const good = invert ? dir === "down" : dir === "up";
  return { dir: dir === "flat" ? "flat" : good ? "up" : "down", pct: Math.abs(pct) };
}

// ── SVG chart helpers ─────────────────────────────────────────────────────

export interface LinePoint { x: number; y: number }

export function buildPoints(values: number[], w: number, h: number, min: number, max: number): LinePoint[] {
  const range = max - min || 1;
  return values.map((v, i) => ({
    x: (i / Math.max(values.length - 1, 1)) * w,
    y: h - ((v - min) / range) * h,
  }));
}

// Smooth Catmull-Rom → cubic Bezier path, so lines read as flowing curves
// rather than jagged polylines. Falls back to a straight segment for <3 pts.
export function smoothPath(pts: LinePoint[]): string {
  if (pts.length === 0) return "";
  if (pts.length < 3) {
    return pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  }
  let d = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)} `;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i === 0 ? 0 : i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2 < pts.length ? i + 2 : i + 1];
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += `C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)} `;
  }
  return d.trim();
}

export function areaPath(pts: LinePoint[], h: number): string {
  if (pts.length === 0) return "";
  const line = smoothPath(pts);
  const last = pts[pts.length - 1];
  const first = pts[0];
  return `${line} L${last.x.toFixed(1)},${h} L${first.x.toFixed(1)},${h} Z`;
}
