import { useTrainingStore, type TrainingState } from "../../../store/trainingStore";
import { buildPoints, smoothPath } from "./chartUtils";

// Several live readings (GPU/CPU/RAM/VRAM/temp/speed) aren't tracked as
// history arrays in the backend, unlike loss/PSNR/SSIM — so the store keeps a
// small capped rolling window of them (see `updateFromHardware` in
// trainingStore.ts) to drive a sparkline + trend for otherwise-static gauges
// and cards. The window advances on every SSE hardware event (backend polls
// every ~0.5s), so a flat reading still produces a fresh sample each tick and
// the sparkline renders as a true sliding time-series window instead of
// freezing into a static line.
//
// Living in the store (rather than local component state) means the window
// survives tab switches: leaving the Live Metrics tab unmounts these
// components, but the history keeps accumulating and is still there on return.
export function useRollingHistory(selector: (s: TrainingState) => number[]): number[] {
  return useTrainingStore(selector);
}

export function Sparkline({ values, color, width = 68, height = 26, padding = 3.5, points = 20 }: {
  values: number[]; color: string; width?: number; height?: number; padding?: number; points?: number;
}) {
  const windowed = values.slice(-points);
  if (windowed.length === 0) return null;
  // Scale to the window actually being drawn, not the full history — using
  // the full history's min/max squashes the visible points whenever an
  // older outlier sits outside the window.
  const min = Math.min(...windowed);
  const max = Math.max(...windowed);
  // Plot into an inset area, not the full width/height: points at the exact
  // top/bottom edge otherwise get their stroke cut off (SVG clips at the
  // viewport by default), so a flat-topped run reads as clipped/cut.
  const plotW = width - padding * 2;
  const plotH = height - padding * 2;
  const pts = buildPoints(windowed, plotW, plotH, min, max);
  const translate = `translate(${padding},${padding})`;
  return (
    <svg width={width} height={height} style={{ display: "block", flexShrink: 0, overflow: "visible" }}>
      {pts.length === 1 ? (
        // A single sample can't draw a line — render a lone dot so a fresh
        // window still has a visible presence.
        <circle cx={pts[0].x} cy={pts[0].y} r={1.6} fill={color} transform={translate} />
      ) : (
        <path
          d={smoothPath(pts)}
          fill="none"
          stroke={color}
          strokeWidth={1.7}
          strokeOpacity={0.85}
          strokeLinecap="round"
          strokeLinejoin="round"
          transform={translate}
        />
      )}
    </svg>
  );
}