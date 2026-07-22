import { useEffect, useState } from "react";
import { buildPoints, smoothPath } from "./chartUtils";

// Several live readings (GPU/CPU/RAM/VRAM/temp/speed) aren't tracked as
// history arrays in the store, unlike loss/PSNR/SSIM — so this keeps a
// small capped local rolling window of readings to drive a sparkline +
// trend for otherwise-static gauges and cards.
export function useRollingHistory(value: number | null, maxLen = 30): number[] {
  const [hist, setHist] = useState<number[]>([]);
  useEffect(() => {
    if (value == null) return;
    setHist((h) => (h[h.length - 1] === value ? h : [...h, value].slice(-maxLen)));
  }, [value, maxLen]);
  return hist;
}

export function Sparkline({ values, color, width = 68, height = 26, padding = 3.5 }: {
  values: number[]; color: string; width?: number; height?: number; padding?: number;
}) {
  const windowed = values.slice(-20);
  if (windowed.length < 2) return null;
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
  return (
    <svg width={width} height={height} style={{ display: "block", flexShrink: 0, overflow: "visible" }}>
      <path
        d={smoothPath(pts)}
        fill="none"
        stroke={color}
        strokeWidth={1.7}
        strokeOpacity={0.85}
        strokeLinecap="round"
        strokeLinejoin="round"
        transform={`translate(${padding},${padding})`}
      />
    </svg>
  );
}