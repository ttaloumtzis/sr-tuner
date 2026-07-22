import { useState, type PointerEvent as ReactPointerEvent } from "react";
import { EmptyChartState } from "./EmptyChartState";
import { niceTicks, fmt, fmtAxisLoss, smoothPath, areaPath } from "./chartUtils";

export function SubChart({ uid, chartKey, series, color, fullSeries, fullColor, windowStart = 0 }: {
  uid: string; chartKey: string; series: number[]; color: string;
  fullSeries?: number[]; fullColor?: string; windowStart?: number;
}) {
  const W = 400; const H = 72; const ML = 42;
  const CW = W - ML - 12;
  // Vertical inset: without this, a point at the series max/min lands at
  // y=0 / y=H exactly, so its marker circle (up to r=3.5) sits half outside
  // the SVG and gets sliced off by the parent's overflow:hidden.
  const MT = 6; const MB = 6;
  const PH = H - MT - MB;
  const R = 2.6;
  const count = series.length;
  const hasFull = fullSeries != null && fullSeries.length > 0;
  const [hoverX, setHoverX] = useState<number | null>(null);

  if (count === 0) {
    return <EmptyChartState label={chartKey.toLowerCase()} />;
  }

  const allVals = hasFull ? [...series, ...fullSeries!] : series;
  const mn = Math.min(...allVals);
  const mx = Math.max(...allVals);
  const ticks = niceTicks(mn, mx, 3);
  const mapX = (i: number) => ML + (i / Math.max(count - 1, 1)) * CW;
  // The "full" validation series is recorded on its own cadence and can have a
  // different length than the per-batch series, so it needs its own x-scale —
  // reusing mapX() here would place each full-validation point at the wrong
  // epoch whenever the two arrays' lengths diverge.
  const mapXFull = (i: number) =>
    ML + (i / Math.max((fullSeries?.length ?? 1) - 1, 1)) * CW;
  const mapY = (v: number) => MT + PH - ((v - mn) / (mx - mn || 1)) * PH;
  const pts = series.map((v, i) => ({ x: mapX(i) - ML, y: mapY(v) }));

  const fullPts = hasFull
    ? fullSeries!.map((v, i) => ({ x: mapXFull(i) - ML, y: mapY(v) }))
    : [];

  const hoverIdx = hoverX == null ? null
    : Math.max(0, Math.min(count - 1, Math.round((hoverX / CW) * (count - 1))));
  const hoverPt = hoverIdx != null ? pts[hoverIdx] : null;
  const hoverFullIdx = hoverX == null || !hasFull
    ? null
    : Math.max(0, Math.min(fullSeries!.length - 1, Math.round((hoverX / CW) * (fullSeries!.length - 1))));
  const hoverFullVal = hoverFullIdx != null ? fullSeries![hoverFullIdx] : null;

  function handlePointer(e: ReactPointerEvent<SVGRectElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * CW;
    setHoverX(Math.max(0, Math.min(CW, relX)));
  }

  const fullCol = fullColor ?? color;

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%", display: "block" }} preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id={`${uid}-${chartKey}-fill`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
          {hasFull && (
            <linearGradient id={`${uid}-${chartKey}-full-fill`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={fullCol} stopOpacity={0.15} />
              <stop offset="100%" stopColor={fullCol} stopOpacity={0} />
            </linearGradient>
          )}
        </defs>
        {ticks.map((t) => {
          const y = mapY(t);
          return (
            <g key={t}>
              <line x1={ML} y1={y} x2={ML + CW} y2={y}
                stroke={color} strokeWidth={0.5} strokeOpacity={0.12} strokeDasharray="3 3" />
              <text x={ML - 5} y={y + 3} textAnchor="end" fill="var(--dim)"
                fontSize={9} fontFamily="var(--font-mono)">
                {fmtAxisLoss(t)}
              </text>
            </g>
          );
        })}
        <g transform={`translate(${ML},0)`}>
          <path d={areaPath(pts, H)} fill={`url(#${uid}-${chartKey}-fill)`} stroke="none" />
          <path d={smoothPath(pts)} fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" />
          {hasFull && (
            <>
              <path d={areaPath(fullPts, H)} fill={`url(#${uid}-${chartKey}-full-fill)`} stroke="none" />
              <path d={smoothPath(fullPts)} fill="none" stroke={fullCol} strokeWidth={1.4} strokeDasharray="4 3" strokeLinecap="round" />
              {fullPts.length > 0 && (
                <circle cx={fullPts[fullPts.length - 1].x} cy={fullPts[fullPts.length - 1].y} r={3} fill={fullCol} />
              )}
            </>
          )}
          {count <= 30 && pts.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={R} fill={color} fillOpacity={0.9} />
          ))}
          {count > 0 && (
            <circle cx={pts[pts.length - 1].x} cy={pts[pts.length - 1].y} r={3.5} fill={color} />
          )}
          {hoverPt && (
            <g pointerEvents="none">
              <line x1={hoverPt.x} y1={0} x2={hoverPt.x} y2={H}
                stroke="var(--border2)" strokeWidth={1} strokeDasharray="2 2" />
              <circle cx={hoverPt.x} cy={hoverPt.y} r={4} fill="var(--bg0)" stroke={color} strokeWidth={1.6} />
            </g>
          )}
          <rect
            x={0} y={0} width={CW} height={H} fill="transparent"
            onPointerMove={handlePointer}
            onPointerLeave={() => setHoverX(null)}
            style={{ cursor: "crosshair" }}
          />
        </g>
      </svg>
      {hoverIdx != null && hoverPt != null && (() => {
        const lines = [
          { text: `epoch ${windowStart + hoverIdx + 1}`, col: "var(--muted)" },
          { text: `patch ${fmt(series[hoverIdx])}`, col: color },
          ...(hoverFullVal != null ? [{ text: `full ${fmt(hoverFullVal)}`, col: fullCol }] : []),
        ];
        const boxW = 90;
        const boxH = lines.length * 13 + 6;
        const overflowsRight = hoverPt.x + 10 + boxW > CW;
        const bx = overflowsRight ? hoverPt.x - 10 - boxW : hoverPt.x + 10;
        const by = Math.max(2, Math.min(H - boxH - 2, hoverPt.y - boxH / 2));
        return (
          <div style={{
            position: "absolute", left: ML + bx, top: by,
            width: boxW, padding: "3px 8px", borderRadius: 4,
            background: "var(--bg2)", border: "1px solid var(--border2)",
            pointerEvents: "none",
          }}>
            {lines.map((l, i) => (
              <div key={i} style={{ fontSize: 9, color: l.col, fontFamily: "var(--font-mono)", fontWeight: i > 0 ? 600 : 400, lineHeight: 1.5 }}>
                {l.text}
              </div>
            ))}
          </div>
        );
      })()}
    </div>
  );
}