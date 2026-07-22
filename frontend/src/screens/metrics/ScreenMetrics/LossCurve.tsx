import { useId, useState, type PointerEvent as ReactPointerEvent } from "react";
import { useModelStore } from "../../../store/modelStore";
import { useTrainingStore } from "../../../store/trainingStore";
import type { RunHistory } from "../../../store/trainingStore";
import { PanelHeader } from "./PanelHeader";
import { EmptyChartState } from "./EmptyChartState";
import { CHART_WINDOW, GAN_ARCH, buildPoints, smoothPath, areaPath, niceTicks, fmtAxisLoss } from "./chartUtils";

export function LossCurve({ history }: { history: RunHistory | null }) {
  const uid = useId();
  const arch        = useModelStore((s) => s.architecture);
  const liveLoss    = useTrainingStore((s) => s.liveLoss);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const isGan = arch === GAN_ARCH;
  const W = 380; const H = 190;
  const ML = 46; const MR = 12; const MB = 18;
  const CW = W - ML - MR;
  const CH = H - MB;

  const fullGLen = history?.gLossHistory?.length ?? 0;
  const windowStart = Math.max(0, fullGLen - CHART_WINDOW);
  const gSeries = (history?.gLossHistory ?? []).slice(windowStart);
  const dSeries = isGan
    ? (history?.dLossHistory ?? []).filter((v): v is number => v != null).slice(windowStart)
    : [];
  const min = 0;
  const max = gSeries.length > 0 ? Math.max(...gSeries) * 1.05 || 1 : 2.0;

  const gChart = buildPoints(gSeries, CW, CH, min, max);
  const dChart = buildPoints(dSeries, CW, CH, min, max);

  const yTicks = niceTicks(min, max, 5);
  const xLabels = gSeries.length;
  const xTicks = xLabels > 1 ? niceTicks(1, xLabels, Math.min(xLabels, 6)) : [];

  const livePoint = liveLoss != null
    ? { x: CW, y: CH - ((liveLoss - min) / (max - min || 1)) * CH }
    : null;

  const empty = gSeries.length === 0;

  // Hover crosshair: snap to the nearest sample index under the cursor.
  const hoverIdx = hoverX == null || xLabels === 0
    ? null
    : Math.max(0, Math.min(xLabels - 1, Math.round((hoverX / CW) * (xLabels - 1))));
  const hoverPoint = hoverIdx != null ? gChart[hoverIdx] : null;
  const hoverDPoint = hoverIdx != null && isGan ? dChart[hoverIdx] ?? null : null;

  function handlePointer(e: ReactPointerEvent<SVGRectElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * CW;
    setHoverX(Math.max(0, Math.min(CW, relX)));
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <PanelHeader
        label="Loss Curve"
        right={
          <div style={{ display: "flex", gap: 10, fontSize: 10, fontFamily: "var(--font-mono)" }}>
            <span style={{ color: "var(--green)" }}>● generator</span>
            {isGan && <span style={{ color: "var(--blue)" }}>● discriminator</span>}
            {liveLoss != null && <span style={{ color: "var(--orange)" }}>◆ live</span>}
          </div>
        }
      />
      <div style={{ flex: 1, padding: "0 14px 10px", minHeight: 90, overflow: "hidden" }}>
        {empty ? (
          <EmptyChartState label="loss data" />
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%", display: "block" }} preserveAspectRatio="xMidYMid meet">
            <defs>
              <linearGradient id={`${uid}-gfill`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--green)" stopOpacity={0.28} />
                <stop offset="100%" stopColor="var(--green)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <g transform={`translate(${ML},0)`}>
              {yTicks.map((t) => {
                const y = CH - ((t - min) / (max - min || 1)) * CH;
                return (
                  <g key={`y${t}`}>
                    <line x1={0} y1={y} x2={CW} y2={y}
                      stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3 3" />
                    <text x={-8} y={y + 3.5} textAnchor="end" fill="var(--dim)"
                      fontSize={9.5} fontFamily="var(--font-mono)">
                      {fmtAxisLoss(t)}
                    </text>
                  </g>
                );
              })}
              {xTicks.map((t) => {
                const i = Math.round(t) - 1;
                if (i < 0 || i >= xLabels) return null;
                const x = (i / Math.max(xLabels - 1, 1)) * CW;
                return (
                  <text key={`x${t}`} x={x} y={CH + 13} textAnchor="middle" fill="var(--dim)"
                    fontSize={9} fontFamily="var(--font-mono)">
                    {windowStart + Math.round(t)}
                  </text>
                );
              })}

              {gChart.length > 1 && (
                <path d={areaPath(gChart, CH)} fill={`url(#${uid}-gfill)`} stroke="none" />
              )}
              {isGan && dChart.length > 1 && (
                <path d={smoothPath(dChart)} fill="none" stroke="var(--blue)" strokeWidth={1.6} strokeOpacity={0.85} strokeLinecap="round" />
              )}
              {gChart.length > 1 && (
                <path d={smoothPath(gChart)} fill="none" stroke="var(--green)" strokeWidth={1.8} strokeLinecap="round" />
              )}
              {/* A single epoch of data has nothing to draw a line through —
                  show it as a plain marker instead of an invisible 0-length path. */}
              {gChart.length === 1 && (
                <circle cx={gChart[0].x} cy={gChart[0].y} r={3.5} fill="var(--green)" />
              )}
              {livePoint && (
                <>
                  <circle cx={livePoint.x} cy={livePoint.y} r={7} fill="var(--orange)" fillOpacity={0.18}>
                    <animate attributeName="r" values="5;9;5" dur="1.6s" repeatCount="indefinite" />
                    <animate attributeName="fill-opacity" values="0.3;0;0.3" dur="1.6s" repeatCount="indefinite" />
                  </circle>
                  <circle cx={livePoint.x} cy={livePoint.y} r={3.5} fill="var(--orange)" />
                </>
              )}

              {hoverPoint && (
                <g pointerEvents="none">
                  <line x1={hoverPoint.x} y1={0} x2={hoverPoint.x} y2={CH}
                    stroke="var(--border2)" strokeWidth={1} strokeDasharray="2 2" />
                  <circle cx={hoverPoint.x} cy={hoverPoint.y} r={3.2} fill="var(--bg0)" stroke="var(--green)" strokeWidth={1.6} />
                  {hoverDPoint && (
                    <circle cx={hoverDPoint.x} cy={hoverDPoint.y} r={3.2} fill="var(--bg0)" stroke="var(--blue)" strokeWidth={1.6} />
                  )}
                </g>
              )}

              {hoverIdx != null && hoverPoint && (() => {
                const lines = [
                  { text: `epoch ${windowStart + hoverIdx + 1}`, color: "var(--muted)" },
                  { text: `gen ${fmtAxisLoss(gSeries[hoverIdx])}`, color: "var(--green)" },
                  ...(isGan && dSeries[hoverIdx] != null
                    ? [{ text: `disc ${fmtAxisLoss(dSeries[hoverIdx])}`, color: "var(--blue)" }]
                    : []),
                ];
                const boxW = 78;
                const boxH = lines.length * 12 + 8;
                const overflowsRight = hoverPoint.x + 10 + boxW > CW;
                const bx = overflowsRight ? hoverPoint.x - 10 - boxW : hoverPoint.x + 10;
                const by = Math.max(2, Math.min(CH - boxH - 2, hoverPoint.y - boxH / 2));
                return (
                  <g pointerEvents="none">
                    <rect x={bx} y={by} width={boxW} height={boxH} rx={4}
                      fill="var(--bg2)" stroke="var(--border2)" strokeWidth={1} />
                    {lines.map((l, i) => (
                      <text key={i} x={bx + 8} y={by + 14 + i * 12} fill={l.color}
                        fontSize={9.5} fontFamily="var(--font-mono)">
                        {l.text}
                      </text>
                    ))}
                  </g>
                );
              })()}

              {/* Transparent capture surface for hover/tooltip interaction. */}
              <rect
                x={0} y={0} width={CW} height={CH} fill="transparent"
                onPointerMove={handlePointer}
                onPointerLeave={() => setHoverX(null)}
                style={{ cursor: "crosshair" }}
              />
            </g>
          </svg>
        )}
      </div>
    </div>
  );
}