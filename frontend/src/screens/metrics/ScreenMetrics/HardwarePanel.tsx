import type { ReactNode } from "react";
import { useTrainingStore } from "../../../store/trainingStore";
import { RadialGauge, TempBadge } from "../../../components/metrics/Gauges";
import { PanelHeader } from "./PanelHeader";
import { fmtGb, fmtPct, trendOf } from "./chartUtils";
import { useRollingHistory, Sparkline } from "./MetricPrimitives";

function hwColor(val: number | null, warn: number, crit: number): string {
  if (val == null) return "var(--dim)";
  if (val >= crit) return "var(--red)";
  if (val >= warn) return "var(--amber)";
  return "var(--green)";
}

function HardwareTile({ children, color, history }: {
  children: ReactNode; color: string; history: number[];
}) {
  const trend = trendOf(history);
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
      background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
      padding: "18px 20px 14px", minWidth: 130,
    }}>
      {children}
      <div style={{ display: "flex", alignItems: "center", gap: 6, minHeight: 20 }}>
        <Sparkline values={history} color={color} width={54} height={20} padding={2.5} />
        {trend && trend.dir !== "flat" && (
          <span style={{
            fontSize: 9.5, fontFamily: "var(--font-mono)",
            color: trend.dir === "up" ? "var(--green)" : "var(--red)",
          }}>
            {trend.dir === "up" ? "▲" : "▼"} {trend.pct.toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

export function HardwarePanel() {
  const gpuUtil = useTrainingStore((s) => s.gpuUtil);
  const vram    = useTrainingStore((s) => s.vram);
  const vramTotalGb = useTrainingStore((s) => s.vramTotalGb);
  const temp    = useTrainingStore((s) => s.temp);
  const cpuUtil = useTrainingStore((s) => s.cpuUtil);
  const ramGb   = useTrainingStore((s) => s.ramGb);
  const ramTotalGb = useTrainingStore((s) => s.ramTotalGb);

  const hasGpu = gpuUtil != null || vram != null || temp != null;

  const vramPct = vram != null && vramTotalGb != null && vramTotalGb > 0 ? (vram / vramTotalGb) * 100 : null;
  const vramLabel = vram != null && vramTotalGb != null
    ? `${vram.toFixed(1)} / ${vramTotalGb.toFixed(0)} GB`
    : fmtGb(vram);
  const ramPct = ramGb != null && ramTotalGb != null && ramTotalGb > 0 ? (ramGb / ramTotalGb) * 100 : null;
  const ramLabel = ramGb != null && ramTotalGb != null
    ? `${ramGb.toFixed(1)} / ${ramTotalGb.toFixed(0)} GB`
    : fmtGb(ramGb);

  // Rolling windows drive the trend arrow + sparkline under each gauge —
  // these readings aren't stored as history arrays, just live snapshots.
  const gpuHistory  = useRollingHistory(gpuUtil);
  const vramHistory = useRollingHistory(vramPct);
  const tempHistory = useRollingHistory(temp);
  const cpuHistory  = useRollingHistory(cpuUtil);
  const ramHistory  = useRollingHistory(ramPct);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <PanelHeader label="Hardware" right={
        hasGpu ? (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ position: "relative", width: 6, height: 6 }}>
              <div style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "var(--green)" }} />
              <div style={{
                position: "absolute", inset: 0, borderRadius: "50%", background: "var(--green)",
                animation: "hw-ping 1.8s cubic-bezier(0,0,0.2,1) infinite",
              }} />
            </div>
            <span style={{ fontSize: 9.5, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>live</span>
          </div>
        ) : (
          <span style={{ fontSize: 9.5, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>no GPU detected</span>
        )
      } />
      <div style={{
        padding: "10px 28px 20px", display: "flex", flexWrap: "wrap", gap: 14,
        flex: 1, justifyContent: "center", alignContent: "center",
      }}>
        {hasGpu && (
          <HardwareTile color="var(--red)" history={gpuHistory}>
            <RadialGauge size={104} label="GPU" value={fmtPct(gpuUtil)} pct={gpuUtil} color={hwColor(gpuUtil, 80, 95)} />
          </HardwareTile>
        )}
        {hasGpu && (
          <HardwareTile color="var(--blue)" history={vramHistory}>
            <RadialGauge size={104} label="VRAM" value={vramPct != null ? `${Math.round(vramPct)}%` : "—"}
              pct={vramPct} color="var(--blue)" sub={vramLabel} />
          </HardwareTile>
        )}
        {hasGpu && (
          <HardwareTile color="var(--green)" history={tempHistory}>
            <TempBadge size={104} temp={temp} />
          </HardwareTile>
        )}
        <HardwareTile color="var(--amber)" history={cpuHistory}>
          <RadialGauge size={104} label="CPU" value={fmtPct(cpuUtil)} pct={cpuUtil} color={hwColor(cpuUtil, 80, 95)} />
        </HardwareTile>
        <HardwareTile color="var(--pink)" history={ramHistory}>
          <RadialGauge size={104} label="RAM" value={ramPct != null ? `${Math.round(ramPct)}%` : "—"}
            pct={ramPct} color="var(--pink)" sub={ramLabel} />
        </HardwareTile>
      </div>

      <style>{`
        @keyframes hw-ping {
          0%   { opacity: 0.55; transform: scale(1); }
          75%, 100% { opacity: 0; transform: scale(3); }
        }
      `}</style>
    </div>
  );
}