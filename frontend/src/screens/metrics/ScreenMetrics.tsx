import { useEffect, useId, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { useTrainingStore } from "../../store/trainingStore";
import { useModelStore } from "../../store/modelStore";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useToast } from "../../components/shell/ToastProvider";
import { Btn } from "../../components/ui/Btn";
import { cancelJob } from "../../lib/api";
import type { RunHistory } from "../../store/trainingStore";
import { ValidationPanel } from "../../components/metrics/ValidationPanel";
import { RadialGauge, TempBadge } from "../../components/metrics/Gauges";

const CHART_WINDOW = 60;
const GAN_ARCH = "rrdb_esrgan";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(n: number | null, decimals = 4): string {
  if (n == null) return "—";
  return n.toFixed(decimals);
}

function fmtPct(n: number | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)}%`;
}

function fmtGb(n: number | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)} GB`;
}

function formatEta(sec: number | null): string {
  if (sec == null || sec <= 0) return "";
  if (sec < 60) return `ETA ${Math.round(sec)}s`;
  if (sec < 3600) return `ETA ${Math.round(sec / 60)}m`;
  return `ETA ${(sec / 3600).toFixed(1)}h`;
}

function computeEtaSec(
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

function niceNumber(x: number): number {
  const exp = Math.floor(Math.log10(x));
  const frac = x / 10 ** exp;
  if (frac <= 1) return 10 ** exp;
  if (frac <= 2) return 2 * 10 ** exp;
  if (frac <= 5) return 5 * 10 ** exp;
  return 10 * 10 ** exp;
}

function niceTicks(min: number, max: number, count: number): number[] {
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

function fmtAxisLoss(v: number): string {
  if (v >= 100) return v.toFixed(0);
  if (v >= 1) return v.toFixed(2);
  if (v >= 0.01) return v.toFixed(4);
  return v.toExponential(1);
}

// Simple trend indicator: compares last value against value ~5 samples back.
function trendOf(series: number[], invert = false): { dir: "up" | "down" | "flat"; pct: number } | null {
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

interface LinePoint { x: number; y: number }

function buildPoints(values: number[], w: number, h: number, min: number, max: number): LinePoint[] {
  const range = max - min || 1;
  return values.map((v, i) => ({
    x: (i / Math.max(values.length - 1, 1)) * w,
    y: h - ((v - min) / range) * h,
  }));
}

// Smooth Catmull-Rom → cubic Bezier path, so lines read as flowing curves
// rather than jagged polylines. Falls back to a straight segment for <3 pts.
function smoothPath(pts: LinePoint[]): string {
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

function areaPath(pts: LinePoint[], h: number): string {
  if (pts.length === 0) return "";
  const line = smoothPath(pts);
  const last = pts[pts.length - 1];
  const first = pts[0];
  return `${line} L${last.x.toFixed(1)},${h} L${first.x.toFixed(1)},${h} Z`;
}

// ── Section header (shared) ───────────────────────────────────────────────

function PanelHeader({ label, right }: { label: string; right?: React.ReactNode }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "9px 14px 7px", flexShrink: 0,
    }}>
      <span style={{
        fontSize: 10, letterSpacing: "0.06em", color: "var(--muted)",
        fontFamily: "var(--font-mono)", fontWeight: 600, textTransform: "uppercase",
      }}>
        {label}
      </span>
      {right}
    </div>
  );
}

// ── Training Status Bar ───────────────────────────────────────────────────

const STATUS_META: Record<string, { label: string; color: string }> = {
  running: { label: "Training active", color: "var(--green)" },
  paused: { label: "Paused", color: "var(--amber)" },
  done: { label: "Complete", color: "var(--blue)" },
  failed: { label: "Failed", color: "var(--red)" },
  disconnected: { label: "Disconnected", color: "var(--red)" },
  idle: { label: "Idle", color: "var(--dim)" },
};

function TrainingStatusBar() {
  const status = useTrainingStore((s) => s.status);
  const iter = useTrainingStore((s) => s.iter);
  const finalEpoch = useTrainingStore((s) => s.finalEpoch);
  const activeTrainingRunId = useTrainingStore((s) => s.activeTrainingRunId);
  const bestPsnr = useTrainingStore((s) => s.bestPsnr);
  const { show } = useToast();
  const toastFiredRef = useRef(false);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    if (status === "done" && finalEpoch != null && !toastFiredRef.current) {
      toastFiredRef.current = true;
      show("success", `Training complete — ${finalEpoch} epochs finished`, 6000);
    }
    if (status !== "done") toastFiredRef.current = false;
  }, [status, finalEpoch, show]);

  const handleStop = async () => {
    if (!activeTrainingRunId || cancelling) return;
    setCancelling(true);
    try {
      await cancelJob(activeTrainingRunId);
    } catch {
      show("error", "Failed to cancel training");
    }
    setCancelling(false);
  };

  const meta = STATUS_META[status] ?? STATUS_META.idle;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 14,
      padding: "10px 16px", flexShrink: 0,
      background: "linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%)",
      borderBottom: "1px solid var(--border)",
    }}>
      <div style={{ position: "relative", width: 10, height: 10, flexShrink: 0 }}>
        <div style={{
          position: "absolute", inset: 0, borderRadius: "50%", background: meta.color,
        }} />
        {status === "running" && (
          <div style={{
            position: "absolute", inset: 0, borderRadius: "50%", background: meta.color,
            animation: "metrics-ping 1.6s cubic-bezier(0,0,0.2,1) infinite",
          }} />
        )}
      </div>

      <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", flexShrink: 0 }}>
        {meta.label}
      </span>

      <span style={{
        fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--dim)",
        padding: "2px 7px", background: "var(--bg2)", borderRadius: 20, flexShrink: 0,
      }}>
        iter {iter.toLocaleString()}
      </span>

      {bestPsnr != null && (
        <span style={{
          fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--green)",
          padding: "2px 7px", background: "var(--green-dim)", borderRadius: 20, flexShrink: 0,
          border: "1px solid rgba(77,186,127,0.25)",
        }}>
          ★ best PSNR {bestPsnr.toFixed(2)} dB
        </span>
      )}

      <div style={{ flex: 1 }} />

      {status === "running" && (
        <Btn variant="solid" color="var(--red)" onClick={handleStop} disabled={cancelling}>
          {cancelling ? "Cancelling…" : "Stop training"}
        </Btn>
      )}

      {status === "done" && (
        <span style={{
          fontSize: 10, padding: "3px 10px", borderRadius: 20,
          background: "var(--blue-dim)", color: "var(--blue)",
          border: "1px solid rgba(90,171,240,0.27)", fontFamily: "var(--font-mono)",
        }}>
          training.complete received
        </span>
      )}

      <style>{`
        @keyframes metrics-ping {
          0%   { opacity: 0.55; transform: scale(1); }
          75%, 100% { opacity: 0; transform: scale(2.4); }
        }
        @keyframes metrics-fade-in {
          from { opacity: 0; transform: translateY(3px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

// ── Progress Row ───────────────────────────────────────────────────────────

function ProgressRow() {
  const status = useTrainingStore((s) => s.status);
  const epoch = useTrainingStore((s) => s.epoch);
  const batch = useTrainingStore((s) => s.batch);
  const totalBatch = useTrainingStore((s) => s.totalBatch);
  const speed = useTrainingStore((s) => s.speed);
  const validationRunning = useTrainingStore((s) => s.validationRunning);
  const totalEpochs = useRunConfigStore((s) => s.schedule.totalEpochs);

  const epochPct = totalBatch > 0 ? (batch / totalBatch) * 100 : 0;
  const overallPct = totalEpochs > 0 ? ((epoch - 1 + epochPct / 100) / totalEpochs) * 100 : 0;
  const etaSec = status === "running" ? computeEtaSec(batch, totalBatch, epoch, totalEpochs, speed) : null;

  if (status === "idle") return null;

  return (
    <div style={{
      display: "flex", flexDirection: "column", gap: 6,
      padding: "9px 16px", background: "var(--bg1)",
      borderBottom: "1px solid var(--border)", flexShrink: 0,
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 10,
        fontSize: 10.5, fontFamily: "var(--font-mono)", color: "var(--dim)",
      }}>
        <span style={{ color: "var(--muted)" }}>Epoch {epoch} / {totalEpochs}</span>
        <span style={{ color: "var(--border)" }}>·</span>
        <span>
          {validationRunning ? `validating epoch ${epoch}…` : `batch ${batch} / ${totalBatch}`}
        </span>
        <div style={{ flex: 1 }} />
        {speed != null && <span>{speed.toFixed(2)} it/s</span>}
        {etaSec != null && (
          <span style={{
            color: "var(--amber)", background: "var(--amber-dim)",
            padding: "1px 7px", borderRadius: 20,
          }}>
            {formatEta(etaSec)}
          </span>
        )}
      </div>

      {/* Overall run progress — modern gradient track with epoch progress overlaid */}
      <div style={{ position: "relative", height: 6, borderRadius: 3, background: "var(--bg3)", overflow: "hidden" }}>
        <div style={{
          position: "absolute", inset: 0, width: `${Math.min(100, overallPct)}%`,
          background: "linear-gradient(90deg, var(--green) 0%, var(--cyan) 100%)",
          borderRadius: 3, transition: "width 0.4s ease",
        }} />
        {validationRunning && (
          <div style={{
            position: "absolute", inset: 0, background: "var(--blue)", opacity: 0.35,
            animation: "metrics-scan 1.4s linear infinite",
          }} />
        )}
      </div>

      <style>{`
        @keyframes metrics-scan {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}

// ── Metric Cards ────────────────────────────────────────────────────────────

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  accent: string;
  trend?: { dir: "up" | "down" | "flat"; pct: number } | null;
  sparkline?: number[];
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null;
  const w = 56; const h = 20;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pts = buildPoints(values.slice(-20), w, h, min, max);
  return (
    <svg width={w} height={h} style={{ display: "block", flexShrink: 0 }}>
      <path d={smoothPath(pts)} fill="none" stroke={color} strokeWidth={1.4} strokeOpacity={0.8} strokeLinecap="round" />
    </svg>
  );
}

function MetricCard({ label, value, sub, accent, trend, sparkline }: MetricCardProps) {
  const trendColor = trend?.dir === "up" ? "var(--green)" : trend?.dir === "down" ? "var(--red)" : "var(--dim)";
  const trendArrow = trend?.dir === "up" ? "▲" : trend?.dir === "down" ? "▼" : "";

  return (
    <div style={{
      flex: "1 1 130px", minWidth: 130,
      background: "linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 130%)",
      border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
      padding: "10px 12px", position: "relative", overflow: "hidden",
      animation: "metrics-fade-in 0.25s ease",
    }}>
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 2,
        background: accent, opacity: 0.85,
      }} />
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{
            fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--font-mono)",
            letterSpacing: "0.05em", marginBottom: 5, whiteSpace: "nowrap",
          }}>
            {label}
          </div>
          <div style={{
            fontSize: 18, color: "var(--text)", fontFamily: "var(--font-mono)",
            fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.01em",
          }}>
            {value}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 3, minHeight: 12 }}>
            {sub && (
              <span style={{ fontSize: 9.5, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>{sub}</span>
            )}
            {trend && trend.dir !== "flat" && (
              <span style={{ fontSize: 9.5, color: trendColor, fontFamily: "var(--font-mono)" }}>
                {trendArrow} {trend.pct.toFixed(1)}%
              </span>
            )}
          </div>
        </div>
        {sparkline && sparkline.length >= 2 && (
          <Sparkline values={sparkline} color={accent} />
        )}
      </div>
    </div>
  );
}

function MetricCards() {
  const gLoss   = useTrainingStore((s) => s.gLoss);
  const dLoss   = useTrainingStore((s) => s.dLoss);
  const psnr    = useTrainingStore((s) => s.psnr);
  const ssim    = useTrainingStore((s) => s.ssim);
  const fullPsnr = useTrainingStore((s) => s.fullPsnr);
  const fullSsim = useTrainingStore((s) => s.fullSsim);
  const cpuUtil = useTrainingStore((s) => s.cpuUtil);
  const ramGb   = useTrainingStore((s) => s.ramGb);
  const epoch   = useTrainingStore((s) => s.epoch);
  const speed   = useTrainingStore((s) => s.speed);
  const gpuUtil = useTrainingStore((s) => s.gpuUtil);
  const lossHistory = useTrainingStore((s) => s.lossHistory);
  const psnrHistory = useTrainingStore((s) => s.psnrHistory);
  const ssimHistory = useTrainingStore((s) => s.ssimHistory);
  const arch    = useModelStore((s) => s.architecture);
  const isGan   = arch === GAN_ARCH;

  return (
    <div style={{ display: "flex", gap: 8, padding: "10px 16px", flexShrink: 0, flexWrap: "wrap" }}>
      <MetricCard label="G LOSS" value={fmt(gLoss)} sub={isGan ? `disc ${fmt(dLoss)}` : "no disc"}
        accent="var(--green)" trend={trendOf(lossHistory, true)} sparkline={lossHistory} />
      <MetricCard label="PSNR (dB)" value={fmt(psnr, 2)} accent="var(--blue)"
        trend={trendOf(psnrHistory)} sparkline={psnrHistory} />
      <MetricCard label="SSIM" value={fmt(ssim)} accent="var(--cyan)"
        trend={trendOf(ssimHistory)} sparkline={ssimHistory} />
      <MetricCard label="FULL PSNR" value={fmt(fullPsnr, 2)} accent="var(--green)" />
      <MetricCard label="FULL SSIM" value={fmt(fullSsim)} accent="var(--teal)" />
      <MetricCard label="GPU" value={fmtPct(gpuUtil)} accent="var(--amber)" />
      <MetricCard label="EPOCH" value={String(epoch)} accent="var(--purple)" />
      <MetricCard label="SPEED" value={`${fmt(speed, 2)} it/s`} accent="var(--muted)" />
      <MetricCard label="CPU" value={fmtPct(cpuUtil)} accent="var(--teal)" />
      <MetricCard label="RAM" value={fmtGb(ramGb)} accent="var(--pink)" />
    </div>
  );
}

// ── Loss Curve ──────────────────────────────────────────────────────────────

function LossCurve({ history }: { history: RunHistory | null }) {
  const uid = useId();
  const arch        = useModelStore((s) => s.architecture);
  const liveLoss    = useTrainingStore((s) => s.liveLoss);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const isGan = arch === GAN_ARCH;
  const W = 380; const H = 190;
  const ML = 46; const MR = 12; const MB = 18;
  const CW = W - ML - MR;
  const CH = H - MB;

  const gSeries = (history?.gLossHistory ?? []).slice(-CHART_WINDOW);
  const dSeries = isGan
    ? (history?.dLossHistory ?? []).filter((v): v is number => v != null).slice(-CHART_WINDOW)
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
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
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
      <div style={{ flex: 1, padding: "0 14px 10px", minHeight: 90 }}>
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
                    {Math.round(t)}
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
                  { text: `epoch ${hoverIdx + 1}`, color: "var(--muted)" },
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

// ── PSNR / SSIM ──────────────────────────────────────────────────────────────

function EmptyChartState({ label }: { label: string }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      height: "100%", flexDirection: "column", gap: 4,
    }}>
      <div style={{
        width: 26, height: 26, borderRadius: "50%", border: "1.5px dashed var(--border2)",
      }} />
      <span style={{ fontSize: 10.5, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
        awaiting {label}…
      </span>
    </div>
  );
}

function SubChart({ uid, chartKey, series, color, fullSeries, fullColor }: {
  uid: string; chartKey: string; series: number[]; color: string;
  fullSeries?: number[]; fullColor?: string;
}) {
  const W = 400; const H = 72; const ML = 42;
  const CW = W - ML - 12;
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
  const mapY = (v: number) => H - ((v - mn) / (mx - mn || 1)) * H;
  const pts = series.map((v, i) => ({ x: mapX(i) - ML, y: mapY(v) }));

  const fullPts = hasFull
    ? fullSeries!.map((v, i) => ({ x: mapX(i) - ML, y: mapY(v) }))
    : [];

  const hoverIdx = hoverX == null ? null
    : Math.max(0, Math.min(count - 1, Math.round((hoverX / CW) * (count - 1))));
  const hoverPt = hoverIdx != null ? pts[hoverIdx] : null;
  const hoverFullVal = hoverIdx != null && hasFull ? fullSeries![hoverIdx] : null;

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
                {fmt(t)}
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
          { text: `epoch ${hoverIdx + 1}`, col: "var(--muted)" },
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

function PsnrSsimChart({ history }: { history: RunHistory | null }) {
  const uid = useId();
  const fullPsnrHistory = useTrainingStore((s) => s.fullPsnrHistory);
  const fullSsimHistory = useTrainingStore((s) => s.fullSsimHistory);

  const psnrSeries = (history?.psnrHistory ?? []).slice(-CHART_WINDOW);
  const ssimSeries = (history?.ssimHistory ?? []).slice(-CHART_WINDOW);
  const fullPsnrSeries = fullPsnrHistory.slice(-CHART_WINDOW);
  const fullSsimSeries = fullSsimHistory.slice(-CHART_WINDOW);

  const hasFull = fullPsnrSeries.length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <PanelHeader
        label="Quality Metrics"
        right={hasFull ? (
          <div style={{ display: "flex", gap: 8, fontSize: 9.5, fontFamily: "var(--font-mono)" }}>
            <span style={{ color: "var(--green)" }}>● PSNR</span>
            <span style={{ color: "var(--teal)" }}>▬▬ full</span>
            <span style={{ color: "var(--blue)" }}>● SSIM</span>
            <span style={{ color: "var(--purple)" }}>▬▬ full</span>
          </div>
        ) : undefined}
      />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, padding: "0 14px 10px", gap: 8 }}>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--font-mono)", marginBottom: 2 }}>
            PSNR (dB)
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <SubChart uid={uid} chartKey="psnr" series={psnrSeries} color="var(--green)"
              fullSeries={fullPsnrSeries} fullColor="var(--teal)" />
          </div>
        </div>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--font-mono)", marginBottom: 2 }}>
            SSIM
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <SubChart uid={uid} chartKey="ssim" series={ssimSeries} color="var(--blue)"
              fullSeries={fullSsimSeries} fullColor="var(--purple)" />
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Hardware Panel ────────────────────────────────────────────────────────

function hwColor(val: number | null, warn: number, crit: number): string {
  if (val == null) return "var(--dim)";
  if (val >= crit) return "var(--red)";
  if (val >= warn) return "var(--amber)";
  return "var(--green)";
}

function HardwarePanel() {
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

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <PanelHeader label="Hardware" right={!hasGpu ? (
        <span style={{ fontSize: 9.5, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>no GPU detected</span>
      ) : undefined} />
      <div style={{
        padding: "2px 28px 10px", display: "flex", flexWrap: "wrap", gap: 10,
        flex: 1, justifyContent: "space-evenly", alignContent: "center",
      }}>
        {hasGpu && (
          <RadialGauge size={88} label="GPU" value={fmtPct(gpuUtil)} pct={gpuUtil} color={hwColor(gpuUtil, 80, 95)} />
        )}
        {hasGpu && (
          <RadialGauge size={88} label="VRAM" value={vramPct != null ? `${Math.round(vramPct)}%` : "—"}
            pct={vramPct} color="var(--blue)" sub={vramLabel} />
        )}
        {hasGpu && <TempBadge size={88} temp={temp} />}
        <RadialGauge size={88} label="CPU" value={fmtPct(cpuUtil)} pct={cpuUtil} color={hwColor(cpuUtil, 80, 95)} />
        <RadialGauge size={88} label="RAM" value={ramPct != null ? `${Math.round(ramPct)}%` : "—"}
          pct={ramPct} color="var(--pink)" sub={ramLabel} />
      </div>
    </div>
  );
}



// ── Resizable split (vertical) ───────────────────────────────────────────

function ResizableSplit({ top, bottom, defaultRatio = 0.5, minPx = 80 }: {
  top: ReactNode;
  bottom: ReactNode;
  defaultRatio?: number;
  minPx?: number;
}) {
  const [ratio, setRatio] = useState(defaultRatio);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startY: number; startRatio: number } | null>(null);

  const onMouseDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragRef.current = { startY: e.clientY, startRatio: ratio };
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const delta = ev.clientY - dragRef.current.startY;
      let r = dragRef.current.startRatio + delta / rect.height;
      r = Math.max(minPx / rect.height, Math.min(1 - minPx / rect.height, r));
      setRatio(r);
    };
    const onUp = () => {
      dragRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  return (
    <div ref={containerRef} style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ flex: ratio, minHeight: 0, overflow: "hidden" }}>
        {top}
      </div>
      <div
        onMouseDown={onMouseDown}
        style={{
          flexShrink: 0, height: 8, cursor: "row-resize", position: "relative",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        <div style={{
          width: 28, height: 3, borderRadius: 2, background: "var(--border2)",
          transition: "background 0.15s ease",
        }} />
      </div>
      <div style={{ flex: 1 - ratio, minHeight: 0, overflow: "hidden" }}>
        {bottom}
      </div>
    </div>
  );
}

// ── ScreenMetrics ────────────────────────────────────────────────────────

function IdleState() {
  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", gap: 10, color: "var(--dim)",
    }}>
      <div style={{
        width: 44, height: 44, borderRadius: "50%",
        border: "2px dashed var(--border2)",
      }} />
      <span style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: "var(--muted)" }}>
        No training run is active
      </span>
      <span style={{ fontSize: 10.5, fontFamily: "var(--font-mono)", color: "var(--dim)", maxWidth: 320, textAlign: "center" }}>
        Start a run from the Training tab to see live loss, quality, and hardware metrics here.
      </span>
    </div>
  );
}

export function ScreenMetrics() {
  const status          = useTrainingStore((s) => s.status);
  const lossHistory     = useTrainingStore((s) => s.lossHistory);
  const dLossHistory    = useTrainingStore((s) => s.dLossHistory);
  const totalLossHist   = useTrainingStore((s) => s.totalLossHistory);
  const psnrHistory     = useTrainingStore((s) => s.psnrHistory);
  const ssimHistory     = useTrainingStore((s) => s.ssimHistory);

  const displayedHistory: RunHistory | null = {
    gLossHistory: lossHistory, dLossHistory, totalLossHistory: totalLossHist, psnrHistory, ssimHistory,
  };

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      width: "100%", height: "100%", overflow: "hidden", background: "var(--bg0)",
    }}>
      <TrainingStatusBar />
      <ProgressRow />

      {status === "idle" ? (
        <IdleState />
      ) : (
        <>
          <MetricCards />
          <div style={{
            flex: 1, display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gridTemplateRows: "1fr 1fr",
            gap: 10, minHeight: 0,
            padding: "0 16px 16px",
          }}>
            <div style={{
              background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
              overflow: "hidden", display: "flex", flexDirection: "column",
            }}>
              <LossCurve history={displayedHistory} />
            </div>
            <div style={{
              background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
              overflow: "hidden", display: "flex", flexDirection: "column",
            }}>
              <PsnrSsimChart history={displayedHistory} />
            </div>
            <div style={{
              gridRow: "1 / 3", gridColumn: "2",
              background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
              overflow: "hidden", display: "flex", flexDirection: "column",
            }}>
              <ResizableSplit
                top={<HardwarePanel />}
                bottom={<ValidationPanel />}
                defaultRatio={0.25}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}