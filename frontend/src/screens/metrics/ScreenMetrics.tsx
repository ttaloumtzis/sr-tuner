import { useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useTrainingStore } from "../../store/trainingStore";
import { useModelStore } from "../../store/modelStore";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useToast } from "../../components/shell/ToastProvider";
import { PBar } from "../../components/ui/PBar";
import { Btn } from "../../components/ui/Btn";
import { cancelJob } from "../../lib/api";
import type { RunHistory } from "../../store/trainingStore";

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
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.5; v += step) {
    if (ticks.length >= count) break;
    if (v >= min - step * 0.01) ticks.push(v);
  }
  return ticks.length < 2 && max > min ? [min, max] : ticks;
}

function fmtAxisLoss(v: number): string {
  if (v >= 100) return v.toFixed(0);
  if (v >= 1) return v.toFixed(2);
  if (v >= 0.01) return v.toFixed(4);
  return v.toExponential(1);
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

function pointsToPath(pts: LinePoint[]): string {
  if (pts.length === 0) return "";
  return pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

// ── Training Status Bar (§12.2) ───────────────────────────────────────────

function TrainingStatusBar() {
  const status = useTrainingStore((s) => s.status);
  const iter = useTrainingStore((s) => s.iter);
  const finalEpoch = useTrainingStore((s) => s.finalEpoch);
  const activeTrainingRunId = useTrainingStore((s) => s.activeTrainingRunId);
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

  const dotColor =
    status === "running" ? "var(--green)" :
    status === "paused"  ? "var(--amber)" : "var(--dim)";

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12,
      padding: "8px 14px", background: "var(--bg1)",
      borderBottom: "1px solid var(--border)", flexShrink: 0,
    }}>
      <div style={{
        width: 8, height: 8, borderRadius: "50%", background: dotColor, flexShrink: 0,
        animation: status === "running" ? "pulse-dot 1.2s ease-in-out infinite" : "none",
      }} />

      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--muted)", flexShrink: 0 }}>
        {status === "running" ? "active" : status}
      </span>

      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--dim)", flexShrink: 0 }}>
        iter {iter.toLocaleString()}
      </span>

      <div style={{ flex: 1 }} />

      {status === "running" && (
        <Btn variant="solid" color="var(--red)" onClick={handleStop} disabled={cancelling}>
          {cancelling ? "cancelling…" : "Stop"}
        </Btn>
      )}

      {status === "done" && (
        <span style={{
          fontSize: 10, padding: "2px 8px", borderRadius: 3,
          background: "var(--blue-dim)", color: "var(--blue)",
          border: "1px solid rgba(90,171,240,0.27)", fontFamily: "var(--font-mono)",
        }}>
          training.complete received
        </span>
      )}

      <style>{`
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.4; transform: scale(0.75); }
        }
      `}</style>
    </div>
  );
}

// ── Progress Row (§new) ───────────────────────────────────────────────────

function ProgressRow() {
  const status = useTrainingStore((s) => s.status);
  const epoch = useTrainingStore((s) => s.epoch);
  const batch = useTrainingStore((s) => s.batch);
  const totalBatch = useTrainingStore((s) => s.totalBatch);
  const speed = useTrainingStore((s) => s.speed);
  const validationRunning = useTrainingStore((s) => s.validationRunning);
  const totalEpochs = useRunConfigStore((s) => s.schedule.totalEpochs);

  const epochPct = totalBatch > 0 ? (batch / totalBatch) * 100 : 0;
  const etaSec = status === "running" ? computeEtaSec(batch, totalBatch, epoch, totalEpochs, speed) : null;

  if (status === "idle") return null;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "6px 14px", background: "var(--bg1)",
      borderBottom: "1px solid var(--border)", flexShrink: 0,
      fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--dim)",
    }}>
      <span>Epoch {epoch} / {totalEpochs}</span>
      <span style={{ color: "var(--border)" }}>|</span>
      <span>
        {validationRunning
          ? `Validating epoch ${epoch}…`
          : `Batch ${batch} / ${totalBatch}`
        }
      </span>
      <div style={{ flex: 1, maxWidth: 160 }}>
        <PBar value={epochPct} max={100} color="var(--green)" height={5} />
      </div>
      <span style={{ minWidth: 32, textAlign: "right" }}>{epochPct.toFixed(0)}%</span>
      {etaSec != null && (
        <>
          <span style={{ color: "var(--border)" }}>|</span>
          <span>{formatEta(etaSec)}</span>
        </>
      )}
    </div>
  );
}

// ── Metric Cards (§12.3) ─────────────────────────────────────────────────

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  accent: string;
}

function MetricCard({ label, value, sub, accent }: MetricCardProps) {
  return (
    <div style={{
      flex: 1, minWidth: 80,
      background: "var(--bg1)", border: "1px solid var(--border)",
      borderTop: `2px solid ${accent}`, borderRadius: "var(--radius-md)",
      padding: "8px 10px",
    }}>
      <div style={{ fontSize: 9, color: "var(--muted)", fontFamily: "var(--font-mono)", marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontSize: 16, color: "var(--text)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)", marginTop: 1 }}>
          {sub}
        </div>
      )}
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
  const arch    = useModelStore((s) => s.architecture);
  const isGan   = arch === GAN_ARCH;

  return (
    <div style={{ display: "flex", gap: 6, padding: "6px 14px", flexShrink: 0, flexWrap: "wrap" }}>
      <MetricCard label="G LOSS"      value={fmt(gLoss)}          sub={isGan ? `disc ${fmt(dLoss)}` : "no disc"} accent="var(--green)"  />
      <MetricCard label="PSNR (dB)"   value={fmt(psnr, 2)}        accent="var(--blue)"   />
      <MetricCard label="SSIM"        value={fmt(ssim)}           accent="var(--cyan)"   />
      <MetricCard label="FULL PSNR"   value={fmt(fullPsnr, 2)}    accent="var(--green)"  />
      <MetricCard label="FULL SSIM"   value={fmt(fullSsim)}       accent="var(--teal)"   />
      <MetricCard label="GPU"         value={fmtPct(gpuUtil)}     accent="var(--amber)"  />
      <MetricCard label="EPOCH"       value={String(epoch)}       accent="var(--purple)" />
      <MetricCard label="SPEED"       value={fmt(speed, 2)}       accent="var(--muted)"  />
      <MetricCard label="CPU"         value={fmtPct(cpuUtil)}     accent="var(--teal)"   />
      <MetricCard label="RAM"         value={fmtGb(ramGb)}        accent="var(--pink)"   />
    </div>
  );
}

// ── Loss Curve SVG (§12.4) ────────────────────────────────────────────────

function LossCurve({ history }: { history: RunHistory | null }) {
  const arch        = useModelStore((s) => s.architecture);
  const liveLoss    = useTrainingStore((s) => s.liveLoss);

  const isGan = arch === GAN_ARCH;
  const W = 380; const H = 200;
  const ML = 48; const MR = 12;
  const CW = W - ML - MR;

  const gSeries = (history?.gLossHistory ?? []).slice(-CHART_WINDOW);
  const dSeries = isGan
    ? (history?.dLossHistory ?? []).filter((v): v is number => v != null).slice(-CHART_WINDOW)
    : [];
  const min = 0;
  const max = gSeries.length > 0 ? Math.max(...gSeries) * 1.05 : 2.0;

  const gChart = buildPoints(gSeries, CW, H, min, max);
  const dChart = buildPoints(dSeries, CW, H, min, max);

  const yTicks = niceTicks(min, max, 5);
  const xLabels = gSeries.length;
  const xTicks = xLabels > 0 ? niceTicks(1, xLabels, Math.min(xLabels, 6)) : [];

  const shiftX = (i: number) => ML + (i / Math.max(xLabels - 1, 1)) * CW;

  const livePoint = liveLoss != null
    ? { x: ML + CW, y: H - ((liveLoss - min) / (max - min || 1)) * H }
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ fontSize: 10, color: "var(--muted)", padding: "8px 12px 2px", fontFamily: "var(--font-mono)" }}>
        LOSS CURVE
      </div>
      <div style={{ display: "flex", gap: 10, padding: "0 12px 4px", fontSize: 10, fontFamily: "var(--font-mono)" }}>
        <span style={{ color: "var(--green)" }}>■ total</span>
        {isGan && <span style={{ color: "var(--blue)" }}>■ disc</span>}
        {liveLoss != null && <span style={{ color: "var(--orange)" }}>● live</span>}
      </div>
      <div style={{ flex: 1, padding: "4px 12px 8px", minHeight: 0 }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%", display: "block" }} preserveAspectRatio="xMidYMid meet">
          {/* Y-axis grid and labels */}
          {yTicks.map((t) => {
            const y = H - ((t - min) / (max - min || 1)) * H;
            return (
              <g key={`y${t}`}>
                <line x1={ML} y1={y} x2={ML + CW} y2={y}
                  stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3 3" />
                <text x={ML - 5} y={y + 3.5} textAnchor="end" fill="var(--dim)"
                  fontSize={10} fontFamily="var(--font-mono)">
                  {fmtAxisLoss(t)}
                </text>
              </g>
            );
          })}
          {/* X-axis epoch labels */}
          {xTicks.map((t) => {
            const i = Math.round(t) - 1;
            if (i < 0 || i >= xLabels) return null;
            const x = ML + shiftX(i) - ML;
            return (
              <text key={`x${t}`} x={x} y={H - 4} textAnchor="middle" fill="var(--dim)"
                fontSize={9} fontFamily="var(--font-mono)">
                {Math.round(t)}
              </text>
            );
          })}
          {/* Data lines */}
          {isGan && dChart.length > 1 && (
            <path d={pointsToPath(dChart)} fill="none" stroke="var(--blue)" strokeWidth={1.5} strokeOpacity={0.8} />
          )}
          {gChart.length > 1 && (
            <path d={pointsToPath(gChart)} fill="none" stroke="var(--green)" strokeWidth={1.5} />
          )}
          {livePoint && (
            <circle cx={livePoint.x} cy={livePoint.y} r={4}
              fill="var(--orange)" fillOpacity={0.9} />
          )}
        </svg>
      </div>
    </div>
  );
}

// ── PSNR / SSIM — two vertically stacked charts (§12.5) ──────────────────

function _subChart(
  label: string,
  series: number[],
  color: string,
  fmt: (v: number) => string,
  W: number, H: number, ML: number,
) {
  const CW = W - ML - 10;
  const R = 3;
  const count = series.length;

  if (count === 0) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flex: 1 }}>
        <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
          awaiting {label.toLowerCase()}…
        </span>
      </div>
    );
  }

  const mn = count > 0 ? Math.min(...series) : 0;
  const mx = count > 0 ? Math.max(...series) : 1;
  const ticks = niceTicks(mn, mx, 4);
  const mapX = (i: number) => ML + (i / Math.max(count - 1, 1)) * CW;
  const mapY = (v: number) => H - ((v - mn) / (mx - mn || 1)) * H;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%", display: "block" }} preserveAspectRatio="xMidYMid meet">
      {ticks.map((t) => {
        const y = mapY(t);
        return (
          <g key={t}>
            <line x1={ML} y1={y} x2={ML + CW} y2={y}
              stroke={color} strokeWidth={0.5} strokeOpacity={0.12} strokeDasharray="3 3" />
            <text x={ML - 4} y={y + 3} textAnchor="end" fill="var(--dim)"
              fontSize={9} fontFamily="var(--font-mono)">
              {fmt(t)}
            </text>
          </g>
        );
      })}
      {series.map((v, i) => (
        <circle key={i} cx={mapX(i)} cy={mapY(v)} r={R}
          fill={color} fillOpacity={0.8} />
      ))}
    </svg>
  );
}

function PsnrSsimChart({ history }: { history: RunHistory | null }) {
  const W = 400; const H = 75; const ML = 45;

  const psnrSeries = (history?.psnrHistory ?? []).slice(-CHART_WINDOW);
  const ssimSeries = (history?.ssimHistory ?? []).slice(-CHART_WINDOW);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div style={{ fontSize: 10, color: "var(--muted)", padding: "6px 12px 1px", fontFamily: "var(--font-mono)" }}>
          PSNR
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          {_subChart("PSNR", psnrSeries, "var(--green)", (v) => v.toFixed(2), W, H, ML)}
        </div>
      </div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div style={{ fontSize: 10, color: "var(--muted)", padding: "6px 12px 1px", fontFamily: "var(--font-mono)" }}>
          SSIM
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          {_subChart("SSIM", ssimSeries, "var(--blue)", (v) => v.toFixed(4), W, H, ML)}
        </div>
      </div>
    </div>
  );
}

// ── Hardware Panel (§12.6) ────────────────────────────────────────────────

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

  const rows = [
    ...(hasGpu ? [
      { label: "GPU %",
        value: fmtPct(gpuUtil),
        pct: gpuUtil,
        color: hwColor(gpuUtil, 80, 95) },
      { label: "VRAM",
        value: vram != null && vramTotalGb != null
          ? `${vram.toFixed(1)} / ${vramTotalGb.toFixed(0)} GB`
          : fmtGb(vram),
        pct: vram != null && vramTotalGb != null && vramTotalGb > 0
          ? (vram / vramTotalGb) * 100 : null,
        color: "var(--blue)" },
      { label: "GPU TEMP °C",
        value: temp != null ? `${temp.toFixed(0)}°C` : "—",
        pct: temp != null ? (temp / 110) * 100 : null,
        color: hwColor(temp, 75, 90) },
    ] : []),
    { label: "CPU %",
      value: fmtPct(cpuUtil),
      pct: cpuUtil,
      color: hwColor(cpuUtil, 80, 95) },
    { label: "RAM",
      value: ramGb != null && ramTotalGb != null
        ? `${ramGb.toFixed(1)} / ${ramTotalGb.toFixed(0)} GB`
        : fmtGb(ramGb),
      pct: ramGb != null && ramTotalGb != null && ramTotalGb > 0
        ? (ramGb / ramTotalGb) * 100 : null,
      color: "var(--pink)" },
  ];

  return (
    <div style={{ padding: "8px 12px", display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>HARDWARE</div>
      {rows.map(({ label, value, pct, color }) => (
        <div key={label}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
            <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>{label}</span>
            <span style={{ fontSize: 10, color: "var(--text)", fontFamily: "var(--font-mono)" }}>{value}</span>
          </div>
          <PBar value={pct ?? 0} max={100} color={color} height={4} />
        </div>
      ))}
    </div>
  );
}

// ── Validation Frames Panel (§12.7, §12.11) ───────────────────────────────

function ValidationPanel() {
  const frames = useTrainingStore((s) => s.validationFrames);

  const cells: { label: string; path: string | null }[] = [
    { label: "LR",   path: frames?.lrPath   ?? null },
    { label: "SR",   path: frames?.srPath   ?? null },
    { label: "GT",   path: frames?.gtPath   ?? null },
    { label: "Diff", path: frames?.diffPath ?? null },
  ];

  return (
    <div style={{ padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6, height: "100%" }}>
      <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>VALIDATION FRAMES</div>
      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr",
        gap: 4, flex: 1, overflow: "auto", minHeight: 0,
      }}>
        {cells.map(({ label, path }) => (
          <div key={label} style={{
            background: "var(--bg2)", border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)", overflow: "hidden",
            display: "flex", alignItems: "center", justifyContent: "center",
            position: "relative", minHeight: 60,
          }}>
            {path ? (
              <img
                src={convertFileSrc(path)}
                alt={label}
                loading="lazy"
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
              />
            ) : (
              <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
                {frames ? "no GT" : label}
              </span>
            )}
            <span style={{
              position: "absolute", top: 3, left: 4, fontSize: 9,
              color: "var(--muted)", fontFamily: "var(--font-mono)",
              background: "rgba(13,15,17,0.7)", padding: "1px 4px", borderRadius: 2,
            }}>
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── ScreenMetrics — §12.1 layout ─────────────────────────────────────────

export function ScreenMetrics() {
  const lossHistory    = useTrainingStore((s) => s.lossHistory);
  const dLossHistory   = useTrainingStore((s) => s.dLossHistory);
  const totalLossHist  = useTrainingStore((s) => s.totalLossHistory);
  const psnrHistory    = useTrainingStore((s) => s.psnrHistory);
  const ssimHistory    = useTrainingStore((s) => s.ssimHistory);

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
      <MetricCards />
      <div style={{
        flex: 1, display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gridTemplateRows: "1fr 1fr",
        gap: 1, minHeight: 0,
        background: "var(--border)",
      }}>
        <div style={{ background: "var(--bg1)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <LossCurve history={displayedHistory} />
        </div>
        <div style={{ background: "var(--bg1)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <PsnrSsimChart history={displayedHistory} />
        </div>
        <div style={{ background: "var(--bg1)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <HardwarePanel />
        </div>
        <div style={{ background: "var(--bg1)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <ValidationPanel />
        </div>
      </div>
    </div>
  );
}
