// §12 Live Metrics Screen
// Tasks: 12.1–12.12, §23.2 RunSelectorPanel sidebar

import { useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useTrainingStore } from "../../store/trainingStore";
import { useUiStore } from "../../store/uiStore";
import { useProjectStore } from "../../store/projectStore";
import { useModelStore } from "../../store/modelStore";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useToast } from "../../components/shell/ToastProvider";
import { StopConfirmDialog } from "../../components/shell/StopConfirmDialog";
import { RunSelectorPanel } from "../../components/shell/RunSelectorPanel";
import { RunComparisonPanel } from "../../components/metrics/RunComparisonPanel";
import { PBar } from "../../components/ui/PBar";
import { Btn } from "../../components/ui/Btn";
import { sendToSidecar } from "../../lib/ipc";
import type { RunHistory } from "../../store/trainingStore";

// ── Constants ─────────────────────────────────────────────────────────────

const CHART_WINDOW = 60;
const GAN_ARCH = "Real-ESRGAN";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(n: number | null, decimals = 4): string {
  if (n == null) return "—";
  return n.toFixed(decimals);
}

function fmtPct(n: number | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)}%`;
}

function formatEta(sec: number | null): string {
  if (sec == null || sec <= 0) return "";
  if (sec < 60) return `ETA ${Math.round(sec)}s`;
  if (sec < 3600) return `ETA ${Math.round(sec / 60)}m`;
  return `ETA ${(sec / 3600).toFixed(1)}h`;
}

// §12.9 — (totalIter - currentIter) / currentSpeed
function computeEtaSec(
  iter: number,
  epoch: number,
  speed: number | null,
  totalEpochs: number
): number | null {
  if (!speed || speed <= 0 || epoch <= 0 || iter <= 0) return null;
  const itersPerEpoch = iter / epoch;
  const totalIter = totalEpochs * itersPerEpoch;
  const remaining = totalIter - iter;
  return remaining > 0 ? remaining / speed : null;
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
  const activeId = useTrainingStore((s) => s.activeTrainingRunId);
  const iter = useTrainingStore((s) => s.iter);
  const finalEpoch = useTrainingStore((s) => s.finalEpoch);
  const displayedId = useUiStore((s) => s.displayedRunId);
  const project = useProjectStore((s) => s.project);
  const { show } = useToast();
  const toastFiredRef = useRef(false);
  // §19.6 — Stop confirmation dialog state
  const [stopDialogOpen, setStopDialogOpen] = useState(false);

  // Emit completion toast once when status transitions to "done"
  useEffect(() => {
    if (status === "done" && finalEpoch != null && !toastFiredRef.current) {
      toastFiredRef.current = true;
      show("success", `Training complete — ${finalEpoch} epochs finished`, 6000);
    }
    if (status !== "done") toastFiredRef.current = false;
  }, [status, finalEpoch, show]);

  const activeRun = project?.runs.find((r) => r.run_id === activeId);
  const isHistorical = displayedId !== null && displayedId !== activeId;
  const isActive = status === "running" || status === "paused";

  const handlePause = () => {
    if (!activeId) return;
    sendToSidecar({ type: "training.pause", run_id: activeId }).catch(() => {});
  };
  const handleResume = () => {
    if (!activeId) return;
    sendToSidecar({ type: "training.resume", run_id: activeId }).catch(() => {});
  };
  // §19.6 — Stop button opens confirmation dialog instead of stopping immediately
  const handleStop = () => {
    if (!activeId) return;
    setStopDialogOpen(true);
  };

  const handleStopSave = () => {
    if (!activeId) return;
    setStopDialogOpen(false);
    sendToSidecar({ type: "training.stop", run_id: activeId, save_checkpoint: true }).catch(() => {});
  };

  const handleStopDiscard = () => {
    if (!activeId) return;
    setStopDialogOpen(false);
    sendToSidecar({ type: "training.stop", run_id: activeId, save_checkpoint: false }).catch(() => {});
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
      {/* Pulsing dot */}
      <div style={{
        width: 8, height: 8, borderRadius: "50%", background: dotColor, flexShrink: 0,
        animation: status === "running" ? "pulse-dot 1.2s ease-in-out infinite" : "none",
      }} />

      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--muted)", flexShrink: 0 }}>
        {status === "running" ? "active" : status}
      </span>

      {activeRun && (
        <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-sans)", flexShrink: 0 }}>
          {activeRun.name}
        </span>
      )}

      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--dim)", flexShrink: 0 }}>
        iter {iter.toLocaleString()}
      </span>

      <div style={{ flex: 1 }} />

      {isHistorical && (
        <span style={{
          fontSize: 10, padding: "2px 8px", borderRadius: 3,
          background: "var(--blue-dim)", color: "var(--blue)",
          border: "1px solid rgba(90,171,240,0.27)", fontFamily: "var(--font-mono)",
        }}>
          historical view
        </span>
      )}

      {status === "done" && !isHistorical && (
        <span style={{
          fontSize: 10, padding: "2px 8px", borderRadius: 3,
          background: "var(--blue-dim)", color: "var(--blue)",
          border: "1px solid rgba(90,171,240,0.27)", fontFamily: "var(--font-mono)",
        }}>
          training.complete received
        </span>
      )}

      {!isHistorical && isActive && (
        <>
          {status === "paused" ? (
            <Btn small onClick={handleResume}>Resume</Btn>
          ) : (
            <Btn small onClick={handlePause}>Pause</Btn>
          )}
          <Btn small color="var(--red)" onClick={handleStop}>Stop</Btn>
        </>
      )}

      <style>{`
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.4; transform: scale(0.75); }
        }
      `}</style>

      {/* §19.6 — Stop confirmation dialog */}
      <StopConfirmDialog
        open={stopDialogOpen}
        onSaveAndStop={handleStopSave}
        onDiscardAndStop={handleStopDiscard}
        onCancel={() => setStopDialogOpen(false)}
      />
    </div>
  );
}

// ── Metric Cards (§12.3) ──────────────────────────────────────────────────

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  accent: string;
}

function MetricCard({ label, value, sub, accent }: MetricCardProps) {
  return (
    <div style={{
      flex: 1, minWidth: 90,
      background: "var(--bg1)", border: "1px solid var(--border)",
      borderTop: `2px solid ${accent}`, borderRadius: "var(--radius-md)",
      padding: "10px 12px",
    }}>
      <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 18, color: "var(--text)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)", marginTop: 2 }}>
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
  const gpuUtil = useTrainingStore((s) => s.gpuUtil);
  const epoch   = useTrainingStore((s) => s.epoch);
  const speed   = useTrainingStore((s) => s.speed);
  const arch    = useModelStore((s) => s.architecture);
  const isGan   = arch === GAN_ARCH;

  return (
    <div style={{ display: "flex", gap: 8, padding: "8px 14px", flexShrink: 0 }}>
      <MetricCard label="G LOSS"      value={fmt(gLoss)}    sub={isGan ? `disc ${fmt(dLoss)}` : "no disc"} accent="var(--green)"  />
      <MetricCard label="PSNR (dB)"   value={fmt(psnr, 2)}  accent="var(--blue)"   />
      <MetricCard label="SSIM"        value={fmt(ssim)}      accent="var(--cyan)"   />
      <MetricCard label="GPU UTIL (%)" value={fmtPct(gpuUtil)} accent="var(--amber)" />
      <MetricCard label="EPOCH"       value={String(epoch)}  accent="var(--purple)" />
      <MetricCard label="SPEED (it/s)" value={fmt(speed, 2)} accent="var(--muted)"  />
    </div>
  );
}

// ── Loss Curve SVG (§12.4) ────────────────────────────────────────────────

function LossCurve({ history }: { history: RunHistory | null }) {
  const arch        = useModelStore((s) => s.architecture);
  const status      = useTrainingStore((s) => s.status);
  const epoch       = useTrainingStore((s) => s.epoch);
  const iter        = useTrainingStore((s) => s.iter);
  const speed       = useTrainingStore((s) => s.speed);
  const totalEpochs = useRunConfigStore((s) => s.schedule.totalEpochs);

  const isGan  = arch === GAN_ARCH;
  const W = 340; const H = 160;

  const gSeries = (history?.gLossHistory ?? []).slice(-CHART_WINDOW);
  const dSeries = isGan
    ? (history?.dLossHistory ?? []).filter((v): v is number => v != null).slice(-CHART_WINDOW)
    : [];
  const tSeries = (history?.totalLossHistory ?? []).slice(-CHART_WINDOW);

  const allVals = [...gSeries, ...dSeries, ...tSeries];
  const min = allVals.length ? Math.min(...allVals) : 0;
  const max = allVals.length ? Math.max(...allVals) : 1;

  const gPts = buildPoints(gSeries, W, H, min, max);
  const dPts = buildPoints(dSeries, W, H, min, max);
  const tPts = buildPoints(tSeries, W, H, min, max);

  const etaSec   = status === "running" ? computeEtaSec(iter, epoch, speed, totalEpochs) : null;
  const epochPct = totalEpochs > 0 ? Math.min(100, (epoch / totalEpochs) * 100) : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ fontSize: 10, color: "var(--muted)", padding: "8px 12px 2px", fontFamily: "var(--font-mono)" }}>
        LOSS CURVE
      </div>
      <div style={{ display: "flex", gap: 10, padding: "0 12px 4px", fontSize: 10, fontFamily: "var(--font-mono)" }}>
        <span style={{ color: "var(--green)" }}>■ gen</span>
        {isGan && <span style={{ color: "var(--blue)" }}>■ disc</span>}
        <span style={{ color: "var(--purple)" }}>■ total</span>
      </div>
      <div style={{ flex: 1, padding: "0 8px", minHeight: 0 }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
          {[H * 0.25, H * 0.5, H * 0.75].map((y) => (
            <line key={y} x1={0} y1={y} x2={W} y2={y}
              stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3 3" />
          ))}
          {tSeries.length > 1 && (
            <path d={pointsToPath(tPts)} fill="none" stroke="var(--purple)" strokeWidth={1.5} strokeOpacity={0.8} />
          )}
          {isGan && dSeries.length > 1 && (
            <path d={pointsToPath(dPts)} fill="none" stroke="var(--blue)" strokeWidth={1.5} strokeOpacity={0.8} />
          )}
          {gSeries.length > 1 && (
            <path d={pointsToPath(gPts)} fill="none" stroke="var(--green)" strokeWidth={1.5} />
          )}
        </svg>
      </div>
      {/* Epoch progress bar */}
      <div style={{ padding: "4px 12px 10px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
          <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
            Epoch {epoch} / {totalEpochs}
          </span>
          {etaSec != null && (
            <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
              {formatEta(etaSec)}
            </span>
          )}
        </div>
        <PBar value={epochPct} max={100} color="var(--green)" height={5} />
      </div>
    </div>
  );
}

// ── PSNR/SSIM Scatter SVG (§12.5) ────────────────────────────────────────

function PsnrSsimChart({ history }: { history: RunHistory | null }) {
  const W = 340; const H = 160; const R = 3;

  const psnrSeries = (history?.psnrHistory ?? []).slice(-CHART_WINDOW);
  const ssimSeries = (history?.ssimHistory ?? []).slice(-CHART_WINDOW);
  const count = Math.max(psnrSeries.length, ssimSeries.length);

  const psnrMin = psnrSeries.length ? Math.min(...psnrSeries) : 0;
  const psnrMax = psnrSeries.length ? Math.max(...psnrSeries) : 1;
  const ssimMin = ssimSeries.length ? Math.min(...ssimSeries) : 0;
  const ssimMax = ssimSeries.length ? Math.max(...ssimSeries) : 1;

  const mapY = (v: number, lo: number, hi: number) => H - ((v - lo) / (hi - lo || 1)) * H;
  const mapX = (i: number) => (i / Math.max(count - 1, 1)) * W;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ fontSize: 10, color: "var(--muted)", padding: "8px 12px 2px", fontFamily: "var(--font-mono)" }}>
        PSNR / SSIM
      </div>
      <div style={{ display: "flex", gap: 10, padding: "0 12px 4px", fontSize: 10, fontFamily: "var(--font-mono)" }}>
        <span style={{ color: "var(--green)" }}>● PSNR (dB)</span>
        <span style={{ color: "var(--blue)" }}>● SSIM</span>
      </div>
      <div style={{ flex: 1, padding: "0 8px", minHeight: 0 }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
          {[H * 0.25, H * 0.5, H * 0.75].map((y) => (
            <line key={y} x1={0} y1={y} x2={W} y2={y}
              stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3 3" />
          ))}
          {psnrSeries.map((v, i) => (
            <circle key={`p${i}`} cx={mapX(i)} cy={mapY(v, psnrMin, psnrMax)} r={R}
              fill="var(--green)" fillOpacity={0.8} />
          ))}
          {ssimSeries.map((v, i) => (
            <circle key={`s${i}`} cx={mapX(i)} cy={mapY(v, ssimMin, ssimMax)} r={R}
              fill="var(--blue)" fillOpacity={0.8} />
          ))}
        </svg>
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
  const temp    = useTrainingStore((s) => s.temp);
  const cpuUtil = useTrainingStore((s) => s.cpuUtil);

  const rows = [
    { label: "GPU %",   value: fmtPct(gpuUtil),                      pct: gpuUtil,                     color: hwColor(gpuUtil, 80, 95) },
    { label: "VRAM GB", value: vram != null ? `${vram.toFixed(1)} GB` : "—", pct: vram != null ? (vram / 24) * 100 : null, color: "var(--blue)"    },
    { label: "TEMP °C", value: temp != null ? `${temp.toFixed(0)}°C`  : "—", pct: temp != null ? (temp / 110) * 100 : null, color: hwColor(temp, 75, 90) },
    { label: "CPU %",   value: fmtPct(cpuUtil),                      pct: cpuUtil,                     color: hwColor(cpuUtil, 80, 95) },
  ];

  return (
    <div style={{ padding: "8px 12px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>HARDWARE</div>
      {rows.map(({ label, value, pct, color }) => (
        <div key={label}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
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
        gap: 4, flex: 1,
      }}>
        {cells.map(({ label, path }) => (
          <div key={label} style={{
            background: "var(--bg2)", border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)", overflow: "hidden",
            display: "flex", alignItems: "center", justifyContent: "center",
            position: "relative",
          }}>
            {path ? (
              <img
                src={convertFileSrc(path)}
                alt={label}
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

// ── ScreenMetrics — §12.1 layout + §23.2 RunSelectorPanel sidebar ────────

export function ScreenMetrics() {
  const activeId       = useTrainingStore((s) => s.activeTrainingRunId);
  const runHistories   = useTrainingStore((s) => s.runHistories);
  const lossHistory    = useTrainingStore((s) => s.lossHistory);
  const dLossHistory   = useTrainingStore((s) => s.dLossHistory);
  const totalLossHist  = useTrainingStore((s) => s.totalLossHistory);
  const psnrHistory    = useTrainingStore((s) => s.psnrHistory);
  const ssimHistory    = useTrainingStore((s) => s.ssimHistory);
  const displayedId    = useUiStore((s) => s.displayedRunId);

  // §23.5 — multi-select state for comparison
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [showComparison, setShowComparison] = useState(false);

  // §12.12 — charts source from displayedRunId; metrics.update always writes to activeTrainingRunId
  const isLive = displayedId === null || displayedId === activeId;
  const displayedHistory: RunHistory | null = isLive
    ? { gLossHistory: lossHistory, dLossHistory, totalLossHistory: totalLossHist, psnrHistory, ssimHistory }
    : (runHistories[displayedId ?? ""] ?? null);

  return (
    <div style={{
      display: "flex", flexDirection: "row",
      width: "100%", height: "100%", overflow: "hidden", background: "var(--bg0)",
    }}>
      {/* §23.2 — RunSelectorPanel left sidebar */}
      <RunSelectorPanel
        forTab="metrics"
        selectedRunIds={selectedRunIds}
        onSelectionChange={setSelectedRunIds}
        onCompare={() => setShowComparison(true)}
      />

      {/* Main content area — comparison panel or live metrics */}
      <div style={{
        flex: 1, display: "flex", flexDirection: "column",
        overflow: "hidden", minWidth: 0,
      }}>
        {showComparison ? (
          <RunComparisonPanel
            runIds={selectedRunIds}
            onClose={() => setShowComparison(false)}
          />
        ) : (
          <>
            {/* §12.2 Training status bar */}
            <TrainingStatusBar />

            {/* §12.3 Metric cards row */}
            <MetricCards />

            {/* 2-col × 2-row chart grid */}
            <div style={{
              flex: 1, display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gridTemplateRows: "1fr 1fr",
              gap: 1, minHeight: 0,
              background: "var(--border)",
            }}>
              <div style={{ background: "var(--bg1)", overflow: "hidden" }}>
                <LossCurve history={displayedHistory} />
              </div>
              <div style={{ background: "var(--bg1)", overflow: "hidden" }}>
                <PsnrSsimChart history={displayedHistory} />
              </div>
              <div style={{ background: "var(--bg1)", overflow: "hidden" }}>
                <HardwarePanel />
              </div>
              <div style={{ background: "var(--bg1)", overflow: "hidden" }}>
                <ValidationPanel />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
