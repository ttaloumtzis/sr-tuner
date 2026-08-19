import { useEffect, useRef, useState } from "react";
import { useTrainingStore, type TrainingStage } from "../../../store/trainingStore";
import { useToast } from "../../../components/shell/ToastProvider";
import { Btn } from "../../../components/ui/Btn";
import { cancelJob } from "../../../lib/api";
import { computeEtaSec, formatEta } from "./chartUtils";

const STATUS_META: Record<string, { label: string; color: string }> = {
  running: { label: "Training active", color: "var(--green)" },
  paused: { label: "Paused", color: "var(--amber)" },
  done: { label: "Complete", color: "var(--blue)" },
  failed: { label: "Failed", color: "var(--red)" },
  disconnected: { label: "Disconnected", color: "var(--red)" },
  idle: { label: "Idle", color: "var(--dim)" },
};

// Sub-step of a run with no step events (pre-training kernel warmup, dataset
// scan, checkpoint save). Shows what the GPU subprocess is busy with.
const STAGE_META: Record<TrainingStage, { label: string }> = {
  starting: { label: "Starting…" },
  preparing: { label: "Preparing" },
  warmup: { label: "Warmup · MIOpen kernel search" },
  training: { label: "Training" },
  validating: { label: "Validating" },
  saving: { label: "Saving checkpoint" },
};

// Stages that replace the epoch/batch pills entirely (epoch/batch are
// meaningless at 0/0 before the loop starts).
const STANDALONE_STAGES: ReadonlySet<TrainingStage> = new Set(["starting", "preparing", "warmup"]);

const pill: React.CSSProperties = {
  fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--dim)",
  padding: "2px 7px", background: "var(--bg2)", borderRadius: 20, flexShrink: 0,
  whiteSpace: "nowrap",
};

const stagePill: React.CSSProperties = {
  fontSize: 10, fontFamily: "var(--font-mono)", flexShrink: 0,
  color: "var(--amber)", background: "var(--amber-dim)",
  padding: "2px 7px", borderRadius: 20, whiteSpace: "nowrap",
};

// Merges the previous status bar + progress row into one compact band:
// a single meta row (status, epoch/batch, speed, ETA, best PSNR, action) with
// the overall-run progress strip underneath.
export function HeaderBand() {
  const status = useTrainingStore((s) => s.status);
  const epoch = useTrainingStore((s) => s.epoch);
  const batch = useTrainingStore((s) => s.batch);
  const totalBatch = useTrainingStore((s) => s.totalBatch);
  const speed = useTrainingStore((s) => s.speed);
  const finalEpoch = useTrainingStore((s) => s.finalEpoch);
  const activeTrainingRunId = useTrainingStore((s) => s.activeTrainingRunId);
  const bestPsnr = useTrainingStore((s) => s.bestPsnr);
  const totalEpochs = useTrainingStore((s) => s.launchConfig?.totalEpochs ?? 0);
  const validationRunning = useTrainingStore((s) => s.validationRunning);
  const validationProgress = useTrainingStore((s) => s.validationProgress);
  const stage = useTrainingStore((s) => s.stage);
  const preparingProgress = useTrainingStore((s) => s.preparingProgress);
  const { show } = useToast();
  const toastFiredRef = useRef(false);
  const [cancelling, setCancelling] = useState(false);

  const isIdle = status === "idle";

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

  const epochPct = totalBatch > 0 ? (batch / totalBatch) * 100 : 0;
  const overallPct = totalEpochs > 0 ? ((epoch - 1 + epochPct / 100) / totalEpochs) * 100 : 0;
  const etaSec = status === "running" ? computeEtaSec(batch, totalBatch, epoch, totalEpochs, speed) : null;
  const valPct = validationProgress && validationProgress.total > 0
    ? (validationProgress.done / validationProgress.total) * 100
    : null;

  return (
    <div style={{
      display: "flex", flexDirection: "column", flexShrink: 0,
      background: "linear-gradient(180deg, var(--bg1) 0%, var(--bg0) 100%)",
      borderBottom: "1px solid var(--border)",
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
        padding: "9px 16px",
      }}>
        <div style={{ position: "relative", width: 10, height: 10, flexShrink: 0 }}>
          <div style={{ position: "absolute", inset: 0, borderRadius: "50%", background: meta.color }} />
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

        {!isIdle && (
          <>
            {stage && STANDALONE_STAGES.has(stage) ? (
              <span style={stagePill}>
                {stage === "preparing" && preparingProgress
                  ? `Preparing · scanning ${preparingProgress.done} / ${preparingProgress.total}`
                  : STAGE_META[stage].label}
              </span>
            ) : (
              <>
                <span style={pill}>
                  Epoch {epoch} / {totalEpochs}
                </span>
                <span style={pill}>
                  {validationRunning
                    ? validationProgress
                      ? `validating e${epoch} · ${validationProgress.done} / ${validationProgress.total} images`
                      : `validating e${epoch}…`
                    : `batch ${batch} / ${totalBatch}`}
                </span>
                {speed != null && <span style={pill}>{speed.toFixed(2)} it/s</span>}
                {etaSec != null && (
                  <span style={{
                    fontSize: 10, fontFamily: "var(--font-mono)", flexShrink: 0,
                    color: "var(--amber)", background: "var(--amber-dim)",
                    padding: "2px 7px", borderRadius: 20, whiteSpace: "nowrap",
                  }}>
                    {formatEta(etaSec)}
                  </span>
                )}
              </>
            )}
            {stage && stage === "saving" && (
              <span style={stagePill}>{STAGE_META[stage].label}</span>
            )}
          </>
        )}

        {bestPsnr != null && (
          <span style={{
            fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--green)",
            padding: "2px 7px", background: "var(--green-dim)", borderRadius: 20, flexShrink: 0,
            border: "1px solid rgba(77,186,127,0.25)", whiteSpace: "nowrap",
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
      </div>

      {!isIdle && (
        <div style={{ position: "relative", height: 6, margin: "0 16px 10px", borderRadius: 3, background: "var(--bg3)", overflow: "hidden" }}>
          <div style={{
            position: "absolute", inset: 0, width: `${Math.min(100, overallPct)}%`,
            background: "linear-gradient(90deg, var(--green) 0%, var(--cyan) 100%)",
            borderRadius: 3, transition: "width 0.4s ease",
          }} />
          {validationRunning && !valPct && (
            <div style={{
              position: "absolute", inset: 0, background: "var(--blue)", opacity: 0.35,
              animation: "metrics-scan 1.4s linear infinite",
            }} />
          )}
          {validationRunning && valPct != null && (
            <div style={{
              position: "absolute", inset: 0, background: "var(--blue)", opacity: 0.45,
              width: `${Math.min(100, valPct)}%`,
              borderRadius: 3, transition: "width 0.25s ease",
            }} />
          )}
        </div>
      )}

      <style>{`
        @keyframes metrics-ping {
          0%   { opacity: 0.55; transform: scale(1); }
          75%, 100% { opacity: 0; transform: scale(2.4); }
        }
        @keyframes metrics-scan {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}
