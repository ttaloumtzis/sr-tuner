import { useEffect, useRef, useState } from "react";
import { useTrainingStore } from "../../../store/trainingStore";
import { useToast } from "../../../components/shell/ToastProvider";
import { Btn } from "../../../components/ui/Btn";
import { cancelJob } from "../../../lib/api";

const STATUS_META: Record<string, { label: string; color: string }> = {
  running: { label: "Training active", color: "var(--green)" },
  paused: { label: "Paused", color: "var(--amber)" },
  done: { label: "Complete", color: "var(--blue)" },
  failed: { label: "Failed", color: "var(--red)" },
  disconnected: { label: "Disconnected", color: "var(--red)" },
  idle: { label: "Idle", color: "var(--dim)" },
};

export function TrainingStatusBar() {
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
