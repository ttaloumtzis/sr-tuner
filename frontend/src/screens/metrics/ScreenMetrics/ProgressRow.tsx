import { useTrainingStore } from "../../../store/trainingStore";
import { computeEtaSec, formatEta } from "./chartUtils";

export function ProgressRow() {
  const status = useTrainingStore((s) => s.status);
  const epoch = useTrainingStore((s) => s.epoch);
  const batch = useTrainingStore((s) => s.batch);
  const totalBatch = useTrainingStore((s) => s.totalBatch);
  const speed = useTrainingStore((s) => s.speed);
  const validationRunning = useTrainingStore((s) => s.validationRunning);
  const totalEpochs = useTrainingStore((s) => s.launchConfig?.totalEpochs ?? 0);

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
