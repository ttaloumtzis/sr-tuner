import { useTrainingStore } from "../../../store/trainingStore";
import type { RunHistory } from "../../../store/trainingStore";
import { ValidationPanel } from "../../../components/metrics/ValidationPanel";
import { TrainingStatusBar } from "./TrainingStatusBar";
import { ProgressRow } from "./ProgressRow";
import { MetricCards } from "./MetricCards";
import { LossCurve } from "./LossCurve";
import { PsnrSsimChart } from "./PsnrSsimChart";
import { HardwarePanel } from "./HardwarePanel";
import { ResizableSplit } from "./ResizableSplit";

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
              overflow: "hidden", display: "flex", flexDirection: "column", minHeight: 0,
            }}>
              <LossCurve history={displayedHistory} />
            </div>
            <div style={{
              background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
              overflow: "hidden", display: "flex", flexDirection: "column", minHeight: 0,
            }}>
              <PsnrSsimChart history={displayedHistory} />
            </div>
            <div style={{
              gridRow: "1 / 3", gridColumn: "2",
              background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
              overflow: "hidden", display: "flex", flexDirection: "column", minHeight: 0,
            }}>
              <ResizableSplit
                top={<HardwarePanel />}
                bottom={<ValidationPanel />}
                defaultRatio={0.35} minTopPx={300}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
