import { useId } from "react";
import { useTrainingStore } from "../../../store/trainingStore";
import type { RunHistory } from "../../../store/trainingStore";
import { PanelHeader } from "./PanelHeader";
import { SubChart } from "./SubChart";
import { CHART_WINDOW } from "./chartUtils";

export function PsnrSsimChart({ history }: { history: RunHistory | null }) {
  const uid = useId();
  const fullPsnrHistory = useTrainingStore((s) => s.fullPsnrHistory);
  const fullSsimHistory = useTrainingStore((s) => s.fullSsimHistory);

  const fullPsnrLen = history?.psnrHistory?.length ?? 0;
  const windowStart = Math.max(0, fullPsnrLen - CHART_WINDOW);
  const psnrSeries = (history?.psnrHistory ?? []).slice(windowStart);
  const ssimSeries = (history?.ssimHistory ?? []).slice(windowStart);
  // The full-validation histories are recorded on their own cadence, so they
  // are windowed independently rather than sharing the per-batch offset.
  const fullPsnrSeries = fullPsnrHistory.slice(-CHART_WINDOW);
  const fullSsimSeries = fullSsimHistory.slice(-CHART_WINDOW);

  const hasFull = fullPsnrSeries.length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
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
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden", padding: "0 14px 10px", gap: 8 }}>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
          <div style={{ fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--font-mono)", marginBottom: 2, flexShrink: 0 }}>
            PSNR (dB)
          </div>
          <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
            <SubChart uid={uid} chartKey="psnr" series={psnrSeries} color="var(--green)"
              fullSeries={fullPsnrSeries} fullColor="var(--teal)" windowStart={windowStart} />
          </div>
        </div>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
          <div style={{ fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--font-mono)", marginBottom: 2, flexShrink: 0 }}>
            SSIM
          </div>
          <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
            <SubChart uid={uid} chartKey="ssim" series={ssimSeries} color="var(--blue)"
              fullSeries={fullSsimSeries} fullColor="var(--purple)" windowStart={windowStart} />
          </div>
        </div>
      </div>
    </div>
  );
}