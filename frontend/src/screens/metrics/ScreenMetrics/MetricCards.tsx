import { useTrainingStore } from "../../../store/trainingStore";
import { useModelStore } from "../../../store/modelStore";
import { fmt, fmtPct, fmtGb, trendOf, GAN_ARCH } from "./chartUtils";
import { useRollingHistory, Sparkline } from "./MetricPrimitives";

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  accent: string;
  trend?: { dir: "up" | "down" | "flat"; pct: number } | null;
  sparkline?: number[];
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

export function MetricCards() {
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
  const fullPsnrHistory = useTrainingStore((s) => s.fullPsnrHistory);
  const fullSsimHistory = useTrainingStore((s) => s.fullSsimHistory);
  const arch    = useModelStore((s) => s.architecture);
  const isGan   = arch === GAN_ARCH;

  const gpuHistory   = useRollingHistory(gpuUtil);
  const cpuHistory   = useRollingHistory(cpuUtil);
  const ramHistory   = useRollingHistory(ramGb);
  const speedHistory = useRollingHistory(speed);

  return (
    <div style={{ display: "flex", gap: 8, padding: "10px 16px", flexShrink: 0, flexWrap: "wrap" }}>
      <MetricCard label="G LOSS" value={fmt(gLoss)} sub={isGan ? `disc ${fmt(dLoss)}` : "no disc"}
        accent="var(--green)" trend={trendOf(lossHistory, true)} sparkline={lossHistory} />
      <MetricCard label="PSNR (dB)" value={fmt(psnr, 2)} accent="var(--blue)"
        trend={trendOf(psnrHistory)} sparkline={psnrHistory} />
      <MetricCard label="SSIM" value={fmt(ssim)} accent="var(--cyan)"
        trend={trendOf(ssimHistory)} sparkline={ssimHistory} />
      <MetricCard label="FULL PSNR" value={fmt(fullPsnr, 2)} accent="var(--green)"
        trend={trendOf(fullPsnrHistory)} sparkline={fullPsnrHistory} />
      <MetricCard label="FULL SSIM" value={fmt(fullSsim)} accent="var(--teal)"
        trend={trendOf(fullSsimHistory)} sparkline={fullSsimHistory} />
      <MetricCard label="GPU" value={fmtPct(gpuUtil)} accent="var(--amber)"
        trend={trendOf(gpuHistory)} sparkline={gpuHistory} />
      <MetricCard label="EPOCH" value={String(epoch)} accent="var(--purple)" />
      <MetricCard label="SPEED" value={`${fmt(speed, 2)} it/s`} accent="var(--muted)"
        trend={trendOf(speedHistory)} sparkline={speedHistory} />
      <MetricCard label="CPU" value={fmtPct(cpuUtil)} accent="var(--teal)"
        trend={trendOf(cpuHistory)} sparkline={cpuHistory} />
      <MetricCard label="RAM" value={fmtGb(ramGb)} accent="var(--pink)"
        trend={trendOf(ramHistory)} sparkline={ramHistory} />
    </div>
  );
}