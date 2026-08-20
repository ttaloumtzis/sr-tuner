import { useTrainingStore } from "../../../store/trainingStore";
import { useModelStore } from "../../../store/modelStore";
import { fmt, fmtPct, GAN_ARCH } from "./chartUtils";
import { useRollingHistory, Sparkline } from "./MetricPrimitives";

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  accent: string;
  sparkline?: number[];
}

function MetricCard({ label, value, sub, accent, sparkline }: MetricCardProps) {
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
          {sub && (
            <div style={{ fontSize: 9.5, color: "var(--dim)", fontFamily: "var(--font-mono)", marginTop: 3, whiteSpace: "nowrap" }}>
              {sub}
            </div>
          )}
        </div>
        {sparkline && sparkline.length >= 2 && (
          <Sparkline values={sparkline} color={accent} width={150} height={30} padding={3} points={60} />
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
  const fullEpoch = useTrainingStore((s) => s.fullEpoch);
  const gpuUtil = useTrainingStore((s) => s.gpuUtil);
  const vram    = useTrainingStore((s) => s.vram);
  const temp    = useTrainingStore((s) => s.temp);
  const lossHistory = useTrainingStore((s) => s.lossHistory);
  const psnrHistory = useTrainingStore((s) => s.psnrHistory);
  const ssimHistory = useTrainingStore((s) => s.ssimHistory);
  const arch    = useModelStore((s) => s.architecture);
  const isGan   = arch === GAN_ARCH;

  const hasGpu = gpuUtil != null || vram != null || temp != null;
  const gpuHistory = useRollingHistory((s) => s.gpuUtilHistory);

  const fullEpochLabel = fullEpoch != null ? ` @ e${fullEpoch}` : "";

  return (
    <div style={{ display: "flex", gap: 8, padding: "10px 16px", flexShrink: 0, flexWrap: "wrap" }}>
      <MetricCard label="G LOSS" value={fmt(gLoss, 4)} sub={isGan ? `disc ${fmt(dLoss, 4)}` : "no disc"}
        accent="var(--green)" sparkline={lossHistory} />
      <MetricCard label="PSNR (dB)" value={fmt(psnr, 2)}
        sub={fullPsnr != null ? `full ${fmt(fullPsnr, 2)}${fullEpochLabel}` : undefined}
        accent="var(--green)" sparkline={psnrHistory} />
      <MetricCard label="SSIM" value={fmt(ssim, 4)}
        sub={fullSsim != null ? `full ${fmt(fullSsim, 4)}${fullEpochLabel}` : undefined}
        accent="var(--blue)" sparkline={ssimHistory} />
      {hasGpu && (
        <MetricCard label="GPU" value={fmtPct(gpuUtil)}
          sub={temp != null ? `${Math.round(temp)}°C` : undefined}
          accent="var(--amber)" sparkline={gpuHistory} />
      )}
    </div>
  );
}
