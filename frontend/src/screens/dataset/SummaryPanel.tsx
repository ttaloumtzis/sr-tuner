import { Panel } from "../../components/ui/Panel";
import { useDatasetStore } from "../../store/datasetStore";

export function SummaryPanel() {
  const s = useDatasetStore();
  const modeLabel: Record<string, string> = { image_folder: "Pre-extracted", video_extract: "Video Extract", on_the_fly: "On-the-fly" };

  const rows: { label: string; value: string }[] = [{ label: "Mode", value: modeLabel[s.mode] || s.mode }];

  if (s.mode === "video_extract") {
    rows.push({ label: "Scale", value: `×${s.scale}` });
    rows.push({ label: "Downsample", value: s.kernel });
    rows.push({ label: "FPS", value: String(s.frameRate) });
    const activeDegs = [s.degBlur && "blur", s.degNoise && "noise", s.degJpeg && "jpeg", s.degJpeg2000 && "jpeg2000", s.degColorJitter && "color-jitter"].filter(Boolean);
    rows.push({ label: "Degradations", value: activeDegs.length ? activeDegs.join(", ") : "none" });
  }
  if (s.mode === "image_folder" && s.rootPath) {
    rows.push({ label: "Source", value: s.rootPath });
  }

  return (
    <div style={{ flex: 1, minWidth: 180, maxWidth: 300 }}>
      <Panel title="Dataset Summary">
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          {rows.map(({ label, value }) => (
            <div key={label} style={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <span style={{ fontSize: 9, color: "var(--dim)", textTransform: "uppercase", letterSpacing: 0.8, fontWeight: 600 }}>{label}</span>
              <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", wordBreak: "break-all" }}>{value}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}