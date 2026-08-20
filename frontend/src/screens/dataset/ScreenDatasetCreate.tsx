import { Panel } from "../../components/ui/Panel";
import "./ScreenDatasetCreate.css";
import { useDatasetStore, type DatasetMode } from "../../store/datasetStore";
import { TypeCard } from "./TypeCard";
import { PreExistingMode } from "./PreExistingMode";
import { VideoExtractMode } from "./VideoExtractMode";
import { OnTheFlyMode } from "./OnTheFlyMode";
import { SummaryPanel } from "./SummaryPanel";

export function ScreenDatasetCreate() {
  const s = useDatasetStore();

  const typeCards: { id: DatasetMode; label: string; description: string }[] = [
    { id: "image_folder", label: "Pre-existing", description: "Import existing HR/LR dataset folders into the project." },
    { id: "video_extract", label: "Video Extract", description: "Extract frames from video files. Full degradation pipeline." },
    { id: "on_the_fly", label: "On-the-fly", description: "Decode video during training. ~90% less disk usage. (Coming soon)" },
  ];

  return (
    <div className="dsc-layout">
      <div className="dsc-main">
        <div className="dsc-cards">
          {typeCards.map((c) => (
            <TypeCard key={c.id} label={c.label} description={c.description} active={s.mode === c.id} disabled={c.id === "on_the_fly"} onClick={() => s.setMode(c.id)} />
          ))}
        </div>

        <Panel title={s.mode === "image_folder" ? "Pre-existing Dataset" : s.mode === "video_extract" ? "Video Extraction" : "On-the-fly"}>
          {s.mode === "image_folder" && <PreExistingMode />}
          {s.mode === "video_extract" && <VideoExtractMode />}
          {s.mode === "on_the_fly" && <OnTheFlyMode />}
        </Panel>
      </div>
      <SummaryPanel />
    </div>
  );
}