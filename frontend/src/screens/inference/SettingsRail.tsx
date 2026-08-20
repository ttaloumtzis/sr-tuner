import { useInferenceStore } from "../../store/inferenceStore";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Field } from "../../components/ui/Field";
import { PathInput } from "../../components/ui/PathInput";
import { Dropdown } from "../../components/ui/Dropdown";
import { PBar } from "../../components/ui/PBar";
import { DropZone } from "../../components/ui/DropZone";
import { NumberInput } from "../../components/ui/NumberInput";
import { ModelPanel } from "./ModelPanel";

const TILE_OPTIONS = [
  { value: "0", label: "No tiling" },
  { value: "128", label: "128 px" },
  { value: "256", label: "256 px" },
  { value: "512", label: "512 px" },
];

const FORMAT_OPTIONS = ["png", "jpeg", "webp", "tiff"];

const DEVICE_OPTIONS = ["auto", "cuda", "cpu"];

export function SettingsRail({ onRun, onCancel }: { onRun: () => void; onCancel: () => void }) {
  const store = useInferenceStore();

  const outputDirError = store.outputDir ? null : "Select an output directory";
  const modelReady = !!store.modelPath;
  const canRun =
    !!store.inputPath && modelReady && !outputDirError && store.status !== "running";

  return (
    <div className="si-rail">
      {/* Input image */}
      <Panel title="Input Image" shrink>
        <DropZone
          compact
          label="Drop image here"
          path={store.inputPath}
          onSelect={store.setInputPath}
          browseTitle="Select Input Image"
        />
      </Panel>

      {/* Ground truth */}
      <Panel title="Ground Truth" subtitle="for quality metrics" shrink>
        <DropZone
          compact
          label="Drop GT image here"
          path={store.gtPath}
          accent="var(--blue)"
          onSelect={store.setGtPath}
          onClear={() => store.setGtPath(null)}
          browseTitle="Select Ground Truth Image"
        />
      </Panel>

      {/* Model */}
      <Panel title="Model" shrink>
        <ModelPanel />
      </Panel>

      {/* Output */}
      <Panel title="Output" shrink>
        <div className="si-stack">
          <Field label="Save Directory">
            <PathInput
              value={store.outputDir}
              onChange={store.setOutputDir}
              browseTitle="Select Output Directory"
              compact
            />
            {outputDirError && (
              <div className="si-error">
                {outputDirError}
              </div>
            )}
          </Field>
          <Field label="Format">
            <Dropdown
              value={store.outputFormat}
              options={FORMAT_OPTIONS}
              onChange={(v) => store.setOutputFormat(v as "png" | "jpeg" | "webp" | "tiff")}
            />
          </Field>
        </div>
      </Panel>

      {/* Tiling + device */}
      <Panel title="Advanced" shrink>
        <div className="si-stack">
          <Field label="Tile Size">
            <Dropdown
              value={String(store.tileSize)}
              options={TILE_OPTIONS}
              onChange={(v) => store.setTileSize(Number(v))}
            />
          </Field>
          <Field label="Tile Overlap" hint="px">
            <NumberInput
              value={store.overlap}
              min={0}
              max={store.tileSize > 0 ? store.tileSize - 1 : 512}
              onChange={store.setOverlap}
              title="Overlap must be less than tile size"
            />
          </Field>
          <Field label="Device">
            <Dropdown
              value={store.device}
              options={DEVICE_OPTIONS}
              onChange={(v) => store.setDevice(v as "auto" | "cuda" | "cpu")}
            />
          </Field>
        </div>
      </Panel>

      {/* Run / cancel + progress */}
      <div className="si-actions">
        {store.status === "running" ? (
          <Btn variant="solid" color="var(--red)" full onClick={onCancel}>
            Cancel
          </Btn>
        ) : (
          <Btn variant="solid" full disabled={!canRun} onClick={onRun}>
            Run Inference
          </Btn>
        )}

        {store.status === "running" && (
          <div className="si-progress">
            <PBar value={store.tilesDone} max={Math.max(store.tilesTotal, 1)} />
            <span className="si-progress-label">
              {store.tilesTotal > 1
                ? `${store.tilesDone} / ${store.tilesTotal} tiles`
                : "Processing…"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}