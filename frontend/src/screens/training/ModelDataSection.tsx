import { useCallback } from "react";
import { Field } from "../../components/ui/Field";
import { Dropdown, type DropdownOption } from "../../components/ui/Dropdown";
import { Tag } from "../../components/ui/Tag";
import { Toggle } from "../../components/ui/Toggle";
import { NumberInput } from "../../components/ui/NumberInput";
import { InlineAlert } from "../../components/ui/InlineAlert";
import { GroupLabel } from "../../components/ui/GroupLabel";
import { IconRewind } from "../../components/ui/icons";
import { useRunConfigStore } from "../../store/runConfigStore";

export interface InstanceOption {
  value: string;
  label: string;
}

export interface DatasetOption {
  value: string;
  label: string;
  path: string;
  pairs: number;
  scale: number;
}

interface ModelDataSectionProps {
  instances: InstanceOption[];
  instancesError: string | null;
  datasets: DatasetOption[];
  datasetsError: string | null;
  onRefresh: () => void;
}

export function ModelDataSection({ instances, instancesError, datasets, datasetsError, onRefresh }: ModelDataSectionProps) {
  const s = useRunConfigStore();

  const handleInstanceSelect = useCallback((name: string) => {
    s.setSelectedInstance(name || null);
    s.setSelectedDataset(null);
    s.setSelectedDatasetPath(null);
    s.setSelectedDatasetPairs(null);
    s.setSelectedValidationDataset(null);
  }, [s]);

  const handleDatasetSelect = useCallback((name: string) => {
    s.setSelectedDataset(name || null);
    if (!name) {
      s.setSelectedDatasetPath(null);
      s.setSelectedDatasetPairs(null);
      return;
    }
    const ds = datasets.find((d) => d.value === name);
    if (ds) {
      s.setSelectedDatasetPath(ds.path);
      s.setSelectedDatasetPairs(ds.pairs);
    }
  }, [s, datasets]);

  const versionOptions: DropdownOption[] = [
    { value: "latest", label: "latest" },
    ...s.instanceVersions.map((v) => ({ value: v.tag, label: v.tag })),
  ];

  const valDatasetOptions: DropdownOption[] = [
    { value: "", label: "none (use split ratio)" },
    ...datasets
      .filter((d) => d.value !== s.selectedDataset)
      .map((d) => ({ value: d.value, label: d.label })),
  ];

  const selectedDatasetMeta = datasets.find((d) => d.value === s.selectedDataset);
  const scaleMismatch = !!(s.instanceScale && selectedDatasetMeta && selectedDatasetMeta.scale !== s.instanceScale);

  return (
    <div className="ts-col-10">
      <div className="ts-group">
        <GroupLabel>
          <span className="ts-label-row">
            Model
            <button onClick={onRefresh} className="ts-ghost-btn">
              Refresh
            </button>
          </span>
        </GroupLabel>
        {instancesError ? (
          <InlineAlert tone="red">
            Failed to load instances: {instancesError}
            <div className="ts-mt-6">
              <button onClick={onRefresh} className="ts-retry-btn">
                Retry
              </button>
            </div>
          </InlineAlert>
        ) : instances.length === 0 ? (
          <InlineAlert tone="amber">
            No model instances found. Create one in the Model Config tab first.
          </InlineAlert>
        ) : (
          <div className="ts-col-8">
            <Field label="Instance">
              <Dropdown
                value={s.selectedInstance ?? ""}
                options={[{ value: "", label: "— Select Instance —" }, ...instances]}
                onChange={handleInstanceSelect}
              />
            </Field>
            {s.instanceArchitecture && (
              <div className="ts-row-wrap">
                <Tag color="blue">{s.instanceArchitecture}</Tag>
                <Tag color="purple">{s.instanceScale ?? "?"}×</Tag>
                <Tag color="cyan">{s.instanceVersions.length} version{s.instanceVersions.length === 1 ? "" : "s"}</Tag>
              </div>
            )}
            {s.instanceVersions.length > 0 && (
              <div
                className="ts-info-box"
                style={{ borderLeft: `3px solid ${s.resumeFrom ? "var(--green)" : "var(--border2)"}` }}
              >
                <IconRewind size={12} color={s.resumeFrom ? "var(--green)" : "var(--muted)"} />
                <span className="ts-muted-label">Resume from</span>
                <div className="ts-flex-140">
                  <Dropdown
                    value={s.resumeFrom ?? "latest"}
                    options={versionOptions}
                    onChange={(v) => s.setResumeFrom(v)}
                  />
                </div>
                <button onClick={() => s.setResumeFrom(null)} className="ts-ghost-btn">
                  Start fresh
                </button>
              </div>
            )}
            {s.instanceVersions.length === 0 && s.selectedInstance && (
              <div className="ts-note-sm">
                No prior training — fresh start
              </div>
            )}
          </div>
        )}
      </div>

      <div className="ts-group">
        <GroupLabel>Data</GroupLabel>
        <div className="ts-col-8">
          {datasetsError ? (
            <InlineAlert tone="red">
              Failed to load datasets: {datasetsError}
              <div className="ts-mt-6">
                <button onClick={onRefresh} className="ts-retry-btn">
                  Retry
                </button>
              </div>
            </InlineAlert>
          ) : null}
          <Field label="Training Data">
            <Dropdown
              value={s.selectedDataset ?? ""}
              options={[{ value: "", label: "— Select Dataset —" }, ...datasets.map((d) => ({ value: d.value, label: d.label }))]}
              onChange={handleDatasetSelect}
            />
          </Field>
          {selectedDatasetMeta && (
            <div className="ts-info-box ts-info-box-8">
              <Tag color={scaleMismatch ? "amber" : "green"}>{selectedDatasetMeta.scale}×</Tag>
              <span className="ts-muted-label">{selectedDatasetMeta.pairs.toLocaleString()} pairs</span>
              <span className="ts-path" title={selectedDatasetMeta.path}>
                {selectedDatasetMeta.path}
              </span>
            </div>
          )}
          <Field label="Validation Data">
            <Dropdown
              value={s.selectedValidationDataset ?? ""}
              options={valDatasetOptions}
              onChange={(v) => s.setSelectedValidationDataset(v || null)}
            />
          </Field>
          <div className="ts-grid" style={{ "--ts-grid-min": "140px", "--ts-grid-max": "220px" } as React.CSSProperties}>
            <div className="ts-toggle-6">
              <Toggle on={s.validationEnabled} onChange={() => s.setValidationEnabled(!s.validationEnabled)} />
              <span className="ts-muted-label">Val enabled</span>
            </div>
            <Field label="Val split">
              <NumberInput compact value={s.validationSplit} onChange={s.setValidationSplit}
                min={0} max={1} step={0.05}
                disabled={s.selectedValidationDataset !== null} />
            </Field>
          </div>
          {s.selectedValidationDataset !== null && (
            <InlineAlert tone="muted" icon={false}>
              Split ratio ignored — using separate validation dataset
            </InlineAlert>
          )}
          {scaleMismatch && (
            <InlineAlert tone="amber">
              Dataset scale does not match model scale ({s.instanceScale}×)
            </InlineAlert>
          )}
        </div>
      </div>
    </div>
  );
}