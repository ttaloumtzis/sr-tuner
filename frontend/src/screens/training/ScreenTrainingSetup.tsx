// §11 Training Setup Screen
// Tasks: 11.1–11.14

import { useState, useCallback } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Field } from "../../components/ui/Field";
import { Toggle } from "../../components/ui/Toggle";
import { PathInput } from "../../components/ui/PathInput";
import { Dropdown } from "../../components/ui/Dropdown";
import type { DropdownOption } from "../../components/ui/Dropdown";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useModelStore } from "../../store/modelStore";
import { useDatasetStore } from "../../store/datasetStore";
import { useProjectStore } from "../../store/projectStore";
import type { Architecture } from "../../lib/srproj";
import { estimateVram } from "../../lib/vramEstimate";

// ── §24.3 — filename from path ────────────────────────────────────────────

function basename(filePath: string): string {
  return filePath.split(/[\\/]/).pop() ?? filePath;
}

// ── §11.5a Architecture pill selector ────────────────────────────────────

const ARCH_OPTIONS: Architecture[] = ["rrdb_esrgan", "swinir"];

interface ArchPillProps {
  id: Architecture;
  selected: boolean;
  onSelect: (id: Architecture) => void;
}

function ArchPill({ id, selected, onSelect }: ArchPillProps) {
  return (
    <div
      onClick={() => onSelect(id)}
      style={{
        padding: "5px 10px",
        borderRadius: "var(--radius-sm)",
        border: `1px solid ${selected ? "var(--green)" : "var(--border)"}`,
        background: selected ? "var(--green-dim)" : "var(--bg3)",
        color: selected ? "var(--green)" : "var(--muted)",
        fontSize: 11,
        fontWeight: selected ? 600 : 400,
        cursor: "pointer",
        textAlign: "center" as const,
        transition: "var(--transition-fast)",
        userSelect: "none" as const,
      }}
    >
      {id}
    </div>
  );
}

// ── §11.9 Inline dataset validation status dot ────────────────────────────

interface ValidationDotProps {
  valid: boolean | null;
}

function ValidationDot({ valid }: ValidationDotProps) {
  const color =
    valid === null ? "var(--dim)" : valid ? "var(--green)" : "var(--red, #ef4444)";
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: color,
        flexShrink: 0,
      }}
    />
  );
}

// ── Numeric input helper ──────────────────────────────────────────────────

interface NumInputProps {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

function NumInput({ value, onChange, min, max, step = 1 }: NumInputProps) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{
        background: "var(--bg3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
        padding: "5px 8px",
        fontSize: 12,
        color: "var(--text)",
        fontFamily: "var(--font-mono)",
        width: "100%",
        outline: "none",
        boxSizing: "border-box" as const,
      }}
    />
  );
}

// ── Text input helper ─────────────────────────────────────────────────────

interface TextInputProps {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}

function TextInput({ value, onChange, placeholder }: TextInputProps) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      style={{
        background: "var(--bg3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
        padding: "5px 8px",
        fontSize: 12,
        color: "var(--text)",
        fontFamily: "var(--font-sans)",
        width: "100%",
        outline: "none",
        boxSizing: "border-box" as const,
      }}
    />
  );
}

// ── EstimateRow helper ────────────────────────────────────────────────────

interface EstimateRowProps {
  label: string;
  value: string;
  color?: string;
}

function EstimateRow({ label, value, color }: EstimateRowProps) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <span style={{ fontSize: 10, color: "var(--muted)" }}>{label}</span>
      <span
        style={{
          fontSize: 12,
          fontFamily: "var(--font-mono)",
          color: color ?? "var(--text)",
          fontWeight: 500,
        }}
      >
        {value}
      </span>
    </div>
  );
}

// ── Main screen ───────────────────────────────────────────────────────────

export function ScreenTrainingSetup() {
  // runConfigStore
  const runName = useRunConfigStore((s) => s.runName);
  const outputDir = useRunConfigStore((s) => s.outputDir);
  const checkpointDir = useRunConfigStore((s) => s.checkpointDir);
  const device = useRunConfigStore((s) => s.device);
  const schedule = useRunConfigStore((s) => s.schedule);
  const tensorboard = useRunConfigStore((s) => s.tensorboard);
  const fp16 = useRunConfigStore((s) => s.fp16);
  const compile = useRunConfigStore((s) => s.compile);
  const resumeFrom = useRunConfigStore((s) => s.resumeFrom);
  const setRunName = useRunConfigStore((s) => s.setRunName);
  const setOutputDir = useRunConfigStore((s) => s.setOutputDir);
  const setCheckpointDir = useRunConfigStore((s) => s.setCheckpointDir);
  const setDevice = useRunConfigStore((s) => s.setDevice);
  const setSchedule = useRunConfigStore((s) => s.setSchedule);
  const setTensorboard = useRunConfigStore((s) => s.setTensorboard);
  const setFp16 = useRunConfigStore((s) => s.setFp16);
  const setCompile = useRunConfigStore((s) => s.setCompile);
  const setResumeFrom = useRunConfigStore((s) => s.setResumeFrom);

  // modelStore
  const architecture = useModelStore((s) => s.architecture);
  const hyperparameters = useModelStore((s) => s.hyperparameters);
  const pretrainedPath = useModelStore((s) => s.pretrainedPath);
  const augmentations = useModelStore((s) => s.augmentations);
  const setArchitecture = useModelStore((s) => s.setArchitecture);
  const setHyperparameters = useModelStore((s) => s.setHyperparameters);

  // projectStore - saved models
  const project = useProjectStore((s) => s.project);
  const savedModels = project?.models ?? [];
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // datasetStore
  const hrPath = useDatasetStore((s) => s.hrPath);
  const validationPath = useDatasetStore((s) => s.validationPath);
  const strategy = useDatasetStore((s) => s.strategy);
  const setHrPath = useDatasetStore((s) => s.setHrPath);
  const setValidationPath = useDatasetStore((s) => s.setValidationPath);

  // §11.9 — local dataset validation state
  const [datasetValid, setDatasetValid] = useState<boolean | null>(null);
  const [datasetErrors, setDatasetErrors] = useState<string[]>([]);

  // §11.9 — send dataset.validate.request
  const sendValidate = useCallback(() => {
    setDatasetValid(null);
    setDatasetErrors([]);
    // TODO: replace with api call
  }, [hrPath, validationPath, strategy]);

// §11.14 — GPU device dropdown options
  const deviceOptions: DropdownOption[] = [
    { value: "cuda:0", label: "cuda:0" },
    { value: "cpu", label: "cpu" },
  ];

  // §11.7 — iter/epoch counts
  const itersPerEpoch = schedule.totalEpochs > 0
    ? Math.round(hyperparameters.totalIter / schedule.totalEpochs)
    : 0;

  // §11.6 / §11.7 — VRAM estimate using formula
  const vramEst = estimateVram(
    architecture,
    hyperparameters.batchSize,
    hyperparameters.patchSize,
    fp16
  );

  // §11.8 — OOM check
  const isOom = false;

  // §11.10 / §11.9 — launch gate
  const canLaunch = datasetValid === true && runName.trim().length > 0;

  // §11.12 / §11.11 / §11.13 — build and send training.start
  const handleLaunch = useCallback(async () => {
    // TODO: build payload and call startTraining()
  }, [architecture, hyperparameters, pretrainedPath, schedule, fp16, compile, tensorboard, augmentations, hrPath, validationPath, strategy, checkpointDir, outputDir, resumeFrom]);

  // §11.10 — save config as YAML (wired in §13)
  const handleSaveYaml = useCallback(() => {
    // YAML export implemented in checkpoint manager (§13)
  }, []);

  return (
    // §11.1 — flex-1 main + 220px right panel
    <div
      style={{
        display: "flex",
        flex: 1,
        gap: 12,
        padding: 12,
        overflow: "hidden",
        minHeight: 0,
      }}
    >
      {/* ── Left: scrollable main column ── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          gap: 10,
          overflowY: "auto",
          minWidth: 0,
        }}
      >
        {/* §11.2 — Run configuration panel */}
        <Panel title="Run Configuration">
          {/* 3-col grid: run name, output dir, checkpoint dir */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 10,
              marginBottom: 10,
            }}
          >
            <Field label="Run Name">
              <TextInput
                value={runName}
                onChange={setRunName}
                placeholder="my-run-001"
              />
            </Field>
            <Field label="Output Dir">
              <PathInput
                value={outputDir}
                onChange={setOutputDir}
                browseTitle="Select output directory"
                placeholder="outputs/"
                compact
              />
            </Field>
            <Field label="Checkpoint Dir">
              <PathInput
                value={checkpointDir}
                onChange={setCheckpointDir}
                browseTitle="Select checkpoint directory"
                placeholder="checkpoints/"
                compact
              />
            </Field>
          </div>
          {/* Toggle row: tensorboard / fp16 / compile + resume badge */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Toggle on={tensorboard} onChange={() => setTensorboard(!tensorboard)} />
              <span style={{ fontSize: 11, color: "var(--muted)" }}>TensorBoard</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Toggle on={fp16} onChange={() => setFp16(!fp16)} />
              <span style={{ fontSize: 11, color: "var(--muted)" }}>FP16</span>
            </div>
            <div
              style={{ display: "flex", alignItems: "center", gap: 6 }}
              title={undefined}
            >
              <Toggle on={compile} onChange={() => setCompile(!compile)} disabled={false} />
              <span style={{ fontSize: 11, color: "var(--muted)" }}>
                torch.compile
              </span>
            </div>
            {/* §24.3 — Resume badge: filename (green) or "fresh run" (dim) + ✕ clear */}
            <div
              style={{
                marginLeft: "auto",
                padding: "3px 8px",
                borderRadius: "var(--radius-sm)",
                background: resumeFrom ? "var(--green-dim)" : "transparent",
                border: `1px solid ${resumeFrom ? "var(--green)" : "var(--border)"}`,
                color: resumeFrom ? "var(--green)" : "var(--dim)",
                fontSize: 10,
                fontWeight: resumeFrom ? 600 : 400,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              {resumeFrom ? basename(resumeFrom.checkpoint_path) : "fresh run"}
              {resumeFrom && (
                <button
                  onClick={() => setResumeFrom(null)}
                  title="Clear resume checkpoint"
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--green)",
                    cursor: "pointer",
                    fontSize: 12,
                    lineHeight: 1,
                    padding: 0,
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          </div>
        </Panel>

        {/* §11.3 — Schedule panel: 6-col single row */}
        <Panel title="Training Schedule">
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(6, 1fr)",
              gap: 10,
            }}
          >
            <Field label="Total Epochs">
              <NumInput
                value={schedule.totalEpochs}
                onChange={(v) => setSchedule({ totalEpochs: v })}
                min={1}
              />
            </Field>
            <Field label="Save Every">
              <NumInput
                value={schedule.saveEvery}
                onChange={(v) => setSchedule({ saveEvery: v })}
                min={1}
              />
            </Field>
            <Field label="Validate Every">
              <NumInput
                value={schedule.validateEvery}
                onChange={(v) => setSchedule({ validateEvery: v })}
                min={1}
              />
            </Field>
            <Field label="Warmup Iters">
              <NumInput
                value={schedule.warmupIter}
                onChange={(v) => setSchedule({ warmupIter: v })}
                min={0}
              />
            </Field>
            <Field label="LR Decay At">
              <NumInput
                value={parseInt(schedule.lrDecay, 10) || 0}
                onChange={(v) => setSchedule({ lrDecay: String(v) })}
                min={0}
              />
            </Field>
            <Field label="Decay Factor">
              <NumInput
                value={0.5}
                onChange={() => { /* decay factor stored separately in future */ }}
                min={0.01}
                max={1}
                step={0.01}
              />
            </Field>
          </div>
        </Panel>

        {/* §11.4 — Dataset panel */}
        <Panel
          title="Dataset"
          actions={
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {/* §11.9 — inline validation status dot + label */}
              <ValidationDot valid={datasetValid} />
              <span style={{ fontSize: 10, color: "var(--muted)" }}>
                {datasetValid === null
                  ? "Not validated"
                  : datasetValid
                  ? "Dataset paths validated"
                  : "Dataset validation failed — fix paths before launch"}
              </span>
              {/* §11.9 — Re-validate ghost button */}
              <Btn small onClick={sendValidate}>
                Re-validate
              </Btn>
            </div>
          }
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <Field label="Training Data">
              <PathInput
                value={hrPath}
                onChange={setHrPath}
                browseTitle="Select training dataset folder"
                placeholder="path/to/training/hr"
              />
            </Field>
            {strategy !== "none" && (
              <Field label="Validation Data">
                <PathInput
                  value={validationPath ?? ""}
                  onChange={setValidationPath}
                  browseTitle="Select validation dataset folder"
                  placeholder="path/to/validation/hr"
                />
              </Field>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <Field label="Workers">
                <NumInput value={4} onChange={() => {}} min={0} max={16} />
              </Field>
              <Field label="Prefetch Queue">
                <NumInput value={2} onChange={() => {}} min={1} max={8} />
              </Field>
            </div>
          </div>
        </Panel>

        {/* §11.5 — Hardware panel */}
        <Panel title="Hardware">
          <Field label="GPU Device">
            <Dropdown
              value={device}
              options={deviceOptions}
              onChange={setDevice}
            />
          </Field>
        </Panel>

        {/* §11.5a — Architecture & Batch Config panel */}
        <Panel title="Architecture & Batch Config">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* 4-col arch pill selector */}
            <Field label="Architecture">
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {savedModels.length > 0 && (
                  <select
                    value={selectedModelId ?? ""}
                    onChange={(e) => {
                      const model = savedModels.find((m) => m.id === e.target.value);
                      if (model) {
                        setSelectedModelId(model.id);
                        setArchitecture(model.architecture);
                        setHyperparameters({
                          batchSize: model.hyperparameters.batch_size,
                          patchSize: model.hyperparameters.patch_size,
                          learningRate: model.hyperparameters.learning_rate,
                          optimizer: model.hyperparameters.optimizer,
                          lrScheduler: model.hyperparameters.lr_scheduler,
                          totalIter: model.hyperparameters.total_iter,
                        });
                      }
                    }}
                    style={{ background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11, padding: "4px 6px", borderRadius: "var(--radius-sm)" }}
                  >
                    <option value="">— Saved Models —</option>
                    {savedModels.map((m) => (
                      <option key={m.id} value={m.id}>{m.name} ({m.architecture})</option>
                    ))}
                  </select>
                )}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6 }}>
                  {ARCH_OPTIONS.map((arch) => (
                    <ArchPill
                      key={arch}
                      id={arch}
                      selected={architecture === arch}
                      onSelect={setArchitecture}
                    />
                  ))}
                </div>
              </div>
            </Field>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <Field label="Batch Size">
                <NumInput
                  value={hyperparameters.batchSize}
                  onChange={(v) => setHyperparameters({ batchSize: v })}
                  min={1}
                  max={64}
                />
              </Field>
              <Field label="Patch Size (px)">
                <NumInput
                  value={hyperparameters.patchSize}
                  onChange={(v) => setHyperparameters({ patchSize: v })}
                  min={32}
                  max={512}
                  step={8}
                />
              </Field>
            </div>
          </div>
        </Panel>
      </div>

      {/* ── Right: 220px estimate + launch panel ── */}
      <div
        style={{
          width: 220,
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        {/* §11.7 — Estimate panel */}
        <Panel title="Estimate" style={{ flex: 1 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <EstimateRow label="Iters / epoch" value={itersPerEpoch.toLocaleString()} />
            <EstimateRow label="Total iters" value={hyperparameters.totalIter.toLocaleString()} />
            <EstimateRow
              label="Est. time"
              value="—"
              color="var(--amber, #f59e0b)"
            />
            {/* §11.7 / §11.8 — VRAM estimate, red when OOM */}
            <EstimateRow
              label="VRAM est."
              value={`${vramEst.toFixed(1)} GB`}
              color={isOom ? "var(--red, #ef4444)" : undefined}
            />
            {false && (
              <EstimateRow
                label="GPU VRAM"
                value={`? GB`}
              />
            )}

            {/* §11.8 — OOM warning */}
            {isOom && (
              <div
                style={{
                  padding: "6px 8px",
                  background: "rgba(239,68,68,0.1)",
                  border: "1px solid rgba(239,68,68,0.4)",
                  borderRadius: "var(--radius-sm)",
                  fontSize: 10,
                  color: "#f87171",
                  lineHeight: 1.4,
                }}
              >
                ⚠ Estimated VRAM exceeds GPU capacity — may OOM
              </div>
            )}

            {/* §11.9 — Validation errors */}
            {datasetErrors.length > 0 && (
              <div
                style={{
                  padding: "6px 8px",
                  background: "rgba(239,68,68,0.08)",
                  border: "1px solid rgba(239,68,68,0.3)",
                  borderRadius: "var(--radius-sm)",
                  fontSize: 10,
                  color: "#f87171",
                  lineHeight: 1.5,
                }}
              >
                {datasetErrors.map((err, i) => (
                  <div key={i}>{err}</div>
                ))}
              </div>
            )}

            {/* Config validation hint */}
            {!runName.trim() && (
              <div
                style={{
                  fontSize: 10,
                  color: "var(--amber, #f59e0b)",
                  lineHeight: 1.4,
                }}
              >
                ⚠ Run name required
              </div>
            )}
          </div>
        </Panel>

        {/* §11.10 — Launch + Save YAML buttons */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {/* §11.8 — OOM changes button to amber "Launch Anyway" */}
          {isOom ? (
            <Btn
              variant="solid"
              color="var(--amber, #f59e0b)"
              full
              onClick={handleLaunch}
              disabled={!runName.trim() || datasetValid !== true}
            >
              ⚠ Launch Anyway (may OOM)
            </Btn>
          ) : (
            <Btn
              variant="solid"
              color="var(--green)"
              full
              onClick={handleLaunch}
              // §11.9 / §11.10 — gated by dataset validation + run name
              disabled={!canLaunch}
            >
              Launch Training
            </Btn>
          )}
          <Btn full onClick={handleSaveYaml}>
            Save Config as YAML
          </Btn>
        </div>
      </div>
    </div>
  );
}
