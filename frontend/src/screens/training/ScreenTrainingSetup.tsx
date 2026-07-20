import { useState, useCallback, useEffect } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Field } from "../../components/ui/Field";
import { Toggle } from "../../components/ui/Toggle";
import { Dropdown, type DropdownOption } from "../../components/ui/Dropdown";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useTrainingStore } from "../../store/trainingStore";
import { useUiStore } from "../../store/uiStore";
import { estimateVramBreakdown, type VramBreakdown } from "../../lib/vramEstimate";

interface ValidationDotProps {
  valid: boolean | null;
}
function ValidationDot({ valid }: ValidationDotProps) {
  const color = valid === null ? "var(--dim)" : valid ? "var(--green)" : "var(--red, #ef4444)";
  return <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />;
}

interface NumInputProps {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}
function NumInput({ value, onChange, min, max, step = 1, disabled }: NumInputProps) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{
        background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
        padding: "5px 8px", fontSize: 12, color: "var(--text)", fontFamily: "var(--font-mono)",
        width: "100%", outline: "none", boxSizing: "border-box" as const,
        opacity: disabled ? 0.4 : 1, cursor: disabled ? "not-allowed" : undefined,
      }}
    />
  );
}

interface TextInputProps {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}
function TextInput({ value, onChange, placeholder }: TextInputProps) {
  return (
    <input
      type="text" value={value} placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      style={{
        background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
        padding: "5px 8px", fontSize: 12, color: "var(--text)", fontFamily: "var(--font-sans)",
        width: "100%", outline: "none", boxSizing: "border-box" as const,
      }}
    />
  );
}

interface EstimateRowProps {
  label: string;
  value: string;
  color?: string;
}
function EstimateRow({ label, value, color }: EstimateRowProps) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <span style={{ fontSize: 10, color: "var(--muted)" }}>{label}</span>
      <span style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: color ?? "var(--text)", fontWeight: 500 }}>{value}</span>
    </div>
  );
}

export function ScreenTrainingSetup() {
  const s = useRunConfigStore();

  const [instances, setInstances] = useState<{ value: string; label: string }[]>([]);
  const [datasets, setDatasets] = useState<{ value: string; label: string; path: string; pairs: number; scale: number }[]>([]);
  const [datasetValid, setDatasetValid] = useState<boolean | null>(null);
  const [datasetErrors, setDatasetErrors] = useState<string[]>([]);
  const [customConfigPath, setCustomConfigPath] = useState("");
  const [scaleMismatch, setScaleMismatch] = useState(false);
  const [gpuTotalVramGb, setGpuTotalVramGb] = useState<number | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const { getEnv } = await import("../../lib/api");
        const env = await getEnv();
        if (env.vram_total_mb) {
          setGpuTotalVramGb(env.vram_total_mb / 1024);
        }
      } catch { console.warn("getEnv failed in training setup"); }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const { listInstances } = await import("../../lib/api");
        const list = await listInstances();
        setInstances(list.map((i: { name: string; architecture: string | null }) => ({
          value: i.name,
          label: `${i.name}${i.architecture ? ` (${i.architecture})` : ""}`,
        })));
      } catch { console.warn("listInstances failed in training setup"); }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const { listDatasets } = await import("../../lib/api");
        const list = await listDatasets();
        setDatasets(list.map((d: { name: string; path: string; num_pairs: number; scale: number }) => ({
          value: d.name,
          label: `${d.name} (${d.scale}× · ${d.num_pairs} pairs)`,
          path: d.path,
          pairs: d.num_pairs,
          scale: d.scale,
        })));
      } catch { console.warn("listDatasets failed in training setup"); }
    })();
  }, []);

  const handleInstanceSelect = useCallback(async (name: string) => {
    s.setSelectedInstance(name);
    s.setResumeFrom(null);
    s.setInstanceVersions([]);
    s.setSelectedDataset(null);
    s.setSelectedDatasetPath(null);
    s.setSelectedDatasetPairs(null);
    s.setInstanceConfig(null);
    s.setSelectedValidationDataset(null);
    setDatasetValid(null);
    setDatasetErrors([]);
    setScaleMismatch(false);
    if (!name) {
      s.setInstanceArchitecture(null);
      s.setInstanceScale(null);
      return;
    }
    try {
      const { getInstance, getInstanceVersions } = await import("../../lib/api");
      const inst = await getInstance(name);
      s.setInstanceArchitecture(inst.architecture);
      s.setInstanceScale(inst.scale ?? null);
      s.setInstanceConfig(inst.config ?? null);
      const versions = await getInstanceVersions(name);
      const versionList = versions.map((v: { tag: string }) => ({ tag: v.tag, path: "" }));
      s.setInstanceVersions(versionList);
      if (versionList.length > 0) {
        s.setResumeFrom("latest");
      }
    } catch { console.warn("getInstance/getInstanceVersions failed in training setup"); }
  }, [s]);

  const handleDatasetSelect = useCallback((name: string) => {
    s.setSelectedDataset(name || null);
    setDatasetValid(null);
    setDatasetErrors([]);
    if (!name) {
      s.setSelectedDatasetPath(null);
      s.setSelectedDatasetPairs(null);
      setScaleMismatch(false);
      return;
    }
    const ds = datasets.find((d) => d.value === name);
    if (ds) {
      s.setSelectedDatasetPath(ds.path);
      s.setSelectedDatasetPairs(ds.pairs);
      setScaleMismatch(s.instanceScale !== null && ds.scale !== s.instanceScale);
    }
  }, [s, datasets]);

  const handleValidate = useCallback(async () => {
    if (!s.selectedDatasetPath) return;
    setDatasetValid(null);
    setDatasetErrors([]);
    try {
      const { validateDatasetPath } = await import("../../lib/api");
      const res = await validateDatasetPath({ path: s.selectedDatasetPath });
      setDatasetValid(res.valid);
      setDatasetErrors(res.problems);
    } catch (e) {
      setDatasetValid(false);
      setDatasetErrors([String(e)]);
    }
  }, [s.selectedDatasetPath]);

  const itersPerEpoch = s.selectedDatasetPairs && s.batchSize > 0
    ? Math.ceil(s.selectedDatasetPairs / s.batchSize)
    : 0;
  const totalIters = itersPerEpoch * s.schedule.totalEpochs;

  const instCfg = s.instanceConfig as Record<string, unknown> | undefined;
  const vramBreakdown: VramBreakdown = s.instanceArchitecture
    ? estimateVramBreakdown(
        s.instanceArchitecture as any,
        s.batchSize,
        s.patchSize,
        s.fp16,
        s.instanceScale ?? 4,
        instCfg?.num_feat as number | undefined,
        instCfg?.num_block as number | undefined,
        instCfg?.embed_dim as number | undefined,
        instCfg?.depths as number[] | undefined,
      )
    : { totalGb: 0, weightsGb: 0, gradsGb: 0, adamGb: 0, activationsGb: 0, inputGb: 0, overheadGb: 0 };
  const vramEst = vramBreakdown.totalGb;
  const isOom = gpuTotalVramGb !== null && vramEst > gpuTotalVramGb;

  const canLaunch = s.selectedInstance && s.selectedDataset;

  const handleLaunch = useCallback(async () => {
    try {
      const { startTraining } = await import("../../lib/api");
      const res = await startTraining({
        model_name: s.instanceArchitecture ?? "",
        instance: s.selectedInstance ?? "",
        dataset: s.selectedDataset ?? "",
        resume: s.resumeFrom ?? undefined,
        config: customConfigPath || undefined,
        device: s.device === "auto" ? undefined : s.device,
        batch_size: s.batchSize,
        learning_rate: s.learningRate,
        max_epochs: s.schedule.totalEpochs,
        patch_size: s.patchSize,
        fp16: s.fp16 || undefined,
        seed: s.seed,
        weight_decay: s.weightDecay,
        betas: s.betas,
        num_workers: s.numWorkers,
        save_per_epoch: s.schedule.saveEvery,
        validation_enabled: s.validationEnabled,
        validation_split: s.validationSplit,
        validation_dataset: s.selectedValidationDataset ?? undefined,
        metrics_frequency: s.metricsFrequency,
        write_metrics_file: s.writeMetricsFile,
        perceptual_weight: s.perceptualWeight,
        warmup_steps: s.schedule.warmupSteps,
      });

      useTrainingStore.getState().reset();
      useTrainingStore.getState().setActiveRun(res.job_id);
      useTrainingStore.getState().setStatus("running");
      useUiStore.getState().setActiveTab("metrics");
      setLaunchError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLaunchError(msg);
      useUiStore.getState().setLastApiError({
        type: "error",
        code: "LAUNCH_FAILED",
        message: msg,
      });
    }
  }, [s, customConfigPath]);

  const valDatasetOptions: DropdownOption[] = [
    { value: "", label: "none (use split ratio)" },
    ...datasets
      .filter((d) => d.value !== s.selectedDataset)
      .map((d) => ({ value: d.value, label: d.label })),
  ];

  const deviceOptions: DropdownOption[] = [
    { value: "cuda:0", label: "cuda:0" },
    { value: "cpu", label: "cpu" },
    { value: "auto", label: "auto" },
  ];

  const versionOptions: DropdownOption[] = [
    { value: "latest", label: "latest" },
    ...s.instanceVersions.map((v) => ({ value: v.tag, label: v.tag })),
  ];

  return (
    <div style={{ display: "flex", flex: 1, gap: 12, padding: 12, overflow: "hidden", minHeight: 0 }}>
      <div style={{ flex: 3, display: "flex", flexDirection: "column", gap: 10, overflowY: "auto", minWidth: 0 }}>
        {/* Run Configuration */}
        <Panel title="Run Configuration">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
            <Field label="Device">
              <Dropdown value={s.device} options={deviceOptions} onChange={s.setDevice} />
            </Field>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 8, paddingBottom: 6 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Toggle on={s.fp16} onChange={() => s.setFp16(!s.fp16)} />
                <span style={{ fontSize: 11, color: "var(--muted)" }}>BF16</span>
              </div>
              <div style={{ marginLeft: "auto", padding: "3px 8px", borderRadius: "var(--radius-sm)", background: s.resumeFrom ? "var(--green-dim)" : "transparent", border: `1px solid ${s.resumeFrom ? "var(--green)" : "var(--border)"}`, color: s.resumeFrom ? "var(--green)" : "var(--dim)", fontSize: 10, fontWeight: s.resumeFrom ? 600 : 400, display: "flex", alignItems: "center", gap: 6 }}>
                {s.resumeFrom ? `Resume: ${s.resumeFrom}` : "fresh run"}
                {s.resumeFrom && (
                  <button onClick={() => s.setResumeFrom(null)} title="Start fresh" style={{ background: "none", border: "none", color: "var(--green)", cursor: "pointer", fontSize: 12, lineHeight: 1, padding: 0 }}>
                    ✕
                  </button>
                )}
              </div>
            </div>
          </div>
        </Panel>

        {/* Model Instance */}
        <Panel title="Model Instance">
          {instances.length === 0 ? (
            <div style={{ color: "var(--amber)", fontSize: 11, padding: "4px 0" }}>
              No model instances found. Create one in the Model Config tab first.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <Field label="Instance">
                <Dropdown
                  value={s.selectedInstance ?? ""}
                  options={[{ value: "", label: "— Select Instance —" }, ...instances]}
                  onChange={handleInstanceSelect}
                />
              </Field>
              {s.instanceArchitecture && (
                <div style={{ fontSize: 11, color: "var(--muted)" }}>
                  {s.instanceArchitecture} · {s.instanceScale ?? "?"}× · {s.instanceVersions.length} version(s)
                </div>
              )}
              {s.instanceVersions.length > 0 && (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 11, color: "var(--muted)" }}>Resume from</span>
                  <div style={{ width: 120 }}>
                    <Dropdown
                      value={s.resumeFrom ?? "latest"}
                      options={versionOptions}
                      onChange={(v) => s.setResumeFrom(v)}
                    />
                  </div>
                  <button onClick={() => s.setResumeFrom(null)} style={{ background: "none", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--muted)", cursor: "pointer", fontSize: 10, padding: "3px 8px" }}>
                    Start fresh
                  </button>
                </div>
              )}
              {s.instanceVersions.length === 0 && s.selectedInstance && (
                <div style={{ fontSize: 10, color: "var(--dim)", padding: "2px 0" }}>
                  No prior training — fresh start
                </div>
              )}
            </div>
          )}
        </Panel>

        {/* Dataset */}
        <Panel
          title="Dataset"
          actions={
            s.selectedDatasetPath ? (
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <ValidationDot valid={datasetValid} />
                <span style={{ fontSize: 10, color: "var(--muted)" }}>
                  {datasetValid === null ? "Not validated" : datasetValid ? "Valid" : "Invalid"}
                </span>
                <Btn small onClick={handleValidate}>Validate</Btn>
              </div>
            ) : undefined
          }
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <Field label="Training Data">
              <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                <Dropdown
                  value={s.selectedDataset ?? ""}
                  options={[{ value: "", label: "— Select Dataset —" }, ...datasets.map((d) => ({ value: d.value, label: d.label }))]}
                  onChange={handleDatasetSelect}
                />
              </div>
            </Field>
            <Field label="Validation Data">
              <Dropdown
                value={s.selectedValidationDataset ?? ""}
                options={valDatasetOptions}
                onChange={(v) => s.setSelectedValidationDataset(v || null)}
              />
            </Field>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Toggle on={s.validationEnabled} onChange={() => s.setValidationEnabled(!s.validationEnabled)} />
                <span style={{ fontSize: 11, color: "var(--muted)" }}>Val enabled</span>
              </div>
              <Field label="Val split">
                <NumInput value={s.validationSplit} onChange={s.setValidationSplit}
                  min={0} max={1} step={0.05}
                  disabled={s.selectedValidationDataset !== null} />
              </Field>
              <Field label="Workers">
                <NumInput value={s.numWorkers} onChange={s.setNumWorkers} min={0} max={16} />
              </Field>
            </div>
            {s.selectedValidationDataset !== null && (
              <div style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
                Split ratio ignored — using separate validation dataset
              </div>
            )}
            {scaleMismatch && (
              <div style={{ padding: "6px 8px", background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.4)", borderRadius: "var(--radius-sm)", fontSize: 10, color: "#f59e0b", lineHeight: 1.4 }}>
                ⚠ Dataset scale does not match model scale ({s.instanceScale}×)
              </div>
            )}
          </div>
        </Panel>

        {/* Hyperparameters */}
        <Panel title="Hyperparameters">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 10 }}>
            <Field label="Total Epochs"><NumInput value={s.schedule.totalEpochs} onChange={(v) => s.setSchedule({ totalEpochs: v })} min={1} /></Field>
            <Field label="Batch Size"><NumInput value={s.batchSize} onChange={s.setBatchSize} min={1} max={128} /></Field>
            <Field label="Patch Size"><NumInput value={s.patchSize} onChange={s.setPatchSize} min={16} max={512} step={8} /></Field>
            <Field label="Save Every"><NumInput value={s.schedule.saveEvery} onChange={(v) => s.setSchedule({ saveEvery: v })} min={1} /></Field>
            <Field label="Warmup Steps"><NumInput value={s.schedule.warmupSteps} onChange={(v) => s.setSchedule({ warmupSteps: v })} min={0} /></Field>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 10 }}>
            <Field label="Seed"><NumInput value={s.seed} onChange={s.setSeed} min={0} /></Field>
            <Field label="Learning Rate"><NumInput value={s.learningRate} onChange={s.setLearningRate} min={0} step={1e-5} /></Field>
            <Field label="Weight Decay"><NumInput value={s.weightDecay} onChange={s.setWeightDecay} min={0} step={0.01} /></Field>
            <Field label="β₁"><NumInput value={s.betas[0]} onChange={(v) => s.setBetas([v, s.betas[1]])} min={0} max={1} step={0.01} /></Field>
            <Field label="β₂"><NumInput value={s.betas[1]} onChange={(v) => s.setBetas([s.betas[0], v])} min={0} max={1} step={0.001} /></Field>
          </div>
          <Field label="Perceptual Weight (0 = disabled)">
            <NumInput value={s.perceptualWeight} onChange={s.setPerceptualWeight} min={0} step={0.01} />
          </Field>
        </Panel>

        {/* Advanced */}
        <Panel title="Advanced">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <Field label="Metrics Frequency">
              <NumInput value={s.metricsFrequency} onChange={s.setMetricsFrequency} min={1} />
            </Field>
            <Field label="Custom Config YAML">
              <TextInput value={customConfigPath} onChange={setCustomConfigPath} placeholder="path/to/config.yaml" />
            </Field>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8 }}>
            <Toggle on={s.writeMetricsFile} onChange={() => s.setWriteMetricsFile(!s.writeMetricsFile)} />
            <span style={{ fontSize: 11, color: "var(--muted)" }}>Write metrics.jsonl file</span>
          </div>
        </Panel>
      </div>

      {/* Sidebar */}
      <div style={{ flex: 1, minWidth: 200, maxWidth: 320, display: "flex", flexDirection: "column", gap: 10 }}>
        <Panel title="Estimate" style={{ flex: 1 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <EstimateRow label="Iters / epoch" value={itersPerEpoch.toLocaleString()} />
            <EstimateRow label="Total iters" value={totalIters.toLocaleString()} />
            <EstimateRow label="Est. time" value="—" color="var(--amber, #f59e0b)" />
            <EstimateRow label="VRAM est." value={`${vramBreakdown.totalGb.toFixed(1)} GB`} color={isOom ? "var(--red, #ef4444)" : undefined} />
            {vramBreakdown.totalGb > 0 && (
              <div style={{ paddingLeft: 14, display: "flex", flexDirection: "column", gap: 2 }}>
                <EstimateRow label="  Model weights" value={`${vramBreakdown.weightsGb.toFixed(2)} GB`} />
                <EstimateRow label="  Gradients" value={`${vramBreakdown.gradsGb.toFixed(2)} GB`} />
                <EstimateRow label="  Adam optimizer" value={`${vramBreakdown.adamGb.toFixed(2)} GB`} />
                <EstimateRow label="  Activations" value={`${vramBreakdown.activationsGb.toFixed(2)} GB`} />
                <EstimateRow label="  Input batch" value={`${vramBreakdown.inputGb.toFixed(2)} GB`} />
                <EstimateRow label="  CUDA overhead" value={`${vramBreakdown.overheadGb.toFixed(2)} GB`} />
              </div>
            )}
            {isOom && (
              <div style={{ padding: "6px 8px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.4)", borderRadius: "var(--radius-sm)", fontSize: 10, color: "#f87171", lineHeight: 1.4 }}>
                ⚠ Estimated VRAM exceeds GPU capacity — may OOM
              </div>
            )}
            {datasetErrors.length > 0 && (
              <div style={{ padding: "6px 8px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "var(--radius-sm)", fontSize: 10, color: "#f87171", lineHeight: 1.5 }}>
                {datasetErrors.map((err, i) => <div key={i}>{err}</div>)}
              </div>
            )}
            {launchError && (
              <div style={{ padding: "6px 8px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "var(--radius-sm)", fontSize: 10, color: "#f87171", lineHeight: 1.5 }}>
                {launchError}
              </div>
            )}

          </div>
        </Panel>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <Btn
            variant="solid"
            color={isOom ? "var(--amber, #f59e0b)" : "var(--green)"}
            full
            onClick={handleLaunch}
            disabled={!canLaunch}
          >
            {isOom ? "⚠ Launch Anyway (may OOM)" : "Launch Training"}
          </Btn>
        </div>
      </div>
    </div>
  );
}