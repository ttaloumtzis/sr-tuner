import { useState, useCallback, useEffect, useRef, type CSSProperties } from "react";
import ReactDOM from "react-dom";
import { LOSS_TYPE_OPTIONS, type LossType } from "../../lib/api-types";
import "./ScreenTrainingSetup.css";
import { Panel } from "../../components/ui/Panel";
import { InlineAlert } from "../../components/ui/InlineAlert";
import { Btn } from "../../components/ui/Btn";
import { Field } from "../../components/ui/Field";
import { Toggle } from "../../components/ui/Toggle";
import { Dropdown, type DropdownOption } from "../../components/ui/Dropdown";
import { Tag } from "../../components/ui/Tag";
import { Tooltip } from "../../components/ui/Tooltip";
import { StackedBar, type StackedBarSegment } from "../../components/ui/StackedBar";
import { CollapsibleSection } from "../../components/ui/CollapsibleSection";
import {
  IconCheck, IconCpu, IconDatabase, IconSliders, IconSettings,
  IconRocket, IconRewind,
} from "../../components/ui/icons";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useTrainingStore } from "../../store/trainingStore";
import { useUiStore } from "../../store/uiStore";
import { estimateVramBreakdown, type VramBreakdown, type VramEstimateOptions } from "../../lib/vramEstimate";

const VGG_LAYERS = [
  "relu1_1","relu1_2","relu2_1","relu2_2",
  "relu3_1","relu3_2","relu3_3","relu3_4",
  "relu4_1","relu4_2","relu4_3","relu4_4",
  "relu5_1","relu5_2","relu5_3","relu5_4",
];

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
        padding: "4px 7px", fontSize: 11.5, color: "var(--text)", fontFamily: "var(--font-mono)",
        width: "100%", outline: "none", boxSizing: "border-box" as const,
        opacity: disabled ? 0.4 : 1, cursor: disabled ? "not-allowed" : undefined,
        transition: "border-color 0.15s",
      }}
      onFocus={(e) => (e.currentTarget.style.borderColor = "var(--green)")}
      onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
    />
  );
}

interface WeightInputProps {
  value: number;
  onChange: (v: number) => void;
}
function WeightInput({ value, onChange }: WeightInputProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (Number.isFinite(value) ? String(value) : "");
  const sliderVal = Math.min(1, Math.max(0.01, Number.isFinite(value) ? value : 0.01));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <input
        type="range"
        min={0.01}
        max={1}
        step={0.01}
        value={sliderVal}
        onChange={(e) => {
          setDraft(null);
          onChange(Number(e.target.value));
        }}
        style={{ flex: 1, minWidth: 0, accentColor: "var(--green)", height: 4, cursor: "pointer" }}
      />
      <input
        type="number"
        min={0}
        step={0.01}
        value={shown}
        onChange={(e) => {
          setDraft(e.target.value);
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(n);
        }}
        onBlur={() => setDraft(null)}
        style={{
          width: 76, flexShrink: 0,
          background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
          padding: "4px 7px", fontSize: 11.5, color: "var(--text)", fontFamily: "var(--font-mono)",
          outline: "none", boxSizing: "border-box" as const,
          transition: "border-color 0.15s",
        }}
        onFocus={(e) => (e.currentTarget.style.borderColor = "var(--green)")}
        onBlurCapture={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
      />
    </div>
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
        padding: "4px 7px", fontSize: 11.5, color: "var(--text)", fontFamily: "var(--font-sans)",
        width: "100%", outline: "none", boxSizing: "border-box" as const,
        transition: "border-color 0.15s",
      }}
      onFocus={(e) => (e.currentTarget.style.borderColor = "var(--green)")}
      onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
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

/** Small uppercase divider used to break Hyperparameters into scannable groups. */
function GroupLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 9.5, color: "var(--dim)", textTransform: "uppercase", letterSpacing: "0.08em",
      fontFamily: "var(--font-sans)", fontWeight: 600, marginBottom: 5,
    }}>
      {children}
    </div>
  );
}

/** A field label plus an inline hover hint. */
function LabelWithHint({ label, hint }: { label: string; hint: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      {label} <Tooltip text={hint} />
    </span>
  );
}

interface LayerMultiSelectProps {
  layers: string[];
  onChange: (layers: string[]) => void;
}
function LayerMultiSelect({ layers, onChange }: LayerMultiSelectProps) {
  const [open, setOpen] = useState(false);
  const [menuStyle, setMenuStyle] = useState<CSSProperties>({});
  const triggerRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (triggerRef.current && !triggerRef.current.contains(target) &&
          menuRef.current && !menuRef.current.contains(target)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("click", handleClick);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("click", handleClick);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const handleOpen = () => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setMenuStyle({
      position: "fixed",
      top: Math.min(rect.bottom + 2, window.innerHeight - 250),
      left: rect.left,
      width: Math.max(rect.width, 200),
      zIndex: 2000,
    });
    setOpen((o) => !o);
  };

  const toggle = (layer: string) => {
    const next = layers.includes(layer)
      ? layers.filter((l) => l !== layer)
      : [...layers, layer];
    onChange(next.length > 0 ? next : layers);
  };

  const triggerBase: CSSProperties = {
    background: "var(--bg3)",
    border: `1px solid ${open ? "var(--green)" : "var(--border)"}`,
    borderRadius: "var(--radius-sm)",
    padding: "5px 8px",
    fontSize: 12,
    color: layers.length > 0 ? "var(--text)" : "var(--dim)",
    width: "100%",
    outline: "none",
    cursor: "pointer",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    userSelect: "none",
    boxSizing: "border-box",
    transition: "border-color 0.15s",
  };

  const menu = open
    ? ReactDOM.createPortal(
        <div ref={menuRef} style={{
          ...menuStyle,
          background: "var(--bg2)",
          border: "1px solid var(--border2)",
          borderRadius: "var(--radius-sm)",
          boxShadow: "0 8px 28px rgba(0,0,0,0.5)",
          maxHeight: 220,
          overflowY: "auto",
        }}>
          {VGG_LAYERS.map((layer) => {
            const active = layers.includes(layer);
            return (
              <div
                key={layer}
                onClick={() => toggle(layer)}
                style={{
                  padding: "5px 9px",
                  fontSize: 12,
                  cursor: "pointer",
                  color: active ? "var(--green)" : "var(--text)",
                  background: active ? "var(--green-dim)" : "transparent",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  transition: "background 0.1s",
                  fontFamily: "var(--font-mono)",
                }}
                onMouseEnter={(e) => {
                  if (!active)
                    (e.currentTarget as HTMLDivElement).style.background = "var(--bg3)";
                }}
                onMouseLeave={(e) => {
                  if (!active)
                    (e.currentTarget as HTMLDivElement).style.background = "transparent";
                }}
              >
                <span style={{
                  width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: active ? "var(--green)" : "transparent",
                  border: `1px solid ${active ? "var(--green)" : "var(--border2)"}`,
                  fontSize: 9, color: active ? "#fff" : "transparent",
                  transition: "all 0.1s",
                }}>
                  {active ? "✓" : ""}
                </span>
                {layer}
              </div>
            );
          })}
        </div>,
        document.body,
      )
    : null;

  return (
    <>
      <div ref={triggerRef} style={triggerBase} onClick={handleOpen}>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
          {layers.length === 0
            ? "Select layers…"
            : `${layers.length} layer${layers.length === 1 ? "" : "s"}`}
        </span>
        <span style={{
          color: "var(--muted)", marginLeft: 8, flexShrink: 0,
          transition: "transform 0.15s",
          transform: open ? "rotate(180deg)" : "none",
          fontSize: 12,
        }}>▾</span>
      </div>
      {menu}
    </>
  );
}

interface ReadinessItemProps {
  done: boolean;
  label: string;
  optional?: boolean;
}
function ReadinessItem({ done, label, optional }: ReadinessItemProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 13, height: 13, borderRadius: "50%", flexShrink: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: done ? "var(--green-dim)" : "transparent",
          border: `1px solid ${done ? "var(--green)" : "var(--border2)"}`,
        }}
      >
        {done && <IconCheck size={8} color="var(--green)" strokeWidth={3} />}
      </span>
      <span style={{ fontSize: 10.5, color: done ? "var(--text)" : "var(--muted)" }}>
        {label}
        {optional && <span style={{ color: "var(--dim)" }}> (optional)</span>}
      </span>
    </div>
  );
}

const VRAM_SEGMENT_COLORS: Record<string, string> = {
  "Model weights": "var(--blue)",
  "Gradients": "var(--purple)",
  "Adam optimizer": "var(--cyan)",
  "Activations": "var(--amber)",
  "Input batch": "var(--pink)",
  "Upsampler": "var(--green)",
  "Allocator overhead": "var(--orange)",
  "CUDA context": "var(--dim)",
};

export function ScreenTrainingSetup() {
  const s = useRunConfigStore();
  const workspaceReady = useUiStore((s) => s.workspaceReady);

  const [instances, setInstances] = useState<{ value: string; label: string }[]>([]);
  const [datasets, setDatasets] = useState<{ value: string; label: string; path: string; pairs: number; scale: number }[]>([]);
  const [customConfigPath, setCustomConfigPath] = useState("");
  const [gpuTotalVramGb, setGpuTotalVramGb] = useState<number | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [instancesError, setInstancesError] = useState<string | null>(null);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [fetchKey, setFetchKey] = useState(0);

  const refreshLists = useCallback(() => setFetchKey((k) => k + 1), []);

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
        setInstancesError(null);
        const storedInstance = useRunConfigStore.getState().selectedInstance;
        if (storedInstance && !list.some((i: { name: string }) => i.name === storedInstance)) {
          useRunConfigStore.getState().setSelectedInstance(null);
        }
      } catch (e) {
        setInstances([]);
        setInstancesError(e instanceof Error ? e.message : String(e));
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceReady, fetchKey]);

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
        setDatasetsError(null);
        const storedDataset = useRunConfigStore.getState().selectedDataset;
        if (storedDataset) {
          const match = list.find((d: { name: string }) => d.name === storedDataset);
          if (match) {
            useRunConfigStore.getState().setSelectedDatasetPath(match.path);
            useRunConfigStore.getState().setSelectedDatasetPairs(match.num_pairs);
          } else {
            useRunConfigStore.getState().setSelectedDataset(null);
            useRunConfigStore.getState().setSelectedDatasetPath(null);
            useRunConfigStore.getState().setSelectedDatasetPairs(null);
          }
        }
      } catch (e) {
        setDatasets([]);
        setDatasetsError(e instanceof Error ? e.message : String(e));
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceReady, fetchKey]);

  // Re-hydrate instance details when selectedInstance changes
  const prevInstanceRef = useRef(s.selectedInstance);
  const selInst = useRunConfigStore((st) => st.selectedInstance);
  useEffect(() => {
    const name = selInst;
    const changed = name !== prevInstanceRef.current;
    prevInstanceRef.current = name;
    const st = useRunConfigStore.getState();
    if (!name) {
      st.setInstanceArchitecture(null);
      st.setInstanceScale(null);
      st.setInstanceConfig(null);
      st.setInstanceVersions([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const { getInstance, getInstanceVersions } = await import("../../lib/api");
        const inst = await getInstance(name);
        if (cancelled) return;
        const cur = useRunConfigStore.getState();
        cur.setInstanceArchitecture(inst.architecture);
        cur.setInstanceScale(inst.scale ?? null);
        cur.setInstanceConfig(inst.config ?? null);
        const versions = await getInstanceVersions(name);
        if (cancelled) return;
        const available = versions.filter((v: { has_weights?: boolean }) => v.has_weights !== false);
        const versionList = available.map((v: { tag: string }) => ({ tag: v.tag, path: "" }));
        const cur2 = useRunConfigStore.getState();
        cur2.setInstanceVersions(versionList);
        if (changed && cur2.resumeFrom === null && versionList.length > 0) {
          cur2.setResumeFrom("latest");
        }
      } catch {
        if (cancelled) return;
        const cur = useRunConfigStore.getState();
        cur.setInstanceArchitecture(null);
        cur.setInstanceScale(null);
        cur.setInstanceConfig(null);
        cur.setInstanceVersions([]);
        cur.setSelectedInstance(null);
      }
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selInst]);

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

  const itersPerEpoch = s.selectedDatasetPairs && s.batchSize > 0
    ? Math.ceil(s.selectedDatasetPairs / s.batchSize)
    : 0;
  const totalIters = itersPerEpoch * s.schedule.totalEpochs;

  const instCfg = s.instanceConfig as Record<string, unknown> | undefined;
  const vramOptions: VramEstimateOptions = {
    arch: s.instanceArchitecture ?? "",
    batchSize: s.batchSize,
    patchSize: s.patchSize,
    fp16: s.fp16,
    scale: s.instanceScale ?? 4,
    config: instCfg,
    gradientCheckpointing: s.gradientCheckpointing === "auto"
      ? s.instanceArchitecture === "swinir"
      : s.gradientCheckpointing === "true",
  };
  const vramBreakdown: VramBreakdown = s.instanceArchitecture
    ? estimateVramBreakdown(vramOptions)
    : { totalGb: 0, weightsGb: 0, gradsGb: 0, adamGb: 0, activationsGb: 0, inputGb: 0, upsamplerGb: 0, overheadGb: 0 };
  const vramEst = vramBreakdown.totalGb;

  const vm = s.vramMeasure;
  const isMeasuredStale =
    vm.status === "done" &&
    vm.measuredFor !==
      JSON.stringify([
        s.instanceArchitecture,
        s.instanceConfig,
        s.instanceScale,
        s.batchSize,
        s.patchSize,
        s.fp16,
        s.gradientCheckpointing,
      ]);
  const hasFreshMeasure = vm.status === "done" && !isMeasuredStale;
  const displayGb = hasFreshMeasure
    ? (vm.reservedMb ?? vm.allocatedMb ?? 0) / 1024
    : vramEst;
  const isOom = gpuTotalVramGb !== null && displayGb > (gpuTotalVramGb - 1.0);
  const vramEstColor = isOom
    ? "var(--red, #ef4444)"
    : hasFreshMeasure
      ? "var(--green, #22c55e)"
      : undefined;

  const trainingActive = useTrainingStore((st) => st.status !== "idle");
  const measureDisabled = !s.instanceArchitecture || !s.instanceConfig || vm.status === "running" || trainingActive;

  const handleMeasureVram = useCallback(async () => {
    if (measureDisabled) return;
    s.setVramMeasure({ status: "running", allocatedMb: null, reservedMb: null, error: null, measuredFor: null, measuredAt: null });
    const signature = JSON.stringify([
      s.instanceArchitecture, s.instanceConfig, s.instanceScale,
      s.batchSize, s.patchSize, s.fp16, s.gradientCheckpointing,
    ]);
    try {
      const { estimateTrainingVram } = await import("../../lib/api");
      const res = await estimateTrainingVram({
        model_name: s.instanceArchitecture!,
        config: s.instanceConfig ?? undefined,
        batch_size: s.batchSize,
        patch_size: s.patchSize,
        dtype: s.fp16 ? "bf16" : "float32",
        scale: s.instanceScale ?? 4,
        gradient_checkpointing: s.gradientCheckpointing,
        loss_config: s.lossConfig,
      });
      if (res.error) {
        s.setVramMeasure({ status: "error", allocatedMb: null, reservedMb: null, error: res.error, measuredFor: null, measuredAt: null });
      } else if (res.peak_reserved_mb ?? res.peak_allocated_mb) {
        s.setVramMeasure({
          status: "done",
          allocatedMb: res.peak_allocated_mb ?? null,
          reservedMb: res.peak_reserved_mb ?? null,
          error: null,
          measuredFor: signature,
          measuredAt: new Date().toLocaleTimeString(),
        });
      }
    } catch {
      s.setVramMeasure({ status: "error", allocatedMb: null, reservedMb: null, error: "Probe failed", measuredFor: null, measuredAt: null });
    }
  }, [measureDisabled, s]);

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
        validation_split_seed: s.validationSplitSeed,
        validation_full_image_limit: s.validationFullImageLimit,
        validation_dataset: s.selectedValidationDataset ?? undefined,
        metrics_frequency: s.metricsFrequency,
        write_metrics_file: s.writeMetricsFile,
        perceptual_weight: undefined,
        losses: s.lossConfig,
        warmup_steps: s.schedule.warmupSteps,
        gradient_checkpointing: s.gradientCheckpointing,
      });

      useTrainingStore.getState().reset();
      useTrainingStore.getState().setActiveRun(res.job_id);
      useTrainingStore.getState().setActiveRunDir(res.run_id ?? null);
      useTrainingStore.getState().setStatus("running");
      useTrainingStore.getState().setLaunchConfig({
        totalEpochs: s.schedule.totalEpochs,
        batchSize: s.batchSize,
        learningRate: s.learningRate,
        fp16: s.fp16,
        patchSize: s.patchSize,
        validationEnabled: s.validationEnabled,
      });
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

  const selectedDatasetMeta = datasets.find((d) => d.value === s.selectedDataset);
  const scaleMismatch = !!(s.instanceScale && selectedDatasetMeta && selectedDatasetMeta.scale !== s.instanceScale);

  const vramCoreGb =
    vramBreakdown.weightsGb +
    vramBreakdown.gradsGb +
    vramBreakdown.adamGb +
    vramBreakdown.activationsGb +
    vramBreakdown.inputGb +
    vramBreakdown.upsamplerGb;
  const allocatorOverheadGb = Math.max(0, vramBreakdown.totalGb - vramCoreGb - vramBreakdown.overheadGb);

  const vramSegments: StackedBarSegment[] = [
    { label: "Model weights", value: vramBreakdown.weightsGb, color: VRAM_SEGMENT_COLORS["Model weights"] },
    { label: "Gradients", value: vramBreakdown.gradsGb, color: VRAM_SEGMENT_COLORS["Gradients"] },
    { label: "Adam optimizer", value: vramBreakdown.adamGb, color: VRAM_SEGMENT_COLORS["Adam optimizer"] },
    { label: "Activations", value: vramBreakdown.activationsGb, color: VRAM_SEGMENT_COLORS["Activations"] },
    { label: "Input batch", value: vramBreakdown.inputGb, color: VRAM_SEGMENT_COLORS["Input batch"] },
    { label: "Upsampler", value: vramBreakdown.upsamplerGb, color: VRAM_SEGMENT_COLORS["Upsampler"] },
    { label: "Allocator overhead", value: allocatorOverheadGb, color: VRAM_SEGMENT_COLORS["Allocator overhead"] },
    { label: "CUDA context", value: vramBreakdown.overheadGb, color: VRAM_SEGMENT_COLORS["CUDA context"] },
  ];

  const totalLossWeight = Object.values(s.lossConfig).reduce((sum, e) => sum + (e.weight || 0), 0);

  const launchSummary = [
    s.instanceArchitecture,
    s.selectedDataset,
    `${s.schedule.totalEpochs} epochs`,
  ].filter(Boolean).join(" · ");

  return (
    <div className="ts-layout">
      <div className="ts-main">

        {/* Model & Data — unified panel: instance, dataset and run basics */}
        <Panel title="Model & Data" icon={<><IconCpu size={13} /><IconDatabase size={13} /></>}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div className="ts-group">
              <GroupLabel>
                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  Model
                  <button
                    onClick={refreshLists}
                    style={{
                      marginLeft: "auto",
                      background: "none", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
                      color: "var(--muted)", cursor: "pointer", fontSize: 10, padding: "3px 8px",
                      textTransform: "none",
                    }}
                  >
                    Refresh
                  </button>
                </span>
              </GroupLabel>
              {instancesError ? (
                <InlineAlert tone="red">
                  Failed to load instances: {instancesError}
                  <div style={{ marginTop: 6 }}>
                    <button
                      onClick={refreshLists}
                      style={{
                        background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
                        color: "var(--text)", cursor: "pointer", fontSize: 10, padding: "3px 10px",
                      }}
                    >
                      Retry
                    </button>
                  </div>
                </InlineAlert>
              ) : instances.length === 0 ? (
                <InlineAlert tone="amber">
                  No model instances found. Create one in the Model Config tab first.
                </InlineAlert>
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
                    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                      <Tag color="blue">{s.instanceArchitecture}</Tag>
                      <Tag color="purple">{s.instanceScale ?? "?"}×</Tag>
                      <Tag color="cyan">{s.instanceVersions.length} version{s.instanceVersions.length === 1 ? "" : "s"}</Tag>
                    </div>
                  )}
                  {s.instanceVersions.length > 0 && (
                    <div style={{
                      display: "flex", alignItems: "center", gap: 7, padding: "6px 9px",
                      background: "var(--bg2)",
                      border: "1px solid var(--border)",
                      borderLeft: `3px solid ${s.resumeFrom ? "var(--green)" : "var(--border2)"}`,
                      borderRadius: "var(--radius-sm)",
                    }}>
                      <IconRewind size={12} color={s.resumeFrom ? "var(--green)" : "var(--muted)"} />
                      <span style={{ fontSize: 11, color: "var(--muted)", flexShrink: 0 }}>Resume from</span>
                      <div style={{ flex: "0 1 140px", minWidth: 90 }}>
                        <Dropdown
                          value={s.resumeFrom ?? "latest"}
                          options={versionOptions}
                          onChange={(v) => s.setResumeFrom(v)}
                        />
                      </div>
                      <button
                        onClick={() => s.setResumeFrom(null)}
                        style={{
                          marginLeft: "auto", background: "none", border: "1px solid var(--border)",
                          borderRadius: "var(--radius-sm)", color: "var(--muted)", cursor: "pointer",
                          fontSize: 10, padding: "3px 8px", flexShrink: 0,
                        }}
                      >
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
            </div>

            <div className="ts-group">
              <GroupLabel>Data</GroupLabel>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {datasetsError ? (
                  <InlineAlert tone="red">
                    Failed to load datasets: {datasetsError}
                    <div style={{ marginTop: 6 }}>
                      <button
                        onClick={refreshLists}
                        style={{
                          background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
                          color: "var(--text)", cursor: "pointer", fontSize: 10, padding: "3px 10px",
                        }}
                      >
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
                  <div style={{
                    display: "flex", alignItems: "center", gap: 8, padding: "6px 9px",
                    background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
                  }}>
                    <Tag color={scaleMismatch ? "amber" : "green"}>{selectedDatasetMeta.scale}×</Tag>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>{selectedDatasetMeta.pairs.toLocaleString()} pairs</span>
                    <span style={{
                      fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)", overflow: "hidden",
                      textOverflow: "ellipsis", whiteSpace: "nowrap", marginLeft: "auto", minWidth: 0,
                    }} title={selectedDatasetMeta.path}>
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
                <div className="ts-grid" style={{ "--ts-grid-min": "140px", "--ts-grid-max": "220px" } as CSSProperties}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, height: 23 }}>
                    <Toggle on={s.validationEnabled} onChange={() => s.setValidationEnabled(!s.validationEnabled)} />
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>Val enabled</span>
                  </div>
                  <Field label="Val split">
                    <NumInput value={s.validationSplit} onChange={s.setValidationSplit}
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
        </Panel>

        {/* Hyperparameters */}
        <Panel title="Hyperparameters" icon={<IconSliders size={13} />}>
          <div className="ts-group" style={{ marginBottom: 10 }}>
            <GroupLabel>Schedule</GroupLabel>
            <div className="ts-grid" style={{ "--ts-grid-min": "100px", "--ts-grid-max": "160px" } as CSSProperties}>
              <Field label={<LabelWithHint label="Total Epochs" hint="Number of full passes over the training dataset." />}>
                <NumInput value={s.schedule.totalEpochs} onChange={(v) => s.setSchedule({ totalEpochs: v })} min={1} />
              </Field>
              <Field label="Batch Size"><NumInput value={s.batchSize} onChange={s.setBatchSize} min={1} max={128} /></Field>
              <Field label={<LabelWithHint label="Patch Size" hint="Crop size (px) fed to the model each step. Larger patches use more VRAM." />}>
                <NumInput value={s.patchSize} onChange={s.setPatchSize} min={16} max={512} step={8} />
              </Field>
              <Field label="Save Every"><NumInput value={s.schedule.saveEvery} onChange={(v) => s.setSchedule({ saveEvery: v })} min={1} /></Field>
              <Field label={<LabelWithHint label="Learning Rate" hint="Step size the optimizer takes each update. Too high can destabilize training; too low slows convergence." />}>
                <NumInput value={s.learningRate} onChange={s.setLearningRate} min={0} step={1e-5} />
              </Field>
            </div>
          </div>
          <div className="ts-group">
            <GroupLabel>Losses</GroupLabel>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {Object.entries(s.lossConfig).map(([name, entry]) => {
                const typeOpt = LOSS_TYPE_OPTIONS.find((o) => o.value === entry.type);
                return (
                  <div key={name} style={{
                    padding: "8px 10px", background: "var(--bg2)",
                    border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                      <span style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase" }}>
                        {name}
                      </span>
                      <div style={{ flex: 1, height: 4, borderRadius: 2, background: "var(--bg3)", overflow: "hidden" }}>
                        <div style={{
                          height: "100%",
                          width: `${totalLossWeight > 0 ? Math.min(100, (entry.weight / totalLossWeight) * 100) : 0}%`,
                          background: "var(--green)", borderRadius: 2, transition: "width 0.15s",
                        }} />
                      </div>
                      <span style={{ fontSize: 9.5, color: "var(--dim)", fontFamily: "var(--font-mono)", flexShrink: 0, width: 32, textAlign: "right" }}>
                        {totalLossWeight > 0 ? `${Math.round((entry.weight / totalLossWeight) * 100)}%` : "—"}
                      </span>
                      {!(name === "pixel" && ["l1", "l2"].includes(entry.type)) && (
                        <button
                          onClick={() => s.removeLoss(name)}
                          style={{
                            background: "none", border: "none", color: "var(--red, #ef4444)",
                            cursor: "pointer", fontSize: 11, padding: "2px 6px",
                            borderRadius: "var(--radius-sm)", lineHeight: 1,
                          }}
                          title="Remove loss"
                        >✕</button>
                      )}
                    </div>
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-start" }}>
                      <div style={{ flex: "1 1 150px", minWidth: 150 }}>
                        <Field label="Type">
                          <div style={{ position: "relative" }}>
                            <Dropdown
                              value={entry.type}
                              options={LOSS_TYPE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
                              onChange={(newType) => {
                                const opt = LOSS_TYPE_OPTIONS.find((o) => o.value === newType);
                                const next = { ...s.lossConfig };
                                next[name] = { type: newType as LossType, weight: entry.weight };
                                if (opt?.needsLayers) {
                                  next[name].layers = newType === "vgg"
                                    ? ["relu5_4"]
                                    : ["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_2"];
                                } else {
                                  delete next[name].layers;
                                }
                                s.setLossConfig(next);
                              }}
                            />
                          </div>
                        </Field>
                      </div>
                      <div style={{ flex: "1 1 200px", minWidth: 200 }}>
                        <Field label="Weight">
                          <WeightInput value={entry.weight} onChange={(v) => s.setLossWeight(name, v)} />
                        </Field>
                      </div>
                    </div>
                    {typeOpt?.needsLayers && entry.layers && (
                      <div style={{ marginTop: 6 }}>
                        <Field label={<LabelWithHint label="Layers"
                          hint={entry.type === "vgg"
                            ? "Which VGG19 feature layers to compare. Lower layers (relu1_1–2_2) capture edges & textures; higher layers (relu4_4–5_4) capture semantic content."
                            : "Compute Gram matrices on these VGG19 layers. Lower layers (relu1_1–2_2) capture fine texture patterns; higher layers (relu4_4–5_2) capture spatial layout."}
                        />}>
                          <LayerMultiSelect
                            layers={entry.layers}
                            onChange={(layers) => {
                              const next = { ...s.lossConfig };
                              next[name] = { ...next[name], layers };
                              s.setLossConfig(next);
                            }}
                          />
                        </Field>
                      </div>
                    )}
                  </div>
                );
              })}
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {LOSS_TYPE_OPTIONS
                  .filter((opt) => !Object.values(s.lossConfig).some((e) => e.type === opt.value))
                  .map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => {
                        s.addLoss(opt.value);
                      }}
                      style={{
                        background: "var(--bg3)", border: "1px dashed var(--border)",
                        borderRadius: "var(--radius-sm)", color: "var(--muted)", cursor: "pointer",
                        fontSize: 10, padding: "3px 8px", lineHeight: 1,
                      }}
                    >
                      + {opt.label}
                    </button>
                  ))}
              </div>
            </div>
          </div>
        </Panel>

        {/* Advanced — collapsed by default to keep the common path uncluttered */}
        <CollapsibleSection
          title="Advanced"
          icon={<IconSettings size={13} />}
          subtitle="run · optimizer · validation · monitoring"
          defaultOpen={false}
          contentStyle={{ maxHeight: "min(48vh, 320px)", overflowY: "auto", overscrollBehavior: "contain" }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div className="ts-group">
              <GroupLabel>Run &amp; Schedule</GroupLabel>
              <div className="ts-grid" style={{ "--ts-grid-min": "150px", "--ts-grid-max": "240px" } as CSSProperties}>
                <Field label="Device">
                  <Dropdown value={s.device} options={deviceOptions} onChange={s.setDevice} />
                </Field>
                <Field label="Warmup Steps">
                  <NumInput value={s.schedule.warmupSteps} onChange={(v) => s.setSchedule({ warmupSteps: v })} min={0} />
                </Field>
                <Field label="Precision">
                  <div style={{ display: "flex", alignItems: "center", gap: 8, height: 23 }}>
                    <Toggle on={s.fp16} onChange={() => s.setFp16(!s.fp16)} />
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>BF16 mixed precision</span>
                  </div>
                </Field>
                <Field label={<LabelWithHint label="Checkpointing" hint="Recompute activations during backward to save VRAM (~4-6x for transformers). On for transformer archs by default." />}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, height: 23 }}>
                    <Toggle on={s.gradientCheckpointing === "auto" ? s.instanceArchitecture === "swinir" : s.gradientCheckpointing === "true"} onChange={() => s.setGradientCheckpointing(s.gradientCheckpointing === "auto" ? "true" : s.gradientCheckpointing === "true" ? "false" : "auto")} />
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>{s.gradientCheckpointing === "auto" ? "Auto" : s.gradientCheckpointing === "true" ? "On" : "Off"}</span>
                  </div>
                </Field>
              </div>
            </div>

            <div className="ts-group">
              <GroupLabel>Optimizer</GroupLabel>
              <div className="ts-grid" style={{ "--ts-grid-min": "100px", "--ts-grid-max": "160px" } as CSSProperties}>
                <Field label="Seed"><NumInput value={s.seed} onChange={s.setSeed} min={0} /></Field>
                <Field label={<LabelWithHint label="Weight Decay" hint="L2 regularization strength applied by the Adam optimizer." />}>
                  <NumInput value={s.weightDecay} onChange={s.setWeightDecay} min={0} step={0.01} />
                </Field>
                <Field label="β₁"><NumInput value={s.betas[0]} onChange={(v) => s.setBetas([v, s.betas[1]])} min={0} max={1} step={0.01} /></Field>
                <Field label="β₂"><NumInput value={s.betas[1]} onChange={(v) => s.setBetas([s.betas[0], v])} min={0} max={1} step={0.001} /></Field>
              </div>
            </div>

            <div className="ts-group">
              <GroupLabel>Validation</GroupLabel>
              <div className="ts-grid" style={{ "--ts-grid-min": "140px", "--ts-grid-max": "220px" } as CSSProperties}>
                <Field label={<LabelWithHint label="Split seed" hint="Seeds the train/validation split only. Independent of the general seed — keeps the same images in train and validation across train phases." />}>
                  <NumInput value={s.validationSplitSeed} onChange={s.setValidationSplitSeed}
                    min={0}
                    disabled={s.selectedValidationDataset !== null} />
                </Field>
                <Field label={<LabelWithHint label="Full-image val" hint="Images per epoch for the slow tiled full-image validation pass (0 disables it). Patch-based PSNR/SSIM/loss still cover the whole validation set." />}>
                  <NumInput value={s.validationFullImageLimit} onChange={s.setValidationFullImageLimit}
                    min={0} max={64} />
                </Field>
                <Field label="Workers">
                  <NumInput value={s.numWorkers} onChange={s.setNumWorkers} min={0} max={16} />
                </Field>
              </div>
            </div>

            <div className="ts-group">
              <GroupLabel>Monitoring</GroupLabel>
              <div className="ts-grid" style={{ "--ts-grid-min": "160px", "--ts-grid-max": "280px" } as CSSProperties}>
                <Field label="Metrics Frequency">
                  <NumInput value={s.metricsFrequency} onChange={s.setMetricsFrequency} min={1} />
                </Field>
                <Field label="Custom Config YAML">
                  <TextInput value={customConfigPath} onChange={setCustomConfigPath} placeholder="path/to/config.yaml" />
                </Field>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 10 }}>
                <Toggle on={s.writeMetricsFile} onChange={() => s.setWriteMetricsFile(!s.writeMetricsFile)} />
                <span style={{ fontSize: 11, color: "var(--muted)" }}>Write metrics.jsonl file</span>
              </div>
            </div>
          </div>
        </CollapsibleSection>
      </div>

      {/* Sidebar */}
      <div className="ts-sidebar">
        {/* Readiness */}
        <Panel
          title="Readiness"
          subtitle={`${[!!s.selectedInstance, !!s.selectedDataset].filter(Boolean).length}/2`}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <ReadinessItem done={!!s.selectedInstance} label="Model instance selected" />
            <ReadinessItem done={!!s.selectedDataset} label="Dataset selected" />
          </div>
        </Panel>

        <Panel title="Estimate" style={{ flex: 1 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              <EstimateRow label="Iters / epoch" value={itersPerEpoch.toLocaleString()} />
              <EstimateRow label="Total iters" value={totalIters.toLocaleString()} />
            </div>

            <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
              <EstimateRow
                label="GPU VRAM"
                value={gpuTotalVramGb ? `${gpuTotalVramGb.toFixed(0)} GB total` : "unknown"}
                color="var(--text)"
              />
              <EstimateRow
                label="Heuristic estimate"
                value={`${vramEst.toFixed(2)} GB`}
              />
              {vramBreakdown.totalGb > 0 && (
                <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 7 }}>
                  <StackedBar segments={vramSegments} />
                  <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    {vramSegments.filter((seg) => seg.value > 0).map((seg) => (
                      <div key={seg.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ width: 7, height: 7, borderRadius: 2, background: seg.color, flexShrink: 0 }} />
                        <span style={{
                          fontSize: 10, color: "var(--muted)", flex: 1, minWidth: 0,
                          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                        }}>
                          {seg.label}
                        </span>
                        <span style={{ fontSize: 10, color: "var(--text)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>
                          {seg.value.toFixed(2)} GB
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {hasFreshMeasure && (
                <EstimateRow
                  label="Measured (real pass)"
                  value={`${displayGb.toFixed(1)} GB${vm.reservedMb ? " (reserved)" : ""}`}
                  color={vramEstColor}
                />
              )}
              {hasFreshMeasure && vramEst > 0 && (() => {
                const ratio = displayGb / vramEst;
                if (ratio > 1.5) {
                  return (
                    <span style={{ fontSize: 9.5, color: "var(--amber)", lineHeight: 1.3 }}>
                      Measured exceeds heuristic — includes MIOpen kernel workspace and allocator fragmentation the heuristic can't model.
                    </span>
                  );
                }
                if (ratio < 0.6) {
                  return (
                    <span style={{ fontSize: 9.5, color: "var(--muted)", lineHeight: 1.3 }}>
                      Measured is well below heuristic — heuristic is conservative.
                    </span>
                  );
                }
                return null;
              })()}

              <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 5 }}>
                <Btn
                  variant="solid"
                  color="var(--green)"
                  onClick={handleMeasureVram}
                  disabled={measureDisabled}
                  full
                  style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}
                  title={trainingActive ? "Disabled while training runs" : undefined}
                >
                  {vm.status === "running" ? "Measuring…" : "Measure VRAM"}
                </Btn>
                {vm.status === "running" && (
                  <span style={{ fontSize: 9.5, color: "var(--amber)", lineHeight: 1.3 }}>
                    Running one dry forward+backward on the GPU — first run autotunes kernels, can take ~1 min.
                  </span>
                )}
                {vm.status === "idle" && s.instanceArchitecture && s.instanceConfig && (
                  <span style={{ fontSize: 9.5, color: "var(--dim)", lineHeight: 1.3 }}>
                    Heuristic only. Measuring runs one real training step for an accurate number.
                  </span>
                )}
                {vm.status === "done" && isMeasuredStale && (
                  <span style={{ fontSize: 9.5, color: "var(--amber)", lineHeight: 1.3 }}>
                    Config changed since last measure — re-measure.
                  </span>
                )}
                {vm.status === "done" && vm.measuredAt && !isMeasuredStale && (
                  <span style={{ fontSize: 9.5, color: "var(--muted)", lineHeight: 1.3 }}>
                    Measured at {vm.measuredAt}
                  </span>
                )}
                {vm.status === "error" && (
                  <span style={{ fontSize: 9.5, color: "var(--red)", lineHeight: 1.3 }}>
                    {vm.error || "Probe failed"}
                  </span>
                )}
                {trainingActive && (
                  <span style={{ fontSize: 9.5, color: "var(--dim)", lineHeight: 1.3 }}>
                    Disabled while training runs
                  </span>
                )}
              </div>
            </div>

            {isOom && (
              <InlineAlert tone="red">
                {hasFreshMeasure ? "Measured" : "Estimated"} VRAM exceeds GPU capacity — may OOM
              </InlineAlert>
            )}
            {launchError && <InlineAlert tone="red">{launchError}</InlineAlert>}
          </div>
        </Panel>

        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {canLaunch && (
            <div style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)", padding: "0 2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {launchSummary}{s.resumeFrom ? ` · resume: ${s.resumeFrom}` : " · fresh run"}
            </div>
          )}
          <Btn
            variant="solid"
            color={isOom ? "var(--amber, #f59e0b)" : "var(--green)"}
            full
            onClick={handleLaunch}
            disabled={!canLaunch}
            style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 7 }}
          >
            {isOom ? (
              <>⚠ Launch Anyway (may OOM)</>
            ) : (
              <><IconRocket size={13} color="#0d0f11" /> Launch Training</>
            )}
          </Btn>
        </div>
      </div>
    </div>
  );
}