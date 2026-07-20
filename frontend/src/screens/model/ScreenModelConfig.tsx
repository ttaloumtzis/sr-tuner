import { useState, useEffect, useCallback, useMemo } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { useModelStore } from "../../store/modelStore";
import { listInstances, createInstance, getInstanceVersions, deleteInstance } from "../../lib/api";
import { useToast } from "../../components/shell/ToastProvider";
import type { Architecture } from "../../lib/srproj";
import type { ModelInstance, ModelVersion } from "../../lib/api-types";

// ── Config field descriptors ─────────────────────────────────────────

interface SliderField {
  type: "slider";
  kind: "int" | "float";
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

interface DropdownField {
  type: "dropdown";
  key: string;
  label: string;
  options: (number | string)[];
  default: number | string;
}

interface TextField {
  type: "text";
  key: string;
  label: string;
}

type ConfigField = SliderField | DropdownField | TextField;

interface ArchDef {
  id: Architecture;
  tag: string;
  description: string;
  vram: string;
  params: string;
  fields: ConfigField[];
}

// ── Architecture definitions ──────────────────────────────────────────

const RRDB_FIELDS: ConfigField[] = [
  { type: "dropdown", key: "scale", label: "Scale Factor", options: [1, 2, 4, 8], default: 4 },
  { type: "slider", kind: "int", key: "num_feat", label: "Base Features", min: 32, max: 256, step: 8, default: 64 },
  { type: "slider", kind: "int", key: "num_block", label: "RRDB Blocks", min: 4, max: 48, step: 1, default: 23 },
  { type: "slider", kind: "int", key: "num_grow_ch", label: "Growth Channels", min: 16, max: 128, step: 8, default: 32 },
  { type: "dropdown", key: "num_in_ch", label: "Input Channels", options: [1, 3], default: 3 },
  { type: "dropdown", key: "num_out_ch", label: "Output Channels", options: [1, 3], default: 3 },
];

const SWINIR_FIELDS: ConfigField[] = [
  { type: "dropdown", key: "scale", label: "Scale Factor", options: [1, 2, 4, 8], default: 4 },
  { type: "slider", kind: "int", key: "embed_dim", label: "Embedding Dim", min: 60, max: 384, step: 12, default: 180 },
  { type: "slider", kind: "int", key: "window_size", label: "Window Size", min: 4, max: 16, step: 2, default: 8 },
  { type: "slider", kind: "float", key: "mlp_ratio", label: "MLP Ratio", min: 1.0, max: 4.0, step: 0.1, default: 2.0 },
  { type: "text", key: "depths", label: "Depths" },
  { type: "text", key: "num_heads", label: "Num Heads" },
  { type: "dropdown", key: "upsampler", label: "Upsampler", options: ["pixelshuffle", "nearest+conv"], default: "pixelshuffle" },
  { type: "slider", kind: "float", key: "img_range", label: "Image Range", min: 0.5, max: 2.0, step: 0.1, default: 1.0 },
  { type: "dropdown", key: "num_in_ch", label: "Input Channels", options: [1, 3], default: 3 },
  { type: "dropdown", key: "num_out_ch", label: "Output Channels", options: [1, 3], default: 3 },
  { type: "text", key: "rgb_mean", label: "RGB Mean" },
];

const ARCH_DEFS: ArchDef[] = [
  {
    id: "rrdb_esrgan",
    tag: "GAN · Perceptual",
    description: "Best visual quality for textures and fine detail. Uses a discriminator network.",
    vram: "~8 GB",
    params: "~16M",
    fields: RRDB_FIELDS,
  },
  {
    id: "swinir",
    tag: "Transformer · PSNR",
    description: "Swin Transformer backbone. Best PSNR/SSIM scores, no adversarial training.",
    vram: "~6 GB",
    params: "~11M",
    fields: SWINIR_FIELDS,
  },
];

const FIELD_DEFAULTS: Record<Architecture, Record<string, unknown>> = {
  rrdb_esrgan: {
    scale: 4, num_feat: 64, num_block: 23, num_grow_ch: 32,
    num_in_ch: 3, num_out_ch: 3,
  },
  swinir: {
    scale: 4, embed_dim: 180, window_size: 8, mlp_ratio: 2.0,
    depths: "6,6,6,6,6,6", num_heads: "6,6,6,6,6,6",
    upsampler: "pixelshuffle", img_range: 1.0,
    num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040",
  },
};

// ── Param estimation ──────────────────────────────────────────────────

function parseCSV(s: string): number[] {
  return s.split(",").map((v) => parseFloat(v.trim())).filter((n) => !isNaN(n));
}

function formatWeightMB(paramsM: number): string {
  return ((paramsM * 4) / 1024).toFixed(1);
}

function estimateParams(arch: Architecture, values: Record<string, unknown>): number {
  if (arch === "rrdb_esrgan") {
    const nf = (values.num_feat as number) ?? 64;
    const nb = (values.num_block as number) ?? 23;
    const ng = (values.num_grow_ch as number) ?? 32;
    return 16.7 * (nf / 64) ** 2 * (nb / 23) * Math.sqrt(ng / 32);
  }
  if (arch === "swinir") {
    const ed = (values.embed_dim as number) ?? 180;
    const depths = parseCSV(String(values.depths ?? "6,6,6,6,6,6"));
    const avgDepth = depths.length > 0 ? depths.reduce((a, b) => a + b, 0) / depths.length : 6;
    return 11.8 * (ed / 180) ** 2 * (avgDepth / 6);
  }
  return 0;
}

function formatParamCount(paramsM: number): string {
  if (paramsM >= 1000) return `${(paramsM / 1000).toFixed(1)} B`;
  if (paramsM >= 1) return `${paramsM.toFixed(1)} M`;
  return `${(paramsM * 1000).toFixed(0)} K`;
}

// ── Config value → YAML value serialization ──────────────────────────

function serializeConfigValue(field: ConfigField, raw: unknown): unknown {
  if (field.type === "text") {
    const s = String(raw ?? "").trim();
    if (field.key === "rgb_mean") {
      if (!s || s.toLowerCase() === "null") return null;
      const nums = parseCSV(s);
      return nums.length > 0 ? nums : null;
    }
    const nums = parseCSV(s);
    return nums.length > 0 ? nums : s;
  }
  return raw;
}

// ── Sub-pill tab ──────────────────────────────────────────────────────

function SubTabPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} role="tab" aria-selected={active}
      style={{
        background: active ? "var(--green)" : "var(--bg3)",
        border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
        color: active ? "#0d0f11" : "var(--muted)",
        fontSize: 11, fontWeight: active ? 600 : 400,
        padding: "4px 16px", borderRadius: 12,
        cursor: "pointer", transition: "var(--transition-fast)",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}

// ── Main screen ───────────────────────────────────────────────────────

export function ScreenModelConfig() {
  const subTab = useModelStore((s) => s.subTab);
  const setSubTab = useModelStore((s) => s.setSubTab);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ display: "flex", gap: 6, padding: "8px 16px 0", flexShrink: 0 }}>
        <SubTabPill label="Create Model" active={subTab === "create"} onClick={() => setSubTab("create")} />
        <SubTabPill label="Model View" active={subTab === "view"} onClick={() => setSubTab("view")} />
      </div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, paddingTop: 16 }}>
        {subTab === "create" && <ScreenModelCreate />}
        {subTab === "view" && <ScreenModelView />}
      </div>
    </div>
  );
}

// ── ScreenModelView ────────────────────────────────────────────────────

function ScreenModelView() {
  const setSubTab = useModelStore((s) => s.setSubTab);
  const { show } = useToast();

  const [instances, setInstances] = useState<ModelInstance[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [loading, setLoading] = useState(true);
  const [deletingName, setDeletingName] = useState<string | null>(null);

  const selectedModel = instances.find((m) => m.name === selectedName) ?? null;

  const fetchInstances = useCallback(async () => {
    try {
      const list = await listInstances();
      setInstances(list);
      if (selectedName && !list.find((i) => i.name === selectedName)) {
        setSelectedName(null);
        setVersions([]);
      }
    } catch {
      // keep previous state on transient errors; poll will retry
    }
    setLoading(false);
  }, [selectedName, loading]);

  const fetchVersions = useCallback(async (name: string) => {
    setLoadingVersions(true);
    try {
      const v = await getInstanceVersions(name);
      setVersions(v);
    } catch {
      setVersions([]);
    }
    setLoadingVersions(false);
  }, []);

  useEffect(() => {
    fetchInstances();
    const interval = setInterval(fetchInstances, 5000);
    return () => clearInterval(interval);
  }, [fetchInstances]);

  useEffect(() => {
    if (selectedModel) {
      fetchVersions(selectedModel.name);
    } else {
      setVersions([]);
    }
  }, [selectedModel, fetchVersions]);

  const handleDeleteConfirm = async () => {
    if (!deletingName) return;
    await deleteInstance(deletingName).catch(() => {});
    show("success", `Model "${deletingName}" deleted`);
    if (selectedName === deletingName) setSelectedName(null);
    setDeletingName(null);
    fetchInstances();
  };

  const scaleLabel = (m: ModelInstance): string => (m.scale ? `${m.scale}x` : "—");

  return (
    <div style={{ flex: 1, display: "flex", minHeight: 0, overflow: "hidden", position: "relative" }}>
      <ModelListPanel
        instances={instances}
        loading={loading}
        selectedName={selectedName}
        onSelect={setSelectedName}
        onCreateClick={() => setSubTab("create")}
        scaleLabel={scaleLabel}
      />
      <ModelDetailPanel
        model={selectedModel}
        versions={versions}
        loadingVersions={loadingVersions}
        scaleLabel={scaleLabel}
        onRefresh={() => selectedModel && fetchVersions(selectedModel.name)}
        onDeleteRequest={(name) => setDeletingName(name)}
      />
      {deletingName && (
        <DeleteConfirmScrim
          name={deletingName}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeletingName(null)}
        />
      )}
    </div>
  );
}

// ── ModelListPanel ──────────────────────────────────────────────────────

interface ModelListPanelProps {
  instances: ModelInstance[];
  loading: boolean;
  selectedName: string | null;
  onSelect: (name: string) => void;
  onCreateClick: () => void;
  scaleLabel: (m: ModelInstance) => string;
}

function ModelListPanel({ instances, loading, selectedName, onSelect, onCreateClick, scaleLabel }: ModelListPanelProps) {
  if (loading) {
    return (
      <div style={{ flex: 1, minWidth: 0, display: "flex", alignItems: "center", justifyContent: "center", borderRight: "1px solid var(--border)" }}>
        <span style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>Loading...</span>
      </div>
    );
  }

  if (instances.length === 0) {
    return (
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10, borderRight: "1px solid var(--border)" }}>
        <span style={{ fontSize: 12, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
          No model instances yet
        </span>
        <Btn variant="solid" small onClick={onCreateClick}>
          Create Model →
        </Btn>
      </div>
    );
  }

  const COL = "3fr 2fr 1fr";

  return (
    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden", borderRight: "1px solid var(--border)" }}>
      <div style={{ display: "grid", gridTemplateColumns: COL, gap: 8, padding: "6px 10px", borderBottom: "1px solid var(--border)", background: "var(--bg2)", flexShrink: 0, fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--muted)" }}>
        <span>NAME</span>
        <span>ARCH</span>
        <span>VER</span>
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        {instances.map((m) => (
          <div
            key={m.name}
            onClick={() => onSelect(m.name)}
            style={{
              display: "grid", gridTemplateColumns: COL, gap: 8, padding: "5px 10px",
              borderBottom: "1px solid var(--border)",
              background: m.name === selectedName ? "var(--bg2)" : "transparent",
              cursor: "pointer", alignItems: "center", transition: "var(--transition-fast)",
            }}
          >
            <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {m.name}
            </span>
            <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
              {m.architecture ?? scaleLabel(m)}
            </span>
            <span style={{ fontSize: 10, color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
              {m.latest_version ?? "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── ModelDetailPanel ────────────────────────────────────────────────────

interface ModelDetailPanelProps {
  model: ModelInstance | null;
  versions: ModelVersion[];
  loadingVersions: boolean;
  scaleLabel: (m: ModelInstance) => string;
  onRefresh: () => void;
  onDeleteRequest: (name: string) => void;
}

function ModelDetailPanel({ model, versions, loadingVersions, scaleLabel, onRefresh, onDeleteRequest }: ModelDetailPanelProps) {
  const fmtTimestamp = (ts: number): string => {
    try { return new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
    catch { return String(ts); }
  };

  const paramsM = useMemo(() => {
    if (!model?.config || !model?.architecture) return 0;
    return estimateParams(model.architecture as Architecture, model.config);
  }, [model]);

  if (!model) {
    return (
      <div style={{ flex: 1, background: "var(--bg1)", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>Select a model</span>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, background: "var(--bg1)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", padding: "10px 12px 6px", borderBottom: "1px solid var(--border)", flexShrink: 0, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Model Detail
      </div>

      <div style={{ flex: 1, overflow: "auto" }}>
        <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
          <DetailRow label="Name" value={model.name} />
          <DetailRow label="Arch" value={model.architecture ?? "—"} />
          <DetailRow label="Scale" value={scaleLabel(model)} />
          {model.latest_version && <DetailRow label="Latest" value={model.latest_version} />}
        </div>

        {/* Model Size */}
        {paramsM > 0 && (
          <>
            <div style={{ borderTop: "1px solid var(--border)" }} />
            <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 3 }}>
              <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>
                Model Size
              </div>
              <SizeRow label="Params" value={formatParamCount(paramsM)} />
              <SizeRow label="Weights f32" value={`${formatWeightMB(paramsM)} MB`} />
              <SizeRow label="Weights f16" value={`${((paramsM * 2) / 1024).toFixed(1)} MB`} />
            </div>
          </>
        )}

        {/* Versions */}
        <div style={{ borderTop: "1px solid var(--border)" }} />
        <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>Versions</span>
            <Btn small variant="ghost" onClick={onRefresh} disabled={loadingVersions}>
              ↻
            </Btn>
          </div>
          {loadingVersions ? (
            <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>Loading...</span>
          ) : versions.length === 0 ? (
            <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>No trained versions yet</span>
          ) : (
            versions.map((v) => (
              <VersionCard key={v.tag} version={v} fmtTimestamp={fmtTimestamp} />
            ))
          )}
        </div>

        {/* Delete */}
        <div style={{ borderTop: "1px solid var(--border)" }} />
        <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
          <Btn variant="ghost" color="var(--red)" full onClick={() => onDeleteRequest(model.name)}>
            Delete Model
          </Btn>
        </div>
      </div>
    </div>
  );
}

function DetailRow({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
      <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", minWidth: 64, flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: 11, color: "var(--text)", fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)" }}>
        {value}
      </span>
    </div>
  );
}

// ── VersionCard ─────────────────────────────────────────────────────────

interface VersionCardProps {
  version: ModelVersion;
  fmtTimestamp: (ts: number) => string;
}

function VersionCard({ version, fmtTimestamp }: VersionCardProps) {
  const meta = version.metadata;
  const ts = meta?.timestamp as number | undefined;
  const run = meta?.run as string | undefined;
  const trainCfg = meta?.train_cfg as Record<string, unknown> | undefined;

  return (
    <div style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "8px 10px", display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--green)", fontFamily: "var(--font-mono)" }}>
          {version.tag}
        </span>
        {ts && <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>{fmtTimestamp(ts)}</span>}
      </div>
      {run && <span style={{ fontSize: 9, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>{run}</span>}
      {trainCfg && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "2px 8px", marginTop: 2 }}>
          {trainCfg.max_epochs != null && <CfgChip label="epochs" value={String(trainCfg.max_epochs)} />}
          {trainCfg.batch_size != null && <CfgChip label="bs" value={String(trainCfg.batch_size)} />}
          {trainCfg.learning_rate != null && <CfgChip label="lr" value={String(trainCfg.learning_rate)} />}
          {trainCfg.patch_size != null && <CfgChip label="patch" value={String(trainCfg.patch_size)} />}
          {trainCfg.dtype != null && <CfgChip label="dtype" value={String(trainCfg.dtype)} />}
          {trainCfg.seed != null && <CfgChip label="seed" value={String(trainCfg.seed)} />}
        </div>
      )}
    </div>
  );
}

function CfgChip({ label, value }: { label: string; value: string }) {
  return (
    <span style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)", whiteSpace: "nowrap" }}>
      {label}:{value}
    </span>
  );
}

// ── DeleteConfirmScrim ──────────────────────────────────────────────────

interface DeleteConfirmScrimProps {
  name: string;
  onConfirm: () => void;
  onCancel: () => void;
}

function DeleteConfirmScrim({ name, onConfirm, onCancel }: DeleteConfirmScrimProps) {
  return (
    <div style={{ position: "absolute", inset: 0, background: "rgba(13,15,17,0.72)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }}>
      <div style={{ background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "20px 24px", width: 320, display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ fontSize: 13, color: "var(--text)", fontFamily: "var(--font-sans)", fontWeight: 600 }}>
          Delete model "{name}"?
        </div>
        <div style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)", lineHeight: 1.8 }}>
          This will permanently remove the model and all its trained versions from the workspace.
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <Btn onClick={onCancel}>Cancel</Btn>
          <Btn variant="solid" color="var(--red)" onClick={onConfirm}>Delete</Btn>
        </div>
      </div>
    </div>
  );
}

// ── ScreenModelCreate ─────────────────────────────────────────────────

function ScreenModelCreate() {
  const arch = useModelStore((s) => s.architecture);
  const setArchitecture = useModelStore((s) => s.setArchitecture);
  const setSubTab = useModelStore((s) => s.setSubTab);
  const { show } = useToast();

  const def = ARCH_DEFS.find((d) => d.id === arch) ?? ARCH_DEFS[0];
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({ ...FIELD_DEFAULTS[arch] });
  const [modelNameInput, setModelNameInput] = useState("");
  const [yaml, setYaml] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setConfigValues({ ...FIELD_DEFAULTS[arch] });
  }, [arch]);

  const buildYaml = useCallback((name: string, values: Record<string, unknown>) => {
    const lines = [`name: ${name || "<unnamed>"}`, `architecture: ${arch}`];
    const defs = ARCH_DEFS.find((d) => d.id === arch)?.fields ?? [];
    for (const f of defs) {
      if (f.key === "num_in_ch" || f.key === "num_out_ch") continue;
      if (f.type === "text") {
        const serialized = serializeConfigValue(f, values[f.key]);
        if (serialized === null) {
          lines.push(`${f.key}: null`);
        } else if (Array.isArray(serialized)) {
          lines.push(`${f.key}: [${serialized.join(", ")}]`);
        } else {
          lines.push(`${f.key}: ${serialized}`);
        }
      } else if (f.key in values) {
        lines.push(`${f.key}: ${values[f.key]}`);
      }
    }
    return lines.join("\n");
  }, [arch]);

  useEffect(() => {
    setYaml(buildYaml(modelNameInput, configValues));
  }, [modelNameInput, configValues, buildYaml]);

  const handleArchSelect = (a: Architecture) => {
    setArchitecture(a);
    setConfigValues({ ...FIELD_DEFAULTS[a] });
  };

  const handleCopyYaml = async () => {
    try {
      await navigator.clipboard.writeText(yaml);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard not available
    }
  };

  const handleChange = (key: string, value: unknown) => {
    setConfigValues((prev) => ({ ...prev, [key]: value }));
  };

  const handleCreateInstance = async () => {
    const name = modelNameInput.trim();
    if (!name) return;

    const defs = ARCH_DEFS.find((d) => d.id === arch)?.fields ?? [];
    const config: Record<string, unknown> = {};
    for (const f of defs) {
      const raw = configValues[f.key];
      if (f.type === "text") {
        config[f.key] = serializeConfigValue(f, raw);
      } else {
        config[f.key] = raw;
      }
    }

    try {
      await createInstance(name, arch, config);
      show("success", `Model "${name}" created`);
      setModelNameInput("");
      setSubTab("view");
    } catch (err) {
      show("error", `Failed to create model: ${err}`);
    }
  };

  const paramsM = estimateParams(arch, configValues);
  const weightFp32MB = formatWeightMB(paramsM);
  const weightFp16MB = ((paramsM * 2) / 1024).toFixed(1);

  return (
    <div style={{ display: "flex", flex: 1, minHeight: 0, overflow: "hidden", background: "var(--bg0)" }}>
      <ArchSelector selected={arch} onSelect={handleArchSelect} />
      <div style={{ flex: 3, display: "flex", flexDirection: "column", gap: 8, padding: "10px 8px", overflow: "auto", minWidth: 0 }}>
        <ConfigPanel fields={def.fields} values={configValues} onChange={handleChange} />
      </div>
      <div style={{ flex: 2, minWidth: 240, maxWidth: 480, borderLeft: "1px solid var(--border)", display: "flex", flexDirection: "column", padding: "0 8px", gap: 0, overflow: "hidden" }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center", paddingTop: 10, paddingBottom: 8 }}>
          <input
            value={modelNameInput}
            onChange={(e) => setModelNameInput(e.target.value)}
            placeholder="Model name"
            style={{ flex: 1, background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11, padding: "4px 6px", borderRadius: "var(--radius-sm)" }}
          />
          <Btn variant="solid" onClick={handleCreateInstance} disabled={!modelNameInput.trim()}>
            Create Instance
          </Btn>
        </div>

        <div style={{ borderTop: "1px solid var(--border)", padding: "8px 12px", flexShrink: 0 }}>
          <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>Model Size</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <SizeRow label="Params" value={formatParamCount(paramsM)} />
            <SizeRow label="Weights f32" value={`${weightFp32MB} MB`} />
            <SizeRow label="Weights f16" value={`${weightFp16MB} MB`} />
          </div>
        </div>

        <YamlPanel yaml={yaml} copied={copied} onCopy={handleCopyYaml} />
      </div>
    </div>
  );
}

function SizeRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 10, color: "var(--dim)" }}>{label}</span>
      <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text)", fontWeight: 500 }}>{value}</span>
    </div>
  );
}

// ── Architecture Config Panel ─────────────────────────────────────────

interface ConfigPanelProps {
  fields: ConfigField[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}

function ConfigPanel({ fields, values, onChange }: ConfigPanelProps) {
  return (
    <Panel title="Architecture Config">
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {fields.map((f) => (
          <ConfigFieldRow key={f.key} field={f} value={values[f.key]} onChange={(v) => onChange(f.key, v)} />
        ))}
      </div>
    </Panel>
  );
}

interface ConfigFieldRowProps {
  field: ConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}

function ConfigFieldRow({ field, value, onChange }: ConfigFieldRowProps) {
  if (field.type === "text") {
    const strVal = String(value ?? "");
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 28 }}>
        <span style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 500, minWidth: 130, flexShrink: 0 }}>{field.label}</span>
        <input
          value={strVal}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.key === "rgb_mean" ? "0.4488, 0.4371, 0.4040" : field.key === "depths" ? "6,6,6,6,6,6" : "6,6,6,6,6,6"}
          style={{
            flex: 1, background: "var(--bg3)", border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)", padding: "4px 8px",
            fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)",
            outline: "none",
          }}
        />
      </div>
    );
  }

  if (field.type === "dropdown") {
    const strValue = String(value ?? field.default);
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 28 }}>
        <span style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 500, minWidth: 130, flexShrink: 0 }}>{field.label}</span>
            <div style={{ display: "flex", gap: 4 }}>
          {field.options.map((opt) => (
            <button
              key={String(opt)}
              onClick={() => onChange(opt)}
              style={{
                background: strValue === String(opt) ? "var(--green)" : "var(--bg3)",
                border: `1px solid ${strValue === String(opt) ? "var(--green)" : "var(--border)"}`,
                color: strValue === String(opt) ? "#0d0f11" : "var(--muted)",
                fontSize: 11, fontWeight: strValue === String(opt) ? 600 : 400,
                padding: "4px 6px", borderRadius: "var(--radius-sm)",
                cursor: "pointer", transition: "var(--transition-fast)",
                flex: 1, textAlign: "center",
              }}
            >
              {String(opt)}
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (field.type === "slider") {
    const numVal = (value as number) ?? field.default;
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, minHeight: 28 }}>
          <span style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 500, minWidth: 130, flexShrink: 0 }}>{field.label}</span>
          <span style={{ fontSize: 12, color: "var(--text)", fontFamily: "var(--font-mono)", fontWeight: 600, minWidth: 40 }}>{numVal}</span>
          <input
            type="range"
            min={field.min}
            max={field.max}
            step={field.step}
            value={numVal}
            onChange={(e) => onChange(field.kind === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
            style={{ flex: 1, width: "100%", accentColor: "var(--green)", height: 4, cursor: "pointer" }}
          />
          <span style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
            {field.min} – {field.max}
          </span>
        </div>
      </div>
    );
  }

  return null;
}

// ── Architecture cards ────────────────────────────────────────────────

interface ArchSelectorProps {
  selected: Architecture;
  onSelect: (a: Architecture) => void;
}

function ArchSelector({ selected, onSelect }: ArchSelectorProps) {
  return (
    <div style={{ flex: 1, minWidth: 180, maxWidth: 280, borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 0, overflow: "auto", background: "var(--bg1)" }}>
      <div style={{ padding: "8px 12px 6px", borderBottom: "1px solid var(--border)", fontSize: 11, color: "var(--text)", fontWeight: 500, flexShrink: 0 }}>
        Architecture
      </div>
      <div style={{ display: "flex", flexDirection: "column", padding: 8, gap: 6 }}>
        {ARCH_DEFS.map((def) => (
          <ArchCard key={def.id} def={def} active={selected === def.id} onClick={() => onSelect(def.id)} />
        ))}
      </div>
    </div>
  );
}

interface ArchCardProps {
  def: ArchDef;
  active: boolean;
  onClick: () => void;
}

function ArchCard({ def, active, onClick }: ArchCardProps) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: active ? "#1a3d2a" : hovered ? "#2a2d32" : "#1a1d21",
        border: `1px solid ${active ? "#4dba7f" : hovered ? "#5d6470" : "#3a3e46"}`,
        borderRadius: 6, padding: "10px 12px", cursor: "pointer", transition: "0.15s",
        display: "flex", flexDirection: "column", gap: 5,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "space-between" }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: active ? "#4dba7f" : "#dde3ea" }}>
          {def.id}
        </span>
        {active && <span style={{ fontSize: 8, color: "var(--green)" }}>●</span>}
      </div>
      <span style={{ fontSize: 9, color: active ? "var(--green)" : "var(--accent)", fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {def.tag}
      </span>
      <span style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>
        {def.description}
      </span>
      <div style={{ display: "flex", gap: 10, marginTop: 2 }}>
        <Stat label="VRAM" value={def.vram} />
        <Stat label="Params" value={def.params} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      <span style={{ fontSize: 9, color: "var(--dim)", textTransform: "uppercase" }}>{label}</span>
      <span style={{ fontSize: 10, color: "var(--text)", fontFamily: "var(--font-mono)" }}>{value}</span>
    </div>
  );
}

// ── YAML Preview panel ────────────────────────────────────────────────

interface YamlPanelProps {
  yaml: string;
  copied: boolean;
  onCopy: () => void;
}

function YamlPanel({ yaml, copied, onCopy }: YamlPanelProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "hidden", borderTop: "1px solid var(--border)" }}>
      <div style={{ padding: "7px 12px", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
        <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>YAML Preview</span>
        <Btn small variant="ghost" onClick={onCopy}>
          {copied ? "Copied!" : "Copy"}
        </Btn>
      </div>
      <pre
        style={{
          flex: 1, overflow: "auto", margin: 0, padding: "10px 12px",
          fontSize: 9.5, lineHeight: 1.6, fontFamily: "var(--font-mono)",
          color: "var(--muted)", whiteSpace: "pre", background: "transparent",
        }}
        dangerouslySetInnerHTML={{ __html: syntaxHighlight(yaml) }}
      />
    </div>
  );
}

function syntaxHighlight(yaml: string): string {
  return yaml
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .split("\n")
    .map((line) => {
      if (/^\s*#/.test(line)) {
        return `<span style="color:var(--dim)">${line}</span>`;
      }
      const keyMatch = line.match(/^(\s*)([^:]+)(:)(.*)$/);
      if (keyMatch) {
        const [, indent, key, colon, rest] = keyMatch;
        const coloredKey = `<span style="color:var(--green)">${key}</span>`;
        const coloredRest = rest
          ? `<span style="color:var(--text)">${rest}</span>`
          : "";
        return `${indent}${coloredKey}${colon}${coloredRest}`;
      }
      if (/^\s*- /.test(line)) {
        return `<span style="color:var(--text)">${line}</span>`;
      }
      return line;
    })
    .join("\n");
}
