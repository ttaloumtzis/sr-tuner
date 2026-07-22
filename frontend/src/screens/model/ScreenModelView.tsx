import { useState, useEffect, useCallback, useMemo } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { InfoRow } from "../../components/ui/InfoRow";
import { useModelStore } from "../../store/modelStore";
import { listInstances, getInstanceVersions, deleteInstance } from "../../lib/api";
import { useToast } from "../../components/shell/ToastProvider";
import type { Architecture } from "../../lib/srproj";
import type { ModelInstance, ModelVersion } from "../../lib/api-types";
import { estimateParams, formatParamCount, formatWeightMB } from "./templates";

function CfgChip({ label, value }: { label: string; value: string }) {
  return (
    <span style={{
      fontSize: 10, background: "var(--bg3)", color: "var(--muted)",
      padding: "1px 6px", borderRadius: 4, whiteSpace: "nowrap",
    }}>
      {label}: <b style={{ color: "var(--text)", fontWeight: 600 }}>{value}</b>
    </span>
  );
}

function VersionCard({ version, fmtTimestamp }: { version: ModelVersion; fmtTimestamp: (ts: number) => string }) {
  const m = version.metadata ?? {};
  const createdAt = (m as any).created_at;
  const runName = (m as any).run_name;
  const tc = (m as any).training_config;
  return (
    <div style={{
      background: "var(--bg2)", border: "1px solid var(--border)",
      borderRadius: 6, padding: 10,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text)", fontFamily: "var(--font-mono)" }}>
          {version.tag}
        </span>
        <span style={{ fontSize: 10, color: "var(--muted)" }}>
          {fmtTimestamp(createdAt)}
        </span>
      </div>
      {runName && (
        <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 4 }}>
          Run: <span style={{ color: "var(--text)" }}>{runName}</span>
        </div>
      )}
      {tc && (
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 4 }}>
          {tc.epochs != null && <CfgChip label="Epochs" value={String(tc.epochs)} />}
          {tc.batch_size != null && <CfgChip label="BS" value={String(tc.batch_size)} />}
          {tc.learning_rate != null && <CfgChip label="LR" value={String(tc.learning_rate)} />}
          {tc.patch_size != null && <CfgChip label="Patch" value={`${tc.patch_size}`} />}
          {tc.dtype != null && <CfgChip label="DType" value={tc.dtype} />}
          {tc.seed != null && <CfgChip label="Seed" value={String(tc.seed)} />}
        </div>
      )}
    </div>
  );
}

function DeleteConfirmScrim({ name, onConfirm, onCancel }: { name: string; onConfirm: () => void; onCancel: () => void }) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  return (
    <div style={{
      position: "absolute", inset: 0, background: "rgba(0,0,0,0.6)",
      display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50,
    }}>
      <div style={{
        background: "var(--bg1)", border: "1px solid var(--border)", borderRadius: 12,
        padding: 24, maxWidth: 400, textAlign: "center",
      }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text)", marginBottom: 8 }}>Delete Model?</div>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 16 }}>
          This will permanently delete "<b>{name}</b>" and all its checkpoints and versions.
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          <Btn onClick={onCancel}>Cancel</Btn>
          <Btn onClick={onConfirm} variant="solid" color="var(--red)">Delete</Btn>
        </div>
      </div>
    </div>
  );
}

function ModelListPanel({ instances, loading, selectedName, onSelect, onCreateClick, scaleLabel }: {
  instances: ModelInstance[];
  loading: boolean;
  selectedName: string | null;
  onSelect: (name: string) => void;
  onCreateClick: () => void;
  scaleLabel: (m: ModelInstance) => string;
}) {
  if (loading) {
    return (
      <Panel title="Model Instances" style={{ flex: "0 0 280px", overflow: "hidden" }}>
        <div style={{ padding: 16, textAlign: "center", fontSize: 12, color: "var(--muted)" }}>Loading...</div>
      </Panel>
    );
  }
  if (instances.length === 0) {
    return (
      <Panel title="Model Instances" style={{ flex: "0 0 280px", overflow: "hidden" }}>
        <div style={{ padding: 16, textAlign: "center" }}>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>No model instances yet</div>
          <Btn variant="solid" small onClick={onCreateClick}>Create Model</Btn>
        </div>
      </Panel>
    );
  }
  return (
    <Panel title="Model Instances" style={{ flex: "0 0 280px", overflow: "hidden" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: "4px 8px", padding: "0 0 6px", fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.3px" }}>
        <span>Name</span>
        <span>Arch</span>
        <span>Scale</span>
      </div>
      <div style={{ overflowY: "auto", flex: 1 }}>
        {instances.map((m) => (
          <button
            key={m.name}
            onClick={() => onSelect(m.name)}
            style={{
              display: "grid", gridTemplateColumns: "1fr auto auto", gap: "4px 8px",
              width: "100%", textAlign: "left", padding: "6px 0",
              border: "none", borderBottom: "1px solid var(--border)",
              background: selectedName === m.name ? "var(--bg3)" : "transparent",
              cursor: "pointer", fontSize: 11, color: "var(--text)",
              fontFamily: "var(--font-mono)", transition: "var(--transition-fast)",
            }}
          >
            <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{m.name}</span>
            <span style={{ color: "var(--muted)", fontSize: 10 }}>{m.architecture}</span>
            <span style={{ color: "var(--muted)" }}>{scaleLabel(m)}</span>
          </button>
        ))}
      </div>
    </Panel>
  );
}

function ModelDetailPanel({ model, versions, loadingVersions, scaleLabel, onRefresh, onDeleteRequest }: {
  model: ModelInstance | null;
  versions: ModelVersion[];
  loadingVersions: boolean;
  scaleLabel: (m: ModelInstance) => string;
  onRefresh: () => void;
  onDeleteRequest: (name: string) => void;
}) {
  const fmtTimestamp = (ts: number): string => {
    const d = new Date(ts * 1000);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  };

  const paramsM = useMemo(() => {
    if (!model) return 0;
    const config = model.config as Record<string, unknown> | undefined;
    if (!config) return 0;
    return estimateParams(model.architecture as Architecture, config);
  }, [model]);

  if (!model) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Select a model instance</span>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8, overflow: "hidden" }}>
      <Panel title={model.name} style={{ flexShrink: 0 }}>
        <InfoRow label="Architecture" value={model.architecture === "swinir" ? "SwinIR" : "RRDB-ESRGAN"} />
        <InfoRow label="Scale" value={scaleLabel(model)} />
        <InfoRow label="Latest Version" value={model.latest_version ?? "—"} mono />
        <div style={{ marginTop: 2 }}>
          <InfoRow label="Parameters" value={formatParamCount(paramsM)} mono />
          <InfoRow label="Weights (f32)" value={`${formatWeightMB(paramsM)} MB`} mono />
          <InfoRow label="Weights (f16)" value={`${(parseFloat(formatWeightMB(paramsM)) / 2).toFixed(1)} MB`} mono />
        </div>
      </Panel>

      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
            Versions
          </span>
          <Btn small variant="ghost" onClick={onRefresh} disabled={loadingVersions}>&#x21bb;</Btn>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, overflowY: "auto", flex: 1 }}>
          {loadingVersions ? (
            <span style={{ fontSize: 11, color: "var(--muted)" }}>Loading...</span>
          ) : versions.length === 0 ? (
            <span style={{ fontSize: 11, color: "var(--muted)" }}>No versions yet</span>
          ) : (
            versions.map((v) => <VersionCard key={v.tag} version={v} fmtTimestamp={fmtTimestamp} />)
          )}
        </div>
      </div>

      <div style={{ flexShrink: 0, paddingTop: 4 }}>
        <Btn variant="ghost" color="var(--red)" full onClick={() => onDeleteRequest(model.name)}>Delete Model</Btn>
      </div>
    </div>
  );
}

export function ScreenModelView() {
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
  }, [selectedName]);

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

  // Depends on `selectedName` (a stable primitive) rather than `selectedModel` (an object
  // that gets a fresh identity every time the 5s instance poll refreshes `instances`, even
  // when nothing actually changed). Depending on the object previously re-fetched versions
  // every 5 seconds regardless of whether the selection changed.
  useEffect(() => {
    if (selectedName) {
      fetchVersions(selectedName);
    } else {
      setVersions([]);
    }
  }, [selectedName, fetchVersions]);

  const handleDeleteConfirm = async () => {
    if (!deletingName) return;
    const name = deletingName;
    setDeletingName(null);
    try {
      await deleteInstance(name);
      show("success", `Model "${name}" deleted`);
      if (selectedName === name) setSelectedName(null);
    } catch (e: any) {
      show("error", e?.message ?? `Failed to delete "${name}"`);
    }
    fetchInstances();
  };

  const scaleLabel = (m: ModelInstance): string => (m.scale ? `${m.scale}x` : "—");

  return (
    <div style={{ flex: 1, display: "flex", gap: 16, minHeight: 0, overflow: "hidden", position: "relative" }}>
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
