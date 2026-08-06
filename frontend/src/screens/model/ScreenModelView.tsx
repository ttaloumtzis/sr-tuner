import { useState, useEffect, useCallback, useMemo } from "react";
import "./ScreenModelView.css";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Tag } from "../../components/ui/Tag";
import { InfoRow } from "../../components/ui/InfoRow";
import { InlineAlert } from "../../components/ui/InlineAlert";
import { useModelStore } from "../../store/modelStore";
import { useTrainingStore } from "../../store/trainingStore";
import { listInstances, getInstanceVersions, deleteInstance, deleteVersion } from "../../lib/api";
import { useToast } from "../../components/shell/ToastProvider";
import type { ModelInstance, ModelVersion } from "../../lib/api-types";
import { estimateParamsFor, formatParamCount, formatWeightMB } from "../../lib/architectures";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmtTimestamp(ts: number | undefined | null): string {
  if (ts == null || Number.isNaN(ts)) return "—";
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

// ── Confirm Scrim (version + instance deletion) ───────────────────────────

function ConfirmScrim({
  title, message, confirmLabel, onConfirm, onCancel, danger,
}: {
  title: string;
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
  danger?: boolean;
}) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  return (
    <div className="mv-scrim">
      <div className="mv-scrim-box">
        <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text)" }}>{title}</div>
        <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>{message}</div>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <Btn onClick={onCancel}>Cancel</Btn>
          <Btn
            variant="solid"
            color={danger ? "var(--red)" : undefined}
            onClick={onConfirm}
          >
            {confirmLabel}
          </Btn>
        </div>
      </div>
    </div>
  );
}

// ── Instances Sidebar ─────────────────────────────────────────────────────

function InstancesSidebar({
  instances, loading, selectedName, onSelect, onCreateClick,
}: {
  instances: ModelInstance[];
  loading: boolean;
  selectedName: string | null;
  onSelect: (name: string) => void;
  onCreateClick: () => void;
}) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(
    () => instances.filter((m) => m.name.toLowerCase().includes(query.toLowerCase())),
    [instances, query],
  );

  if (loading) {
    return (
      <Panel title="Model Instances" style={{ flexShrink: 0 }}>
        <div style={{ padding: 16, textAlign: "center", fontSize: 12, color: "var(--muted)" }}>Loading...</div>
      </Panel>
    );
  }

  if (instances.length === 0) {
    return (
      <Panel title="Model Instances" style={{ flexShrink: 0 }}>
        <div style={{ padding: 16, textAlign: "center" }}>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>No model instances yet</div>
          <Btn variant="solid" small onClick={onCreateClick}>Create Model</Btn>
        </div>
      </Panel>
    );
  }

  return (
    <div className="mv-sidebar">
      <Panel title="Model Instances" style={{ flexShrink: 0 }} noPadding>
        <div style={{ padding: "8px 10px 4px", display: "flex", flexDirection: "column", gap: 6 }}>
          <div className="mv-sidebar-header">
            <span>Instances</span>
            <span className="mv-sidebar-count">{filtered.length} / {instances.length}</span>
          </div>
          <input
            className="mv-sidebar-search"
            type="text"
            placeholder="Search…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
          {filtered.map((m) => (
            <button
              key={m.name}
              className={`mv-instance-row${selectedName === m.name ? " selected" : ""}`}
              onClick={() => onSelect(m.name)}
            >
              <span className="mv-instance-name">{m.name}</span>
              {m.architecture && (
                <Tag color="cyan">{m.architecture}</Tag>
              )}
              {m.scale != null && (
                <span style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>{m.scale}x</span>
              )}
            </button>
          ))}
        </div>
      </Panel>
    </div>
  );
}

// ── Version Card ──────────────────────────────────────────────────────────

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

function VersionCard({
  version, onDelete, trainingActive,
}: {
  version: ModelVersion;
  onDelete: () => void;
  trainingActive: boolean;
}) {
  const m = version.metadata ?? {};
  const ts = (m as any).timestamp ?? (m as any).created_at;
  const runName = (m as any).run_name;
  const tc = (m as any).training_config;
  const missingWeights = version.has_weights === false;

  return (
    <div className={`mv-version-card${missingWeights ? " missing" : ""}`}>
      <div className="mv-version-card-top">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="mv-version-tag">{version.tag}</span>
          {missingWeights && <Tag color="red">missing</Tag>}
        </div>
        <div className="mv-version-card-actions">
          <span className="mv-version-date">{fmtTimestamp(ts)}</span>
          <button
            onClick={(ev) => { ev.stopPropagation(); onDelete(); }}
            disabled={trainingActive}
            title={trainingActive ? "Cannot delete versions while training is active" : `Delete ${version.tag}`}
            style={{
              background: "none", border: "none",
              color: trainingActive ? "var(--dim)" : "var(--red)",
              cursor: trainingActive ? "default" : "pointer",
              fontSize: 14, lineHeight: 1, padding: "0 4px",
              opacity: trainingActive ? 0.4 : 1,
              transition: "var(--transition-fast)",
            }}
          >
            ✕
          </button>
        </div>
      </div>
      {runName && (
        <div className="mv-version-run">
          Run: <span>{runName}</span>
        </div>
      )}
      {tc && (
        <div className="mv-version-chips">
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

// ── Detail Panel ──────────────────────────────────────────────────────────

function DetailPanel({
  model, versions, loadingVersions, trainingActive, onRefresh, onDeleteInstance, onDeleteVersion,
}: {
  model: ModelInstance | null;
  versions: ModelVersion[];
  loadingVersions: boolean;
  trainingActive: boolean;
  onRefresh: () => void;
  onDeleteInstance: () => void;
  onDeleteVersion: (v: ModelVersion) => void;
}) {
  const paramsM = useMemo(() => {
    if (!model) return 0;
    const config = model.config as Record<string, unknown> | undefined;
    if (!config) return 0;
    return estimateParamsFor(model.architecture ?? "", config);
  }, [model]);

  const missingCount = versions.filter((v) => v.has_weights === false).length;

  if (!model) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Select a model instance</span>
      </div>
    );
  }

  return (
    <div className="mv-detail">
      <Panel title={model.name} style={{ flexShrink: 0 }}>
        <InfoRow label="Architecture" value={model.architecture === "swinir" ? "SwinIR" : "RRDB-ESRGAN"} />
        <InfoRow label="Scale" value={model.scale ? `${model.scale}x` : "—"} />
        <InfoRow label="Latest Version" value={model.latest_version ?? "—"} mono />
        <div style={{ marginTop: 2 }}>
          <InfoRow label="Parameters" value={formatParamCount(paramsM)} mono />
          <InfoRow label="Weights (f32)" value={`${formatWeightMB(paramsM)} MB`} mono />
          <InfoRow label="Weights (f16)" value={`${(parseFloat(formatWeightMB(paramsM)) / 2).toFixed(1)} MB`} mono />
        </div>
      </Panel>

      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <div className="mv-versions-header">
          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
            Versions
          </span>
          <Btn small variant="ghost" onClick={onRefresh} disabled={loadingVersions}>&#x21bb;</Btn>
        </div>

        {loadingVersions ? (
          <span style={{ fontSize: 11, color: "var(--muted)" }}>Loading...</span>
        ) : versions.length === 0 ? (
          <span style={{ fontSize: 11, color: "var(--muted)" }}>No versions yet</span>
        ) : (
          <div className="mv-versions-list">
            {missingCount > 0 && (
              <InlineAlert tone="muted">
                {missingCount} version(s) missing weights — deleted or incomplete.
              </InlineAlert>
            )}
            {versions.map((v) => (
              <VersionCard
                key={v.tag}
                version={v}
                onDelete={() => onDeleteVersion(v)}
                trainingActive={trainingActive}
              />
            ))}
          </div>
        )}
      </div>

      <div style={{ flexShrink: 0, paddingTop: 4 }}>
        <Btn
          variant="ghost"
          color="var(--red)"
          full
          onClick={onDeleteInstance}
          disabled={trainingActive}
          title={trainingActive ? "Cannot delete model while training is active" : undefined}
        >
          Delete Model
        </Btn>
      </div>
    </div>
  );
}

// ── Root Screen ───────────────────────────────────────────────────────────

export function ScreenModelView() {
  const setSubTab = useModelStore((s) => s.setSubTab);
  const trainingActive = useTrainingStore((s) => s.status === "running");
  const { show } = useToast();

  const [instances, setInstances] = useState<ModelInstance[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [loading, setLoading] = useState(true);
  const [confirmDeleteInstance, setConfirmDeleteInstance] = useState(false);
  const [deletingVersion, setDeletingVersion] = useState<ModelVersion | null>(null);

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

  useEffect(() => {
    if (selectedName) {
      fetchVersions(selectedName);
    } else {
      setVersions([]);
    }
  }, [selectedName, fetchVersions]);

  // ── Handlers ────────────────────────────────────────────────────────────

  const handleDeleteInstanceConfirm = async () => {
    if (!selectedName) return;
    const name = selectedName;
    setConfirmDeleteInstance(false);
    try {
      await deleteInstance(name);
      show("success", `Model "${name}" deleted`);
      if (selectedName === name) setSelectedName(null);
    } catch (e: any) {
      show("error", e?.message ?? `Failed to delete "${name}"`);
    }
    fetchInstances();
  };

  const handleDeleteVersionConfirm = async () => {
    const v = deletingVersion;
    if (!v || !selectedName) return;
    const instance = selectedName;
    setDeletingVersion(null);
    try {
      await deleteVersion(instance, v.tag);
      show("success", `Version "${v.tag}" deleted`);
    } catch (e: any) {
      show("error", e?.message ?? `Failed to delete version "${v.tag}"`);
    }
    fetchVersions(instance);
    fetchInstances();
  };

  return (
    <div className="mv-layout">
      <InstancesSidebar
        instances={instances}
        loading={loading}
        selectedName={selectedName}
        onSelect={setSelectedName}
        onCreateClick={() => setSubTab("create")}
      />
      <DetailPanel
        model={selectedModel}
        versions={versions}
        loadingVersions={loadingVersions}
        trainingActive={trainingActive}
        onRefresh={() => selectedModel && fetchVersions(selectedModel.name)}
        onDeleteInstance={() => setConfirmDeleteInstance(true)}
        onDeleteVersion={(v) => setDeletingVersion(v)}
      />

      {confirmDeleteInstance && selectedModel && (
        <ConfirmScrim
          title="Delete Model?"
          message={`This will permanently delete "${selectedModel.name}" and all its checkpoints and versions.`}
          confirmLabel="Delete"
          onConfirm={handleDeleteInstanceConfirm}
          onCancel={() => setConfirmDeleteInstance(false)}
          danger
        />
      )}

      {deletingVersion && selectedModel && (
        <ConfirmScrim
          title={`Delete version "${deletingVersion.tag}"?`}
          message="This will permanently delete this version and its weights."
          confirmLabel="Delete"
          onConfirm={handleDeleteVersionConfirm}
          onCancel={() => setDeletingVersion(null)}
          danger
        />
      )}
    </div>
  );
}
