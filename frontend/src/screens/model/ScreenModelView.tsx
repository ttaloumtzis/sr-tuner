import { useState, useEffect, useCallback } from "react";
import "./ScreenModelView.css";
import { ConfirmScrim } from "../../components/ui/ConfirmScrim";
import { useModelStore } from "../../store/modelStore";
import { useTrainingStore } from "../../store/trainingStore";
import { listInstances, getInstanceVersions, deleteInstance, deleteVersion } from "../../lib/api";
import { useToast } from "../../components/shell/ToastProvider";
import type { ModelInstance, ModelVersion } from "../../lib/api-types";
import { InstancesSidebar } from "./InstancesSidebar";
import { DetailPanel } from "./DetailPanel";

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
          escToClose
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
          escToClose
        />
      )}
    </div>
  );
}