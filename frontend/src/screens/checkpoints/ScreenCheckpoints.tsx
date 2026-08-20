// Runs & Checkpoints Screen — 3-column layout:
//   left   — models & their runs (disk-derived status, run deletion)
//   center — checkpoints + metrics for the selected run
//   right  — selected checkpoint detail + inference stub

import { useState, useEffect, useMemo } from "react";
import "./ScreenCheckpoints.css";
import { useTrainingStore } from "../../store/trainingStore";
import { useRunsStore } from "../../store/runsStore";
import { useUiStore } from "../../store/uiStore";
import { useInferenceStore } from "../../store/inferenceStore";
import { useModelStore } from "../../store/modelStore";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useToast } from "../../components/shell/ToastProvider";
import { Tag } from "../../components/ui/Tag";
import { StatusDot } from "../../components/ui/StatusDot";

import type { CheckpointEntry, RunInfo } from "../../lib/api-types";
import type { Hyperparameters } from "../../store/modelStore";
import { buildRunDisplays, shortRunId } from "../../lib/runLabel";
import { fmtSize } from "../../lib/format";
import { CheckpointsTable, sortEntries, type SortCol, type SortDir } from "./CheckpointsTable";
import { DetailPanel } from "./DetailPanel";
import { DeleteConfirmScrim } from "./DeleteConfirmScrim";
import { ModelsRunsSidebar } from "./ModelsRunsSidebar";
import { StorageSummaryPanel } from "./StorageSummaryPanel";

function DeleteDisabledBanner() {
  return (
    <div className="sc-delete-banner">
      Delete disabled — training is active. Stop training to delete checkpoints.
    </div>
  );
}

// ── ScreenCheckpoints ─────────────────────────────────────────────────────

export function ScreenCheckpoints() {
  const status          = useTrainingStore((s) => s.status);
  const activeRunDirId  = useTrainingStore((s) => s.activeRunDirId);
  const setActiveTab    = useUiStore((s) => s.setActiveTab);
  const models          = useRunsStore((s) => s.models);
  const loading         = useRunsStore((s) => s.loading);
  const refreshCounter  = useRunsStore((s) => s.refreshCounter);
  const selectedInstance = useRunsStore((s) => s.selectedInstance);
  const selectedRunId   = useRunsStore((s) => s.selectedRunId);
  const checkpointsByRun = useRunsStore((s) => s.checkpointsByRun);
  const selectedCheckpointPath = useRunsStore((s) => s.selectedCheckpointPath);
  const { show } = useToast();

  const trainingActive = status === "running";
  const allEntries: CheckpointEntry[] = selectedRunId ? (checkpointsByRun[selectedRunId] ?? []) : [];

  const [sortCol, setSortCol]       = useState<SortCol>("epoch");
  const [sortDir, setSortDir]       = useState<SortDir>("desc");
  const [deletingEntry, setDeletingEntry] = useState<CheckpointEntry | null>(null);
  const [deletingRun, setDeletingRun] = useState<{ instance: string; run: RunInfo } | null>(null);

  // Initial load + background refresh every 15s.
  useEffect(() => {
    const { refresh } = useRunsStore.getState();
    refresh();
    const t = setInterval(() => useRunsStore.getState().refresh(), 15000);
    return () => clearInterval(t);
  }, []);

  // SSE-driven refresh (checkpoint_saved, run lifecycle events).
  useEffect(() => {
    if (refreshCounter > 0) useRunsStore.getState().refresh();
  }, [refreshCounter]);

  // A training run just started — surface it in the sidebar immediately.
  useEffect(() => {
    if (trainingActive && activeRunDirId) useRunsStore.getState().refresh();
  }, [trainingActive, activeRunDirId]);

  const selectedRun: RunInfo | null = useMemo(() => {
    if (!selectedInstance || !selectedRunId) return null;
    return models.find((m) => m.name === selectedInstance)?.runs.find((r) => r.run_id === selectedRunId) ?? null;
  }, [models, selectedInstance, selectedRunId]);

  const isActiveRun = selectedRun != null && activeRunDirId === selectedRun.run_id && trainingActive;

  const bestPsnrPath = useMemo(() => {
    let best: CheckpointEntry | null = null;
    for (const e of allEntries) {
      if (e.metrics.psnr != null && (best == null || e.metrics.psnr > (best.metrics.psnr ?? -Infinity))) {
        best = e;
      }
    }
    return best?.path ?? null;
  }, [allEntries]);

  const latestPath = allEntries[allEntries.length - 1]?.path ?? null;

  const sorted = useMemo(
    () => sortEntries(allEntries, sortCol, sortDir),
    [allEntries, sortCol, sortDir],
  );

  const selectedEntry = sorted.find((e) => e.path === selectedCheckpointPath) ?? null;

  // ── Handlers ─────────────────────────────────────────────────────────

  const handleSort = (col: SortCol) => {
    if (col === sortCol) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortCol(col); setSortDir("desc"); }
  };

  const handleDeleteRequest = (e: CheckpointEntry) => {
    if (trainingActive) return;
    setDeletingEntry(e);
  };

  const handleDeleteConfirm = () => {
    if (!deletingEntry) return;
    // TODO: replace with api call (per-checkpoint delete)
    if (selectedCheckpointPath === deletingEntry.path) useRunsStore.getState().selectCheckpoint(null);
    setDeletingEntry(null);
  };

  const handleExportPth = (_e: CheckpointEntry) => {
    // TODO: replace with api call
  };

  const handleExportOnnx = (_e: CheckpointEntry) => {
    // TODO: replace with api call
  };

  const handleRunDeleteConfirm = async () => {
    if (!deletingRun) return;
    const { instance, run } = deletingRun;
    setDeletingRun(null);
    const ok = await useRunsStore.getState().deleteRun(instance, run.run_id);
    if (ok) show("success", `Deleted run ${run.run_id}`, 3000);
  };

  // Resume prefill from the run's on-disk config snapshot (run_config.json)
  // + instance metadata from the models API.
  const handleResume = (e: CheckpointEntry) => {
    const tc = (selectedRun?.config?.train_cfg ?? {}) as Record<string, unknown>;
    const instName = selectedRun?.config?.instance as string | undefined;
    const inst = models.find((m) => m.name === instName);

    if (inst?.architecture) {
      useModelStore.getState().setArchitecture(
        (inst.architecture === "swinir" ? "swinir" : "rrdb_esrgan") as "rrdb_esrgan" | "swinir",
      );
    }

    const hp: Partial<Hyperparameters> = { scale: inst?.scale ?? 4 };
    if (typeof tc.batch_size === "number") hp.batchSize = tc.batch_size;
    if (typeof tc.learning_rate === "number") hp.learningRate = tc.learning_rate;
    if (typeof tc.scheduler === "string") hp.lrScheduler = tc.scheduler;
    if (typeof tc.optimizer === "string") hp.optimizer = tc.optimizer;
    if (typeof tc.patch_size === "number") hp.patchSize = tc.patch_size;
    useModelStore.getState().setHyperparameters(hp);

    useRunConfigStore.getState().setSelectedInstance(instName ?? null);
    useRunConfigStore.getState().setSchedule({
      totalEpochs: typeof tc.max_epochs === "number" ? tc.max_epochs : 100,
    });
    useRunConfigStore.getState().setResumeFrom(e.path);
    setActiveTab("training");
  };

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="sc-layout">
      {/* Left — models & runs */}
      <ModelsRunsSidebar
        models={models}
        selectedInstance={selectedInstance}
        selectedRunId={selectedRunId}
        activeRunDirId={activeRunDirId}
        refreshing={loading}
        onSelectRun={(instance, runId) => useRunsStore.getState().selectRun(instance, runId)}
        onDeleteRequest={(run) => {
          const instance = models.find((m) => m.runs.some((r) => r.run_id === run.run_id))?.name ?? "";
          setDeletingRun({ instance, run });
        }}
        onRefresh={() => useRunsStore.getState().refresh()}
      />

      {/* Center — checkpoints of the selected run */}
      <div className="sc-center">
        <div className="sc-run-header">
          {selectedRun == null ? (
            <span className="sc-mono-dim">
              Select a run from the left to browse its checkpoints
            </span>
          ) : (
            <>
              <StatusDot status={selectedRun.status} />
              <span className="sc-run-id">
                {selectedRun.run_id}
              </span>
              <Tag color={selectedRun.status === "failed" ? "red" : selectedRun.status === "running" ? "green" : selectedRun.status === "finished" ? "green" : "amber"}>
                {selectedRun.status}
              </Tag>
              <span className="sc-spacer" />
              <span className="sc-run-meta">
                {selectedRun.checkpoint_count} ckpt · {fmtSize(selectedRun.total_size_mb)}
              </span>
            </>
          )}
        </div>

        {selectedRun != null && (
          <>
            <CheckpointsTable
              entries={sorted}
              bestPsnrPath={bestPsnrPath}
              latestPath={latestPath}
              selectedPath={selectedCheckpointPath}
              sortCol={sortCol}
              sortDir={sortDir}
              trainingActive={isActiveRun}
              onSort={handleSort}
              onSelect={(e) => useRunsStore.getState().selectCheckpoint(e.path)}
              onDeleteRequest={handleDeleteRequest}
            />
            <StorageSummaryPanel entries={allEntries} />
            {isActiveRun && <DeleteDisabledBanner />}
          </>
        )}
      </div>

      {/* Right — checkpoint detail */}
      <DetailPanel
        entry={selectedEntry}
        deleteDisabled={isActiveRun}
        deleteDisabledTitle={isActiveRun ? "Cannot delete checkpoints of the run that is currently training" : undefined}
        onExportPth={handleExportPth}
        onExportOnnx={handleExportOnnx}
        onDeleteRequest={handleDeleteRequest}
        onResume={handleResume}
        onRunInference={() => {
          if (selectedCheckpointPath) {
            const inf = useInferenceStore.getState();
            inf.setPreselectedInstance(selectedInstance);
            inf.setPreselectedCheckpointPath(selectedCheckpointPath);
            if (selectedRunId) {
              const inst = models.find((m) => m.name === selectedInstance);
              const displays = inst ? buildRunDisplays(inst.runs) : new Map();
              inf.setCheckpointContext(selectedRunId, displays.get(selectedRunId)?.label ?? shortRunId(selectedRunId));
            }
          }
          setActiveTab("inference");
        }}
      />

      {/* Delete confirmations */}
      {deletingEntry != null && (
        <DeleteConfirmScrim
          target={{ kind: "checkpoint", entry: deletingEntry }}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeletingEntry(null)}
        />
      )}
      {deletingRun != null && (
        <DeleteConfirmScrim
          target={{ kind: "run", run: deletingRun.run }}
          onConfirm={handleRunDeleteConfirm}
          onCancel={() => setDeletingRun(null)}
        />
      )}
    </div>
  );
}