// §21 — State Restoration on Project Reopen
//
// Runs once each time a project is opened. Restores Zustand store state from the
// .srproj manifest and sends IPC hydration requests to the sidecar.
//
// Task map:
//   §21.1 — Restore dataset + model stores from default_dataset / default_model
//   §21.2 — Select active run (last_active_run_id or last in runs[])
//   §21.3 — Send checkpoint.list.request for active run
//   §21.4 — Send run.history.request for active run
//   §21.5 — Detect crashed runs (status "running" + sidecar_pid ≠ null), show dialog
//   §21.6 — Restore last_active_tab; default to "metrics" if null

import { useEffect, useState } from "react";
import { useProjectStore } from "../store/projectStore";
import { useDatasetStore } from "../store/datasetStore";
import { useModelStore } from "../store/modelStore";
import { useUiStore, type TabId } from "../store/uiStore";
import { useRunConfigStore } from "../store/runConfigStore";
import { SRProjManager } from "../lib/SRProjManager";
import { sendToSidecar } from "../lib/ipc";

// All tab IDs that can be written to .srproj (the uiStore superset includes "history"
// which .srproj never stores, so every stored value is always a valid uiStore TabId).
const RESTORABLE_TABS = new Set<string>([
  "dataset",
  "model",
  "training",
  "metrics",
  "checkpoints",
  "inference",
]);

export interface CrashRecoveryState {
  open: boolean;
  runId: string;
  runName: string;
  lastEpoch: number;
  lastCheckpointPath: string | null;
}

const CLOSED_CRASH_STATE: CrashRecoveryState = {
  open: false,
  runId: "",
  runName: "",
  lastEpoch: 0,
  lastCheckpointPath: null,
};

export function useProjectRestoration() {
  const project = useProjectStore((s) => s.project);
  const [crashRecovery, setCrashRecovery] =
    useState<CrashRecoveryState>(CLOSED_CRASH_STATE);

  // Keyed on filePath so the effect re-runs exactly once per distinct project open.
  useEffect(() => {
    if (!project) return;

    // ── §21.1 — Restore dataset state ────────────────────────────────────────
    const ds = project.default_dataset;
    const dsStore = useDatasetStore.getState();
    dsStore.setHrPath(ds.training_path);
    dsStore.setValidationPath(ds.validation_path);
    dsStore.setStrategy(ds.validation_strategy);
    dsStore.setType(ds.dataset_type);

    // ── §21.1 — Restore model state ───────────────────────────────────────────
    const dm = project.default_model;
    const mdStore = useModelStore.getState();
    mdStore.setArchitecture(dm.architecture);
    mdStore.setHyperparameters({ scale: dm.upscale_factor });

    // ── §21.2 — Determine active run ──────────────────────────────────────────
    const runs = project.runs;
    const lastActiveId = project.ui_state.last_active_run_id;
    const activeRun =
      runs.find((r) => r.run_id === lastActiveId) ??
      runs[runs.length - 1] ??
      null;

    if (activeRun) {
      useUiStore.getState().setDisplayedRunId(activeRun.run_id);
    }

    // ── §21.5 — Detect crashed runs ───────────────────────────────────────────
    // A non-null sidecar_pid on reopen always means the previous session ended
    // abnormally — stdin/stdout pipes don't survive across app restarts.
    const crashedRuns = runs.filter(
      (r) => r.status === "running" && r.sidecar_pid !== null
    );

    if (crashedRuns.length > 0) {
      for (const run of crashedRuns) {
        SRProjManager.updateRun(run.run_id, { status: "crashed" });
      }
      SRProjManager.save().catch(() => {
        // Best-effort persist — crash recovery still shows even if save fails
      });

      // Show dialog for the most recently started crashed run
      const mostRecent = crashedRuns.reduce((a, b) =>
        (a.started_at ?? "") > (b.started_at ?? "") ? a : b
      );
      setCrashRecovery({
        open: true,
        runId: mostRecent.run_id,
        runName: mostRecent.name,
        lastEpoch: mostRecent.metrics.current_epoch,
        lastCheckpointPath: mostRecent.checkpoints.last_saved_path,
      });
    }

    // ── §21.3 — Hydrate Checkpoints tab ──────────────────────────────────────
    // ── §21.4 — Hydrate Metrics history charts ────────────────────────────────
    if (activeRun) {
      const { checkpoint_dir, log_dir } = activeRun.paths;

      if (checkpoint_dir) {
        sendToSidecar({
          type: "checkpoint.list.request",
          run_id: activeRun.run_id,
          checkpoint_dir,
        }).catch(() => {
          // Sidecar may not be running yet; hydration is best-effort
        });
      }

      if (log_dir) {
        sendToSidecar({
          type: "run.history.request",
          run_id: activeRun.run_id,
          log_dir,
        }).catch(() => {
          // Same — best-effort; ipc.ts dispatchRunHistory handles the response
        });
      }
    }

    // ── §21.6 — Restore active tab ────────────────────────────────────────────
    const lastTab = project.ui_state.last_active_tab;
    const targetTab: TabId =
      lastTab && RESTORABLE_TABS.has(lastTab)
        ? (lastTab as TabId)
        : "metrics";
    useUiStore.getState().setActiveTab(targetTab);

    // ── §3.8 [Gap I] — Restore expanded panel state ───────────────────────────
    const savedPanels = project.ui_state.expanded_panels;
    if (savedPanels && Object.keys(savedPanels).length > 0) {
      useUiStore.getState().setExpandedPanels(savedPanels);
    }
  }, [project?.filePath]);

  // ── Crash recovery dialog handlers ───────────────────────────────────────────

  function handleResumeCrashedRun() {
    const { lastCheckpointPath, lastEpoch } = crashRecovery;
    if (!lastCheckpointPath) return;

    useRunConfigStore.getState().setResumeFrom({
      checkpoint_path: lastCheckpointPath,
      resume_epoch: lastEpoch,
      resume_optimizer_state: true,
      resume_lr_scheduler_state: true,
    });
    useUiStore.getState().setActiveTab("training");
    setCrashRecovery(CLOSED_CRASH_STATE);
  }

  function handleAbandonCrashedRun() {
    const { runId } = crashRecovery;
    SRProjManager.updateRun(runId, { status: "failed" });
    SRProjManager.save().catch(() => {/* best-effort */});
    setCrashRecovery(CLOSED_CRASH_STATE);
  }

  return { crashRecovery, handleResumeCrashedRun, handleAbandonCrashedRun };
}
