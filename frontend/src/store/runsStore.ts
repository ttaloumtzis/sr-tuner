import { create } from "zustand";
import type { CheckpointEntry, ModelRuns, RunInfo } from "../lib/api-types";
import { deleteRun as apiDeleteRun, listRunCheckpoints, listRuns } from "../lib/api";
import { SRProjManager } from "../lib/SRProjManager";
import { useCheckpointStore } from "./checkpointStore";
import { useProjectStore } from "./projectStore";
import { useTrainingStore } from "./trainingStore";

// Legacy srproj statuses → canonical vocabulary (disk ground truth wins).
const LEGACY_STATUS_MAP: Record<string, RunInfo["status"]> = {
  configured: "stopped",
  paused: "stopped",
  completed: "finished",
  running: "running",
  failed: "failed",
};

export function mapLegacyStatus(status: string | undefined): RunInfo["status"] {
  return LEGACY_STATUS_MAP[status ?? ""] ?? "interrupted";
}

// Pure reconciliation — testable without zustand.
export function reconcileSrprojRuns(
  proj: Parameters<typeof SRProjManager.setProject>[1],
  models: ModelRuns[],
): Parameters<typeof SRProjManager.setProject>[1] {
  const diskRuns = new Map<string, RunInfo>();
  for (const m of models) {
    for (const run of m.runs) diskRuns.set(run.run_id, run);
  }

  const nextRuns = (proj.runs ?? [])
    .map((r) => {
      const disk = diskRuns.get(r.run_id);
      if (!disk) return null; // folder gone → prune
      return {
        ...r,
        status: mapLegacyStatus(disk.status),
        completed_at: disk.finished_at ?? r.completed_at,
        checkpoints: {
          total_count: disk.checkpoint_count,
          last_saved_epoch: disk.last_epoch > 0 ? disk.last_epoch : r.checkpoints?.last_saved_epoch ?? null,
          last_saved_path: r.checkpoints?.last_saved_path ?? null,
          best_checkpoint_path: r.checkpoints?.best_checkpoint_path ?? null,
        },
        metrics: {
          ...r.metrics,
          current_epoch: disk.last_epoch > 0 ? disk.last_epoch : r.metrics?.current_epoch ?? 0,
          epochs_completed: disk.last_epoch > 0 ? disk.last_epoch : r.metrics?.epochs_completed ?? 0,
        },
      };
    })
    .filter((r): r is NonNullable<typeof r> => r !== null);

  // Discover runs that exist on disk but are missing from srproj.
  const known = new Set(nextRuns.map((r) => r.run_id));
  for (const m of models) {
    for (const run of m.runs) {
      if (known.has(run.run_id)) continue;
      const tc = (run.config?.train_cfg ?? {}) as Record<string, unknown>;
      nextRuns.push({
        run_id: run.run_id,
        name: run.run_id,
        status: mapLegacyStatus(run.status),
        created_at: run.created_at ?? new Date().toISOString(),
        started_at: run.created_at ?? null,
        completed_at: run.finished_at ?? null,
        architecture: {
          type: (m.architecture === "swinir" ? "swinir" : "rrdb_esrgan") as "rrdb_esrgan" | "swinir",
          upscale_factor: m.scale ?? 4,
          custom_config: {},
        },
        training_config: {
          num_epochs: typeof tc.max_epochs === "number" ? tc.max_epochs : 0,
          batch_size: typeof tc.batch_size === "number" ? tc.batch_size : 0,
          learning_rate: typeof tc.learning_rate === "number" ? tc.learning_rate : 0,
          scheduler: typeof tc.scheduler === "string" ? tc.scheduler : "",
          optimizer: typeof tc.optimizer === "string" ? tc.optimizer : "",
          patch_size: typeof tc.patch_size === "number" ? tc.patch_size : 0,
          augmentations: {
            horizontal_flip: false, vertical_flip: false, rotation_90: false,
            mixup: false, color_jitter: false, random_degradation: false,
            gaussian_blur: false, noise_injection: false,
          },
        },
        paths: {
          training_data: typeof run.config?.dataset === "string" ? run.config.dataset : "",
          validation_data: "",
          checkpoint_dir: "",
          log_dir: "",
        },
        metrics: {
          current_epoch: run.last_epoch,
          epochs_completed: run.last_epoch,
          best_loss: null, best_loss_epoch: null,
          best_psnr: null, best_psnr_epoch: null,
          last_loss: null, last_psnr: null, last_ssim: null,
        },
        checkpoints: {
          total_count: run.checkpoint_count,
          last_saved_epoch: run.last_epoch > 0 ? run.last_epoch : null,
          last_saved_path: null,
          best_checkpoint_path: null,
        },
      });
    }
  }

  return { ...proj, runs: nextRuns };
}

interface RunsState {
  models: ModelRuns[];
  loading: boolean;
  error: string | null;
  lastRefreshAt: number | null;
  refreshCounter: number;
  selectedInstance: string | null;
  selectedRunId: string | null;
  checkpointsByRun: Record<string, CheckpointEntry[]>;
  selectedCheckpointPath: string | null;

  refresh: () => Promise<void>;
  bumpRefresh: () => void;
  selectRun: (instance: string, runId: string) => Promise<void>;
  selectCheckpoint: (path: string | null) => void;
  deleteRun: (instance: string, runId: string) => Promise<boolean>;
}

export const useRunsStore = create<RunsState>((set, get) => ({
  models: [],
  loading: false,
  error: null,
  lastRefreshAt: null,
  refreshCounter: 0,
  selectedInstance: null,
  selectedRunId: null,
  checkpointsByRun: {},
  selectedCheckpointPath: null,

  refresh: async () => {
    const st = get();
    if (st.loading) return;
    set({ loading: true, error: null });
    try {
      const models = await listRuns();

      // Auto-select the active training run when nothing is selected.
      let { selectedInstance, selectedRunId } = st;
      if (!selectedRunId) {
        const activeDir = useTrainingStore.getState().activeRunDirId;
        if (activeDir) {
          for (const m of models) {
            if (m.runs.some((r) => r.run_id === activeDir)) {
              selectedInstance = m.name;
              selectedRunId = activeDir;
              break;
            }
          }
        }
      } else if (!models.some((m) => m.name === selectedInstance && m.runs.some((r) => r.run_id === selectedRunId))) {
        selectedInstance = null;
        selectedRunId = null;
      }

      // Fetch checkpoints for the selected run.
      let checkpointsByRun = st.checkpointsByRun;
      if (selectedInstance && selectedRunId) {
        try {
          const entries = await listRunCheckpoints(selectedInstance, selectedRunId);
          checkpointsByRun = { ...checkpointsByRun, [selectedRunId]: entries };
          // Mirror into checkpointStore — the Inference tab reads it for its
          // checkpoint picker (all runs flattened).
          useCheckpointStore.getState().setCheckpointsForRun(selectedRunId, entries);
        } catch {
          // run may have been deleted mid-refresh
        }
      }

      set({
        models,
        checkpointsByRun,
        selectedInstance,
        selectedRunId,
        lastRefreshAt: Date.now(),
        loading: false,
      });

      // Reconcile srproj (fire-and-forget, best effort).
      const proj = SRProjManager.current;
      if (proj) {
        try {
          const next = reconcileSrprojRuns(proj, models);
          SRProjManager.setProject(SRProjManager.filePath ?? "", next);
          await SRProjManager.save();
          useProjectStore.getState().setProject({ ...next, filePath: useProjectStore.getState().project?.filePath ?? "" });
        } catch {
          // never fail the UI over srproj bookkeeping
        }
      }
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), loading: false });
    }
  },

  bumpRefresh: () => set((s) => ({ refreshCounter: s.refreshCounter + 1 })),

  selectRun: async (instance: string, runId: string) => {
    set({ selectedInstance: instance, selectedRunId: runId, selectedCheckpointPath: null });
    try {
      const entries = await listRunCheckpoints(instance, runId);
      set((s) => ({ checkpointsByRun: { ...s.checkpointsByRun, [runId]: entries } }));
      useCheckpointStore.getState().setCheckpointsForRun(runId, entries);
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  selectCheckpoint: (path) => set({ selectedCheckpointPath: path }),

  deleteRun: async (instance: string, runId: string) => {
    try {
      await apiDeleteRun(instance, runId);
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
      return false;
    }
    // Prune srproj entry.
    const proj = SRProjManager.current;
    if (proj) {
      try {
        const next = { ...proj, runs: (proj.runs ?? []).filter((r) => r.run_id !== runId) };
        SRProjManager.setProject(SRProjManager.filePath ?? "", next);
        await SRProjManager.save();
        useProjectStore.getState().setProject({ ...next, filePath: useProjectStore.getState().project?.filePath ?? "" });
      } catch {
        // srproj bookkeeping failure is non-fatal
      }
    }
    const st = get();
    useCheckpointStore.getState().setCheckpointsForRun(runId, []);
    set((s) => {
      const checkpointsByRun = { ...s.checkpointsByRun };
      delete checkpointsByRun[runId];
      return {
        checkpointsByRun,
        selectedInstance: st.selectedRunId === runId ? null : s.selectedInstance,
        selectedRunId: st.selectedRunId === runId ? null : s.selectedRunId,
        selectedCheckpointPath: st.selectedRunId === runId ? null : s.selectedCheckpointPath,
        error: null,
      };
    });
    await get().refresh();
    return true;
  },
}));
