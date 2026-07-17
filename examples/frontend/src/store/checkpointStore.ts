import { create } from "zustand";
import type { CheckpointEntry } from "../lib/ipc-types";

interface CheckpointState {
  // Per-run checkpoint lists
  checkpointsByRun: Record<string, CheckpointEntry[]>;
  // Set by ipc.ts on checkpoint.export.done; screen processes into a toast then clears
  lastExportDone: { path: string; sizeMb: number } | null;

  setCheckpointsForRun: (runId: string, entries: CheckpointEntry[]) => void;
  addCheckpointForRun: (runId: string, entry: CheckpointEntry) => void;
  // Remove by path across all runs (checkpoint.delete.done has no run_id)
  removeCheckpointByPath: (path: string) => void;
  setLastExportDone: (result: { path: string; sizeMb: number } | null) => void;
}

export const useCheckpointStore = create<CheckpointState>((set) => ({
  checkpointsByRun: {},
  lastExportDone: null,

  setCheckpointsForRun: (runId, entries) =>
    set((s) => ({ checkpointsByRun: { ...s.checkpointsByRun, [runId]: entries } })),

  addCheckpointForRun: (runId, entry) =>
    set((s) => {
      const existing = s.checkpointsByRun[runId] ?? [];
      if (existing.some((e) => e.path === entry.path)) return s;
      return {
        checkpointsByRun: { ...s.checkpointsByRun, [runId]: [...existing, entry] },
      };
    }),

  removeCheckpointByPath: (path) =>
    set((s) => {
      const updated: Record<string, CheckpointEntry[]> = {};
      for (const [runId, entries] of Object.entries(s.checkpointsByRun)) {
        updated[runId] = entries.filter((e) => e.path !== path);
      }
      return { checkpointsByRun: updated };
    }),

  setLastExportDone: (result) => set({ lastExportDone: result }),
}));
