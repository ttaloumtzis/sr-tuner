import { create } from "zustand";
import type { SRProjFile } from "../lib/srproj";
import { SRProjManager } from "../lib/SRProjManager";
import { addToRecent } from "../screens/ProjectScreen";
import { sendToSidecar } from "../lib/ipc";

export interface SRProject extends SRProjFile {
  filePath: string;
}

interface ProjectState {
  project: SRProject | null;
  openProject: (filePath: string) => Promise<void>;
  closeProject: () => void;
}

export const useProjectStore = create<ProjectState>((set) => ({
  project: null,

  // §8.7: Load .srproj from disk; SRProjManager.load handles §8.10 version check
  openProject: async (filePath: string) => {
    if (!filePath) return;
    const proj = await SRProjManager.load(filePath);
    const srProject: SRProject = { ...proj, filePath };
    set({ project: srProject });

    // §8.8: Persist to recent list
    addToRecent({
      name: proj.name,
      filePath,
      lastOpened: new Date().toISOString(),
    });

    // §12.10: Hydrate history for any run that was training or paused
    for (const run of proj.runs) {
      if (run.status === "running" || run.status === "paused") {
        sendToSidecar({
          type: "run.history.request",
          run_id: run.run_id,
          log_dir: run.paths.log_dir,
        }).catch(() => {
          // Sidecar may not be running yet; history hydration is best-effort
        });
      }
    }
  },

  closeProject: () => {
    SRProjManager.close();
    set({ project: null });
  },
}));
