import { create } from "zustand";
import type { SRProjFile } from "../lib/srproj";
import { SRProjManager } from "../lib/SRProjManager";
import { addToRecent } from "../screens/ProjectScreen";

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

  openProject: async (filePath: string) => {
    if (!filePath) return;
    const proj = await SRProjManager.load(filePath);
    const srProject: SRProject = { ...proj, filePath };
    set({ project: srProject });

    addToRecent({
      name: proj.name,
      filePath,
      lastOpened: new Date().toISOString(),
    });
  },

  closeProject: () => {
    SRProjManager.close();
    set({ project: null });
  },
}));
