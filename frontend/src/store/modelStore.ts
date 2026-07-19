import { create } from "zustand";
import type { Architecture } from "../lib/srproj";

export type ModelSubTab = "create" | "view";

export interface Hyperparameters {
  scale: number;
  lrScheduler: string;
  optimizer: string;
  learningRate: number;
  batchSize: number;
  patchSize: number;
  totalIter: number;
}

const DEFAULT_HYPERPARAMETERS: Hyperparameters = {
  scale: 4,
  lrScheduler: "cosine",
  optimizer: "Adam",
  learningRate: 1e-4,
  batchSize: 16,
  patchSize: 128,
  totalIter: 300000,
};

interface ModelState {
  subTab: ModelSubTab;
  architecture: Architecture;
  hyperparameters: Hyperparameters;
  setSubTab: (tab: ModelSubTab) => void;
  setArchitecture: (arch: Architecture) => void;
  setHyperparameters: (hp: Partial<Hyperparameters>) => void;
  resetHyperparameters: () => void;
}

export const useModelStore = create<ModelState>((set) => ({
  subTab: "create",
  architecture: "rrdb_esrgan",
  hyperparameters: DEFAULT_HYPERPARAMETERS,
  setSubTab: (subTab) => set({ subTab }),
  setArchitecture: (architecture) => set({ architecture }),
  setHyperparameters: (hp) =>
    set((s) => ({ hyperparameters: { ...s.hyperparameters, ...hp } })),
  resetHyperparameters: () => set({ hyperparameters: DEFAULT_HYPERPARAMETERS }),
}));
