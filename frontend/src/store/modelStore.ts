import { create } from "zustand";
import type { Architecture, AugmentationConfig } from "../lib/srproj";

export interface Hyperparameters {
  scale: number;
  lrScheduler: string;
  optimizer: string;
  learningRate: number;
  batchSize: number;
  patchSize: number;
  totalIter: number;
}

export interface LossWeights {
  pixel: number;
  perceptual: number;
  adversarial: number;
}

const DEFAULT_AUGMENTATIONS: AugmentationConfig = {
  horizontal_flip: true,
  vertical_flip: false,
  rotation_90: false,
  mixup: false,
  color_jitter: false,
  random_degradation: false,
  gaussian_blur: false,
  noise_injection: false,
};

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
  architecture: Architecture;
  hyperparameters: Hyperparameters;
  lossWeights: LossWeights;
  augmentations: AugmentationConfig;
  pretrainedPath: string | null;
  setArchitecture: (arch: Architecture) => void;
  setHyperparameters: (hp: Partial<Hyperparameters>) => void;
  setLossWeights: (lw: Partial<LossWeights>) => void;
  setAugmentations: (aug: Partial<AugmentationConfig>) => void;
  setPretrainedPath: (path: string | null) => void;
  resetHyperparameters: () => void;
}

export const useModelStore = create<ModelState>((set) => ({
  architecture: "rrdb_esrgan",
  hyperparameters: DEFAULT_HYPERPARAMETERS,
  lossWeights: { pixel: 1.0, perceptual: 1.0, adversarial: 0.1 },
  augmentations: DEFAULT_AUGMENTATIONS,
  pretrainedPath: null,
  setArchitecture: (architecture) => set({ architecture }),
  setHyperparameters: (hp) =>
    set((s) => ({ hyperparameters: { ...s.hyperparameters, ...hp } })),
  setLossWeights: (lw) =>
    set((s) => ({ lossWeights: { ...s.lossWeights, ...lw } })),
  setAugmentations: (aug) =>
    set((s) => ({ augmentations: { ...s.augmentations, ...aug } })),
  setPretrainedPath: (path) => set({ pretrainedPath: path }),
  resetHyperparameters: () => set({ hyperparameters: DEFAULT_HYPERPARAMETERS }),
}));
