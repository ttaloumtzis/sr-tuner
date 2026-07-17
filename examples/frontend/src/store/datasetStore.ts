import { create } from "zustand";
import type { DatasetType } from "../lib/srproj";

export type DownscaleKernel = "bicubic" | "bilinear" | "real-world";
export type ValidationStrategy = "auto_split" | "separate_folder" | "none";

export interface ExtractionProgress {
  framesDone: number;
  framesTotal: number;
  fps: number;
  etaSec: number;
}

interface DatasetState {
  type: DatasetType;
  scale: number;
  kernel: DownscaleKernel;
  hrPath: string;
  lrPath: string;
  namingPattern: string;
  extractionProgress: ExtractionProgress | null;
  strategy: ValidationStrategy;
  validationSplitRatio: number;
  validationPath: string | null;
  setType: (type: DatasetType) => void;
  setScale: (scale: number) => void;
  setKernel: (kernel: DownscaleKernel) => void;
  setHrPath: (path: string) => void;
  setLrPath: (path: string) => void;
  setNamingPattern: (pattern: string) => void;
  setExtractionProgress: (progress: ExtractionProgress | null) => void;
  setStrategy: (strategy: ValidationStrategy) => void;
  setValidationSplitRatio: (ratio: number) => void;
  setValidationPath: (path: string | null) => void;
}

export const useDatasetStore = create<DatasetState>((set) => ({
  type: "image_folder",
  scale: 4,
  kernel: "bicubic",
  hrPath: "",
  lrPath: "",
  namingPattern: "%06d",
  extractionProgress: null,
  strategy: "auto_split",
  validationSplitRatio: 0.1,
  validationPath: null,
  setType: (type) => set({ type }),
  setScale: (scale) => set({ scale }),
  setKernel: (kernel) => set({ kernel }),
  setHrPath: (path) => set({ hrPath: path }),
  setLrPath: (path) => set({ lrPath: path }),
  setNamingPattern: (pattern) => set({ namingPattern: pattern }),
  setExtractionProgress: (progress) => set({ extractionProgress: progress }),
  setStrategy: (strategy) => set({ strategy }),
  setValidationSplitRatio: (ratio) => set({ validationSplitRatio: ratio }),
  setValidationPath: (path) => set({ validationPath: path }),
}));
