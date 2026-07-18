import { create } from "zustand";
import type { InferenceResult } from "../lib/api-types";

interface InferenceState {
  // Input
  inputPath: string | null;
  gtPath: string | null;

  // Model config
  preselectedCheckpointPath: string | null; // §13.9b
  checkpointPath: string | null;
  architecture: string;
  scaleFactor: number;
  tileSize: number; // 0 = no tiling
  fp16: boolean;

  // Output
  outputDir: string;
  outputFormat: "png" | "jpeg" | "webp" | "tiff";

  // Run state
  isRunning: boolean;
  tilesDone: number;
  tilesTotal: number;
  result: InferenceResult | null;

  // Actions
  setInputPath: (path: string | null) => void;
  setGtPath: (path: string | null) => void;
  setPreselectedCheckpointPath: (path: string | null) => void;
  setCheckpointPath: (path: string | null) => void;
  setArchitecture: (arch: string) => void;
  setScaleFactor: (scale: number) => void;
  setTileSize: (size: number) => void;
  setFp16: (on: boolean) => void;
  setOutputDir: (dir: string) => void;
  setOutputFormat: (fmt: "png" | "jpeg" | "webp" | "tiff") => void;
  setRunning: (running: boolean) => void;
  setTileProgress: (done: number, total: number) => void;
  setResult: (result: InferenceResult | null) => void;
}

export const useInferenceStore = create<InferenceState>((set) => ({
  inputPath: null,
  gtPath: null,
  preselectedCheckpointPath: null,
  checkpointPath: null,
  architecture: "rrdb_esrgan",
  scaleFactor: 4,
  tileSize: 0,
  fp16: false,
  outputDir: "",
  outputFormat: "png",
  isRunning: false,
  tilesDone: 0,
  tilesTotal: 0,
  result: null,

  setInputPath: (path) => set({ inputPath: path }),
  setGtPath: (path) => set({ gtPath: path }),
  setPreselectedCheckpointPath: (path) => set({ preselectedCheckpointPath: path }),
  setCheckpointPath: (path) => set({ checkpointPath: path }),
  setArchitecture: (architecture) => set({ architecture }),
  setScaleFactor: (scaleFactor) => set({ scaleFactor }),
  setTileSize: (tileSize) => set({ tileSize }),
  setFp16: (fp16) => set({ fp16 }),
  setOutputDir: (outputDir) => set({ outputDir }),
  setOutputFormat: (outputFormat) => set({ outputFormat }),
  setRunning: (isRunning) => set({ isRunning }),
  setTileProgress: (tilesDone, tilesTotal) => set({ tilesDone, tilesTotal }),
  setResult: (result) => set({ result }),
}));
