import { create } from "zustand";
import type { InferenceResult } from "../lib/api-types";

export type InferenceStatus = "idle" | "running" | "done" | "error";

interface InferenceState {
  // Input
  inputPath: string | null;
  gtPath: string | null;

  // Model (either a model instance + version, or a raw checkpoint path)
  preselectedCheckpointPath: string | null; // §13.9b — set by Checkpoints tab
  instance: string | null;
  version: string | null;
  modelPath: string | null;

  // Output
  outputDir: string;
  outputFormat: "png" | "jpeg" | "webp" | "tiff";

  // Tiling
  tileSize: number; // 0 = no tiling
  overlap: number;

  // Device
  device: "auto" | "cuda" | "cpu";

  // Run state
  status: InferenceStatus;
  activeJobId: string | null;
  errorMsg: string | null;
  tilesDone: number;
  tilesTotal: number;
  result: InferenceResult | null;

  // Actions
  setInputPath: (path: string | null) => void;
  setGtPath: (path: string | null) => void;
  setPreselectedCheckpointPath: (path: string | null) => void;
  setInstance: (name: string | null) => void;
  setVersion: (version: string | null) => void;
  setModelPath: (path: string | null) => void;
  setOutputDir: (dir: string) => void;
  setOutputFormat: (fmt: "png" | "jpeg" | "webp" | "tiff") => void;
  setTileSize: (size: number) => void;
  setOverlap: (overlap: number) => void;
  setDevice: (device: "auto" | "cuda" | "cpu") => void;
  setStatus: (status: InferenceStatus) => void;
  setActiveJobId: (jobId: string | null) => void;
  setErrorMsg: (msg: string | null) => void;
  setTileProgress: (done: number, total: number) => void;
  setResult: (result: InferenceResult | null) => void;
  resetRun: () => void;
}

export const useInferenceStore = create<InferenceState>((set) => ({
  inputPath: null,
  gtPath: null,
  preselectedCheckpointPath: null,
  instance: null,
  version: null,
  modelPath: null,
  outputDir: "",
  outputFormat: "png",
  tileSize: 0,
  overlap: 64,
  device: "auto",
  status: "idle",
  activeJobId: null,
  errorMsg: null,
  tilesDone: 0,
  tilesTotal: 0,
  result: null,

  setInputPath: (inputPath) => set({ inputPath }),
  setGtPath: (gtPath) => set({ gtPath }),
  setPreselectedCheckpointPath: (path) => set({ preselectedCheckpointPath: path }),
  setInstance: (instance) => set({ instance, version: null }),
  setVersion: (version) => set({ version }),
  setModelPath: (modelPath) => set({ modelPath }),
  setOutputDir: (outputDir) => set({ outputDir }),
  setOutputFormat: (outputFormat) => set({ outputFormat }),
  setTileSize: (tileSize) => set({ tileSize }),
  setOverlap: (overlap) => set({ overlap }),
  setDevice: (device) => set({ device }),
  setStatus: (status) => set({ status }),
  setActiveJobId: (activeJobId) => set({ activeJobId }),
  setErrorMsg: (errorMsg) => set({ errorMsg }),
  setTileProgress: (tilesDone, tilesTotal) => set({ tilesDone, tilesTotal }),
  setResult: (result) => set({ result }),
  resetRun: () =>
    set({ status: "idle", activeJobId: null, errorMsg: null, tilesDone: 0, tilesTotal: 0, result: null }),
}));
