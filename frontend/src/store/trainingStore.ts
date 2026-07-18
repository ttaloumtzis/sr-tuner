import { create } from "zustand";

export type TrainingStatus =
  | "idle"
  | "running"
  | "paused"
  | "done"
  | "disconnected";

export interface RunHistory {
  gLossHistory: number[];
  dLossHistory: (number | null)[];
  totalLossHistory: number[];
  psnrHistory: number[];
  ssimHistory: number[];
}

export interface ValidationFrames {
  lrPath: string;
  srPath: string;
  gtPath: string | null;
  diffPath: string | null;
}

interface TrainingState {
  status: TrainingStatus;
  activeTrainingRunId: string | null;
  iter: number;
  epoch: number;
  gLoss: number;
  dLoss: number | null;
  psnr: number | null;
  ssim: number | null;
  gpuUtil: number | null;
  vram: number | null;
  temp: number | null;
  cpuUtil: number | null;
  speed: number | null;
  bestPsnr: number | null;
  // Active-run history arrays (rolling, up to 500 pts)
  lossHistory: number[];
  dLossHistory: (number | null)[];
  totalLossHistory: number[];
  psnrHistory: number[];
  ssimHistory: number[];
  etaSec: number | null;
  finalEpoch: number | null;
  validationFrames: ValidationFrames | null;
  // Per-run history cache for historical view (§12.12) and reopen hydration (§12.10)
  runHistories: Record<string, RunHistory>;
}

export const useTrainingStore = create<TrainingState>(() => ({
  status: "idle",
  activeTrainingRunId: null,
  iter: 0,
  epoch: 0,
  gLoss: 0,
  dLoss: null,
  psnr: null,
  ssim: null,
  gpuUtil: null,
  vram: null,
  temp: null,
  cpuUtil: null,
  speed: null,
  bestPsnr: null,
  lossHistory: [],
  dLossHistory: [],
  totalLossHistory: [],
  psnrHistory: [],
  ssimHistory: [],
  etaSec: null,
  finalEpoch: null,
  validationFrames: null,
  runHistories: {},
}));
