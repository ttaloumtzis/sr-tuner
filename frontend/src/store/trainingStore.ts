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

export interface HardwareData {
  cpu_percent: number | null;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  gpu_util_percent: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  temp_c: number | null;
}

interface TrainingState {
  status: TrainingStatus;
  activeTrainingRunId: string | null;
  iter: number;
  epoch: number;
  batch: number;
  totalBatch: number;
  gLoss: number | null;
  dLoss: number | null;
  psnr: number | null;
  ssim: number | null;
  fullPsnr: number | null;
  fullSsim: number | null;
  liveLoss: number | null;
  gpuUtil: number | null;
  vram: number | null;
  vramTotalGb: number | null;
  temp: number | null;
  cpuUtil: number | null;
  ramGb: number | null;
  ramTotalGb: number | null;
  speed: number | null;
  bestPsnr: number | null;
  lossHistory: number[];
  dLossHistory: (number | null)[];
  totalLossHistory: number[];
  psnrHistory: number[];
  ssimHistory: number[];
  etaSec: number | null;
  finalEpoch: number | null;
  validationFrames: ValidationFrames | null;
  validationRunning: boolean;

  setStatus: (status: TrainingStatus) => void;
  setActiveRun: (runId: string | null) => void;
  setValidationFrames: (frames: ValidationFrames | null) => void;
  setValidationRunning: (v: boolean) => void;
  updateFromStep: (epoch: number, batch: number, totalBatch: number, speed: number) => void;
  setLiveLoss: (avg: number | null) => void;
  pushEpochLoss: (avgLoss: number) => void;
  updateFromValidate: (epoch: number, psnr: number, ssim: number, fullPsnr?: number, fullSsim?: number) => void;
  updateFromHardware: (data: HardwareData) => void;
  setFinalEpoch: (epoch: number) => void;
  reset: () => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  status: "idle",
  activeTrainingRunId: null,
  iter: 0,
  epoch: 0,
  batch: 0,
  totalBatch: 0,
  gLoss: null,
  dLoss: null,
  psnr: null,
  ssim: null,
  fullPsnr: null,
  fullSsim: null,
  liveLoss: null,
  gpuUtil: null,
  vram: null,
  vramTotalGb: null,
  temp: null,
  cpuUtil: null,
  ramGb: null,
  ramTotalGb: null,
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
  validationRunning: false,

  setStatus: (status) => set({ status }),
  setActiveRun: (runId) => set({ activeTrainingRunId: runId }),
  setValidationFrames: (frames) => set({ validationFrames: frames }),
  setValidationRunning: (v) => set({ validationRunning: v }),

  updateFromStep: (epoch, batch, totalBatch, speed) =>
    set((s) => ({
      epoch,
      batch,
      totalBatch,
      speed,
      iter: s.iter + 1,
      validationRunning: false,
    })),

  setLiveLoss: (avg) => set({ liveLoss: avg }),

  pushEpochLoss: (avgLoss) =>
    set((s) => ({
      gLoss: avgLoss,
      lossHistory: [...s.lossHistory, avgLoss].slice(-500),
    })),

  updateFromValidate: (epoch, psnr, ssim, fullPsnr, fullSsim) =>
    set((s) => {
      const psnrH = [...s.psnrHistory, psnr].slice(-500);
      const ssimH = [...s.ssimHistory, ssim].slice(-500);
      const best = s.bestPsnr !== null ? Math.max(s.bestPsnr, psnr) : psnr;
      return { epoch, psnr, ssim, fullPsnr: fullPsnr ?? s.fullPsnr, fullSsim: fullSsim ?? s.fullSsim, bestPsnr: best, psnrHistory: psnrH, ssimHistory: ssimH };
    }),

  updateFromHardware: (data) =>
    set({
      cpuUtil: data.cpu_percent,
      ramGb: data.ram_used_gb,
      ramTotalGb: data.ram_total_gb,
      gpuUtil: data.gpu_util_percent,
      vram: data.vram_used_gb,
      vramTotalGb: data.vram_total_gb,
      temp: data.temp_c,
    }),

  setFinalEpoch: (finalEpoch) => set({ finalEpoch }),

  reset: () => set({
    status: "idle",
    activeTrainingRunId: null,
    iter: 0,
    epoch: 0,
    batch: 0,
    totalBatch: 0,
    gLoss: null,
    dLoss: null,
    psnr: null,
    ssim: null,
    fullPsnr: null,
    fullSsim: null,
    liveLoss: null,
    gpuUtil: null,
    vram: null,
    vramTotalGb: null,
    temp: null,
    cpuUtil: null,
    ramGb: null,
    ramTotalGb: null,
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
    validationRunning: false,
  }),
}));
