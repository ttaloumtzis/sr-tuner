import { create } from "zustand";

export type TrainingStatus =
  | "idle"
  | "running"
  | "paused"
  | "done"
  | "failed"
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

// A single epoch's worth of validation frames, as saved by the trainer into
// its own subfolder under validation/epoch_XXX/. Kept around (capped) so the
// UI can scrub back through the run's progression instead of only ever
// showing the latest epoch.
export interface ValidationHistoryEntry extends ValidationFrames {
  epoch: number;
  psnr: number | null;
  ssim: number | null;
  receivedAt: number;
}

const MAX_VALIDATION_HISTORY = 300;

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
  fullPsnrHistory: number[];
  fullSsimHistory: number[];
  etaSec: number | null;
  finalEpoch: number | null;
  validationFrames: ValidationFrames | null;
  validationHistory: ValidationHistoryEntry[];
  validationRunning: boolean;
  errorCode: string | null;
  errorMessage: string | null;

  setStatus: (status: TrainingStatus) => void;
  setError: (code: string, message: string) => void;
  setActiveRun: (runId: string | null) => void;
  setValidationFrames: (frames: ValidationFrames | null) => void;
  pushValidationFrames: (epoch: number, frames: ValidationFrames, psnr?: number | null, ssim?: number | null) => void;
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
  fullPsnrHistory: [],
  fullSsimHistory: [],
  etaSec: null,
  finalEpoch: null,
  validationFrames: null,
  validationHistory: [],
  validationRunning: false,
  errorCode: null,
  errorMessage: null,

  setStatus: (status) => set({ status }),
  setError: (code, message) => set({ errorCode: code, errorMessage: message, status: "failed" }),
  setActiveRun: (runId) => set({ activeTrainingRunId: runId }),
  setValidationFrames: (frames) => set({ validationFrames: frames }),

  pushValidationFrames: (epoch, frames, psnr = null, ssim = null) =>
    set((s) => {
      const entry: ValidationHistoryEntry = { epoch, ...frames, psnr, ssim, receivedAt: Date.now() };
      // Each epoch trains into its own validation/epoch_XXX/ subfolder, so a
      // given epoch should only ever appear once — but re-runs of the same
      // epoch (e.g. resumed training) replace the earlier entry in place.
      const idx = s.validationHistory.findIndex((e) => e.epoch === epoch);
      const nextHistory = idx >= 0
        ? [...s.validationHistory.slice(0, idx), entry, ...s.validationHistory.slice(idx + 1)]
        : [...s.validationHistory, entry].sort((a, b) => a.epoch - b.epoch);
      return {
        validationFrames: frames,
        validationHistory: nextHistory.slice(-MAX_VALIDATION_HISTORY),
      };
    }),

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
      const fullPsnrH = fullPsnr != null
        ? [...s.fullPsnrHistory, fullPsnr].slice(-500) : s.fullPsnrHistory;
      const fullSsimH = fullSsim != null
        ? [...s.fullSsimHistory, fullSsim].slice(-500) : s.fullSsimHistory;
      const best = s.bestPsnr !== null ? Math.max(s.bestPsnr, psnr) : psnr;
      return { epoch, psnr, ssim, fullPsnr: fullPsnr ?? s.fullPsnr, fullSsim: fullSsim ?? s.fullSsim, bestPsnr: best, psnrHistory: psnrH, ssimHistory: ssimH, fullPsnrHistory: fullPsnrH, fullSsimHistory: fullSsimH };
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
    fullPsnrHistory: [],
    fullSsimHistory: [],
    etaSec: null,
    finalEpoch: null,
    validationFrames: null,
    validationHistory: [],
    validationRunning: false,
    errorCode: null,
    errorMessage: null,
  }),
}));