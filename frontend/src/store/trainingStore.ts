import { create } from "zustand";

export type TrainingStatus =
  | "idle"
  | "running"
  | "paused"
  | "done"
  | "failed"
  | "disconnected";

// Current sub-step of an active run, driven by backend `phase` events. Lets the
// UI show what the GPU subprocess is doing between step reports (e.g. the
// up-front MIOpen/cuDNN kernel warmup, which emits no step events).
export type TrainingStage =
  | "starting"
  | "preparing"
  | "warmup"
  | "training"
  | "validating"
  | "saving";

export interface RunHistory {
  gLossHistory: number[];
  dLossHistory: (number | null)[];
  totalLossHistory: number[];
  psnrHistory: number[];
  ssimHistory: number[];
  valLossHistory: number[];
  metricEpochs: number[];
}

export interface ValidationProgress {
  done: number;
  total: number;
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
  fullPsnr: number | null;
  fullSsim: number | null;
  receivedAt: number;
}

export interface LaunchConfig {
  totalEpochs: number;
  batchSize: number;
  learningRate: number;
  fp16: boolean;
  patchSize: number;
  validationEnabled: boolean;
}

const MAX_VALIDATION_HISTORY = 300;
// Hardware readings (GPU/VRAM/temp/CPU/RAM) aren't emitted as history arrays
// by the backend — just live snapshots every ~0.5s. We keep a capped rolling
// window of them in the store so the metrics sparklines persist across tab
// switches instead of living (and dying) in component state.
const MAX_HW_HISTORY = 90;

export interface HardwareData {
  cpu_percent: number | null;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  gpu_util_percent: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  temp_c: number | null;
}

export interface TrainingState {
  status: TrainingStatus;
  activeTrainingRunId: string | null;
  activeRunDirId: string | null;
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
  gpuUtilHistory: number[];
  vramPctHistory: number[];
  tempHistory: number[];
  cpuUtilHistory: number[];
  ramPctHistory: number[];
  lossHistory: number[];
  dLossHistory: (number | null)[];
  totalLossHistory: number[];
  psnrHistory: number[];
  ssimHistory: number[];
  valLossHistory: number[];
  fullPsnrHistory: number[];
  fullSsimHistory: number[];
  metricEpochs: number[];
  fullEpochs: number[];
  fullEpoch: number | null;
  etaSec: number | null;
  finalEpoch: number | null;
  validationFrames: ValidationFrames | null;
  validationHistory: ValidationHistoryEntry[];
  validationRunning: boolean;
  validationProgress: ValidationProgress | null;
  stage: TrainingStage | null;
  preparingProgress: ValidationProgress | null;
  errorCode: string | null;
  errorMessage: string | null;
  launchConfig: LaunchConfig | null;

  setStatus: (status: TrainingStatus) => void;
  setError: (code: string, message: string) => void;
  setActiveRun: (runId: string | null) => void;
  setActiveRunDir: (runDirId: string | null) => void;
  setValidationFrames: (frames: ValidationFrames | null) => void;
  pushValidationFrames: (epoch: number, frames: ValidationFrames, psnr?: number | null, ssim?: number | null, fullPsnr?: number | null, fullSsim?: number | null) => void;
  setValidationRunning: (v: boolean) => void;
  setValidationProgress: (p: ValidationProgress | null) => void;
  setStage: (stage: TrainingStage | null) => void;
  setPreparingProgress: (p: ValidationProgress | null) => void;
  updateFromStep: (epoch: number, batch: number, totalBatch: number, speed: number) => void;
  setLiveLoss: (avg: number | null) => void;
  pushEpochLoss: (avgLoss: number) => void;
  updateFromValidate: (epoch: number, psnr: number, ssim: number, fullPsnr?: number, fullSsim?: number, valLoss?: number) => void;
  updateFromHardware: (data: HardwareData) => void;
  setFinalEpoch: (epoch: number) => void;
  setLaunchConfig: (config: LaunchConfig) => void;
  reset: () => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  status: "idle",
  activeTrainingRunId: null,
  activeRunDirId: null,
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
  gpuUtilHistory: [],
  vramPctHistory: [],
  tempHistory: [],
  cpuUtilHistory: [],
  ramPctHistory: [],
  lossHistory: [],
  dLossHistory: [],
  totalLossHistory: [],
  psnrHistory: [],
  ssimHistory: [],
  valLossHistory: [],
  fullPsnrHistory: [],
  fullSsimHistory: [],
  metricEpochs: [],
  fullEpochs: [],
  fullEpoch: null,
  etaSec: null,
  finalEpoch: null,
  validationFrames: null,
  validationHistory: [],
  validationRunning: false,
  validationProgress: null,
  stage: null,
  preparingProgress: null,
  errorCode: null,
  errorMessage: null,
  launchConfig: null,

  setStatus: (status) => set({ status }),
  setError: (code, message) => set({ errorCode: code, errorMessage: message, status: "failed" }),
  setActiveRun: (runId) => set({ activeTrainingRunId: runId }),
  setActiveRunDir: (runDirId) => set({ activeRunDirId: runDirId }),
  setValidationFrames: (frames) => set({ validationFrames: frames }),
  setLaunchConfig: (config) => set({ launchConfig: config }),
  setStage: (stage) => set({ stage }),
  setPreparingProgress: (p) => set({ preparingProgress: p }),

  pushValidationFrames: (epoch, frames, psnr = null, ssim = null, fullPsnr = null, fullSsim = null) =>
    set((s) => {
      const entry: ValidationHistoryEntry = { epoch, ...frames, psnr, ssim, fullPsnr, fullSsim, receivedAt: Date.now() };
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

  setValidationProgress: (p) => set({ validationProgress: p }),

  updateFromStep: (epoch, batch, totalBatch, speed) =>
    set((s) => ({
      epoch,
      batch,
      totalBatch,
      speed,
      iter: s.iter + 1,
      validationRunning: false,
      validationProgress: null,
    })),

  setLiveLoss: (avg) => set({ liveLoss: avg }),

  pushEpochLoss: (avgLoss) =>
    set((s) => ({
      gLoss: avgLoss,
      lossHistory: [...s.lossHistory, avgLoss].slice(-500),
    })),

  updateFromValidate: (epoch, psnr, ssim, fullPsnr, fullSsim, valLoss) =>
    set((s) => {
      const psnrH = [...s.psnrHistory, psnr].slice(-500);
      const ssimH = [...s.ssimHistory, ssim].slice(-500);
      const epochsH = [...s.metricEpochs, epoch].slice(-500);
      const fullPsnrH = fullPsnr != null
        ? [...s.fullPsnrHistory, fullPsnr].slice(-500) : s.fullPsnrHistory;
      const fullSsimH = fullSsim != null
        ? [...s.fullSsimHistory, fullSsim].slice(-500) : s.fullSsimHistory;
      const fullEpochsH = fullPsnr != null
        ? [...s.fullEpochs, epoch].slice(-500) : s.fullEpochs;
      const valLossH = valLoss != null
        ? [...s.valLossHistory, valLoss].slice(-500) : s.valLossHistory;
      const best = s.bestPsnr !== null ? Math.max(s.bestPsnr, psnr) : psnr;
      return {
        epoch, psnr, ssim,
        fullPsnr: fullPsnr ?? s.fullPsnr,
        fullSsim: fullSsim ?? s.fullSsim,
        fullEpoch: fullPsnr != null ? epoch : s.fullEpoch,
        bestPsnr: best,
        psnrHistory: psnrH, ssimHistory: ssimH, metricEpochs: epochsH,
        fullPsnrHistory: fullPsnrH, fullSsimHistory: fullSsimH, fullEpochs: fullEpochsH,
        valLossHistory: valLossH,
      };
    }),

  updateFromHardware: (data) =>
    set((s) => {
      const push = (arr: number[], v: number | null) =>
        v != null && Number.isFinite(v) ? [...arr, v].slice(-MAX_HW_HISTORY) : arr;
      const pct = (used: number | null, total: number | null) =>
        used != null && total != null && total > 0 ? (used / total) * 100 : null;
      return {
        cpuUtil: data.cpu_percent,
        ramGb: data.ram_used_gb,
        ramTotalGb: data.ram_total_gb,
        gpuUtil: data.gpu_util_percent,
        vram: data.vram_used_gb,
        vramTotalGb: data.vram_total_gb,
        temp: data.temp_c,
        gpuUtilHistory: push(s.gpuUtilHistory, data.gpu_util_percent),
        vramPctHistory: push(s.vramPctHistory, pct(data.vram_used_gb, data.vram_total_gb)),
        tempHistory: push(s.tempHistory, data.temp_c),
        cpuUtilHistory: push(s.cpuUtilHistory, data.cpu_percent),
        ramPctHistory: push(s.ramPctHistory, pct(data.ram_used_gb, data.ram_total_gb)),
      };
    }),

  setFinalEpoch: (finalEpoch) => set({ finalEpoch }),

  reset: () => set({
    status: "idle",
    activeTrainingRunId: null,
    activeRunDirId: null,
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
    gpuUtilHistory: [],
    vramPctHistory: [],
    tempHistory: [],
    cpuUtilHistory: [],
    ramPctHistory: [],
    lossHistory: [],
    dLossHistory: [],
    totalLossHistory: [],
    psnrHistory: [],
    ssimHistory: [],
    valLossHistory: [],
    fullPsnrHistory: [],
    fullSsimHistory: [],
    metricEpochs: [],
    fullEpochs: [],
    fullEpoch: null,
    etaSec: null,
    finalEpoch: null,
    validationFrames: null,
    validationHistory: [],
    validationRunning: false,
    validationProgress: null,
    stage: null,
    preparingProgress: null,
    errorCode: null,
    errorMessage: null,
    launchConfig: null,
  }),
}));