import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { TrainLossConfig, LossType } from "../lib/api-types";
import { getDefaultLosses } from "../lib/api-types";

export interface TrainingSchedule {
  totalEpochs: number;
  saveEvery: number;
  warmupSteps: number;
}

const DEFAULT_SCHEDULE: TrainingSchedule = {
  totalEpochs: 100,
  saveEvery: 5,
  warmupSteps: 2000,
};

export interface VramMeasureState {
  status: "idle" | "running" | "done" | "error";
  allocatedMb: number | null;
  reservedMb: number | null;
  error: string | null;
  measuredFor: string | null;
  measuredAt: string | null;
}

const DEFAULT_VRAM_MEASURE: VramMeasureState = {
  status: "idle",
  allocatedMb: null,
  reservedMb: null,
  error: null,
  measuredFor: null,
  measuredAt: null,
};

interface RunConfigState {
  runName: string;
  device: string;
  fp16: boolean;
  gradientCheckpointing: string;
  schedule: TrainingSchedule;
  batchSize: number;
  patchSize: number;
  learningRate: number;
  seed: number;
  weightDecay: number;
  betas: [number, number];
  numWorkers: number;
  metricsFrequency: number;
  writeMetricsFile: boolean;
  validationEnabled: boolean;
  validationSplit: number;
  validationSplitSeed: number;
  validationFullImageLimit: number;
  lossConfig: TrainLossConfig;

  selectedInstance: string | null;
  instanceArchitecture: string | null;
  instanceScale: number | null;
  instanceConfig: Record<string, unknown> | null;

  selectedDataset: string | null;
  selectedDatasetPath: string | null;
  selectedDatasetPairs: number | null;
  selectedValidationDataset: string | null;

  resumeFrom: string | null;
  instanceVersions: { tag: string; path: string }[];

  vramMeasure: VramMeasureState;

  setRunName: (v: string) => void;
  setDevice: (v: string) => void;
  setFp16: (v: boolean) => void;
  setGradientCheckpointing: (v: string) => void;
  setSchedule: (v: Partial<TrainingSchedule>) => void;
  setBatchSize: (v: number) => void;
  setPatchSize: (v: number) => void;
  setLearningRate: (v: number) => void;
  setSeed: (v: number) => void;
  setWeightDecay: (v: number) => void;
  setBetas: (v: [number, number]) => void;
  setNumWorkers: (v: number) => void;
  setMetricsFrequency: (v: number) => void;
  setWriteMetricsFile: (v: boolean) => void;
  setValidationEnabled: (v: boolean) => void;
  setValidationSplit: (v: number) => void;
  setValidationSplitSeed: (v: number) => void;
  setValidationFullImageLimit: (v: number) => void;
  setLossConfig: (v: TrainLossConfig) => void;
  setLossWeight: (name: string, weight: number) => void;
  addLoss: (type: LossType, name?: string) => void;
  removeLoss: (name: string) => void;
  setSelectedInstance: (v: string | null) => void;
  setInstanceArchitecture: (v: string | null) => void;
  setInstanceScale: (v: number | null) => void;
  setInstanceConfig: (v: Record<string, unknown> | null) => void;
  setSelectedDataset: (v: string | null) => void;
  setSelectedDatasetPath: (v: string | null) => void;
  setSelectedDatasetPairs: (v: number | null) => void;
  setSelectedValidationDataset: (v: string | null) => void;
  setResumeFrom: (v: string | null) => void;
  setInstanceVersions: (v: { tag: string; path: string }[]) => void;
  setVramMeasure: (v: VramMeasureState | ((prev: VramMeasureState) => VramMeasureState)) => void;
}

// Resilient localStorage wrapper — falls back to in‑memory map when
// localStorage is unavailable (node without --localstorage-file, vitest, etc.).
function safeStorage(): Storage {
  let mem: Record<string, string> = {};
  return {
    getItem: (k: string) => mem[k] ?? null,
    setItem: (k: string, v: string) => { mem[k] = v; },
    removeItem: (k: string) => { delete mem[k]; },
    clear: () => { mem = {}; },
    get length() { return Object.keys(mem).length; },
    key: (i: number) => Object.keys(mem)[i] ?? null,
  };
}

let _storage: Storage | null = null;
function getStorage(): Storage {
  if (_storage) return _storage;
  try {
    if (typeof localStorage !== "undefined") {
      void localStorage.length; // throws in node without flag
      _storage = localStorage;
      return _storage;
    }
  } catch { /* fall through */ }
  _storage = safeStorage();
  return _storage;
}

export const useRunConfigStore = create<RunConfigState>()(
  persist(
    (set) => ({
      runName: "",
      device: "auto",
      fp16: false,
      gradientCheckpointing: "auto",
      schedule: DEFAULT_SCHEDULE,
      batchSize: 16,
      patchSize: 128,
      learningRate: 2e-4,
      seed: 42,
      weightDecay: 0.0,
      betas: [0.9, 0.99] as [number, number],
      numWorkers: 4,
      metricsFrequency: 1,
      writeMetricsFile: true,
      validationEnabled: true,
      validationSplit: 0.1,
      validationSplitSeed: 1234,
      validationFullImageLimit: 8,
      lossConfig: getDefaultLosses(),

      selectedInstance: null,
      instanceArchitecture: null,
      instanceScale: null,
      instanceConfig: null,

      selectedDataset: null,
      selectedDatasetPath: null,
      selectedDatasetPairs: null,
      selectedValidationDataset: null,

      resumeFrom: null,
      instanceVersions: [],

      vramMeasure: { ...DEFAULT_VRAM_MEASURE },

      setRunName: (v) => set({ runName: v }),
      setDevice: (v) => set({ device: v }),
      setFp16: (v) => set({ fp16: v }),
      setGradientCheckpointing: (v) => set({ gradientCheckpointing: v }),
      setSchedule: (v) => set((s) => ({ schedule: { ...s.schedule, ...v } })),
      setBatchSize: (v) => set({ batchSize: v }),
      setPatchSize: (v) => set({ patchSize: v }),
      setLearningRate: (v) => set({ learningRate: v }),
      setSeed: (v) => set({ seed: v }),
      setWeightDecay: (v) => set({ weightDecay: v }),
      setBetas: (v) => set({ betas: v }),
      setNumWorkers: (v) => set({ numWorkers: v }),
      setMetricsFrequency: (v) => set({ metricsFrequency: v }),
      setWriteMetricsFile: (v) => set({ writeMetricsFile: v }),
      setValidationEnabled: (v) => set({ validationEnabled: v }),
      setValidationSplit: (v) => set({ validationSplit: v }),
      setValidationSplitSeed: (v) => set({ validationSplitSeed: v }),
      setValidationFullImageLimit: (v) => set({ validationFullImageLimit: v }),
      setLossConfig: (v) => set({ lossConfig: v }),
      setLossWeight: (name, weight) =>
        set((s) => {
          if (!s.lossConfig[name]) return s;
          return { lossConfig: { ...s.lossConfig, [name]: { ...s.lossConfig[name], weight } } };
        }),
      addLoss: (type, name) =>
        set((s) => {
          const key = name ?? type;
          if (s.lossConfig[key]) return s;
          const entry: TrainLossConfig[string] = { type, weight: 0.1 };
          if (type === "vgg") entry.layers = ["relu5_4"];
          if (type === "style") entry.layers = ["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_2"];
          return { lossConfig: { ...s.lossConfig, [key]: entry } };
        }),
      removeLoss: (name) =>
        set((s) => {
          const next = { ...s.lossConfig };
          delete next[name];
          return { lossConfig: next };
        }),
      setSelectedInstance: (v) => set({ selectedInstance: v }),
      setInstanceArchitecture: (v) => set({ instanceArchitecture: v }),
      setInstanceScale: (v) => set({ instanceScale: v }),
      setInstanceConfig: (v) => set({ instanceConfig: v }),
      setSelectedDataset: (v) => set({ selectedDataset: v }),
      setSelectedDatasetPath: (v) => set({ selectedDatasetPath: v }),
      setSelectedDatasetPairs: (v) => set({ selectedDatasetPairs: v }),
      setSelectedValidationDataset: (v) => set({ selectedValidationDataset: v }),
      setResumeFrom: (v) => set({ resumeFrom: v }),
      setInstanceVersions: (v) => set({ instanceVersions: v }),
      setVramMeasure: (v) =>
        set((s) => ({
          vramMeasure:
            typeof v === "function"
              ? (v as (prev: VramMeasureState) => VramMeasureState)(s.vramMeasure)
              : v,
        })),
    }),
    {
      name: "sr-tuner:run-config",
      storage: createJSONStorage(getStorage),
      partialize: (s) => ({
        runName: s.runName,
        device: s.device,
        fp16: s.fp16,
        gradientCheckpointing: s.gradientCheckpointing,
        schedule: s.schedule,
        batchSize: s.batchSize,
        patchSize: s.patchSize,
        learningRate: s.learningRate,
        seed: s.seed,
        weightDecay: s.weightDecay,
        betas: s.betas,
        numWorkers: s.numWorkers,
        metricsFrequency: s.metricsFrequency,
        writeMetricsFile: s.writeMetricsFile,
        validationEnabled: s.validationEnabled,
        validationSplit: s.validationSplit,
        validationSplitSeed: s.validationSplitSeed,
        validationFullImageLimit: s.validationFullImageLimit,
        lossConfig: s.lossConfig,
        selectedInstance: s.selectedInstance,
        selectedDataset: s.selectedDataset,
        selectedDatasetPath: s.selectedDatasetPath,
        selectedDatasetPairs: s.selectedDatasetPairs,
        selectedValidationDataset: s.selectedValidationDataset,
        resumeFrom: s.resumeFrom,
      }),
    },
  ),
);