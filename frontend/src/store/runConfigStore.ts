import { create } from "zustand";
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

interface RunConfigState {
  runName: string;
  device: string;
  fp16: boolean;
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

  setRunName: (v: string) => void;
  setDevice: (v: string) => void;
  setFp16: (v: boolean) => void;
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
}

export const useRunConfigStore = create<RunConfigState>((set) => ({
  runName: "",
  device: "auto",
  fp16: false,
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

  setRunName: (v) => set({ runName: v }),
  setDevice: (v) => set({ device: v }),
  setFp16: (v) => set({ fp16: v }),
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
}));
