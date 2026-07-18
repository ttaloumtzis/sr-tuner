import { create } from "zustand";

export interface ResumeFrom {
  checkpoint_path: string;
  resume_epoch: number;
  resume_optimizer_state: boolean;
  resume_lr_scheduler_state: boolean;
}

export interface TrainingSchedule {
  totalEpochs: number;
  saveEvery: number;
  validateEvery: number;
  warmupIter: number;
  lrDecay: string;
}

const DEFAULT_SCHEDULE: TrainingSchedule = {
  totalEpochs: 100,
  saveEvery: 5,
  validateEvery: 1,
  warmupIter: 1000,
  lrDecay: "cosine",
};

interface RunConfigState {
  runName: string;
  outputDir: string;
  checkpointDir: string;
  logDir: string;
  device: string;
  schedule: TrainingSchedule;
  tensorboard: boolean;
  fp16: boolean;
  compile: boolean;
  resumeFrom: ResumeFrom | null;
  setRunName: (name: string) => void;
  setOutputDir: (dir: string) => void;
  setCheckpointDir: (dir: string) => void;
  setLogDir: (dir: string) => void;
  setDevice: (device: string) => void;
  setSchedule: (schedule: Partial<TrainingSchedule>) => void;
  setTensorboard: (enabled: boolean) => void;
  setFp16: (enabled: boolean) => void;
  setCompile: (enabled: boolean) => void;
  setResumeFrom: (r: ResumeFrom | null) => void;
}

export const useRunConfigStore = create<RunConfigState>((set) => ({
  runName: "",
  outputDir: "",
  checkpointDir: "",
  logDir: "",
  device: "cpu",
  schedule: DEFAULT_SCHEDULE,
  tensorboard: false,
  fp16: false,
  compile: false,
  resumeFrom: null,
  setRunName: (name) => set({ runName: name }),
  setOutputDir: (dir) => set({ outputDir: dir }),
  setCheckpointDir: (dir) => set({ checkpointDir: dir }),
  setLogDir: (dir) => set({ logDir: dir }),
  setDevice: (device) => set({ device }),
  setSchedule: (schedule) =>
    set((s) => ({ schedule: { ...s.schedule, ...schedule } })),
  setTensorboard: (enabled) => set({ tensorboard: enabled }),
  setFp16: (enabled) => set({ fp16: enabled }),
  setCompile: (enabled) => set({ compile: enabled }),
  setResumeFrom: (r) => set({ resumeFrom: r }),
}));
