import { create } from "zustand";
import type { HealthReport } from "../lib/api-types";

export type DatasetSubTab = "create" | "browse" | "merge";
export type DatasetMode = "image_folder" | "video_extract" | "on_the_fly";
export type DownscaleKernel = "bicubic" | "bilinear" | "area" | "lanczos" | "nearest";
export type JobStatus = "idle" | "running" | "done" | "error";

export interface ProgressStep {
  id: number;
  desc: string;
  total: number | null;
  current: number;
  status: "active" | "done" | "pending";
}

export interface VideoFileEntry {
  name: string;
  path: string;
}

interface DatasetState {
  subTab: DatasetSubTab;
  mode: DatasetMode;

  scale: number;
  kernel: DownscaleKernel;

  rootPath: string;

  frameRate: number;
  startTime: number;
  duration: number | null;
  antialias: boolean;

  degBlur: boolean;
  degNoise: boolean;
  degJpeg: boolean;
  degJpeg2000: boolean;
  degColorJitter: boolean;

  blurKernelSize: number;
  blurSigmaMin: number;
  blurSigmaMax: number;
  blurGaussianProb: number;
  motionBlurEnabled: boolean;
  motionBlurMaxKernel: number;
  blurMotionProb: number;

  noiseSigmaMin: number;
  noiseSigmaMax: number;
  noiseGaussianProb: number;
  poissonScaleMin: number;
  poissonScaleMax: number;
  noisePoissonProb: number;
  saltPepperAmount: number;
  noiseSaltPepperProb: number;

  jpegQualityMin: number;
  jpegQualityMax: number;
  jpegProb: number;

  jpeg2000QualityMin: number;
  jpeg2000QualityMax: number;
  jpeg2000Prob: number;

  jitterHueRange: number;
  jitterSaturationRange: number;
  jitterValueRange: number;
  jitterProb: number;

  videoFile: VideoFileEntry | null;

  mergeCustomName: string;
  mergeKeepSources: boolean;
  mergeScaleFilter: number | null;

  jobId: string | null;
  jobStatus: JobStatus;
  jobError: string | null;
  jobType: "build" | "health" | "merge" | "prune" | "validate" | null;
  jobDatasetPath: string | null;
  jobHealthReport: HealthReport | null;
  progressSteps: ProgressStep[];
  mergeResults: { scale: number; output_path: string; source_datasets: string[] }[] | null;
  validationResult: { valid: boolean; problems: string[]; num_pairs: number } | null;

  setSubTab: (tab: DatasetSubTab) => void;
  setMode: (mode: DatasetMode) => void;
  setScale: (s: number) => void;
  setKernel: (k: DownscaleKernel) => void;
  setRootPath: (p: string) => void;
  setFrameRate: (n: number) => void;
  setStartTime: (t: number) => void;
  setDuration: (d: number | null) => void;
  setAntialias: (v: boolean) => void;

  setDegBlur: (v: boolean) => void;
  setDegNoise: (v: boolean) => void;
  setDegJpeg: (v: boolean) => void;
  setDegJpeg2000: (v: boolean) => void;
  setDegColorJitter: (v: boolean) => void;

  setBlurKernelSize: (n: number) => void;
  setBlurSigmaRange: (min: number, max: number) => void;
  setBlurGaussianProb: (n: number) => void;
  setMotionBlurEnabled: (v: boolean) => void;
  setMotionBlurMaxKernel: (n: number) => void;
  setBlurMotionProb: (n: number) => void;

  setNoiseSigmaRange: (min: number, max: number) => void;
  setNoiseGaussianProb: (n: number) => void;
  setPoissonScaleRange: (min: number, max: number) => void;
  setNoisePoissonProb: (n: number) => void;
  setSaltPepperAmount: (n: number) => void;
  setNoiseSaltPepperProb: (n: number) => void;

  setJpegQualityRange: (min: number, max: number) => void;
  setJpegProb: (n: number) => void;
  setJpeg2000QualityRange: (min: number, max: number) => void;
  setJpeg2000Prob: (n: number) => void;

  setJitterHueRange: (n: number) => void;
  setJitterSaturationRange: (n: number) => void;
  setJitterValueRange: (n: number) => void;
  setJitterProb: (n: number) => void;

  setVideoFile: (file: VideoFileEntry | null) => void;
  clearVideoFile: () => void;

  setMergeCustomName: (n: string) => void;
  setMergeKeepSources: (v: boolean) => void;
  setMergeScaleFilter: (s: number | null) => void;

  setJobId: (id: string | null) => void;
  setJobStatus: (status: JobStatus) => void;
  setJobError: (err: string | null) => void;
  setJobType: (t: "build" | "health" | "merge" | "prune" | "validate" | null) => void;
  setJobDatasetPath: (path: string | null) => void;
  setJobHealthReport: (report: HealthReport | null) => void;
  startProgressStep: (desc: string, total: number | null) => void;
  updateProgressStep: (stepId: number, current: number) => void;
  finishProgressStep: (stepId: number) => void;
  clearJob: () => void;
  setMergeResults: (results: { scale: number; output_path: string; source_datasets: string[] }[] | null) => void;
  setValidationResult: (result: { valid: boolean; problems: string[]; num_pairs: number } | null) => void;
}

export const useDatasetStore = create<DatasetState>((set) => ({
  subTab: "create",
  mode: "image_folder",
  scale: 4,
  kernel: "bicubic",
  rootPath: "",
  frameRate: 10,
  startTime: 0,
  duration: null,
  antialias: true,

  degBlur: true,
  degNoise: false,
  degJpeg: true,
  degJpeg2000: false,
  degColorJitter: false,

  blurKernelSize: 21,
  blurSigmaMin: 0.1,
  blurSigmaMax: 3.0,
  blurGaussianProb: 1.0,
  motionBlurEnabled: true,
  motionBlurMaxKernel: 31,
  blurMotionProb: 0.5,

  noiseSigmaMin: 1,
  noiseSigmaMax: 30,
  noiseGaussianProb: 0.5,
  poissonScaleMin: 0.05,
  poissonScaleMax: 3.0,
  noisePoissonProb: 0.5,
  saltPepperAmount: 0.01,
  noiseSaltPepperProb: 0.3,

  jpegQualityMin: 30,
  jpegQualityMax: 95,
  jpegProb: 1.0,

  jpeg2000QualityMin: 30,
  jpeg2000QualityMax: 95,
  jpeg2000Prob: 0.5,

  jitterHueRange: 0.05,
  jitterSaturationRange: 0.3,
  jitterValueRange: 0.3,
  jitterProb: 0.8,

  videoFile: null,

  mergeCustomName: "",
  mergeKeepSources: false,
  mergeScaleFilter: null,

  jobId: null,
  jobStatus: "idle",
  jobError: null,
  jobType: null,
  jobDatasetPath: null,
  jobHealthReport: null,
  progressSteps: [],
  mergeResults: null,
  validationResult: null,

  setSubTab: (subTab) => set({ subTab }),
  setMode: (mode) => set({ mode }),
  setScale: (scale) => set({ scale }),
  setKernel: (kernel) => set({ kernel }),
  setRootPath: (rootPath) => set({ rootPath }),
  setFrameRate: (frameRate) => set({ frameRate }),
  setStartTime: (startTime) => set({ startTime }),
  setDuration: (duration) => set({ duration }),
  setAntialias: (antialias) => set({ antialias }),

  setDegBlur: (degBlur) => set({ degBlur }),
  setDegNoise: (degNoise) => set({ degNoise }),
  setDegJpeg: (degJpeg) => set({ degJpeg }),
  setDegJpeg2000: (degJpeg2000) => set({ degJpeg2000 }),
  setDegColorJitter: (degColorJitter) => set({ degColorJitter }),

  setBlurKernelSize: (blurKernelSize) => set({ blurKernelSize }),
  setBlurSigmaRange: (min, max) => set({ blurSigmaMin: min, blurSigmaMax: max }),
  setBlurGaussianProb: (blurGaussianProb) => set({ blurGaussianProb }),
  setMotionBlurEnabled: (motionBlurEnabled) => set({ motionBlurEnabled }),
  setMotionBlurMaxKernel: (motionBlurMaxKernel) => set({ motionBlurMaxKernel }),
  setBlurMotionProb: (blurMotionProb) => set({ blurMotionProb }),

  setNoiseSigmaRange: (min, max) => set({ noiseSigmaMin: min, noiseSigmaMax: max }),
  setNoiseGaussianProb: (noiseGaussianProb) => set({ noiseGaussianProb }),
  setPoissonScaleRange: (min, max) => set({ poissonScaleMin: min, poissonScaleMax: max }),
  setNoisePoissonProb: (noisePoissonProb) => set({ noisePoissonProb }),
  setSaltPepperAmount: (saltPepperAmount) => set({ saltPepperAmount }),
  setNoiseSaltPepperProb: (noiseSaltPepperProb) => set({ noiseSaltPepperProb }),

  setJpegQualityRange: (min, max) => set({ jpegQualityMin: min, jpegQualityMax: max }),
  setJpegProb: (jpegProb) => set({ jpegProb }),
  setJpeg2000QualityRange: (min, max) => set({ jpeg2000QualityMin: min, jpeg2000QualityMax: max }),
  setJpeg2000Prob: (jpeg2000Prob) => set({ jpeg2000Prob }),

  setJitterHueRange: (jitterHueRange) => set({ jitterHueRange }),
  setJitterSaturationRange: (jitterSaturationRange) => set({ jitterSaturationRange }),
  setJitterValueRange: (jitterValueRange) => set({ jitterValueRange }),
  setJitterProb: (jitterProb) => set({ jitterProb }),

  setVideoFile: (videoFile) => set({ videoFile }),
  clearVideoFile: () => set({ videoFile: null }),

  setMergeCustomName: (mergeCustomName) => set({ mergeCustomName }),
  setMergeKeepSources: (mergeKeepSources) => set({ mergeKeepSources }),
  setMergeScaleFilter: (mergeScaleFilter) => set({ mergeScaleFilter }),

  setJobId: (jobId) => set({ jobId }),
  setJobStatus: (jobStatus) => set({ jobStatus }),
  setJobError: (jobError) => set({ jobError }),
  setJobType: (jobType) => set({ jobType }),
  setJobDatasetPath: (jobDatasetPath) => set({ jobDatasetPath }),
  setJobHealthReport: (jobHealthReport) => set({ jobHealthReport }),
  startProgressStep: (desc, total) =>
    set((s) => {
      const id = s.progressSteps.length;
      const step: ProgressStep = { id, desc, total, current: 0, status: "active" };
      const steps = s.progressSteps.map((st) =>
        st.status === "active" ? { ...st, status: "done" as const } : st
      );
      return { progressSteps: [...steps, step] };
    }),
  updateProgressStep: (stepId, current) =>
    set((s) => ({
      progressSteps: s.progressSteps.map((st) =>
        st.id === stepId ? { ...st, current } : st
      ),
    })),
  finishProgressStep: (stepId) =>
    set((s) => ({
      progressSteps: s.progressSteps.map((st) =>
        st.id === stepId ? { ...st, status: "done" as const } : st
      ),
    })),
  clearJob: () => set({ jobId: null, jobStatus: "idle", jobError: null, jobType: null, jobDatasetPath: null, jobHealthReport: null, progressSteps: [], mergeResults: null }),
  setMergeResults: (mergeResults) => set({ mergeResults }),
  setValidationResult: (validationResult) => set({ validationResult }),
}));