import { create } from "zustand";
import { basename } from "../lib/path";

export type DatasetSubTab = "create" | "browse" | "merge";
export type DatasetMode = "image_folder" | "video_extract" | "on_the_fly";
export type DownscaleKernel = "bicubic" | "bilinear" | "real-world" | "area" | "lanczos" | "nearest";
export type ResizeMethod = "area" | "bicubic" | "bilinear" | "lanczos" | "nearest";
export type JobStatus = "idle" | "running" | "done" | "error";

export interface ExtractionProgress {
  framesDone: number;
  framesTotal: number;
  fps: number;
  etaSec: number;
}

export interface ProgressStep {
  id: number;
  desc: string;
  total: number;
  current: number;
  status: "active" | "done" | "pending";
}

export interface VideoFileEntry {
  name: string;
  path: string;
  status: "pending" | "extracting" | "done";
}

interface DatasetState {
  subTab: DatasetSubTab;
  mode: DatasetMode;

  scale: number;
  kernel: DownscaleKernel;
  namingPattern: string;

  rootPath: string;

  frameRate: number;
  frameFormat: string;
  startTime: number;
  duration: number | null;
  resizeMethod: ResizeMethod;
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

  videoFiles: VideoFileEntry[];
  extractionProgress: ExtractionProgress | null;

  mergeOutputPath: string;
  mergeCustomName: string;
  mergeKeepSources: boolean;
  mergeScaleFilter: number | null;

  jobId: string | null;
  jobStatus: JobStatus;
  jobError: string | null;
  jobType: "build" | "health" | "merge" | "prune" | "validate" | null;
  progressSteps: ProgressStep[];
  mergeResults: { scale: number; output_path: string; source_datasets: string[] }[] | null;
  healthReport: Record<string, unknown> | null;
  validationResult: { valid: boolean; problems: string[]; num_pairs: number } | null;

  setSubTab: (tab: DatasetSubTab) => void;
  setMode: (mode: DatasetMode) => void;
  setScale: (s: number) => void;
  setKernel: (k: DownscaleKernel) => void;
  setNamingPattern: (p: string) => void;
  setRootPath: (p: string) => void;
  setFrameRate: (n: number) => void;
  setFrameFormat: (f: string) => void;
  setStartTime: (t: number) => void;
  setDuration: (d: number | null) => void;
  setResizeMethod: (m: ResizeMethod) => void;
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

  addVideoFiles: (paths: string[]) => void;
  clearVideoFiles: () => void;
  removeVideoFile: (path: string) => void;
  setExtractionProgress: (p: ExtractionProgress | null) => void;

  setMergeOutputPath: (p: string) => void;
  setMergeCustomName: (n: string) => void;
  setMergeKeepSources: (v: boolean) => void;
  setMergeScaleFilter: (s: number | null) => void;

  setJobId: (id: string | null) => void;
  setJobStatus: (status: JobStatus) => void;
  setJobError: (err: string | null) => void;
  setJobType: (t: "build" | "health" | "merge" | "prune" | "validate" | null) => void;
  startProgressStep: (desc: string, total: number) => void;
  updateProgressStep: (stepId: number, current: number, fps: number, etaSec: number) => void;
  finishProgressStep: (stepId: number) => void;
  clearJob: () => void;
  setMergeResults: (results: { scale: number; output_path: string; source_datasets: string[] }[] | null) => void;
  setHealthReport: (report: Record<string, unknown> | null) => void;
  setValidationResult: (result: { valid: boolean; problems: string[]; num_pairs: number } | null) => void;
}

export const useDatasetStore = create<DatasetState>((set) => ({
  subTab: "create",
  mode: "image_folder",
  scale: 4,
  kernel: "bicubic",
  namingPattern: "%06d",
  rootPath: "",
  frameRate: 10,
  frameFormat: "png",
  startTime: 0,
  duration: null,
  resizeMethod: "area",
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

  videoFiles: [],
  extractionProgress: null,

  mergeOutputPath: "",
  mergeCustomName: "",
  mergeKeepSources: false,
  mergeScaleFilter: null,

  jobId: null,
  jobStatus: "idle",
  jobError: null,
  jobType: null,
  progressSteps: [],
  mergeResults: null,
  healthReport: null,
  validationResult: null,

  setSubTab: (subTab) => set({
    subTab,
    jobId: null,
    jobStatus: "idle",
    jobError: null,
    jobType: null,
    progressSteps: [],
    extractionProgress: null,
  }),
  setMode: (mode) => set({ mode }),
  setScale: (scale) => set({ scale }),
  setKernel: (kernel) => set({ kernel }),
  setNamingPattern: (namingPattern) => set({ namingPattern }),
  setRootPath: (rootPath) => set({ rootPath }),
  setFrameRate: (frameRate) => set({ frameRate }),
  setFrameFormat: (frameFormat) => set({ frameFormat }),
  setStartTime: (startTime) => set({ startTime }),
  setDuration: (duration) => set({ duration }),
  setResizeMethod: (resizeMethod) => set({ resizeMethod }),
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

  addVideoFiles: (paths) =>
    set((s) => {
      const existing = new Set(s.videoFiles.map((f) => f.path));
      const toAdd = paths.filter((p) => !existing.has(p));
      return {
        videoFiles: [
          ...s.videoFiles,
          ...toAdd.map((p) => ({ name: basename(p) ?? p, path: p, status: "pending" as const })),
        ],
      };
    }),
  clearVideoFiles: () => set({ videoFiles: [] }),
  removeVideoFile: (path: string) =>
    set((s) => ({ videoFiles: s.videoFiles.filter((f) => f.path !== path) })),
  setExtractionProgress: (extractionProgress) => set({ extractionProgress }),

  setMergeOutputPath: (mergeOutputPath) => set({ mergeOutputPath }),
  setMergeCustomName: (mergeCustomName) => set({ mergeCustomName }),
  setMergeKeepSources: (mergeKeepSources) => set({ mergeKeepSources }),
  setMergeScaleFilter: (mergeScaleFilter) => set({ mergeScaleFilter }),

  setJobId: (jobId) => set({ jobId }),
  setJobStatus: (jobStatus) => set({ jobStatus }),
  setJobError: (jobError) => set({ jobError }),
  setJobType: (jobType) => set({ jobType }),
  startProgressStep: (desc, total) =>
    set((s) => {
      const id = s.progressSteps.length;
      const step: ProgressStep = { id, desc, total, current: 0, status: "active" };
      const steps = s.progressSteps.map((st) =>
        st.status === "active" ? { ...st, status: "done" as const } : st
      );
      return { progressSteps: [...steps, step] };
    }),
  updateProgressStep: (stepId, current, fps, etaSec) =>
    set((s) => ({
      progressSteps: s.progressSteps.map((st) =>
        st.id === stepId ? { ...st, current } : st
      ),
      extractionProgress: { framesDone: current, framesTotal: s.progressSteps[stepId]?.total ?? 0, fps, etaSec },
    })),
  finishProgressStep: (stepId) =>
    set((s) => ({
      progressSteps: s.progressSteps.map((st) =>
        st.id === stepId ? { ...st, status: "done" as const } : st
      ),
    })),
  clearJob: () => set({ jobId: null, jobStatus: "idle", jobError: null, jobType: null, progressSteps: [], extractionProgress: null, mergeResults: null }),
  setMergeResults: (mergeResults) => set({ mergeResults }),
  setHealthReport: (healthReport) => set({ healthReport }),
  setValidationResult: (validationResult) => set({ validationResult }),
}));