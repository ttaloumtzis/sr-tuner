export interface WorkspaceInfo {
  path: string;
  exists: boolean;
  models: { name: string; path: string }[];
}

export interface ModelInstance {
  name: string;
  path: string;
  architecture: string | null;
  scale: number | null;
  checkpoints: string[];
  latest_version?: string | null;
  config?: Record<string, unknown>;
}

export interface ExportParams {
  instance: string;
  format: "pth" | "onnx" | "torchscript" | "safetensors";
  output?: string;
}

export interface DatasetInfo {
  name: string;
  path: string;
  scale: number;
  num_pairs: number;
}

export interface HealthReport {
  total_images: number;
  resolutions: Record<string, number>;
  aspect_ratios: Record<string, number>;
  channels: Record<string, number>;
  computed_threshold: number;
  black_frames: string[];
}

export type LossType = "l1" | "l2" | "vgg" | "edge" | "style" | "fft" | "ssim" | "lpips";

export interface LossEntry {
  type: LossType;
  weight: number;
  layers?: string[];
}

export type TrainLossConfig = Record<string, LossEntry>;

const DEFAULT_LOSSES: TrainLossConfig = {
  pixel: { type: "l1", weight: 1.0 },
  perceptual: { type: "vgg", weight: 0.1, layers: ["relu5_4"] },
};

export function getDefaultLosses(): TrainLossConfig {
  return JSON.parse(JSON.stringify(DEFAULT_LOSSES));
}

export const LOSS_TYPE_OPTIONS: { value: LossType; label: string; needsLayers: boolean }[] = [
  { value: "l1", label: "L1 (Charbonnier)", needsLayers: false },
  { value: "l2", label: "L2 (MSE)", needsLayers: false },
  { value: "vgg", label: "VGG Perceptual", needsLayers: true },
  { value: "edge", label: "Edge (Sobel)", needsLayers: false },
  { value: "style", label: "Style (Gram)", needsLayers: true },
  { value: "fft", label: "Frequency (FFT)", needsLayers: false },
  { value: "ssim", label: "SSIM", needsLayers: false },
  { value: "lpips", label: "LPIPS", needsLayers: false },
];

export interface TrainParams {
  model_name: string;
  instance: string;
  dataset: string;
  config?: string;
  resume?: string;
  device?: string;
  batch_size?: number;
  learning_rate?: number;
  max_epochs?: number;
  patch_size?: number;
  fp16?: boolean;
  seed?: number;
  weight_decay?: number;
  betas?: [number, number];
  num_workers?: number;
  save_per_epoch?: number;
  validation_enabled?: boolean;
  validation_split?: number;
  validation_dataset?: string;
  metrics_frequency?: number;
  perceptual_weight?: number;
  warmup_steps?: number;
  write_metrics_file?: boolean;
  losses?: TrainLossConfig;
}

export interface InferParams {
  model?: string;
  instance?: string;
  version?: string;
  input: string;
  output: string;
  tile?: number;
  overlap?: number;
  device?: string;
}

export interface JobAccepted {
  job_id: string;
  status: string;
}

export interface JobStatus {
  job_id: string;
  job_type: string;
  status: string;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  error?: string;
  result?: Record<string, unknown>;
  config?: Record<string, unknown>;
}

export interface SystemInfo {
  os: string;
  os_distro: string | null;
  cuda_available: boolean;
  rocm_available: boolean;
  mps_available: boolean;
  has_ffmpeg: boolean;
  has_uv: boolean;
  has_python3: boolean;
  supported_backends: string[];
  default_backend: string;
}

export interface EnvMeta {
  app_version: string;
  backend: string;
  env_type: string;
  env_path: string;
  installed_at: string;
}

export interface EnvInfo {
  torch_version: string;
  device: string;
  cuda_available: boolean;
  rocm: boolean;
  bf16_supported: boolean;
  flash_attn: boolean;
  device_name?: string;
  vram_total_mb?: number;
}

export interface DatasetBuildParams {
  input: string;
  out?: string;
  config?: string;
  degradations?: string;
  config_overrides?: Record<string, unknown>;
}

export interface DatasetValidateParams {
  path: string;
}

export interface DatasetHealthParams {
  path: string;
  yes?: boolean;
}

export interface DatasetMergeParams {
  input: string;
  out?: string;
  scale?: number;
  name?: string;
  keep_sources?: boolean;
  input_datasets?: string[];
}

export interface DatasetManifest {
  config: {
    scale: number;
    frame_rate?: number | null;
    video_source?: string;
    sources?: string[];
  };
  pairs: { hr: string; lr: string }[];
}

export interface ModelVersion {
  tag: string;
  path: string;
  metadata?: Record<string, unknown>;
}

export interface CheckpointEntry {
  epoch: number;
  filename: string;
  path: string;
  created_at: string;
  file_size_mb: number;
  metrics: { loss?: number; psnr?: number; ssim?: number };
}

export interface MetricsEvent {
  iter: number;
  epoch: number;
  g_loss: number;
  d_loss: number | null;
  psnr: number | null;
  ssim: number | null;
  gpu_util: number | null;
  vram_gb: number | null;
  temp_c: number | null;
  cpu_util: number | null;
  speed: number | null;
  timestamp: string;
}

// ── SSE event types (from api/event_manager.py) ─────────────────────────

export type SSEEvent =
  | { type: "progress_start"; total: number | null; desc: string }
  | { type: "progress_update"; n: number }
  | { type: "progress_end" }
  | { type: "postfix"; desc?: string; [key: string]: unknown }
  | { type: "phase"; phase: string; [key: string]: unknown }
  | { type: "step"; epoch: number; batch: number; total_batches: number; [key: string]: unknown }
  | { type: "validate"; epoch: number; frames?: { lrPath: string; srPath: string; gtPath: string; diffPath: string } | null; [key: string]: unknown }
  | { type: "done"; elapsed_seconds: number }
  | { type: "error"; code: string; message: string; context?: Record<string, unknown> };

// ── Inference result ────────────────────────────────────────────────────

export interface InferenceResult {
  success: boolean;
  error?: string;
  preview_input_path?: string;
  preview_output_path?: string;
  input_resolution?: { width: number; height: number };
  output_resolution?: { width: number; height: number };
  inference_time_ms?: number;
  metrics?: {
    psnr: number | null;
    ssim: number | null;
    lpips: number | null;
    ms_ssim: number | null;
  };
}

// ── Architecture name mapping ───────────────────────────────────────────

export const FRONTEND_ARCH_MAP: Record<string, string> = {
  "Real-ESRGAN": "rrdb_esrgan",
  "SwinIR": "swinir",
};

export const ENGINE_ARCH_MAP: Record<string, string> = {
  "rrdb_esrgan": "Real-ESRGAN",
  "swinir": "SwinIR",
};