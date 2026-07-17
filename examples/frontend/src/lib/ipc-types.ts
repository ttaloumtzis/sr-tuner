// IPC message types for NDJSON communication between frontend and Python sidecar.
// Direction: → = frontend to sidecar, ← = sidecar to frontend.
// All messages are single JSON objects terminated with \n (NDJSON).

// ── Lifecycle (←) ────────────────────────────────────────────────────────

export interface SidecarReadyMessage {
  type: "sidecar.ready";
  version: string;
  pid: number;
}

export interface HeartbeatMessage {
  type: "heartbeat";
  timestamp: string; // ISO-8601
  status: "idle" | "training";
}

// Task 7.6 — "rocm" is used for AMD GPUs detected via torch.version.hip
export interface DeviceInfo {
  id: string;
  name: string;
  vram_gb: number | null; // null for CPU and MPS
  type: "cuda" | "rocm" | "mps" | "cpu";
}

export interface HardwareInfoMessage {
  type: "hardware.info";
  devices: DeviceInfo[];
}

// ── Training (→) ─────────────────────────────────────────────────────────

export interface TrainingAugmentations {
  horizontal_flip: boolean;
  vertical_flip: boolean;
  rotation_90: boolean;
  mixup: boolean;
  color_jitter: boolean;
  random_degradation: boolean;
  gaussian_blur: boolean;
  noise_injection: boolean;
}

export interface TrainingResumeFrom {
  checkpoint_path: string;
  resume_epoch: number;
  resume_optimizer_state: boolean;
  resume_lr_scheduler_state: boolean;
}

export interface TrainingStartMessage {
  type: "training.start";
  run_id: string;
  config: {
    architecture: string;
    upscale_factor: number;
    pretrained_path: string | null;
  };
  training_config: {
    num_epochs: number;
    batch_size: number;
    learning_rate: number;
    scheduler: string;
    optimizer: string;
    patch_size: number;
    fp16: boolean;
    torch_compile: boolean;
    tensorboard: boolean;
    augmentations: TrainingAugmentations;
  };
  paths: {
    training_data: string;
    validation_data: string;
    checkpoint_dir: string;
    log_dir: string;
  };
  resume_from: TrainingResumeFrom | null;
}

export interface TrainingPauseMessage {
  type: "training.pause";
  run_id: string;
}

export interface TrainingResumeMessage {
  type: "training.resume";
  run_id: string;
}

export interface TrainingStopMessage {
  type: "training.stop";
  run_id: string;
  save_checkpoint: boolean;
}

// ── Training (←) ─────────────────────────────────────────────────────────

export interface ProjectRunStartedMessage {
  type: "project.run.started";
  run_id: string;
  total_epochs: number;
  sidecar_pid: number;
}

// Task 7.15 — highest-frequency message (every 50 iters during training)
export interface MetricsUpdateMessage {
  type: "metrics.update";
  iter: number;
  epoch: number;
  g_loss: number;
  d_loss: number | null; // null for single-network architectures
  psnr: number;
  ssim: number;
  gpu_util: number;
  vram_gb: number;
  temp_c: number;
  cpu_util: number;
  speed: number; // iterations/sec
}

export interface TrainingCompleteMessage {
  type: "training.complete";
  run_id: string;
  final_epoch: number;
  total_epochs: number;
  final_metrics: { loss: number; psnr: number; ssim: number };
  stopped_at: string; // ISO-8601
  checkpoint_final: string;
  status: "completed";
}

export interface TrainingResumedMessage {
  type: "training.resumed";
  run_id: string;
  resumed_from_epoch: number;
}

// Task 7.8 / 7.12
export interface ValidationSample {
  lr_path: string;
  sr_path: string;
  gt_path: string | null; // null when validation strategy is "none"
  diff_path: string | null;
}

export interface ValidationCompleteMessage {
  type: "validation.complete";
  run_id: string;
  epoch: number;
  samples: (ValidationSample | null)[]; // padded to 4 entries
  avg_psnr: number;
  avg_ssim: number;
}

// ── Checkpoints (←) ──────────────────────────────────────────────────────

export interface CheckpointSavedMessage {
  type: "checkpoint.saved";
  run_id: string;
  epoch: number;
  psnr: number | null;
  ssim: number | null;
  size_mb: number;
  path: string;
}

export interface CheckpointEntry {
  epoch: number;
  filename: string;
  path: string;
  created_at: string; // ISO-8601
  file_size_mb: number;
  metrics: { loss?: number; psnr?: number; ssim?: number };
}

export interface CheckpointListResponseMessage {
  type: "checkpoint.list.response";
  run_id: string;
  checkpoints: CheckpointEntry[];
  status: "success" | "error";
}

// Task 7.7
export interface CheckpointExportDoneMessage {
  type: "checkpoint.export.done";
  out_path: string;
  size_mb: number;
}

// Task 7.13
export interface CheckpointDeleteDoneMessage {
  type: "checkpoint.delete.done";
  checkpoint_path: string;
}

// ── Checkpoints (→) ──────────────────────────────────────────────────────

export interface CheckpointListRequestMessage {
  type: "checkpoint.list.request";
  run_id: string;
  checkpoint_dir: string;
}

// Task 7.7
export interface CheckpointExportMessage {
  type: "checkpoint.export";
  checkpoint_path: string;
  format: "pth" | "onnx";
  out_path: string;
}

// Task 7.13
export interface CheckpointDeleteMessage {
  type: "checkpoint.delete";
  checkpoint_path: string;
}

// ── Dataset (→) ──────────────────────────────────────────────────────────

// Task 7.10 / Gap P — sync_mode used in on_the_fly mode (frame-aligned vs time-aligned mux)
export interface DatasetCreateMessage {
  type: "dataset.create";
  mode: "video_extract" | "image_folder" | "on_the_fly";
  source_paths: string[];
  output_hr_path: string;
  output_lr_path: string;
  scale_factor: number;
  kernel: string;
  naming_pattern: string;
  ffmpeg_fps: number | null;
  sync_mode: "frame" | "time" | null; // on_the_fly only; null for other modes
}

export interface DatasetValidateRequestMessage {
  type: "dataset.validate.request";
  training_path: string;
  validation_path: string | null;
}

export interface DatasetSplitRequestMessage {
  type: "dataset.split.request";
  training_path: string;
  split_ratio: number;
}

// ── Dataset (←) ──────────────────────────────────────────────────────────

export interface DatasetProgressMessage {
  type: "dataset.progress";
  frames_done: number;
  frames_total: number;
  fps: number;
  eta_sec: number;
}

export interface DatasetValidateResultMessage {
  type: "dataset.validate.result";
  valid: boolean;
  errors: string[];
}

// Task 7.16
export interface DatasetSplitResultMessage {
  type: "dataset.split.result";
  validation_path: string;
  training_count: number;
  validation_count: number;
}

// ── Inference (→) ────────────────────────────────────────────────────────

// Task 7.11
export interface InferenceRunMessage {
  type: "inference.run";
  input_path: string;
  checkpoint_path: string;
  architecture: string;
  scale_factor: number;
  tile_size: number; // 0 = no tiling
  fp16: boolean;
  output_path: string;
  output_format: "png" | "jpeg" | "webp" | "tiff";
  gt_path: string | null; // null → metrics fields in result will be null
}

// ── Inference (←) ────────────────────────────────────────────────────────

export interface InferenceProgressMessage {
  type: "inference.progress";
  tiles_done: number;
  tiles_total: number;
}

// Task 7.14 — ms_ssim MUST be present; sidecar §18.6 computes it
export interface InferenceResultMessage {
  type: "inference.result";
  success: boolean;
  input_path: string;
  output_path: string;
  preview_input_path: string;
  preview_output_path: string;
  metrics: {
    psnr: number | null;
    ssim: number | null;
    lpips: number | null;
    ms_ssim: number | null;
  } | null; // null when gt_path was null in the request
  inference_time_ms: number;
  input_resolution: { width: number; height: number };
  output_resolution: { width: number; height: number };
  error?: string; // present only when success is false
}

// ── History ───────────────────────────────────────────────────────────────

// Task 7.9a — element type of run.history.response.metrics[] and metrics.update payload
export interface MetricsEvent {
  iter: number;
  epoch: number;
  g_loss: number;
  d_loss: number | null; // null for single-network architectures
  psnr: number | null; // null until validation interval fires
  ssim: number | null;
  gpu_util: number | null; // 0–100 %; null if pynvml unavailable
  vram_gb: number | null;
  temp_c: number | null;
  cpu_util: number | null;
  speed: number | null; // iterations/sec
  timestamp: string; // ISO-8601
}

// Task 7.9
export interface RunHistoryRequestMessage {
  type: "run.history.request";
  run_id: string;
  log_dir: string;
}

export interface RunHistoryResponseMessage {
  type: "run.history.response";
  run_id: string;
  metrics: MetricsEvent[];
}

// ── GPU variant download (§19.7 ←, §19.8 ←) ─────────────────────────────

// §19.7 — Emitted by CPU-only minimal sidecar on first launch after detecting GPU vendor.
// Frontend uses this to decide whether to download a full GPU-enabled sidecar variant.
export interface GpuDetectionNeededMessage {
  type: "gpu.detection_needed";
  vendor: "nvidia" | "amd" | "cpu";
  variant: "cuda" | "rocm" | "cpu";
}

// §19.8 — Progress events emitted during variant download (fired by frontend Tauri download)
export interface GpuVariantDownloadProgressMessage {
  type: "gpu.variant_download_progress";
  bytes_done: number;
  bytes_total: number | null;
  variant: "cuda" | "rocm" | "cpu";
}

// ── Error (←) ────────────────────────────────────────────────────────────

export interface ErrorMessage {
  type: "error";
  code: string;
  message: string;
  context?: Record<string, unknown>;
}

// ── IPCMessage union (Task 7.5) ───────────────────────────────────────────

export type IPCMessage =
  // Lifecycle (←)
  | SidecarReadyMessage
  | HeartbeatMessage
  | HardwareInfoMessage
  // GPU variant download (←)
  | GpuDetectionNeededMessage
  | GpuVariantDownloadProgressMessage
  // Training (←)
  | ProjectRunStartedMessage
  | MetricsUpdateMessage
  | TrainingCompleteMessage
  | TrainingResumedMessage
  | ValidationCompleteMessage
  | CheckpointSavedMessage
  // Training (→)
  | TrainingStartMessage
  | TrainingPauseMessage
  | TrainingResumeMessage
  | TrainingStopMessage
  // Checkpoints (←)
  | CheckpointListResponseMessage
  | CheckpointExportDoneMessage
  | CheckpointDeleteDoneMessage
  // Checkpoints (→)
  | CheckpointListRequestMessage
  | CheckpointExportMessage
  | CheckpointDeleteMessage
  // Dataset (←)
  | DatasetProgressMessage
  | DatasetValidateResultMessage
  | DatasetSplitResultMessage
  // Dataset (→)
  | DatasetCreateMessage
  | DatasetValidateRequestMessage
  | DatasetSplitRequestMessage
  // Inference (←)
  | InferenceProgressMessage
  | InferenceResultMessage
  // Inference (→)
  | InferenceRunMessage
  // History (←)
  | RunHistoryResponseMessage
  // History (→)
  | RunHistoryRequestMessage
  // Error (←)
  | ErrorMessage;
