export type ValidationStrategy = "auto_split" | "separate_folder" | "none";
export type DatasetType = "image_folder" | "video_extract" | "on_the_fly";
export type Architecture = "rrdb_esrgan" | "swinir";
export type RunStatus = "configured" | "running" | "paused" | "completed" | "failed";
export type TabId = "dataset" | "model" | "training" | "metrics" | "checkpoints" | "inference";

export interface AugmentationConfig {
  horizontal_flip: boolean;
  vertical_flip: boolean;
  rotation_90: boolean;
  mixup: boolean;
  color_jitter: boolean;
  random_degradation: boolean;
  gaussian_blur: boolean;
  noise_injection: boolean;
}

export interface SRProjRun {
  run_id: string;
  name: string;
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  architecture: {
    type: Architecture;
    upscale_factor: number;
    custom_config: Record<string, unknown>;
  };
  training_config: {
    num_epochs: number;
    batch_size: number;
    learning_rate: number;
    scheduler: string;
    optimizer: string;
    patch_size: number;
    augmentations: AugmentationConfig;
  };
  paths: {
    training_data: string;
    validation_data: string;
    checkpoint_dir: string;
    log_dir: string;
  };
  metrics: {
    current_epoch: number;
    epochs_completed: number;
    best_loss: number | null;
    best_loss_epoch: number | null;
    best_psnr: number | null;
    best_psnr_epoch: number | null;
    last_loss: number | null;
    last_psnr: number | null;
    last_ssim: number | null;
  };
  checkpoints: {
    total_count: number;
    last_saved_epoch: number | null;
    last_saved_path: string | null;
    best_checkpoint_path: string | null;
  };
}

export interface SRProjFile {
  version: "1.0.0";
  name: string;
  created_at: string;
  last_modified_at: string;
  default_dataset: {
    training_path: string;
    validation_path: string;
    validation_strategy: ValidationStrategy;
    validation_split_ratio: number | null;
    dataset_type: DatasetType;
  };
  default_model: {
    architecture: Architecture;
    upscale_factor: number;
  };
  models: {
    id: string;
    name: string;
    architecture: Architecture;
    config?: Record<string, unknown>;
    hyperparameters: {
      scale: number;
      batch_size: number;
      patch_size: number;
      learning_rate: number;
      optimizer: string;
      lr_scheduler: string;
      total_iter: number;
      augmentations: AugmentationConfig;
    };
    created_at: string;
  }[];
  runs: SRProjRun[];
  ui_state: {
    last_active_run_id: string | null;
    last_active_tab: TabId | null;
    expanded_panels: Record<string, boolean>;
  };
  metadata: {
    app_version: string;
    notes: string | null;
    tags: string[];
  };
}

export const SRPROJ_SCHEMA_VERSION = "1.0.0" as const;
