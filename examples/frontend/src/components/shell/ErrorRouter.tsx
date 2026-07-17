// §20.9 — Error code → dialog router.
// Listens to IPC error messages stored in uiStore.lastIpcError and renders
// the appropriate pre-configured ErrorDialog.  Covers all canonical error codes
// listed in the task spec.

import { invoke } from "@tauri-apps/api/core";
import { useUiStore } from "../../store/uiStore";
import {
  ErrorDialog,
  type ErrorDialogAction,
} from "./ErrorDialog";

// ── Dialog config ─────────────────────────────────────────────────────────

interface DialogConfig {
  title: string;
  detail: string;
  suggestions?: string[];
  actions?: ErrorDialogAction[];
}

function retryLowerLrAction(dismiss: () => void): ErrorDialogAction {
  return {
    label: "Retry with Lower LR",
    variant: "primary",
    onClick: () => {
      dismiss();
      useUiStore.getState().setActiveTab("training");
    },
  };
}

function openFileManagerAction(path: string | undefined, dismiss: () => void): ErrorDialogAction {
  return {
    label: "Open File Manager",
    variant: "primary",
    onClick: async () => {
      if (path) {
        await invoke("open_in_file_manager", { path }).catch(() => {/* best-effort */});
      }
      dismiss();
    },
  };
}

function buildConfig(
  code: string,
  message: string,
  context: Record<string, unknown> | undefined,
  dismiss: () => void,
): DialogConfig {
  switch (code) {

    // ── §20.1 NaN loss ─────────────────────────────────────────────────────
    case "TRAINING_NAN_LOSS":
      return {
        title: "NaN Loss Detected",
        detail: message || "The training loss became NaN (not-a-number). An emergency checkpoint was saved before the loop exited.",
        suggestions: [
          "Reduce the learning rate by 10× (e.g. 1e-4 → 1e-5)",
          "Disable FP16 mixed precision — it can cause NaN with certain architectures",
          "Increase warmup epochs to stabilise early training",
          "Check your dataset for corrupt or zero-valued images",
        ],
        actions: [retryLowerLrAction(dismiss)],
      };

    // ── §20.10 Gradient explosion ──────────────────────────────────────────
    case "GRADIENT_EXPLOSION":
      return {
        title: "Gradient Explosion Detected",
        detail: message || "Infinite loss values (gradient explosion) were detected. An emergency checkpoint was saved before the loop exited.",
        suggestions: [
          "Reduce the learning rate — gradient explosions often stem from an LR that is too high",
          "Enable gradient clipping — the sidecar already clips at max_norm=1.0, but lower values may help",
          "Use a warmup schedule to ramp up LR gradually",
          "Reduce the batch size to stabilise gradient estimates",
        ],
        actions: [retryLowerLrAction(dismiss)],
      };

    // ── §20.2 Disk full ────────────────────────────────────────────────────
    case "DISK_FULL":
      return {
        title: "Disk Full",
        detail: message || "Less than 500 MB of free space remains. The checkpoint was not saved to avoid a partial write.",
        suggestions: [
          "Free up disk space by deleting old checkpoints from the Checkpoints tab",
          "Change the checkpoint directory to a drive with more space",
          "Reduce the checkpoint save frequency in Training Setup",
        ],
        actions: [openFileManagerAction(String(context?.checkpoint_dir ?? ""), dismiss)],
      };

    // ── §20.3 FFmpeg not found ─────────────────────────────────────────────
    case "FFMPEG_NOT_FOUND":
      return {
        title: "FFmpeg Not Found",
        detail: message || "FFmpeg is required for video extraction but could not be found on PATH.",
        suggestions: [
          "Linux: sudo apt install ffmpeg  (Ubuntu/Debian)  or  sudo dnf install ffmpeg  (Fedora)",
          "macOS: brew install ffmpeg",
          "Windows: download from ffmpeg.org and add the bin/ folder to your PATH environment variable",
          "After installing, restart SR Tuner so the new PATH is picked up",
        ],
      };

    // ── §20.4 ONNX export failed ───────────────────────────────────────────
    case "ONNX_EXPORT_FAILED": {
      const traceback = String(context?.traceback ?? message ?? "");
      // Extract unsupported op lines from the traceback heuristically.
      const unsupportedOps = traceback
        .split("\n")
        .filter((l) => /unsupported|not implemented|aten::/i.test(l))
        .slice(0, 6)
        .map((l) => l.trim())
        .filter(Boolean);
      return {
        title: "ONNX Export Failed",
        detail:
          unsupportedOps.length > 0
            ? `The model contains ops that cannot be exported to ONNX:\n${unsupportedOps.join("\n")}`
            : (message || "The model could not be exported to ONNX format."),
        suggestions: [
          "SwinIR and HAT use operations not yet supported by the ONNX exporter — use .pth export instead",
          "Disable FP16 before exporting — some half-precision ops lack ONNX support",
          "Try a smaller opset version (opset_version=11) if the default fails",
        ],
      };
    }

    // ── §20.5 Dataset loading failed ──────────────────────────────────────
    case "DATASET_LOADING_FAILED":
      return {
        title: "Dataset Loading Failed",
        detail: message || "The training dataset could not be loaded.",
        suggestions: [
          `Check that the path "${String(context?.path ?? "—")}" exists and is readable`,
          "Ensure you have read permission on the directory (chmod +r on Linux/macOS)",
          "Verify the folder contains valid image files (.png / .jpg)",
          "Re-run dataset validation from the Training Setup screen",
        ],
      };

    // ── §20.5 / companion code ────────────────────────────────────────────
    case "DATASET_INVALID_FORMAT":
      return {
        title: "Invalid Dataset Format",
        detail: message || "The dataset directory does not meet the required format.",
        suggestions: [
          "HR and LR folders must contain the same number of files",
          "Images must be valid PNG or JPEG files — corrupted files will fail",
          "Use the Dataset Setup screen to re-extract or re-validate the dataset",
        ],
      };

    // ── §20.6 Pretrained weight incompatible ──────────────────────────────
    case "PRETRAINED_WEIGHT_INCOMPATIBLE":
      return {
        title: "Pretrained Weights Incompatible",
        detail: message || "The selected pretrained checkpoint cannot be loaded into the current architecture — state dict keys do not match.",
        suggestions: [
          "Ensure the .pth file was trained with the same architecture (Real-ESRGAN, SwinIR, HAT, EDSR)",
          "Check that the upscale factor matches (e.g. a 4× checkpoint cannot initialise a 2× model)",
          "Try loading without pretrained weights — clear the pretrained path in Model Config",
        ],
      };

    // ── §20.7 Checkpoint in use ────────────────────────────────────────────
    case "CHECKPOINT_IN_USE":
      return {
        title: "Checkpoint In Use",
        detail: message || "This checkpoint is currently selected as the resume target and cannot be deleted while training is queued.",
        suggestions: [
          "Clear the resume target in Training Setup before deleting this checkpoint",
          "Or stop the active training run first, then delete the checkpoint",
        ],
      };

    // ── §20.8 Inference GT size mismatch ──────────────────────────────────
    case "INFERENCE_GT_SIZE_MISMATCH":
      return {
        title: "Ground Truth Size Mismatch",
        detail: message || "The ground truth image dimensions do not match the super-resolved output dimensions.",
        suggestions: [
          "The GT image must be the same size as the SR output (input × scale factor)",
          "If you upscaled by 4×, a 256×256 input needs a 1024×1024 GT image",
          "Remove the GT image or re-export it at the correct resolution",
        ],
      };

    // ── Checkpoint not found / load failed ────────────────────────────────
    case "CHECKPOINT_NOT_FOUND":
      return {
        title: "Checkpoint Not Found",
        detail: message || `The checkpoint file could not be found: "${String(context?.path ?? "—")}"`,
        suggestions: [
          "The file may have been moved or deleted — check the Checkpoints tab",
          "Select a different checkpoint in Training Setup before resuming",
        ],
      };

    case "CHECKPOINT_LOAD_FAILED":
      return {
        title: "Checkpoint Load Failed",
        detail: message || "The checkpoint file exists but could not be loaded.",
        suggestions: [
          "The file may be corrupt — try a different checkpoint",
          "Ensure the checkpoint was saved with a compatible PyTorch version",
        ],
      };

    // ── OOM ───────────────────────────────────────────────────────────────
    case "OOM":
    case "CUDA_OUT_OF_MEMORY":
      return {
        title: "Out of GPU Memory",
        detail: message || "The GPU ran out of VRAM during training or inference.",
        suggestions: [
          "Enable FP16 mixed precision — halves VRAM usage",
          "Reduce batch size (try halving it)",
          "Reduce patch size (e.g. 128 instead of 192)",
          "Use a smaller architecture (EDSR has the lowest VRAM footprint)",
        ],
        actions: [retryLowerLrAction(dismiss)],
      };

    // ── CUDA / ROCm device errors ──────────────────────────────────────────
    case "CUDA_ERROR":
      return {
        title: "CUDA Error",
        detail: message || "A CUDA runtime error occurred.",
        suggestions: [
          "Update your NVIDIA GPU drivers to the latest version",
          "Ensure the CUDA version shipped with PyTorch matches your driver",
          "Restart the application and try again",
        ],
      };

    case "CUDA_DEVICE_NOT_FOUND":
      return {
        title: "CUDA Device Not Found",
        detail: message || "No CUDA-capable GPU was found.",
        suggestions: [
          "Verify your NVIDIA drivers are installed",
          "Select 'CPU' as the device in Training Setup to proceed without a GPU",
        ],
      };

    // ── Config / infra errors ─────────────────────────────────────────────
    case "BASICSR_CONFIG_INVALID":
      return {
        title: "Invalid BasicSR Configuration",
        detail: message || "The generated BasicSR YAML is not valid.",
        suggestions: [
          "Return to Model Config and reset hyperparameters to defaults",
          "Check that all required fields (scale, architecture) are set",
        ],
      };

    case "SIDECAR_TIMEOUT":
      return {
        title: "Sidecar Timeout",
        detail: message || "The sidecar process did not respond within the expected time.",
        suggestions: [
          "Restart the application — the sidecar may have crashed silently",
          "Check the log files in the app data directory for details",
        ],
      };

    case "IPC_CONNECTION_LOST":
      return {
        title: "IPC Connection Lost",
        detail: message || "Communication with the training process was interrupted.",
        suggestions: [
          "The sidecar process may have exited unexpectedly",
          "Reopen the project to attempt crash recovery",
        ],
      };

    // ── Unknown / fallback ────────────────────────────────────────────────
    default:
      return {
        title: `Error: ${code || "UNKNOWN_ERROR"}`,
        detail: message || "An unexpected error occurred in the training backend.",
        suggestions: [
          "Check the log files in the app data directory for more details",
          "Restart the application and try again",
        ],
      };
  }
}

// ── Component ─────────────────────────────────────────────────────────────

export function ErrorRouter() {
  const lastIpcError = useUiStore((s) => s.lastIpcError);
  const dismiss = () => useUiStore.getState().setLastIpcError(null);

  if (!lastIpcError) return null;

  const config = buildConfig(
    lastIpcError.code,
    lastIpcError.message,
    lastIpcError.context,
    dismiss,
  );

  return (
    <ErrorDialog
      open={true}
      title={config.title}
      detail={config.detail}
      suggestions={config.suggestions}
      actions={config.actions}
      onClose={dismiss}
    />
  );
}
