import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type {
  IPCMessage,
  MetricsUpdateMessage,
  RunHistoryResponseMessage,
  ValidationCompleteMessage,
  CheckpointEntry,
} from "./ipc-types";
import { useTrainingStore } from "../store/trainingStore";
import type { RunHistory } from "../store/trainingStore";
import { useUiStore } from "../store/uiStore";
import { useCheckpointStore } from "../store/checkpointStore";
import { useInferenceStore } from "../store/inferenceStore";
import { SRProjManager } from "./SRProjManager";
import type { SRProjRun } from "./srproj";

// ── §19.12 — Frontend IPC logging ────────────────────────────────────────────

// Log IPC messages at DEBUG level to the browser console and, when available,
// to the Tauri app-data log file via a Rust command.
// High-frequency messages (metrics.update, heartbeat) are suppressed to avoid noise.
const HIGH_FREQ = new Set(["metrics.update", "heartbeat"]);

export function ipcLog(
  direction: "rx" | "tx" | string,
  payload: Record<string, unknown>
): void {
  if (HIGH_FREQ.has(String(payload.type ?? direction))) return;

  const ts = new Date().toISOString();
  const line = `[IPC ${ts}] ${direction.toUpperCase()} ${JSON.stringify(payload)}`;

  // Write to app-data log via Rust command (fire-and-forget)
  invoke<void>("append_frontend_log", { line }).catch(() => {/* Rust command optional */});
}

// ── Listener lifecycle ────────────────────────────────────────────────────

let unlisten: (() => void) | null = null;

export async function startIpcListener(): Promise<void> {
  if (unlisten) return;
  unlisten = await listen<IPCMessage>("sidecar-message", ({ payload }) => {
    dispatch(payload);
  });
}

export function stopIpcListener(): void {
  unlisten?.();
  unlisten = null;
}

// ── Sidecar process control ───────────────────────────────────────────────

export async function startSidecar(): Promise<void> {
  await invoke("start_sidecar");
  await startIpcListener();
}

export async function killSidecar(): Promise<void> {
  await invoke("kill_sidecar");
  stopIpcListener();
}

// ── Send helpers ──────────────────────────────────────────────────────────

export async function sendToSidecar(message: IPCMessage): Promise<void> {
  // §19.12 — Log outbound IPC messages
  ipcLog("tx", message as unknown as Record<string, unknown>);
  await invoke<void>("send_to_sidecar", { payload: JSON.stringify(message) });
}

// ── Dispatch ──────────────────────────────────────────────────────────────

function dispatch(msg: IPCMessage): void {
  // §19.12 — Log inbound IPC messages
  ipcLog("rx", msg as unknown as Record<string, unknown>);

  switch (msg.type) {
    case "heartbeat":
      useUiStore.getState().setLastHeartbeat(Date.now());
      break;

    case "hardware.info": {
      const primary = msg.devices.find((d) => d.type !== "cpu");
      useUiStore.getState().setDeviceName(primary?.name ?? null);
      useUiStore.getState().setDetectedDevices(msg.devices);
      break;
    }

    // §19.7 — GPU variant detection needed: relay as a Tauri event so the
    // OnboardingScreen can react and trigger the download flow (§19.8).
    case "gpu.detection_needed":
      // Re-emit as a distinct Tauri event so ScreenOnboarding can listen for it.
      // The onboarding screen sets up its own listener; we just log here.
      ipcLog("gpu.detection_needed", { vendor: msg.vendor, variant: msg.variant });
      break;

    case "metrics.update":
      dispatchMetrics(msg);
      break;

    case "training.complete":
      useTrainingStore.setState({ status: "done", finalEpoch: msg.final_epoch });
      break;

    case "training.resumed":
      useTrainingStore.setState({ status: "running" });
      break;

    // §17.12 — Reaction: project.run.started
    // (1) Update trainingStore so "Active" badge appears immediately
    // (2) Append a new SRProjRun to the .srproj manifest and persist to disk
    //     before acknowledging — sidecar_pid must survive unexpected app exit
    //     so §21.5 crash recovery can read it on reopen.
    case "project.run.started": {
      useTrainingStore.setState({
        status: "running",
        activeTrainingRunId: msg.run_id,
      });
      const now = new Date().toISOString();
      const newRun: SRProjRun = {
        run_id: msg.run_id,
        name: msg.run_id,
        status: "running",
        created_at: now,
        started_at: now,
        completed_at: null,
        architecture: {
          type: "Real-ESRGAN",
          upscale_factor: 4,
          custom_config: {},
        },
        training_config: {
          num_epochs: msg.total_epochs ?? 100,
          batch_size: 4,
          learning_rate: 1e-4,
          scheduler: "MultiStepLR",
          optimizer: "Adam",
          patch_size: 192,
          augmentations: {
            horizontal_flip: true,
            vertical_flip: false,
            rotation_90: false,
            mixup: false,
            color_jitter: false,
            random_degradation: false,
            gaussian_blur: false,
            noise_injection: false,
          },
        },
        paths: {
          training_data: "",
          validation_data: "",
          checkpoint_dir: "",
          log_dir: "",
        },
        metrics: {
          current_epoch: 0,
          epochs_completed: 0,
          best_loss: null,
          best_loss_epoch: null,
          best_psnr: null,
          best_psnr_epoch: null,
          last_loss: null,
          last_psnr: null,
          last_ssim: null,
        },
        checkpoints: {
          total_count: 0,
          last_saved_epoch: null,
          last_saved_path: null,
          best_checkpoint_path: null,
        },
        sidecar_pid: msg.sidecar_pid ?? null,
        sidecar_log_file: null,
      };
      SRProjManager.addRun(newRun);
      SRProjManager.save().catch(() => {
        // Non-fatal: best-effort persist; crash recovery degrades gracefully
      });
      break;
    }

    case "validation.complete":
      dispatchValidation(msg);
      break;

    case "run.history.response":
      dispatchRunHistory(msg);
      break;

    // §13.7 — Populate checkpoint list from live sidecar events
    // §17.13 — Reaction: checkpoint.saved
    // (1) Update run metadata (current_epoch, best_psnr) in .srproj
    // (2) Auto-save .srproj so Checkpoints tab and crash recovery see accurate state
    case "checkpoint.saved": {
      dispatchCheckpointSaved(msg.run_id, {
        epoch: msg.epoch,
        filename: basename(msg.path),
        path: msg.path,
        created_at: new Date().toISOString(),
        file_size_mb: msg.size_mb,
        metrics: {
          psnr: msg.psnr ?? undefined,
          ssim: msg.ssim ?? undefined,
        },
      });
      const existingRun = SRProjManager.getRun(msg.run_id);
      const prevBestPsnr = existingRun?.metrics.best_psnr ?? null;
      const newPsnr = msg.psnr ?? null;
      const updatedBestPsnr =
        newPsnr !== null && (prevBestPsnr === null || newPsnr > prevBestPsnr)
          ? newPsnr
          : prevBestPsnr;
      SRProjManager.updateRun(msg.run_id, {
        metrics: {
          ...(existingRun?.metrics ?? {
            current_epoch: 0,
            epochs_completed: 0,
            best_loss: null,
            best_loss_epoch: null,
            best_psnr: null,
            best_psnr_epoch: null,
            last_loss: null,
            last_psnr: null,
            last_ssim: null,
          }),
          current_epoch: msg.epoch,
          best_psnr: updatedBestPsnr,
        },
        checkpoints: {
          ...(existingRun?.checkpoints ?? {
            total_count: 0,
            last_saved_epoch: null,
            last_saved_path: null,
            best_checkpoint_path: null,
          }),
          total_count: (existingRun?.checkpoints.total_count ?? 0) + 1,
          last_saved_epoch: msg.epoch,
          last_saved_path: msg.path,
        },
      });
      SRProjManager.save().catch(() => {
        // Non-fatal: best-effort persist
      });
      break;
    }

    case "checkpoint.list.response":
      useCheckpointStore.getState().setCheckpointsForRun(msg.run_id, msg.checkpoints);
      break;

    // §13.5 — Remove deleted checkpoint from store
    case "checkpoint.delete.done":
      useCheckpointStore.getState().removeCheckpointByPath(msg.checkpoint_path);
      break;

    // §13.4 — Signal the screen to fire an export-done toast
    case "checkpoint.export.done":
      useCheckpointStore.getState().setLastExportDone({ path: msg.out_path, sizeMb: msg.size_mb });
      break;

    // §14.11 — Update tile progress bar during inference
    case "inference.progress":
      useInferenceStore.getState().setTileProgress(msg.tiles_done, msg.tiles_total);
      break;

    // §14.10 / §14.12 / §14.13 — Store inference result; screen handles success vs. error
    case "inference.result":
      useInferenceStore.getState().setResult(msg);
      break;

    // §20.9 — Route sidecar error messages to the global error dialog router
    case "error":
      useUiStore.getState().setLastIpcError(msg);
      break;

    default:
      break;
  }
}

function basename(filePath: string): string {
  return filePath.split(/[\\/]/).pop() ?? filePath;
}

function dispatchCheckpointSaved(runId: string, entry: CheckpointEntry): void {
  useCheckpointStore.getState().addCheckpointForRun(runId, entry);
  // §13.8 — Best PSNR tracking is computed reactively in ScreenCheckpoints from the store.
}

// §12.8 — Maps snake_case IPC fields to camelCase store fields.
// Maintains per-run history cache (§12.12) keyed by activeTrainingRunId.
function dispatchMetrics(msg: MetricsUpdateMessage): void {
  const prev = useTrainingStore.getState();
  const totalLoss =
    msg.d_loss != null ? msg.g_loss + msg.d_loss : msg.g_loss;

  const nextGLoss = [...prev.lossHistory, msg.g_loss].slice(-500);
  const nextDLoss = [...prev.dLossHistory, msg.d_loss].slice(-500);
  const nextTotal = [...prev.totalLossHistory, totalLoss].slice(-500);
  const nextPsnr =
    msg.psnr != null
      ? [...prev.psnrHistory, msg.psnr].slice(-500)
      : prev.psnrHistory;
  const nextSsim =
    msg.ssim != null
      ? [...prev.ssimHistory, msg.ssim].slice(-500)
      : prev.ssimHistory;

  const runId = prev.activeTrainingRunId;
  const nextRunHistories = runId
    ? {
        ...prev.runHistories,
        [runId]: {
          gLossHistory: nextGLoss,
          dLossHistory: nextDLoss,
          totalLossHistory: nextTotal,
          psnrHistory: nextPsnr,
          ssimHistory: nextSsim,
        } satisfies RunHistory,
      }
    : prev.runHistories;

  useTrainingStore.setState({
    iter: msg.iter,
    epoch: msg.epoch,
    gLoss: msg.g_loss,
    dLoss: msg.d_loss,
    psnr: msg.psnr,
    ssim: msg.ssim,
    gpuUtil: msg.gpu_util,
    vram: msg.vram_gb,
    temp: msg.temp_c,
    cpuUtil: msg.cpu_util,
    speed: msg.speed,
    lossHistory: nextGLoss,
    dLossHistory: nextDLoss,
    totalLossHistory: nextTotal,
    psnrHistory: nextPsnr,
    ssimHistory: nextSsim,
    runHistories: nextRunHistories,
  });
}

// §12.11 — Stores the first valid sample from validation.complete.
// GT+Diff remain null when gt_path is null (no-GT validation strategy).
function dispatchValidation(msg: ValidationCompleteMessage): void {
  const sample = msg.samples.find((s) => s !== null) ?? null;
  if (!sample) return;
  useTrainingStore.setState({
    validationFrames: {
      lrPath: sample.lr_path,
      srPath: sample.sr_path,
      gtPath: sample.gt_path,
      diffPath: sample.diff_path,
    },
  });
}

// §12.10 — Hydrates runHistories when run.history.response arrives after project reopen.
function dispatchRunHistory(msg: RunHistoryResponseMessage): void {
  const gLossHistory = msg.metrics.map((m) => m.g_loss);
  const dLossHistory = msg.metrics.map((m) => m.d_loss);
  const totalLossHistory = msg.metrics.map((m) =>
    m.d_loss != null ? m.g_loss + m.d_loss : m.g_loss
  );
  const psnrHistory = msg.metrics
    .filter((m) => m.psnr != null)
    .map((m) => m.psnr as number);
  const ssimHistory = msg.metrics
    .filter((m) => m.ssim != null)
    .map((m) => m.ssim as number);

  const prev = useTrainingStore.getState();
  const updatedHistories = {
    ...prev.runHistories,
    [msg.run_id]: {
      gLossHistory,
      dLossHistory,
      totalLossHistory,
      psnrHistory,
      ssimHistory,
    },
  };
  useTrainingStore.setState({ runHistories: updatedHistories });

  // §22.2 — mark this run's comparison history as received
  useUiStore.getState().markComparisonHistoryReceived(msg.run_id);

  // If this is the active run, also hydrate the flat history arrays
  if (prev.activeTrainingRunId === msg.run_id) {
    useTrainingStore.setState({
      lossHistory: gLossHistory,
      dLossHistory,
      totalLossHistory,
      psnrHistory,
      ssimHistory,
    });
  }
}
