import { useEffect, useRef } from "react";
import { useTrainingStore, type HardwareData, type TrainingStage } from "../store/trainingStore";
import { useUiStore } from "../store/uiStore";
import { useRunsStore } from "../store/runsStore";
import { getBaseUrl } from "../lib/api";

const SPEED_WINDOW_MS = 5000;
const SPEED_EMA_ALPHA = 0.2;

// Backend `phase` events → store stage, shown in the metrics HeaderBand so the
// user always knows what the GPU subprocess is doing between step reports.
const STAGE_BY_PHASE: Record<string, TrainingStage> = {
  preparing: "preparing",
  warmup: "warmup",
  training: "training",
  validating: "validating",
  saving: "saving",
};

interface SpeedEntry {
  linear: number;
  time: number;
}

function linearPos(epoch: number, batch: number): number {
  return epoch * 1_000_000 + batch;
}

function flushEpoch(sum: number, count: number) {
  if (count > 0) {
    useTrainingStore.getState().pushEpochLoss(sum / count);
  }
}

export function useTrainingSSE() {
  const activeTrainingRunId = useTrainingStore((s) => s.activeTrainingRunId);
  const esRef = useRef<EventSource | null>(null);
  const speedWindowRef = useRef<SpeedEntry[]>([]);
  const smoothedSpeedRef = useRef(0);
  const epochAccRef = useRef<{ currentEpoch: number; sum: number; count: number }>({
    currentEpoch: -1, sum: 0, count: 0,
  });

  useEffect(() => {
    if (!activeTrainingRunId) return;

    // Fetch launch config from backend (survives page refresh)
    (async () => {
      try {
        const { getJobStatus } = await import("../lib/api");
        const job = await getJobStatus(activeTrainingRunId);
        if (job.config) {
          useTrainingStore.getState().setLaunchConfig({
            totalEpochs: (job.config.max_epochs as number) ?? 100,
            batchSize: (job.config.batch_size as number) ?? 16,
            learningRate: (job.config.learning_rate as number) ?? 2e-4,
            fp16: (job.config.dtype as string) === "bf16",
            patchSize: (job.config.patch_size as number) ?? 128,
            validationEnabled: ((job.config.validation as Record<string, unknown>)?.enabled as boolean) ?? true,
          });
        }
        // Reconnect mid-run: show *some* stage until the next phase event.
        if (job.status === "running" && useTrainingStore.getState().stage === null) {
          useTrainingStore.getState().setStage("preparing");
        }
      } catch {
        // local snapshot from handleLaunch is the fallback
      }
    })();

    const baseUrl = getBaseUrl();
    const es = new EventSource(`${baseUrl}/api/events?job_id=${activeTrainingRunId}`);
    esRef.current = es;

    speedWindowRef.current = [];
    smoothedSpeedRef.current = 0;
    epochAccRef.current = { currentEpoch: -1, sum: 0, count: 0 };

    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as Record<string, unknown>;
        const type = event.type as string;

        switch (type) {
          case "phase": {
            const phase = event.phase as string;
            const stage = STAGE_BY_PHASE[phase];
            if (stage) {
              useTrainingStore.getState().setStage(stage);
              if (phase !== "preparing") {
                useTrainingStore.getState().setPreparingProgress(null);
              }
            }
            if (phase === "training") {
              useTrainingStore.getState().setStatus("running");
              useTrainingStore.getState().setValidationRunning(false);
              useTrainingStore.getState().setValidationProgress(null);
              speedWindowRef.current = [];
              smoothedSpeedRef.current = 0;
            } else if (phase === "validating") {
              useTrainingStore.getState().setValidationRunning(true);
              useTrainingStore.getState().setValidationProgress(null);
            } else if (phase === "saving") {
              useTrainingStore.getState().setValidationRunning(false);
              useTrainingStore.getState().setValidationProgress(null);
            } else if (phase === "complete" || phase === "cancelled") {
              useTrainingStore.getState().setStatus("done");
              useTrainingStore.getState().setValidationRunning(false);
              useTrainingStore.getState().setValidationProgress(null);
              useTrainingStore.getState().setStage(null);
              useTrainingStore.getState().setPreparingProgress(null);
              useRunsStore.getState().bumpRefresh();
            }
            break;
          }
          case "progress_start": {
            // Pre-training dataset-integrity scan (see workers.py). The training
            // epoch loop also emits progress_start/update/end for its per-epoch
            // progress bar — ignore those so they can't hijack the "training"
            // stage and freeze the UI on "Preparing · scanning X/Y".
            const st = useTrainingStore.getState();
            if (st.stage !== "preparing" && st.stage !== null) return;
            const total = (event.total as number) ?? 0;
            useTrainingStore.getState().setPreparingProgress({ done: 0, total });
            break;
          }
          case "progress_update": {
            const st = useTrainingStore.getState();
            if (st.preparingProgress) {
              const n = (event.n as number) ?? 0;
              st.setPreparingProgress({
                done: st.preparingProgress.done + n,
                total: st.preparingProgress.total,
              });
            }
            break;
          }
          case "progress_end": {
            useTrainingStore.getState().setPreparingProgress(null);
            break;
          }
          case "checkpoint_saved": {
            useRunsStore.getState().bumpRefresh();
            break;
          }
          case "step": {
            const epoch = (event.epoch as number) ?? 0;
            const batch = (event.batch as number) ?? 0;
            const totalBatch = (event.total_batches as number) ?? 0;
            const totalLoss = (event.total as number) ?? 0;
            const now = performance.now();

            useTrainingStore.getState().setValidationRunning(false);
            useTrainingStore.getState().setStage("training");
            useTrainingStore.getState().setPreparingProgress(null);

            // Sliding-window speed
            const curLinear = linearPos(epoch, batch);
            speedWindowRef.current = speedWindowRef.current.filter(
              (sw) => now - sw.time < SPEED_WINDOW_MS,
            );
            speedWindowRef.current.push({ linear: curLinear, time: now });

            let speed = 0;
            if (speedWindowRef.current.length >= 2) {
              const first = speedWindowRef.current[0];
              const last = speedWindowRef.current[speedWindowRef.current.length - 1];
              const dt = (last.time - first.time) / 1000;
              const dPos = last.linear - first.linear;
              if (dt > 0 && dPos > 0) {
                const instant = dPos / dt;
                if (smoothedSpeedRef.current === 0) {
                  smoothedSpeedRef.current = instant;
                } else {
                  smoothedSpeedRef.current =
                    (1 - SPEED_EMA_ALPHA) * smoothedSpeedRef.current +
                    SPEED_EMA_ALPHA * instant;
                }
                speed = smoothedSpeedRef.current;
              }
            }

            useTrainingStore.getState().updateFromStep(epoch, batch, totalBatch, speed);

            // Epoch-accumulated loss
            const acc = epochAccRef.current;
            if (epoch !== acc.currentEpoch && acc.currentEpoch >= 0) {
              flushEpoch(acc.sum, acc.count);
              acc.sum = 0;
              acc.count = 0;
            }
            acc.currentEpoch = epoch;
            acc.sum += totalLoss;
            acc.count += 1;

            // Live loss (running average within epoch)
            useTrainingStore.getState().setLiveLoss(acc.sum / acc.count);
            break;
          }
          case "validate": {
            const vepoch = (event.epoch as number) ?? 0;
            const psnr = (event.psnr as number) ?? 0;
            const ssim = (event.ssim as number) ?? 0;
            const fullPsnr = (event.full_psnr as number) ?? undefined;
            const fullSsim = (event.full_ssim as number) ?? undefined;
            const valLoss = (event.val_loss as number) ?? undefined;
            useTrainingStore.getState().updateFromValidate(vepoch, psnr, ssim, fullPsnr, fullSsim, valLoss);
            const frames = event.frames as Record<string, string> | null | undefined;
            if (frames && frames.lrPath && frames.srPath) {
              useTrainingStore.getState().pushValidationFrames(vepoch, {
                lrPath: frames.lrPath,
                srPath: frames.srPath,
                gtPath: frames.gtPath ?? null,
                diffPath: frames.diffPath ?? null,
              }, psnr, ssim, fullPsnr, fullSsim);
            }
            useTrainingStore.getState().setValidationProgress(null);
            break;
          }
          case "validate_progress": {
            const done = (event.done as number) ?? 0;
            const total = (event.total as number) ?? 0;
            useTrainingStore.getState().setValidationRunning(true);
            useTrainingStore.getState().setValidationProgress({ done, total });
            break;
          }
          case "hardware": {
            const hw: HardwareData = {
              cpu_percent: (event.cpu_percent as number) ?? null,
              ram_used_gb: (event.ram_used_gb as number) ?? null,
              ram_total_gb: (event.ram_total_gb as number) ?? null,
              gpu_util_percent: (event.gpu_util_percent as number) ?? null,
              vram_used_gb: (event.vram_used_gb as number) ?? null,
              vram_total_gb: (event.vram_total_gb as number) ?? null,
              temp_c: (event.temp_c as number) ?? null,
            };
            useTrainingStore.getState().updateFromHardware(hw);
            break;
          }
          case "done": {
            const acc = epochAccRef.current;
            flushEpoch(acc.sum, acc.count);
            useTrainingStore.getState().setStatus("done");
            useTrainingStore.getState().setValidationRunning(false);
            useTrainingStore.getState().setValidationProgress(null);
            useTrainingStore.getState().setStage(null);
            useTrainingStore.getState().setPreparingProgress(null);
            useRunsStore.getState().bumpRefresh();
            break;
          }
          case "error": {
            const code = (event.code as string) ?? "UNKNOWN_ERROR";
            const msg = (event.message as string) ?? "An unknown training error occurred";
            useTrainingStore.getState().setError(code, msg);
            useTrainingStore.getState().setStage(null);
            useTrainingStore.getState().setPreparingProgress(null);
            useRunsStore.getState().bumpRefresh();
            useUiStore.getState().setLastApiError({
              type: "error",
              code,
              message: msg,
              context: event.context as Record<string, unknown> | undefined,
            });
            break;
          }
        }
      } catch {
        // ignore malformed events
      }
    };

    es.onerror = () => {
      const state = useTrainingStore.getState();
      if (state.status === "done" || state.status === "failed") return;
      useTrainingStore.getState().setStatus("disconnected");
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [activeTrainingRunId]);

  return esRef;
}
