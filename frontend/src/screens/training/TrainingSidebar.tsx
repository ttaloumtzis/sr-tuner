import { useState, useCallback } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { InlineAlert } from "../../components/ui/InlineAlert";
import { InfoRow } from "../../components/ui/InfoRow";
import { StackedBar, type StackedBarSegment } from "../../components/ui/StackedBar";
import { IconCheck, IconRocket } from "../../components/ui/icons";
import { fmtGb } from "../../lib/format";
import { estimateVramBreakdown, type VramBreakdown, type VramEstimateOptions } from "../../lib/vramEstimate";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useTrainingStore } from "../../store/trainingStore";
import { useUiStore } from "../../store/uiStore";

const VRAM_SEGMENT_COLORS: Record<string, string> = {
  "Model weights": "var(--blue)",
  "Gradients": "var(--purple)",
  "Adam optimizer": "var(--cyan)",
  "Activations": "var(--amber)",
  "Input batch": "var(--pink)",
  "Upsampler": "var(--green)",
  "Allocator overhead": "var(--orange)",
  "CUDA context": "var(--dim)",
};

interface ReadinessItemProps {
  done: boolean;
  label: string;
  optional?: boolean;
}
function ReadinessItem({ done, label, optional }: ReadinessItemProps) {
  return (
    <div className="ts-row-6">
      <span className={`ts-check${done ? " ts-check-done" : ""}`}>
        {done && <IconCheck size={8} color="var(--green)" strokeWidth={3} />}
      </span>
      <span className={`ts-ri-label${done ? " ts-ri-label-done" : ""}`}>
        {label}
        {optional && <span className="ts-optional"> (optional)</span>}
      </span>
    </div>
  );
}

interface TrainingSidebarProps {
  gpuTotalVramGb: number | null;
  customConfigPath: string;
}

export function TrainingSidebar({ gpuTotalVramGb, customConfigPath }: TrainingSidebarProps) {
  const s = useRunConfigStore();
  const [launchError, setLaunchError] = useState<string | null>(null);

  const itersPerEpoch = s.selectedDatasetPairs && s.batchSize > 0
    ? Math.ceil(s.selectedDatasetPairs / s.batchSize)
    : 0;
  const totalIters = itersPerEpoch * s.schedule.totalEpochs;

  const instCfg = s.instanceConfig as Record<string, unknown> | undefined;
  const vramOptions: VramEstimateOptions = {
    arch: s.instanceArchitecture ?? "",
    batchSize: s.batchSize,
    patchSize: s.patchSize,
    fp16: s.fp16,
    scale: s.instanceScale ?? 4,
    config: instCfg,
    gradientCheckpointing: s.gradientCheckpointing === "auto"
      ? s.instanceArchitecture === "swinir"
      : s.gradientCheckpointing === "true",
  };
  const vramBreakdown: VramBreakdown = s.instanceArchitecture
    ? estimateVramBreakdown(vramOptions)
    : { totalGb: 0, weightsGb: 0, gradsGb: 0, adamGb: 0, activationsGb: 0, inputGb: 0, upsamplerGb: 0, overheadGb: 0 };
  const vramEst = vramBreakdown.totalGb;

  const vm = s.vramMeasure;
  const isMeasuredStale =
    vm.status === "done" &&
    vm.measuredFor !==
      JSON.stringify([
        s.instanceArchitecture,
        s.instanceConfig,
        s.instanceScale,
        s.batchSize,
        s.patchSize,
        s.fp16,
        s.gradientCheckpointing,
      ]);
  const hasFreshMeasure = vm.status === "done" && !isMeasuredStale;
  const displayGb = hasFreshMeasure
    ? (vm.reservedMb ?? vm.allocatedMb ?? 0) / 1024
    : vramEst;
  const isOom = gpuTotalVramGb !== null && displayGb > (gpuTotalVramGb - 1.0);
  const vramEstColor = isOom
    ? "var(--red, #ef4444)"
    : hasFreshMeasure
      ? "var(--green, #22c55e)"
      : undefined;

  const trainingActive = useTrainingStore((st) => st.status !== "idle");
  const measureDisabled = !s.instanceArchitecture || !s.instanceConfig || vm.status === "running" || trainingActive;

  const handleMeasureVram = useCallback(async () => {
    if (measureDisabled) return;
    s.setVramMeasure({ status: "running", allocatedMb: null, reservedMb: null, error: null, measuredFor: null, measuredAt: null });
    const signature = JSON.stringify([
      s.instanceArchitecture, s.instanceConfig, s.instanceScale,
      s.batchSize, s.patchSize, s.fp16, s.gradientCheckpointing,
    ]);
    try {
      const { estimateTrainingVram } = await import("../../lib/api");
      const res = await estimateTrainingVram({
        model_name: s.instanceArchitecture!,
        config: s.instanceConfig ?? undefined,
        batch_size: s.batchSize,
        patch_size: s.patchSize,
        dtype: s.fp16 ? "bf16" : "float32",
        scale: s.instanceScale ?? 4,
        gradient_checkpointing: s.gradientCheckpointing,
        loss_config: s.lossConfig,
      });
      if (res.error) {
        s.setVramMeasure({ status: "error", allocatedMb: null, reservedMb: null, error: res.error, measuredFor: null, measuredAt: null });
      } else if (res.peak_reserved_mb ?? res.peak_allocated_mb) {
        s.setVramMeasure({
          status: "done",
          allocatedMb: res.peak_allocated_mb ?? null,
          reservedMb: res.peak_reserved_mb ?? null,
          error: null,
          measuredFor: signature,
          measuredAt: new Date().toLocaleTimeString(),
        });
      }
    } catch {
      s.setVramMeasure({ status: "error", allocatedMb: null, reservedMb: null, error: "Probe failed", measuredFor: null, measuredAt: null });
    }
  }, [measureDisabled, s]);

  const canLaunch = s.selectedInstance && s.selectedDataset;

  const handleLaunch = useCallback(async () => {
    try {
      const { startTraining } = await import("../../lib/api");
      const res = await startTraining({
        model_name: s.instanceArchitecture ?? "",
        instance: s.selectedInstance ?? "",
        dataset: s.selectedDataset ?? "",
        resume: s.resumeFrom ?? undefined,
        config: customConfigPath || undefined,
        device: s.device === "auto" ? undefined : s.device,
        batch_size: s.batchSize,
        learning_rate: s.learningRate,
        max_epochs: s.schedule.totalEpochs,
        patch_size: s.patchSize,
        fp16: s.fp16 || undefined,
        seed: s.seed,
        weight_decay: s.weightDecay,
        betas: s.betas,
        num_workers: s.numWorkers,
        save_per_epoch: s.schedule.saveEvery,
        validation_enabled: s.validationEnabled,
        validation_split: s.validationSplit,
        validation_split_seed: s.validationSplitSeed,
        validation_full_image_limit: s.validationFullImageLimit,
        validation_dataset: s.selectedValidationDataset ?? undefined,
        metrics_frequency: s.metricsFrequency,
        write_metrics_file: s.writeMetricsFile,
        benchmark_warmup: s.benchmarkWarmup,
        perceptual_weight: undefined,
        losses: s.lossConfig,
        warmup_steps: s.schedule.warmupSteps,
        gradient_checkpointing: s.gradientCheckpointing,
      });

      useTrainingStore.getState().reset();
      useTrainingStore.getState().setActiveRun(res.job_id);
      useTrainingStore.getState().setActiveRunDir(res.run_id ?? null);
      useTrainingStore.getState().setStatus("running");
      useTrainingStore.getState().setStage("starting");
      useTrainingStore.getState().setLaunchConfig({
        totalEpochs: s.schedule.totalEpochs,
        batchSize: s.batchSize,
        learningRate: s.learningRate,
        fp16: s.fp16,
        patchSize: s.patchSize,
        validationEnabled: s.validationEnabled,
      });
      useUiStore.getState().setActiveTab("metrics");
      setLaunchError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLaunchError(msg);
      useUiStore.getState().setLastApiError({
        type: "error",
        code: "LAUNCH_FAILED",
        message: msg,
      });
    }
  }, [s, customConfigPath]);

  const vramCoreGb =
    vramBreakdown.weightsGb +
    vramBreakdown.gradsGb +
    vramBreakdown.adamGb +
    vramBreakdown.activationsGb +
    vramBreakdown.inputGb +
    vramBreakdown.upsamplerGb;
  const allocatorOverheadGb = Math.max(0, vramBreakdown.totalGb - vramCoreGb - vramBreakdown.overheadGb);

  const vramSegments: StackedBarSegment[] = [
    { label: "Model weights", value: vramBreakdown.weightsGb, color: VRAM_SEGMENT_COLORS["Model weights"] },
    { label: "Gradients", value: vramBreakdown.gradsGb, color: VRAM_SEGMENT_COLORS["Gradients"] },
    { label: "Adam optimizer", value: vramBreakdown.adamGb, color: VRAM_SEGMENT_COLORS["Adam optimizer"] },
    { label: "Activations", value: vramBreakdown.activationsGb, color: VRAM_SEGMENT_COLORS["Activations"] },
    { label: "Input batch", value: vramBreakdown.inputGb, color: VRAM_SEGMENT_COLORS["Input batch"] },
    { label: "Upsampler", value: vramBreakdown.upsamplerGb, color: VRAM_SEGMENT_COLORS["Upsampler"] },
    { label: "Allocator overhead", value: allocatorOverheadGb, color: VRAM_SEGMENT_COLORS["Allocator overhead"] },
    { label: "CUDA context", value: vramBreakdown.overheadGb, color: VRAM_SEGMENT_COLORS["CUDA context"] },
  ];

  const launchSummary = [
    s.instanceArchitecture,
    s.selectedDataset,
    `${s.schedule.totalEpochs} epochs`,
  ].filter(Boolean).join(" · ");

  return (
    <div className="ts-sidebar">
      <Panel
        title="Readiness"
        subtitle={`${[!!s.selectedInstance, !!s.selectedDataset].filter(Boolean).length}/2`}
      >
        <div className="ts-col-6">
          <ReadinessItem done={!!s.selectedInstance} label="Model instance selected" />
          <ReadinessItem done={!!s.selectedDataset} label="Dataset selected" />
        </div>
      </Panel>

      <Panel title="Estimate" grow>
        <div className="ts-col-8">
          <div className="ts-col-3">
            <InfoRow label="Iters / epoch" value={itersPerEpoch.toLocaleString()} baseline labelSize={10} emphasis border={false} />
            <InfoRow label="Total iters" value={totalIters.toLocaleString()} baseline labelSize={10} emphasis border={false} />
          </div>

          <div className="ts-divider">
            <InfoRow
              label="GPU VRAM"
              value={gpuTotalVramGb ? `${gpuTotalVramGb.toFixed(0)} GB total` : "unknown"}
              color="var(--text)"
              baseline
              labelSize={10}
              emphasis
              border={false}
            />
            <InfoRow
              label="Heuristic estimate"
              value={`${vramEst.toFixed(2)} GB`}
              baseline
              labelSize={10}
              emphasis
              border={false}
            />
            {vramBreakdown.totalGb > 0 && (
              <div className="ts-breakdown">
                <StackedBar segments={vramSegments} />
                <div className="ts-col-3">
                  {vramSegments.filter((seg) => seg.value > 0).map((seg) => (
                    <div key={seg.label} className="ts-row-6">
                      <span className="ts-swatch" style={{ background: seg.color }} />
                      <span className="ts-legend-label">
                        {seg.label}
                      </span>
                      <span className="ts-legend-value">
                        {seg.value.toFixed(2)} GB
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {hasFreshMeasure && (
              <InfoRow
                label="Measured (real pass)"
                value={`${fmtGb(displayGb)}${vm.reservedMb ? " (reserved)" : ""}`}
                color={vramEstColor}
                baseline
                labelSize={10}
                emphasis
                border={false}
              />
            )}
            {hasFreshMeasure && vramEst > 0 && (() => {
              const ratio = displayGb / vramEst;
              if (ratio > 1.5) {
                return (
                  <span className="ts-note ts-note-amber">
                    Measured exceeds heuristic — includes MIOpen kernel workspace and allocator fragmentation the heuristic can't model.
                  </span>
                );
              }
              if (ratio < 0.6) {
                return (
                  <span className="ts-note ts-note-muted">
                    Measured is well below heuristic — heuristic is conservative.
                  </span>
                );
              }
              return null;
            })()}

            <div className="ts-btn-col">
              <Btn
                variant="solid"
                color="var(--green)"
                onClick={handleMeasureVram}
                disabled={measureDisabled}
                full
                centered
                title={trainingActive ? "Disabled while training runs" : undefined}
              >
                {vm.status === "running" ? "Measuring…" : "Measure VRAM"}
              </Btn>
              {vm.status === "running" && (
                <span className="ts-note ts-note-amber">
                  Running one dry forward+backward on the GPU — first run autotunes kernels, can take ~1 min.
                </span>
              )}
              {vm.status === "idle" && s.instanceArchitecture && s.instanceConfig && (
                <span className="ts-note ts-note-dim">
                  Heuristic only. Measuring runs one real training step for an accurate number.
                </span>
              )}
              {vm.status === "done" && isMeasuredStale && (
                <span className="ts-note ts-note-amber">
                  Config changed since last measure — re-measure.
                </span>
              )}
              {vm.status === "done" && vm.measuredAt && !isMeasuredStale && (
                <span className="ts-note ts-note-muted">
                  Measured at {vm.measuredAt}
                </span>
              )}
              {vm.status === "error" && (
                <span className="ts-note ts-note-red">
                  {vm.error || "Probe failed"}
                </span>
              )}
              {trainingActive && (
                <span className="ts-note ts-note-dim">
                  Disabled while training runs
                </span>
              )}
            </div>
          </div>

          {isOom && (
            <InlineAlert tone="red">
              {hasFreshMeasure ? "Measured" : "Estimated"} VRAM exceeds GPU capacity — may OOM
            </InlineAlert>
          )}
          {launchError && <InlineAlert tone="red">{launchError}</InlineAlert>}
        </div>
      </Panel>

      <div className="ts-col-6">
        {canLaunch && (
          <div className="ts-summary">
            {launchSummary}{s.resumeFrom ? ` · resume: ${s.resumeFrom}` : " · fresh run"}
          </div>
        )}
        <Btn
          variant="solid"
          color={isOom ? "var(--amber, #f59e0b)" : "var(--green)"}
          full
          centered
          onClick={handleLaunch}
          disabled={!canLaunch}
          style={{ gap: 7 }}
        >
          {isOom ? (
            <>⚠ Launch Anyway (may OOM)</>
          ) : (
            <><IconRocket size={13} color="#0d0f11" /> Launch Training</>
          )}
        </Btn>
      </div>
    </div>
  );
}