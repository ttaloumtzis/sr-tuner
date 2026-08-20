import type { CSSProperties } from "react";
import { CollapsibleSection } from "../../components/ui/CollapsibleSection";
import { Field } from "../../components/ui/Field";
import { Dropdown, type DropdownOption } from "../../components/ui/Dropdown";
import { NumberInput } from "../../components/ui/NumberInput";
import { TextInput } from "../../components/ui/TextInput";
import { Toggle } from "../../components/ui/Toggle";
import { GroupLabel } from "../../components/ui/GroupLabel";
import { LabelWithHint } from "../../components/ui/LabelWithHint";
import { IconSettings } from "../../components/ui/icons";
import { useRunConfigStore } from "../../store/runConfigStore";

interface AdvancedSectionProps {
  customConfigPath: string;
  onCustomConfigPath: (v: string) => void;
}

export function AdvancedSection({ customConfigPath, onCustomConfigPath }: AdvancedSectionProps) {
  const s = useRunConfigStore();

  const deviceOptions: DropdownOption[] = [
    { value: "cuda:0", label: "cuda:0" },
    { value: "cpu", label: "cpu" },
    { value: "auto", label: "auto" },
  ];

  return (
    <CollapsibleSection
      title="Advanced"
      icon={<IconSettings size={13} />}
      subtitle="run · optimizer · validation · monitoring"
      defaultOpen={false}
      contentStyle={{ maxHeight: "min(48vh, 320px)", overflowY: "auto", overscrollBehavior: "contain" }}
    >
      <div className="ts-col-10">
        <div className="ts-group">
          <GroupLabel>Run &amp; Schedule</GroupLabel>
          <div className="ts-grid" style={{ "--ts-grid-min": "150px", "--ts-grid-max": "240px" } as CSSProperties}>
            <Field label="Device">
              <Dropdown value={s.device} options={deviceOptions} onChange={s.setDevice} />
            </Field>
            <Field label="Warmup Steps">
              <NumberInput compact value={s.schedule.warmupSteps} onChange={(v) => s.setSchedule({ warmupSteps: v })} min={0} />
            </Field>
            <Field label="Precision">
              <div className="ts-toggle-8">
                <Toggle on={s.fp16} onChange={() => s.setFp16(!s.fp16)} />
                <span className="ts-muted-label">BF16 mixed precision</span>
              </div>
            </Field>
            <Field label={<LabelWithHint label="Checkpointing" hint="Recompute activations during backward to save VRAM (~4-6x for transformers). On for transformer archs by default." />}>
              <div className="ts-toggle-8">
                <Toggle on={s.gradientCheckpointing === "auto" ? s.instanceArchitecture === "swinir" : s.gradientCheckpointing === "true"} onChange={() => s.setGradientCheckpointing(s.gradientCheckpointing === "auto" ? "true" : s.gradientCheckpointing === "true" ? "false" : "auto")} />
                <span className="ts-muted-label">{s.gradientCheckpointing === "auto" ? "Auto" : s.gradientCheckpointing === "true" ? "On" : "Off"}</span>
              </div>
            </Field>
          </div>
        </div>

        <div className="ts-group">
          <GroupLabel>Optimizer</GroupLabel>
          <div className="ts-grid" style={{ "--ts-grid-min": "100px", "--ts-grid-max": "160px" } as CSSProperties}>
            <Field label="Seed"><NumberInput compact value={s.seed} onChange={s.setSeed} min={0} /></Field>
            <Field label={<LabelWithHint label="Weight Decay" hint="L2 regularization strength applied by the Adam optimizer." />}>
              <NumberInput compact value={s.weightDecay} onChange={s.setWeightDecay} min={0} step={0.01} />
            </Field>
            <Field label="β₁"><NumberInput compact value={s.betas[0]} onChange={(v) => s.setBetas([v, s.betas[1]])} min={0} max={1} step={0.01} /></Field>
            <Field label="β₂"><NumberInput compact value={s.betas[1]} onChange={(v) => s.setBetas([s.betas[0], v])} min={0} max={1} step={0.001} /></Field>
          </div>
        </div>

        <div className="ts-group">
          <GroupLabel>Validation</GroupLabel>
          <div className="ts-grid" style={{ "--ts-grid-min": "140px", "--ts-grid-max": "220px" } as CSSProperties}>
            <Field label={<LabelWithHint label="Split seed" hint="Seeds the train/validation split only. Independent of the general seed — keeps the same images in train and validation across train phases." />}>
              <NumberInput compact value={s.validationSplitSeed} onChange={s.setValidationSplitSeed}
                min={0}
                disabled={s.selectedValidationDataset !== null} />
            </Field>
            <Field label={<LabelWithHint label="Full-image val" hint="Images per epoch for the slow tiled full-image validation pass (0 disables it). Patch-based PSNR/SSIM/loss still cover the whole validation set." />}>
              <NumberInput compact value={s.validationFullImageLimit} onChange={s.setValidationFullImageLimit}
                min={0} max={64} />
            </Field>
            <Field label="Workers">
              <NumberInput compact value={s.numWorkers} onChange={s.setNumWorkers} min={0} max={16} />
            </Field>
          </div>
        </div>

        <div className="ts-group">
          <GroupLabel>Monitoring</GroupLabel>
          <div className="ts-grid" style={{ "--ts-grid-min": "160px", "--ts-grid-max": "280px" } as CSSProperties}>
            <Field label="Metrics Frequency">
              <NumberInput compact value={s.metricsFrequency} onChange={s.setMetricsFrequency} min={1} />
            </Field>
            <Field label="Custom Config YAML">
              <TextInput compact value={customConfigPath} onChange={onCustomConfigPath} placeholder="path/to/config.yaml" />
            </Field>
          </div>
          <div className="ts-toggle-mt10">
            <Toggle on={s.writeMetricsFile} onChange={() => s.setWriteMetricsFile(!s.writeMetricsFile)} />
            <span className="ts-muted-label">Write metrics.jsonl file</span>
          </div>
          <div className="ts-toggle-mt8">
            <Toggle on={s.benchmarkWarmup} onChange={() => s.setBenchmarkWarmup(!s.benchmarkWarmup)} />
            <span className="ts-muted-label">
              Pre-warm GPU kernels before training
            </span>
          </div>
        </div>
      </div>
    </CollapsibleSection>
  );
}