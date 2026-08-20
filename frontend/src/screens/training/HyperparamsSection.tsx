import type { CSSProperties } from "react";
import { Field } from "../../components/ui/Field";
import { NumberInput } from "../../components/ui/NumberInput";
import { GroupLabel } from "../../components/ui/GroupLabel";
import { LabelWithHint } from "../../components/ui/LabelWithHint";
import { useRunConfigStore } from "../../store/runConfigStore";

export function HyperparamsSection() {
  const s = useRunConfigStore();

  return (
    <div className="ts-group ts-mb-10">
      <GroupLabel>Schedule</GroupLabel>
      <div className="ts-grid" style={{ "--ts-grid-min": "100px", "--ts-grid-max": "160px" } as CSSProperties}>
        <Field label={<LabelWithHint label="Total Epochs" hint="Number of full passes over the training dataset." />}>
          <NumberInput compact value={s.schedule.totalEpochs} onChange={(v) => s.setSchedule({ totalEpochs: v })} min={1} />
        </Field>
        <Field label="Batch Size"><NumberInput compact value={s.batchSize} onChange={s.setBatchSize} min={1} max={128} /></Field>
        <Field label={<LabelWithHint label="Patch Size" hint="Crop size (px) fed to the model each step. Larger patches use more VRAM." />}>
          <NumberInput compact value={s.patchSize} onChange={s.setPatchSize} min={16} max={512} step={8} />
        </Field>
        <Field label="Save Every"><NumberInput compact value={s.schedule.saveEvery} onChange={(v) => s.setSchedule({ saveEvery: v })} min={1} /></Field>
        <Field label={<LabelWithHint label="Learning Rate" hint="Step size the optimizer takes each update. Too high can destabilize training; too low slows convergence." />}>
          <NumberInput compact value={s.learningRate} onChange={s.setLearningRate} min={0} step={1e-5} />
        </Field>
      </div>
    </div>
  );
}