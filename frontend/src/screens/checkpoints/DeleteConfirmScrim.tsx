import type { CheckpointEntry, RunInfo } from "../../lib/api-types";
import { ConfirmScrim } from "../../components/ui/ConfirmScrim";
import { fmtSize } from "../../lib/format";

type DeleteTarget =
  | { kind: "checkpoint"; entry: CheckpointEntry }
  | { kind: "run"; run: RunInfo };

interface DeleteConfirmScrimProps {
  target: DeleteTarget;
  onConfirm: () => void;
  onCancel: () => void;
}

export function DeleteConfirmScrim({ target, onConfirm, onCancel }: DeleteConfirmScrimProps) {
  if (target.kind === "checkpoint") {
    const { entry } = target;
    return (
      <ConfirmScrim
        title="Delete checkpoint?"
        message={
          <>
            The following files will be deleted:
            <div className="sc-file-list">
              {entry.filename}
              <br />
              {entry.filename.replace(/\.pth$/i, "_metrics.json")}
            </div>
          </>
        }
        confirmLabel="Delete"
        onConfirm={onConfirm}
        onCancel={onCancel}
        danger
        width={320}
        zIndex={100}
      />
    );
  }

  const { run } = target;
  return (
<ConfirmScrim
        title={`Delete run ${run.run_id}?`}
        message={
          <>
            The run folder and everything in it will be permanently deleted:
            <div className="sc-file-list">
              {run.checkpoint_count} checkpoint(s) · {fmtSize(run.total_size_mb)}
              {run.has_metrics && <><br />metrics.jsonl</>}
              <br />validation frames
            </div>
          </>
        }
      confirmLabel="Delete"
      onConfirm={onConfirm}
      onCancel={onCancel}
      danger
      width={340}
      zIndex={100}
    />
  );
}