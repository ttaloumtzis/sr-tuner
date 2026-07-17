// §19.3 — Error dialog when sidecar exits unexpectedly mid-training.
// Shows last known epoch and "Resume from last checkpoint?" action.

import { Btn } from "../ui/Btn";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useUiStore } from "../../store/uiStore";
import { useProjectStore } from "../../store/projectStore";

interface Props {
  open: boolean;
  lastEpoch: number;
  runId: string | null;
  onClose: () => void;
}

export function SidecarExitDialog({ open, lastEpoch, runId, onClose }: Props) {
  if (!open) return null;

  function handleResumeFromCheckpoint() {
    // Navigate to Training Setup and pre-fill resumeFrom if we have a matching checkpoint
    const project = useProjectStore.getState().project;
    const run = project?.runs.find((r) => r.run_id === runId);
    const lastCheckpointPath = run?.checkpoints.last_saved_path ?? null;
    const lastEpochNum = run?.metrics.current_epoch ?? lastEpoch;

    if (lastCheckpointPath) {
      useRunConfigStore.getState().setResumeFrom({
        checkpoint_path: lastCheckpointPath,
        resume_epoch: lastEpochNum,
        resume_optimizer_state: true,
        resume_lr_scheduler_state: true,
      });
    }

    useUiStore.getState().setActiveTab("training");
    onClose();
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.70)",
        zIndex: 4000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--red)",
          borderRadius: 6,
          padding: "24px 28px",
          width: 420,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "var(--red)", fontSize: 18 }}>✗</span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--red)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Sidecar process terminated unexpectedly
          </span>
        </div>

        <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>
          The training backend exited while training was in progress.
          {lastEpoch > 0 && (
            <>
              {" "}Last known epoch:{" "}
              <span
                style={{ color: "var(--text)", fontFamily: "var(--font-mono)" }}
              >
                {lastEpoch}
              </span>
              .
            </>
          )}
        </p>

        <div
          style={{
            background: "var(--bg0)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            padding: "10px 12px",
            fontSize: 11,
            color: "var(--dim)",
            fontFamily: "var(--font-mono)",
          }}
        >
          Check logs for details. The sidecar may have crashed due to OOM,
          CUDA error, or an unhandled exception.
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Btn variant="solid" color="var(--green)" onClick={handleResumeFromCheckpoint}>
            Resume from last checkpoint
          </Btn>
          <Btn color="var(--dim)" onClick={onClose}>
            Dismiss
          </Btn>
        </div>
      </div>
    </div>
  );
}
