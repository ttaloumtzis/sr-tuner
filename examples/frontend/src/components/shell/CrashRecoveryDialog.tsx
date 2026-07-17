// §21.5 — Crash recovery dialog shown on project reopen when a run was in-progress.
// Sidecar IPC uses stdin/stdout pipes that don't survive app restarts — reconnect is
// impossible. The user must choose to resume from the last checkpoint or abandon the run.

import { Btn } from "../ui/Btn";

interface Props {
  open: boolean;
  runName: string;
  lastEpoch: number;
  lastCheckpointPath: string | null;
  onResume: () => void;
  onAbandon: () => void;
}

export function CrashRecoveryDialog({
  open,
  runName,
  lastEpoch,
  lastCheckpointPath,
  onResume,
  onAbandon,
}: Props) {
  if (!open) return null;

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
    >
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--amber)",
          borderRadius: 6,
          padding: "24px 28px",
          width: 440,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "var(--amber)", fontSize: 18 }}>⚠</span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--amber)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Training session ended abnormally
          </span>
        </div>

        <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>
          Run{" "}
          <span
            style={{ color: "var(--text)", fontFamily: "var(--font-mono)" }}
          >
            {runName}
          </span>{" "}
          was in progress when this project was last closed.
          {lastEpoch > 0 && (
            <>
              {" "}Last saved epoch:{" "}
              <span
                style={{ color: "var(--text)", fontFamily: "var(--font-mono)" }}
              >
                {lastEpoch}
              </span>
              .
            </>
          )}
        </p>

        {lastCheckpointPath ? (
          <div
            style={{
              background: "var(--bg0)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              padding: "10px 12px",
              fontSize: 11,
              color: "var(--dim)",
              fontFamily: "var(--font-mono)",
              wordBreak: "break-all",
            }}
          >
            {lastCheckpointPath}
          </div>
        ) : (
          <div
            style={{
              background: "var(--bg0)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              padding: "10px 12px",
              fontSize: 11,
              color: "var(--red)",
            }}
          >
            No checkpoint was saved for this run — resume is not possible.
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Btn
            variant="solid"
            color="var(--green)"
            onClick={onResume}
            disabled={!lastCheckpointPath}
          >
            Resume from last checkpoint
          </Btn>
          <Btn color="var(--red)" onClick={onAbandon}>
            Abandon run
          </Btn>
        </div>
      </div>
    </div>
  );
}
