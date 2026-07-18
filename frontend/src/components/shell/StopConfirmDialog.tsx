// §19.6 — "Stop and save checkpoint?" / "Stop and discard?" dialog
// Maps to training.stop { save_checkpoint: true/false }

import { Btn } from "../ui/Btn";

interface Props {
  open: boolean;
  onSaveAndStop: () => void;
  onDiscardAndStop: () => void;
  onCancel: () => void;
}

export function StopConfirmDialog({ open, onSaveAndStop, onDiscardAndStop, onCancel }: Props) {
  if (!open) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.65)",
        zIndex: 4000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onCancel}
    >
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--border)",
          borderRadius: 6,
          padding: "24px 28px",
          width: 380,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "#f5a623", fontSize: 18 }}>⚠</span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: "var(--text)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Stop training?
          </span>
        </div>

        <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6, margin: 0 }}>
          Training is currently in progress. Choose how to stop:
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Btn variant="solid" color="var(--green)" onClick={onSaveAndStop}>
            Stop and save checkpoint
          </Btn>
          <Btn variant="solid" color="var(--red)" onClick={onDiscardAndStop}>
            Stop and discard
          </Btn>
          <Btn color="var(--dim)" onClick={onCancel}>
            Cancel
          </Btn>
        </div>

        <p
          style={{
            fontSize: 10,
            color: "var(--dim)",
            margin: 0,
            fontFamily: "var(--font-mono)",
          }}
        >
          Save checkpoint preserves model weights at the current epoch.
        </p>
      </div>
    </div>
  );
}
