interface ConnectionErrorDialogProps {
  open: boolean;
  onRetry: () => void;
  onExit: () => void;
}

export function ConnectionErrorDialog({ open, onRetry, onExit }: ConnectionErrorDialogProps) {
  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed", inset: 0, zIndex: 10000,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "rgba(0,0,0,0.6)", backdropFilter: "blur(2px)",
      }}
    >
      <div style={{
        background: "var(--bg2)", border: "1px solid var(--border2)",
        borderTop: "3px solid var(--red)", borderRadius: "var(--radius-lg)",
        padding: "24px 28px", width: 400, maxWidth: "90vw",
        boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
      }}>
        <h2 style={{ margin: "0 0 10px", color: "var(--red)", fontSize: 14, fontFamily: "var(--font-sans)", fontWeight: 600 }}>
          Backend Disconnected
        </h2>
        <p style={{ margin: "0 0 16px", color: "var(--muted)", fontSize: 12, fontFamily: "var(--font-mono)", lineHeight: 1.6 }}>
          The Python backend server is not responding. Training will continue in the background but live updates are paused.
        </p>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button onClick={onExit} style={{
            background: "transparent", border: "1px solid var(--border)",
            color: "var(--muted)", fontSize: 12, padding: "6px 16px",
            borderRadius: "var(--radius-sm)", cursor: "pointer",
          }}>
            Exit
          </button>
          <button onClick={onRetry} style={{
            background: "var(--red)", border: "none", color: "#fff",
            fontSize: 12, fontWeight: 600, padding: "6px 16px",
            borderRadius: "var(--radius-sm)", cursor: "pointer",
          }}>
            Retry
          </button>
        </div>
      </div>
    </div>
  );
}