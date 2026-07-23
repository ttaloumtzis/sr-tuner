import { useUiStore } from "../../store/uiStore";

export function SetupStepDone({ onLaunch }: { onLaunch: () => void }) {
  const backend = useUiStore((s) => s.selectedBackend);
  const envType = useUiStore((s) => s.selectedEnvType);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 16,
        textAlign: "center",
      }}
    >
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: "50%",
          background: "var(--green-dim)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 22,
          color: "var(--green)",
        }}
      >
        ✓
      </div>

      <div style={{ fontSize: 14, fontWeight: 600, color: "var(--green)", fontFamily: "var(--font-mono)" }}>
        Installation Complete
      </div>

      <div
        style={{
          background: "var(--bg2)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)",
          padding: "12px 20px",
          fontSize: 11,
          fontFamily: "var(--font-mono)",
          color: "var(--muted)",
          lineHeight: 1.8,
          textAlign: "left",
        }}
      >
        <div>Backend: <span style={{ color: "var(--text)" }}>{backend.toUpperCase()}</span></div>
        <div>Environment: <span style={{ color: "var(--text)" }}>{envType}</span></div>
      </div>

      <div style={{ fontSize: 11, color: "var(--dim)" }}>
        SR Tuner is ready to launch. The Python server will start automatically.
      </div>

      <button
        onClick={onLaunch}
        style={{
          background: "var(--green)",
          border: "none",
          borderRadius: "var(--radius-sm)",
          color: "#0d0f11",
          cursor: "pointer",
          fontSize: 12,
          fontWeight: 600,
          fontFamily: "var(--font-mono)",
          padding: "8px 28px",
          marginTop: 8,
        }}
      >
        Launch SR Tuner
      </button>
    </div>
  );
}
