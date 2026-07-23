import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useUiStore } from "../../store/uiStore";

export function SetupStepInstall({ onCancel, onProceed }: { onCancel: () => void; onProceed?: () => void }) {
  const logEndRef = useRef<HTMLDivElement>(null);
  const log = useUiStore((s) => s.installationLog);
  const error = useUiStore((s) => s.installError);
  const installationDone = useUiStore((s) => s.installationDone);
  const appendInstallLog = useUiStore((s) => s.appendInstallLog);
  const setInstallError = useUiStore((s) => s.setInstallError);
  const setInstallationDone = useUiStore((s) => s.setInstallationDone);
  const [currentStep, setCurrentStep] = useState<string | null>(null);

  useEffect(() => {
    const unsubLog = listen<string>("install-log", (e) => {
      appendInstallLog(e.payload);
    });
    const unsubDone = listen("install-done", () => {
      setInstallationDone(true);
    });
    const unsubErr = listen<string>("install-error", (e) => {
      setInstallError(e.payload);
    });
    const unsubLabel = listen<string>("install-progress-label", (e) => {
      setCurrentStep(e.payload);
    });

    return () => {
      unsubLog.then((f) => f());
      unsubDone.then((f) => f());
      unsubErr.then((f) => f());
      unsubLabel.then((f) => f());
    };
  }, [appendInstallLog, setInstallError, setInstallationDone]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  useEffect(() => {
    if (installationDone) {
      // small delay so the last log lines render first
      const t = setTimeout(() => logEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
      return () => clearTimeout(t);
    }
  }, [installationDone]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%" }}>
      {currentStep && (
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--green)",
            fontFamily: "var(--font-mono)",
            letterSpacing: "0.03em",
            padding: "0 2px",
          }}
        >
          ▸ {currentStep}
        </div>
      )}
      <div
        style={{
          background: "#0a0c0e",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)",
          padding: 12,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          lineHeight: 1.6,
          height: 260,
          overflowY: "auto",
          color: "#4dba7f",
        }}
      >
        {log.length === 0 && !error && (
          <span style={{ color: "var(--dim)" }}>{currentStep ?? "Preparing installation..."}</span>
        )}
        {log.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
        {error && (
          <div style={{ color: "var(--red)", marginTop: 8 }}>
            ✗ {error}
          </div>
        )}
        <div ref={logEndRef} />
      </div>

      {error ? (
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          <button onClick={onCancel} style={btnStyle}>
            Back
          </button>
        </div>
      ) : installationDone ? (
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          <button onClick={onProceed} style={proceedBtnStyle}>
            Proceed
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          <button onClick={onCancel} style={btnStyleMuted}>
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: "var(--bg2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)",
  color: "var(--text)",
  cursor: "pointer",
  fontSize: 11,
  fontFamily: "var(--font-mono)",
  padding: "6px 18px",
};

const btnStyleMuted: React.CSSProperties = {
  ...btnStyle,
  color: "var(--muted)",
};

const proceedBtnStyle: React.CSSProperties = {
  ...btnStyle,
  background: "var(--green)",
  color: "#0d0f11",
  fontWeight: 600,
};
