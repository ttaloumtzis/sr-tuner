import { useUiStore } from "../../store/uiStore";
import type { SystemInfo } from "../../lib/api-types";

interface Props {
  step: number;
  systemInfo: SystemInfo;
  onStart: (backend: string, envType: "venv" | "sidecar") => void;
  onBack: () => void;
  onNext: () => void;
}

export function SetupWizardWindows({ step, systemInfo, onStart, onBack, onNext }: Props) {
  const selectedBackend = useUiStore((s) => s.selectedBackend);
  const setSelectedBackend = useUiStore((s) => s.setSelectedBackend);

  return (
    <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 16 }}>
      {step === 1 && (
        <>
          <div style={labelStyle}>System & Backend</div>
          <div style={infoRowStyle}>
            <span>NVIDIA GPU (CUDA):</span>
            <span style={{ ...valueStyle, color: systemInfo.cuda_available ? "var(--green)" : "var(--dim)" }}>
              {systemInfo.cuda_available ? "Detected" : "Not detected"}
            </span>
          </div>
          <div style={infoRowStyle}>
            <span>ffmpeg:</span>
            <span style={{ ...valueStyle, color: systemInfo.has_ffmpeg ? "var(--green)" : "var(--amber)" }}>
              {systemInfo.has_ffmpeg ? "Found" : "Optional — install for video support"}
            </span>
          </div>
          <div style={infoRowStyle}>
            <span>uv:</span>
            <span style={{ ...valueStyle, color: systemInfo.has_uv ? "var(--green)" : "var(--red)" }}>
              {systemInfo.has_uv ? "Found" : "Not found — required"}
            </span>
          </div>
          <div style={infoRowStyle}>
            <span>Python 3:</span>
            <span style={{ ...valueStyle, color: systemInfo.has_python3 ? "var(--green)" : "var(--red)" }}>
              {systemInfo.has_python3 ? "Found" : "Not found — required"}
            </span>
          </div>

          <div style={{ ...labelStyle, marginTop: 8 }}>Select Backend</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {systemInfo.supported_backends.map((b) => (
              <label
                key={b}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 14px",
                  background: selectedBackend === b ? "var(--bg3)" : "var(--bg2)",
                  border: `1px solid ${selectedBackend === b ? "var(--green)66" : "var(--border)"}`,
                  borderRadius: "var(--radius-sm)",
                  cursor: "pointer",
                  fontSize: 12,
                }}
              >
                <input
                  type="radio"
                  name="backend"
                  value={b}
                  checked={selectedBackend === b}
                  onChange={() => setSelectedBackend(b)}
                  style={{ accentColor: "var(--green)" }}
                />
                <div>
                  <div style={{ color: "var(--text)", fontWeight: 500, textTransform: "uppercase" }}>
                    {b}
                    {b === systemInfo.default_backend && (
                      <span style={{ color: "var(--green)", fontSize: 10, marginLeft: 6 }}>recommended</span>
                    )}
                  </div>
                  <div style={{ color: "var(--dim)", fontSize: 10, marginTop: 2 }}>
                    {b === "cuda" && "NVIDIA GPU acceleration"}
                    {b === "cpu" && "CPU only"}
                  </div>
                </div>
              </label>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
            <button onClick={onNext} style={btnStyle}>
              Continue
            </button>
          </div>
        </>
      )}

      {step === 2 && (
        <>
          <div style={labelStyle}>Environment Type</div>
          <div
            style={{
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              padding: "10px 14px",
              fontSize: 12,
              color: "var(--text)",
              textAlign: "center",
            }}
          >
            Virtual Environment (venv)
          </div>
          <div style={{ fontSize: 11, color: "var(--dim)", textAlign: "center" }}>
            A Python virtual environment will be created at the configured env directory.
            Sidecar is not available on Windows in this version.
          </div>
          <div style={{ display: "flex", gap: 8, justifyContent: "space-between", marginTop: 8 }}>
            <button onClick={onBack} style={btnStyleMuted}>Back</button>
            <button onClick={() => onStart(selectedBackend, "venv")} style={btnStyle}>
              Install
            </button>
          </div>
        </>
      )}
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 600,
  color: "var(--muted)",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  marginBottom: 4,
};

const infoRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  fontSize: 11,
  fontFamily: "var(--font-mono)",
  color: "var(--muted)",
  padding: "4px 0",
};

const valueStyle: React.CSSProperties = {
  color: "var(--text)",
};

const btnStyle: React.CSSProperties = {
  background: "var(--green)",
  border: "none",
  borderRadius: "var(--radius-sm)",
  color: "#0d0f11",
  cursor: "pointer",
  fontSize: 11,
  fontWeight: 600,
  fontFamily: "var(--font-mono)",
  padding: "6px 18px",
};

const btnStyleMuted: React.CSSProperties = {
  background: "var(--bg2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)",
  color: "var(--muted)",
  cursor: "pointer",
  fontSize: 11,
  fontFamily: "var(--font-mono)",
  padding: "6px 18px",
};
