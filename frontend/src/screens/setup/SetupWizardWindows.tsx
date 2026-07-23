import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useUiStore } from "../../store/uiStore";
import type { SystemInfo } from "../../lib/api-types";

interface RocmVenvInfo {
  valid: boolean;
  hip_version: string | null;
  python_version: string | null;
  error: string | null;
}

interface Props {
  step: number;
  systemInfo: SystemInfo;
  onStart: (backend: string, envType: "venv" | "sidecar", rocmVenvPath?: string) => void;
  onBack: () => void;
  onNext: () => void;
}

export function SetupWizardWindows({ step, systemInfo, onStart, onBack, onNext }: Props) {
  const selectedBackend = useUiStore((s) => s.selectedBackend);
  const setSelectedBackend = useUiStore((s) => s.setSelectedBackend);

  const [rocmVenvPath, setRocmVenvPath] = useState(() => {
    const home = typeof process !== "undefined" && process.env.USERPROFILE
      ? process.env.USERPROFILE
      : "C:\\Users\\user";
    return `${home}\\sr-tuner\\env\\venv`;
  });
  const [verifyStatus, setVerifyStatus] = useState<"idle" | "verifying" | "valid" | "invalid">("idle");
  const [verifyInfo, setVerifyInfo] = useState<RocmVenvInfo | null>(null);

  const handleBrowse = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ directory: true, multiple: false, title: "Select ROCm venv folder" });
      if (selected && typeof selected === "string") {
        setRocmVenvPath(selected);
        setVerifyStatus("idle");
        setVerifyInfo(null);
      }
    } catch {
      // Fallback if dialog plugin not available
    }
  };

  const handleVerify = async () => {
    setVerifyStatus("verifying");
    setVerifyInfo(null);
    try {
      const info = await invoke<RocmVenvInfo>("verify_rocm_venv", { venvPath: rocmVenvPath });
      setVerifyInfo(info);
      setVerifyStatus(info.valid ? "valid" : "invalid");
    } catch (err) {
      setVerifyInfo({ valid: false, hip_version: null, python_version: null, error: String(err) });
      setVerifyStatus("invalid");
    }
  };

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
            <span>AMD GPU (ROCm):</span>
            <span style={{ ...valueStyle, color: systemInfo.rocm_available ? "var(--green)" : "var(--dim)" }}>
              {systemInfo.rocm_available ? "Detected" : "Not detected"}
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
                    {b === "rocm" && "AMD GPU acceleration (via AMD Adrenalin)"}
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

      {step === 2 && selectedBackend === "rocm" && (
        <>
          <div style={labelStyle}>AMD GPU (ROCm) Setup</div>
          <div style={{ fontSize: 11, color: "var(--text)", lineHeight: 1.6, padding: "4px 0" }}>
            Create a Python venv with ROCm PyTorch via AMD Adrenalin, then click Verify.
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={labelStyle}>Venv Path</div>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="text"
                value={rocmVenvPath}
                onChange={(e) => { setRocmVenvPath(e.target.value); setVerifyStatus("idle"); setVerifyInfo(null); }}
                style={{
                  flex: 1,
                  background: "var(--bg2)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-sm)",
                  color: "var(--text)",
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                  padding: "6px 10px",
                }}
              />
              <button onClick={handleBrowse} style={btnStyleMuted}>
                Browse
              </button>
            </div>
          </div>

          <button
            onClick={handleVerify}
            disabled={verifyStatus === "verifying"}
            style={{
              ...btnStyle,
              opacity: verifyStatus === "verifying" ? 0.6 : 1,
              alignSelf: "flex-start",
            }}
          >
            {verifyStatus === "verifying" ? "Verifying..." : "Verify"}
          </button>

          {verifyInfo && (
            <div
              style={{
                background: "var(--bg2)",
                border: `1px solid ${verifyInfo.valid ? "var(--green)66" : "var(--red)66"}`,
                borderRadius: "var(--radius-sm)",
                padding: "10px 14px",
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                lineHeight: 1.8,
              }}
            >
              {verifyInfo.valid ? (
                <>
                  <div style={{ color: "var(--green)" }}>✓ ROCm PyTorch detected</div>
                  {verifyInfo.hip_version && (
                    <div style={{ color: "var(--text)" }}>  HIP Version: {verifyInfo.hip_version}</div>
                  )}
                  {verifyInfo.python_version && (
                    <div style={{ color: "var(--text)" }}>  Python: {verifyInfo.python_version}</div>
                  )}
                </>
              ) : (
                <>
                  <div style={{ color: "var(--red)" }}>✗ {verifyInfo.error || "ROCm PyTorch not found"}</div>
                  <div style={{ color: "var(--dim)", marginTop: 4 }}>
                    Create the venv using AMD Adrenalin at the path above, then click Verify.
                  </div>
                </>
              )}
            </div>
          )}

          <div style={{ display: "flex", gap: 8, justifyContent: "space-between", marginTop: 8 }}>
            <button onClick={onBack} style={btnStyleMuted}>Back</button>
            <button
              onClick={() => onStart(selectedBackend, "venv", rocmVenvPath)}
              disabled={verifyStatus !== "valid"}
              style={{
                ...btnStyle,
                opacity: verifyStatus === "valid" ? 1 : 0.4,
              }}
            >
              Install
            </button>
          </div>
        </>
      )}

      {step === 2 && selectedBackend !== "rocm" && (
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
