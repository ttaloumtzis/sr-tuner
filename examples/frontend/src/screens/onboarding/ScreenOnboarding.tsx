import React, { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useDeploymentMode } from "../../hooks/useDeploymentMode";
import { startSidecar } from "../../lib/ipc";
import type { IPCMessage, DeviceInfo } from "../../lib/ipc-types";
import {
  downloadAndInstallVariant,
  VARIANT_MANIFEST_URL,
  type GpuVariant,
} from "../../lib/gpuVariantManager";

// ── Types ──────────────────────────────────────────────────────────────────

type StepStatus = "pending" | "running" | "ok" | "warn" | "error";

interface CheckStep {
  id: string;
  label: string;
  status: StepStatus;
  detail?: string;
}

type ScreenState =
  | "checking"
  | "ready"
  | "incompatible"
  | "timeout"
  | "cpu-fallback"
  | "gpu-download";  // §15.6 — downloading GPU-specific sidecar variant

// Tauri DevCheckResult shape from §6.4
interface DevCheckResult {
  python_ok: boolean;
  python_version: string | null;
  torch_ok: boolean;
  basicsr_ok: boolean;
  ffmpeg_ok: boolean;
  errors: string[];
}

interface Props {
  onComplete: () => void;
}

// ── Constants ──────────────────────────────────────────────────────────────

const DISK_MIN_GB = 5;
const SIDECAR_READY_TIMEOUT_MS = 10_000;
const SIDECAR_EXTRACTING_TIMEOUT_MS = 120_000;
const VRAM_WARN_GB = 4;

// ── Spinner ────────────────────────────────────────────────────────────────

function Spinner() {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setFrame((f) => (f + 1) % frames.length), 80);
    return () => clearInterval(t);
  }, []);
  return (
    <span
      style={{
        color: "var(--green)",
        fontSize: 12,
        fontFamily: "var(--font-mono)",
      }}
    >
      {frames[frame]}
    </span>
  );
}

// ── Step indicator ─────────────────────────────────────────────────────────

function StepIndicator({ step }: { step: CheckStep }) {
  const icon = (() => {
    switch (step.status) {
      case "ok":
        return <span style={{ color: "var(--green)" }}>✓</span>;
      case "warn":
        return <span style={{ color: "#f5a623" }}>⚠</span>;
      case "error":
        return <span style={{ color: "var(--red)" }}>✗</span>;
      case "running":
        return <Spinner />;
      default:
        return (
          <span style={{ color: "var(--dim)", fontSize: 10 }}>○</span>
        );
    }
  })();

  const labelColor =
    step.status === "ok"
      ? "var(--text)"
      : step.status === "warn"
      ? "#f5a623"
      : step.status === "error"
      ? "var(--red)"
      : step.status === "running"
      ? "var(--text)"
      : "var(--dim)";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 12,
        padding: "8px 0",
      }}
    >
      <div
        style={{
          width: 18,
          flexShrink: 0,
          display: "flex",
          justifyContent: "center",
          marginTop: 1,
        }}
      >
        {icon}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 2,
          flex: 1,
        }}
      >
        <span
          style={{
            fontSize: 13,
            color: labelColor,
            fontFamily: "var(--font-sans)",
          }}
        >
          {step.label}
        </span>
        {step.detail && (
          <span
            style={{
              fontSize: 11,
              color: "var(--muted)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {step.detail}
          </span>
        )}
      </div>
    </div>
  );
}

// ── GPU card ───────────────────────────────────────────────────────────────

function GpuCard({ devices }: { devices: DeviceInfo[] }) {
  const primary = devices.find((d) => d.type !== "cpu");
  const cpu = devices.find((d) => d.type === "cpu");
  const lowVram =
    primary != null &&
    primary.vram_gb != null &&
    primary.vram_gb < VRAM_WARN_GB;

  return (
    <div
      style={{
        background: "var(--bg3)",
        border: `1px solid ${lowVram ? "#f5a62344" : "var(--green)44"}`,
        borderRadius: "var(--radius-md)",
        padding: "12px 16px",
        marginTop: 8,
      }}
    >
      {primary ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                fontSize: 11,
                color: "var(--muted)",
                fontFamily: "var(--font-sans)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              GPU
            </span>
            <span
              style={{
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                background:
                  primary.type === "rocm" ? "#7c3aed22" : "#22c55e22",
                color:
                  primary.type === "rocm" ? "#a78bfa" : "var(--green)",
                border: `1px solid ${
                  primary.type === "rocm" ? "#7c3aed44" : "var(--green)44"
                }`,
                borderRadius: "var(--radius-sm)",
                padding: "1px 6px",
              }}
            >
              {primary.type.toUpperCase()}
            </span>
          </div>
          <span
            style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}
          >
            {primary.name}
          </span>
          {primary.vram_gb != null && (
            <span
              style={{
                fontSize: 11,
                color: lowVram ? "#f5a623" : "var(--muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {primary.vram_gb.toFixed(1)} GB VRAM
              {lowVram ? " — low (< 4 GB)" : ""}
            </span>
          )}
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span
            style={{
              fontSize: 11,
              color: "var(--muted)",
              fontFamily: "var(--font-sans)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            CPU fallback
          </span>
          <span style={{ fontSize: 13, color: "var(--text)" }}>
            {cpu?.name ?? "CPU"}
          </span>
          <span style={{ fontSize: 11, color: "#f5a623" }}>
            Training will be slow without a GPU
          </span>
        </div>
      )}
    </div>
  );
}

// ── Shared action button ───────────────────────────────────────────────────

function ActionBtn({
  label,
  variant,
  onClick,
}: {
  label: string;
  variant: "primary" | "ghost";
  onClick: () => void;
}) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background:
          variant === "primary"
            ? hovered
              ? "#1a7f3a"
              : "var(--green)"
            : hovered
            ? "var(--bg3)"
            : "transparent",
        border:
          variant === "primary" ? "none" : "1px solid var(--border)",
        color:
          variant === "primary"
            ? "#000"
            : hovered
            ? "var(--text)"
            : "var(--muted)",
        fontSize: 12,
        fontFamily: "var(--font-sans)",
        fontWeight: variant === "primary" ? 600 : 400,
        padding: "7px 18px",
        borderRadius: "var(--radius-sm)",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
    >
      {label}
    </button>
  );
}

// ── Incompatible panel ─────────────────────────────────────────────────────

function IncompatiblePanel({
  errorCode,
  errorMessage,
  onRetry,
  onCpuFallback,
  fallbackLabel,
}: {
  errorCode: string | null;
  errorMessage: string;
  onRetry: () => void;
  onCpuFallback?: () => void;
  fallbackLabel?: string;
}) {
  const isCuda = errorCode === "CUDA_NOT_FOUND";
  const isRocm = errorCode === "ROCM_NOT_FOUND";

  const title =
    isCuda
      ? "CUDA Not Found"
      : isRocm
      ? "ROCm Not Found"
      : errorCode === "DISK_LOW"
      ? "Insufficient Disk Space"
      : errorCode === "DEV_DEP_MISSING"
      ? "Missing Dependencies"
      : "Incompatible Hardware";

  const suggestions: string[] = isCuda
    ? [
        "Install NVIDIA drivers (≥ 525) from nvidia.com",
        "Install CUDA Toolkit 11.8 or 12.x",
        "Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121",
        "Use CPU fallback for slow but functional training",
      ]
    : isRocm
    ? [
        "Install ROCm 6.x from rocm.docs.amd.com",
        "Add your user to the 'render' and 'video' groups",
        "Reinstall PyTorch ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0",
        "Use CPU fallback for slow but functional training",
      ]
    : errorCode === "DISK_LOW"
    ? [
        "Free up disk space by removing old checkpoints, datasets, or videos",
        "Use an external drive with at least 5 GB free",
      ]
    : errorCode === "DEV_DEP_MISSING"
    ? [
        "Install missing packages with pip inside your virtual environment",
        "Ensure you activated the correct venv before launching",
      ]
    : ["Check the application logs for details"];

  return (
    <div
      style={{
        background: "var(--bg1)",
        border: "1px solid #e05c5c44",
        borderTop: "3px solid var(--red)",
        borderRadius: "var(--radius-md)",
        padding: "16px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <h3
        style={{
          margin: 0,
          fontSize: 13,
          fontWeight: 600,
          color: "var(--red)",
        }}
      >
        {title}
      </h3>
      <p
        style={{
          margin: 0,
          fontSize: 11,
          color: "var(--muted)",
          fontFamily: "var(--font-mono)",
          lineHeight: 1.6,
          wordBreak: "break-word",
          whiteSpace: "pre-wrap",
        }}
      >
        {errorMessage}
      </p>
      {suggestions.length > 0 && (
        <div
          style={{
            background: "var(--bg3)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            padding: "10px 14px",
          }}
        >
          <p
            style={{
              margin: "0 0 6px",
              fontSize: 10,
              fontWeight: 600,
              color: "var(--muted)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            Suggestions
          </p>
          <ul style={{ margin: 0, padding: "0 0 0 16px" }}>
            {suggestions.map((s, i) => (
              <li
                key={i}
                style={{
                  fontSize: 11,
                  color: "var(--text)",
                  lineHeight: 1.6,
                }}
              >
                {s}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
        {onCpuFallback && (
          <ActionBtn
            label={fallbackLabel ?? "Use CPU Fallback"}
            variant="ghost"
            onClick={onCpuFallback}
          />
        )}
        <ActionBtn label="Retry" variant="primary" onClick={onRetry} />
      </div>
    </div>
  );
}

// ── Timeout panel ──────────────────────────────────────────────────────────

function TimeoutPanel({ onRetry }: { onRetry: () => void }) {
  return (
    <div
      style={{
        background: "var(--bg1)",
        border: "1px solid #e05c5c44",
        borderTop: "3px solid var(--red)",
        borderRadius: "var(--radius-md)",
        padding: "16px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <h3
        style={{
          margin: 0,
          fontSize: 13,
          fontWeight: 600,
          color: "var(--red)",
        }}
      >
        Sidecar Failed to Start
      </h3>
      <p
        style={{
          margin: 0,
          fontSize: 11,
          color: "var(--muted)",
          lineHeight: 1.6,
        }}
      >
        The Python sidecar process did not respond within the timeout window.
        Check the application logs for Python errors.
      </p>
      <div
        style={{
          background: "var(--bg3)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)",
          padding: "10px 14px",
        }}
      >
        <p
          style={{
            margin: "0 0 6px",
            fontSize: 10,
            fontWeight: 600,
            color: "var(--muted)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          Suggestions
        </p>
        <ul style={{ margin: 0, padding: "0 0 0 16px" }}>
          {[
            "Check the application logs (~/.sr-tuner/logs/) for Python errors",
            "Ensure the sidecar binary is present and executable",
            "Try restarting the application",
          ].map((s, i) => (
            <li
              key={i}
              style={{ fontSize: 11, color: "var(--text)", lineHeight: 1.6 }}
            >
              {s}
            </li>
          ))}
        </ul>
      </div>
      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
        <ActionBtn label="Retry" variant="primary" onClick={onRetry} />
      </div>
    </div>
  );
}

// ── Main OnboardingScreen ──────────────────────────────────────────────────

export function ScreenOnboarding({ onComplete }: Props) {
  const mode = useDeploymentMode();

  const [steps, setSteps] = useState<CheckStep[]>([]);
  const [screenState, setScreenState] = useState<ScreenState>("checking");
  const [hardwareDevices, setHardwareDevices] = useState<DeviceInfo[]>([]);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  // §15.6 / §19.8 — GPU variant download state
  const [gpuVendor, setGpuVendor] = useState<"nvidia" | "amd" | "cpu" | null>(null);
  const [downloadBytesDone, setDownloadBytesDone] = useState(0);
  const [downloadBytesTotal, setDownloadBytesTotal] = useState<number | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unlistenRef = useRef<(() => void) | null>(null);
  const extractingRef = useRef(false);

  // ── Step helpers ─────────────────────────────────────────────────────

  const updateStep = (
    id: string,
    patch: Partial<CheckStep>,
    setter: React.Dispatch<React.SetStateAction<CheckStep[]>>
  ) =>
    setter((prev) =>
      prev.map((s) => (s.id === id ? { ...s, ...patch } : s))
    );

  // ── Sidecar IPC listener ─────────────────────────────────────────────

  const setupSidecarListener = (
    setStep: (id: string, patch: Partial<CheckStep>) => void
  ) =>
    listen<IPCMessage>("sidecar-message", ({ payload: msg }) => {
      if (msg.type === "sidecar.ready") {
        setStep("sidecar", {
          status: "ok",
          detail: `v${msg.version} · pid ${msg.pid}`,
        });
        setStep("gpu", { status: "running" });
        // Sidecar is up — clear the spawn timeout; hardware.info will arrive next
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
      }

      // §15.6 / §19.7 — GPU variant download needed
      if (msg.type === "gpu.detection_needed") {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        setGpuVendor(msg.vendor as "nvidia" | "amd" | "cpu");
        setScreenState("gpu-download");
        return;
      }

      if (msg.type === "hardware.info") {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        unlistenRef.current?.();

        const primary = msg.devices.find((d) => d.type !== "cpu");
        const lowVram =
          primary?.vram_gb != null && primary.vram_gb < VRAM_WARN_GB;

        setStep("gpu", {
          status: lowVram ? "warn" : "ok",
          detail: primary
            ? `${primary.name} · ${primary.vram_gb?.toFixed(1) ?? "?"} GB VRAM`
            : "No GPU detected — CPU fallback",
        });

        setHardwareDevices(msg.devices);
        setScreenState(primary ? "ready" : "cpu-fallback");
      }

      if (msg.type === "error") {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        unlistenRef.current?.();

        setStep("gpu", { status: "error", detail: msg.message });
        setErrorCode(msg.code);
        setErrorMessage(msg.message);
        setScreenState("incompatible");
      }
    });

  // ── Main check sequence ──────────────────────────────────────────────

  const runChecks = async (
    setStepsLocal: React.Dispatch<React.SetStateAction<CheckStep[]>>
  ) => {
    if (!mode) return;

    const setStep = (id: string, patch: Partial<CheckStep>) =>
      updateStep(id, patch, setStepsLocal);

    // Build initial step list
    const initialSteps: CheckStep[] = [
      { id: "disk", label: "Disk space (≥ 5 GB free)", status: "running" },
      ...(mode === "dev"
        ? [
            {
              id: "venv",
              label: "Python venv (uv)",
              status: "pending" as StepStatus,
            },
            {
              id: "python",
              label: "Python ≥ 3.11",
              status: "pending" as StepStatus,
            },
            {
              id: "torch",
              label: "PyTorch",
              status: "pending" as StepStatus,
            },
            {
              id: "basicsr",
              label: "BasicSR",
              status: "pending" as StepStatus,
            },
            {
              id: "ffmpeg",
              label: "FFmpeg",
              status: "pending" as StepStatus,
            },
          ]
        : []),
      { id: "sidecar", label: "Sidecar process", status: "pending" },
      { id: "gpu", label: "GPU / hardware", status: "pending" },
    ];
    setStepsLocal(initialSteps);

    // 1. Disk space check
    try {
      const freeGb = await invoke<number>("check_disk_space", { path: "." });
      if (freeGb >= DISK_MIN_GB) {
        setStep("disk", {
          status: "ok",
          detail: `${freeGb.toFixed(1)} GB free`,
        });
      } else {
        setStep("disk", {
          status: "error",
          detail: `Only ${freeGb.toFixed(1)} GB free — need ${DISK_MIN_GB} GB`,
        });
        setErrorCode("DISK_LOW");
        setErrorMessage(
          `Only ${freeGb.toFixed(1)} GB of disk space is free. SR Tuner requires at least ${DISK_MIN_GB} GB.`
        );
        setScreenState("incompatible");
        return;
      }
    } catch {
      setStep("disk", {
        status: "warn",
        detail: "Could not check disk space",
      });
    }

    // 2. Dev mode: venv first-run detection and setup (§19.5)
    if (mode === "dev") {
      setStep("venv", { status: "running" });
      try {
        const venvExists = await invoke<boolean>("check_venv_exists");
        if (venvExists) {
          setStep("venv", { status: "ok", detail: ".venv already exists" });
        } else {
          setStep("venv", { status: "running", detail: "Running uv sync…" });
          try {
            const output = await invoke<string>("setup_venv");
            setStep("venv", {
              status: "ok",
              detail: output.trim().split("\n").pop() ?? "venv created",
            });
          } catch (uvErr) {
            setStep("venv", {
              status: "error",
              detail: String(uvErr).slice(0, 120),
            });
            setErrorCode("VENV_SETUP_FAILED");
            setErrorMessage(
              `Failed to set up the Python venv. Ensure 'uv' is installed.\n\n${String(uvErr)}`
            );
            setScreenState("incompatible");
            return;
          }
        }
      } catch {
        // Non-fatal: best-effort check; continue
        setStep("venv", { status: "warn", detail: "Could not check venv" });
      }
    }

    // 3. Dev mode: PATH dependency checks (§6.4)
    if (mode === "dev") {
      setStep("python", { status: "running" });
      setStep("torch", { status: "running" });
      setStep("basicsr", { status: "running" });
      setStep("ffmpeg", { status: "running" });

      try {
        const res = await invoke<DevCheckResult>("check_dev_dependencies");

        setStep("python", {
          status: res.python_ok ? "ok" : "error",
          detail: res.python_version ?? (res.python_ok ? undefined : "Not found on PATH"),
        });
        setStep("torch", {
          status: res.torch_ok ? "ok" : "error",
          detail: res.torch_ok
            ? undefined
            : "Not importable — run: pip install torch",
        });
        setStep("basicsr", {
          status: res.basicsr_ok ? "ok" : "error",
          detail: res.basicsr_ok
            ? undefined
            : "Not importable — run: pip install basicsr",
        });
        setStep("ffmpeg", {
          status: res.ffmpeg_ok ? "ok" : "warn",
          detail: res.ffmpeg_ok
            ? undefined
            : "Not found — video extract will be disabled",
        });

        if (!res.python_ok || !res.torch_ok || !res.basicsr_ok) {
          setErrorCode("DEV_DEP_MISSING");
          setErrorMessage(
            "Required development dependencies are missing:\n" +
              res.errors.join("\n")
          );
          setScreenState("incompatible");
          return;
        }
      } catch (e) {
        setStep("python", { status: "error", detail: String(e) });
        setErrorCode("DEV_CHECK_FAILED");
        setErrorMessage(String(e));
        setScreenState("incompatible");
        return;
      }
    }

    // 4. Spawn sidecar and await sidecar.ready + hardware.info
    setStep("sidecar", { status: "running" });

    // Listen before spawning so we don't miss messages
    unlistenRef.current = await setupSidecarListener(setStep);

    // Timeout — 10 s normally, extended to 120 s if SIDECAR_EXTRACTING received
    const timeoutMs = extractingRef.current
      ? SIDECAR_EXTRACTING_TIMEOUT_MS
      : SIDECAR_READY_TIMEOUT_MS;

    timeoutRef.current = setTimeout(() => {
      unlistenRef.current?.();
      setStep("sidecar", {
        status: "error",
        detail: "Timed out waiting for sidecar.ready",
      });
      setStep("gpu", { status: "error" });
      setScreenState("timeout");
    }, timeoutMs);

    try {
      await startSidecar();
    } catch (e) {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      unlistenRef.current?.();
      setStep("sidecar", { status: "error", detail: String(e) });
      setStep("gpu", { status: "error" });
      setErrorCode("SIDECAR_SPAWN_FAILED");
      setErrorMessage(String(e));
      setScreenState("timeout");
    }
  };

  // ── Retry ────────────────────────────────────────────────────────────

  const handleRetry = () => {
    setScreenState("checking");
    setErrorCode(null);
    setErrorMessage("");
    setHardwareDevices([]);
    extractingRef.current = false;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    unlistenRef.current?.();
    runChecks(setSteps);
  };

  // ── GPU variant download (§25.8) ─────────────────────────────────────
  // Triggered when the minimal sidecar emits gpu.detection_needed and the
  // screen transitions to "gpu-download".  Downloads the matching variant,
  // verifies its checksum, then re-spawns the sidecar from the GPU binary
  // so onboarding can continue with full hardware.info.

  useEffect(() => {
    if (screenState !== "gpu-download" || gpuVendor === null || gpuVendor === "cpu") return;

    let cancelled = false;
    const variant: GpuVariant = gpuVendor === "nvidia" ? "cuda" : "rocm";

    setDownloadBytesDone(0);
    setDownloadBytesTotal(null);
    setDownloadError(null);

    downloadAndInstallVariant(variant, VARIANT_MANIFEST_URL, (progress) => {
      if (cancelled) return;
      setDownloadBytesDone(progress.bytesDone);
      setDownloadBytesTotal(progress.bytesTotal);
    })
      .then(async (binaryPath) => {
        if (cancelled) return;
        // Re-spawn sidecar from the downloaded GPU binary; onboarding listener
        // is still active and will receive hardware.info from the new process.
        await invoke("spawn_sidecar_from_path", { path: binaryPath });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setDownloadError(msg);
      });

    return () => { cancelled = true; };
  }, [screenState, gpuVendor]);

  // ── Initial run ──────────────────────────────────────────────────────

  useEffect(() => {
    if (mode) runChecks(setSteps);
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      unlistenRef.current?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // ── Render ───────────────────────────────────────────────────────────

  const header =
    mode === "bundled"
      ? "Checking system requirements"
      : mode === "dev"
      ? "Checking development environment"
      : "Starting…";

  const showGpuCard =
    (screenState === "ready" || screenState === "cpu-fallback") &&
    hardwareDevices.length > 0;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "var(--bg0)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        overflow: "auto",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 480,
          display: "flex",
          flexDirection: "column",
          gap: 24,
        }}
      >
        {/* Logo / header */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 24,
              fontWeight: 700,
              color: "var(--green)",
              letterSpacing: "0.04em",
            }}
          >
            SR TUNER
          </span>
          <span style={{ fontSize: 12, color: "var(--muted)" }}>
            {header}
          </span>
        </div>

        {/* Animated steps card */}
        <div
          style={{
            background: "var(--bg1)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            padding: "16px 20px",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {steps.map((step, i) => (
            <div key={step.id}>
              {i > 0 && (
                <div
                  style={{ height: 1, background: "var(--border)", margin: "2px 0" }}
                />
              )}
              <StepIndicator step={step} />
            </div>
          ))}
        </div>

        {/* GPU card — shown once hardware.info arrives */}
        {showGpuCard && <GpuCard devices={hardwareDevices} />}

        {/* State-based action area */}
        {screenState === "ready" && (
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <ActionBtn
              label="Get Started →"
              variant="primary"
              onClick={onComplete}
            />
          </div>
        )}

        {screenState === "cpu-fallback" && (
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
            }}
          >
            <span style={{ fontSize: 11, color: "#f5a623" }}>
              No GPU detected — CPU training is supported but slow.
            </span>
            <ActionBtn
              label="Continue Anyway →"
              variant="ghost"
              onClick={onComplete}
            />
          </div>
        )}

        {screenState === "incompatible" && (
          <IncompatiblePanel
            errorCode={errorCode}
            errorMessage={errorMessage}
            onRetry={handleRetry}
            onCpuFallback={
              errorCode === "CUDA_NOT_FOUND" || errorCode === "ROCM_NOT_FOUND"
                ? () => {
                    setHardwareDevices([
                      {
                        id: "cpu",
                        name: "CPU",
                        vram_gb: null,
                        type: "cpu",
                      },
                    ]);
                    setScreenState("cpu-fallback");
                  }
                : errorCode === "DEV_DEP_MISSING"
                ? () => {
                    // Skip dep check — start sidecar so dev can access the UI.
                    // Training will fail at runtime if deps are truly missing.
                    setSteps((prev) =>
                      prev.map((s) =>
                        s.id === "sidecar" ? { ...s, status: "running" } : s
                      )
                    );
                    setScreenState("checking");
                    setupSidecarListener((id, patch) =>
                      setSteps((prev) =>
                        prev.map((s) => (s.id === id ? { ...s, ...patch } : s))
                      )
                    ).then((unlisten) => {
                      unlistenRef.current = unlisten;
                      timeoutRef.current = setTimeout(() => {
                        unlisten();
                        setSteps((prev) =>
                          prev.map((s) =>
                            s.id === "sidecar" || s.id === "gpu"
                              ? { ...s, status: "error" }
                              : s
                          )
                        );
                        setScreenState("timeout");
                      }, SIDECAR_READY_TIMEOUT_MS);
                      startSidecar().catch((e) => {
                        if (timeoutRef.current) clearTimeout(timeoutRef.current);
                        unlisten();
                        setErrorCode("SIDECAR_SPAWN_FAILED");
                        setErrorMessage(String(e));
                        setScreenState("timeout");
                      });
                    });
                  }
                : undefined
            }
            fallbackLabel={
              errorCode === "DEV_DEP_MISSING" ? "Continue anyway →" : undefined
            }
          />
        )}

        {screenState === "timeout" && (
          <TimeoutPanel onRetry={handleRetry} />
        )}

        {/* §15.6 / §19.8 — GPU variant download panel */}
        {screenState === "gpu-download" && (
          <div
            style={{
              background: "var(--bg1)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              padding: "16px 20px",
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>
              {gpuVendor === "nvidia"
                ? "Downloading NVIDIA (CUDA) GPU support…"
                : gpuVendor === "amd"
                ? "Downloading AMD (ROCm) GPU support…"
                : "No GPU detected — continuing with CPU mode"}
            </div>

            {gpuVendor !== "cpu" && (
              <>
                <div
                  style={{
                    height: 6,
                    background: "var(--bg2)",
                    borderRadius: 3,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width:
                        downloadBytesTotal != null && downloadBytesTotal > 0
                          ? `${Math.min(100, (downloadBytesDone / downloadBytesTotal) * 100)}%`
                          : "30%",
                      background: "var(--green)",
                      borderRadius: 3,
                      transition: "width 0.3s ease",
                      animation: downloadBytesTotal == null ? "indeterminate 1.5s ease-in-out infinite" : "none",
                    }}
                  />
                </div>
                <span style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
                  {downloadBytesTotal != null
                    ? `${(downloadBytesDone / 1024 / 1024).toFixed(1)} / ${(downloadBytesTotal / 1024 / 1024).toFixed(1)} MB`
                    : "Downloading…"}
                </span>

                <button
                  onClick={() => {
                    setScreenState("cpu-fallback");
                    setHardwareDevices([{ id: "cpu", name: "CPU", vram_gb: null, type: "cpu" }]);
                  }}
                  style={{
                    alignSelf: "flex-start",
                    background: "none",
                    border: "none",
                    color: "var(--dim)",
                    fontSize: 11,
                    cursor: "pointer",
                    padding: 0,
                    textDecoration: "underline",
                  }}
                >
                  Use CPU fallback instead
                </button>
              </>
            )}

            {gpuVendor === "cpu" && (
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <ActionBtn
                  label="Continue with CPU →"
                  variant="ghost"
                  onClick={() => {
                    setHardwareDevices([{ id: "cpu", name: "CPU", vram_gb: null, type: "cpu" }]);
                    setScreenState("cpu-fallback");
                  }}
                />
              </div>
            )}

            {downloadError && (
              <div
                style={{
                  background: "color-mix(in srgb, var(--red) 10%, var(--bg1))",
                  border: "1px solid var(--red)",
                  borderRadius: 4,
                  padding: "8px 12px",
                  fontSize: 11,
                  color: "var(--red)",
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Download failed</div>
                <div style={{ fontFamily: "var(--font-mono)", wordBreak: "break-all" }}>
                  {downloadError}
                </div>
                <button
                  onClick={() => {
                    setHardwareDevices([{ id: "cpu", name: "CPU", vram_gb: null, type: "cpu" }]);
                    setScreenState("cpu-fallback");
                  }}
                  style={{
                    marginTop: 8,
                    background: "none",
                    border: "none",
                    color: "var(--red)",
                    fontSize: 11,
                    cursor: "pointer",
                    padding: 0,
                    textDecoration: "underline",
                  }}
                >
                  Continue with CPU fallback
                </button>
              </div>
            )}

            <style>{`
              @keyframes indeterminate {
                0%   { transform: translateX(-100%); width: 40%; }
                100% { transform: translateX(250%);  width: 40%; }
              }
            `}</style>
          </div>
        )}

        {screenState === "checking" && steps.length > 0 && (
          <span
            style={{
              fontSize: 11,
              color: "var(--dim)",
              textAlign: "center",
            }}
          >
            This may take a moment on first launch…
          </span>
        )}
      </div>
    </div>
  );
}

// ── Exported runtime error dialogs (15.3, 15.4, 15.5) ─────────────────────
// These are used from ipc.ts or screen-level handlers when errors arrive
// outside of the onboarding flow.

// ── Shared overlay ─────────────────────────────────────────────────────────

interface ErrorOverlayProps {
  title: string;
  detail: string;
  suggestions?: string[];
  actions?: {
    label: string;
    onClick: () => void;
    variant?: "primary" | "ghost";
  }[];
  onClose: () => void;
}

function ErrorOverlay({
  title,
  detail,
  suggestions,
  actions,
  onClose,
}: ErrorOverlayProps) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="err-overlay-title"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 10000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0,0,0,0.6)",
        backdropFilter: "blur(2px)",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          background: "var(--bg2)",
          border: "1px solid var(--border2)",
          borderTop: "3px solid var(--red)",
          borderRadius: "var(--radius-lg)",
          padding: "24px 28px",
          width: 480,
          maxWidth: "90vw",
          boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
        }}
      >
        <h2
          id="err-overlay-title"
          style={{
            margin: "0 0 10px",
            color: "var(--red)",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          {title}
        </h2>
        <p
          style={{
            margin: "0 0 16px",
            color: "var(--muted)",
            fontSize: 12,
            fontFamily: "var(--font-mono)",
            lineHeight: 1.6,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {detail}
        </p>
        {suggestions && suggestions.length > 0 && (
          <div
            style={{
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              padding: "10px 14px",
              marginBottom: 20,
            }}
          >
            <p
              style={{
                margin: "0 0 6px",
                fontSize: 11,
                fontWeight: 600,
                color: "var(--muted)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              Suggestions
            </p>
            <ul style={{ margin: 0, padding: "0 0 0 16px" }}>
              {suggestions.map((s, i) => (
                <li
                  key={i}
                  style={{
                    fontSize: 12,
                    color: "var(--text)",
                    lineHeight: 1.6,
                  }}
                >
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <ActionBtn label="Dismiss" variant="ghost" onClick={onClose} />
          {actions?.map((a, i) => (
            <ActionBtn
              key={i}
              label={a.label}
              variant={a.variant ?? "ghost"}
              onClick={a.onClick}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// 15.3 — OOM error dialog
export interface OomErrorDialogProps {
  open: boolean;
  onClose: () => void;
}

export function OomErrorDialog({ open, onClose }: OomErrorDialogProps) {
  if (!open) return null;
  return (
    <ErrorOverlay
      title="Out of Memory (OOM)"
      detail="The GPU ran out of VRAM during training. The run has been paused."
      suggestions={[
        "Reduce batch size (try halving it first)",
        "Reduce patch size (e.g. 128 → 64)",
        "Enable FP16 mixed precision in Training Setup",
        "Switch to a smaller architecture (e.g. SRCNN instead of Real-ESRGAN)",
      ]}
      onClose={onClose}
    />
  );
}

// 15.4 — CUDA / ROCm error dialog
export interface CudaRocmErrorDialogProps {
  open: boolean;
  code: string;
  message: string;
  onRetry: () => void;
  onClose: () => void;
}

export function CudaRocmErrorDialog({
  open,
  code,
  message,
  onRetry,
  onClose,
}: CudaRocmErrorDialogProps) {
  if (!open) return null;
  const isCuda = code === "CUDA_NOT_FOUND";
  return (
    <ErrorOverlay
      title={isCuda ? "CUDA Not Found" : "ROCm Not Found"}
      detail={message}
      suggestions={
        isCuda
          ? [
              "Install NVIDIA drivers ≥ 525 from nvidia.com",
              "Verify: nvidia-smi should list your GPU",
              "Reinstall PyTorch with CUDA support",
            ]
          : [
              "Install ROCm 6.x from rocm.docs.amd.com",
              "Ensure your user is in the 'render' and 'video' groups",
              "Reinstall PyTorch with ROCm support",
            ]
      }
      actions={[{ label: "Retry", onClick: onRetry, variant: "primary" }]}
      onClose={onClose}
    />
  );
}

// 15.5 — Invalid path error dialog
export interface InvalidPathErrorDialogProps {
  open: boolean;
  path: string;
  onClose: () => void;
}

export function InvalidPathErrorDialog({
  open,
  path,
  onClose,
}: InvalidPathErrorDialogProps) {
  if (!open) return null;
  return (
    <ErrorOverlay
      title="Invalid Path"
      detail={`The configured path does not exist or is not accessible:\n${path}`}
      suggestions={[
        "Verify that the path exists on disk",
        "Ensure you have read permissions for this directory",
        "Update the path in Dataset Setup or Model Config",
      ]}
      onClose={onClose}
    />
  );
}
