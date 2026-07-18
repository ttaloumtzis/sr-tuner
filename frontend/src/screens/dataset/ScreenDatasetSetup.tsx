// §9 Dataset Setup Screen
// Tasks: 9.1–9.15

import { useState, useEffect, useRef } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { PathInput } from "../../components/ui/PathInput";
import { PBar } from "../../components/ui/PBar";
import { useDatasetStore, type ValidationStrategy } from "../../store/datasetStore";
import { useModelStore } from "../../store/modelStore";
import { useProjectStore } from "../../store/projectStore";
import { validateNamingPattern, previewFilename } from "../../lib/namingPattern";

// ── §9.2 Dataset type selector ────────────────────────────────────────────

type DatasetMode = "image_folder" | "video_extract" | "on_the_fly";

interface TypeCardProps {
  id: DatasetMode;
  label: string;
  description: string;
  active: boolean;
  onClick: () => void;
}

function TypeCard({ label, description, active, onClick }: TypeCardProps) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: active ? "var(--greenDim)" : hovered ? "var(--bg2)" : "var(--bg1)",
        border: `1px solid ${active ? "var(--green)" : hovered ? "var(--muted)" : "var(--border)"}`,
        borderRadius: "var(--radius-md)",
        padding: "12px 14px",
        cursor: "pointer",
        transition: "var(--transition-fast)",
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
    >
      <span style={{ fontSize: 12, fontWeight: 600, color: active ? "var(--green)" : "var(--text)" }}>
        {label}
      </span>
      <span style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>
        {description}
      </span>
    </div>
  );
}

// ── §9.3a Scale + Naming inline bar ──────────────────────────────────────

interface ScaleNamingBarProps {
  scale: number;
  onScale: (s: number) => void;
  namingPattern: string;
  onNamingPattern: (p: string) => void;
}

function ScaleNamingBar({ scale, onScale, namingPattern, onNamingPattern }: ScaleNamingBarProps) {
  const presets = [2, 4, 8];
  const [customVal, setCustomVal] = useState(presets.includes(scale) ? "" : String(scale));
  const [customActive, setCustomActive] = useState(!presets.includes(scale));
  const patternError = validateNamingPattern(namingPattern);
  const isValid = !patternError;
  const preview = isValid ? previewFilename(namingPattern) : "invalid";
  const hrRes = 480;
  const lrRes = Math.round(hrRes / scale);

  const handleCustomChange = (v: string) => {
    setCustomVal(v);
    const n = parseInt(v, 10);
    if (!isNaN(n) && n > 0) onScale(n);
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "7px 10px",
        background: "var(--bg2)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
        flexWrap: "wrap",
      }}
    >
      <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>
        Scale
      </span>
      {presets.map((p) => (
        <button
          key={p}
          onClick={() => { onScale(p); setCustomActive(false); setCustomVal(""); }}
          style={{
            background: scale === p && !customActive ? "var(--green)" : "var(--bg3)",
            border: `1px solid ${scale === p && !customActive ? "var(--green)" : "var(--border)"}`,
            color: scale === p && !customActive ? "#0d0f11" : "var(--muted)",
            fontSize: 11,
            fontWeight: 600,
            padding: "2px 9px",
            borderRadius: "var(--radius-sm)",
            cursor: "pointer",
            transition: "var(--transition-fast)",
          }}
        >
          ×{p}
        </button>
      ))}
      <input
        value={customVal}
        onChange={(e) => { setCustomActive(true); handleCustomChange(e.target.value); }}
        onFocus={() => setCustomActive(true)}
        placeholder="custom"
        style={{
          width: 52,
          background: customActive ? "var(--greenDim)" : "var(--bg3)",
          border: `1px solid ${customActive ? "var(--green)" : "var(--border)"}`,
          color: "var(--text)",
          fontSize: 11,
          padding: "2px 7px",
          borderRadius: "var(--radius-sm)",
          outline: "none",
          fontFamily: "var(--font-mono)",
        }}
      />
      <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
        {hrRes}px → {lrRes}px
      </span>

      <div style={{ width: 1, height: 16, background: "var(--border)" }} />

      <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>
        Naming
      </span>
      <input
        value={namingPattern}
        onChange={(e) => onNamingPattern(e.target.value)}
        placeholder="%06d"
        style={{
          width: 68,
          background: patternError ? "color-mix(in srgb, var(--red) 15%, var(--bg3))" : "var(--bg3)",
          border: `1px solid ${patternError ? "var(--red)" : "var(--border)"}`,
          color: "var(--text)",
          fontSize: 11,
          padding: "2px 7px",
          borderRadius: "var(--radius-sm)",
          outline: "none",
          fontFamily: "var(--font-mono)",
        }}
      />
      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: isValid ? "var(--green)" : "var(--red)" }}>
        → {preview}
      </span>
      {patternError && (
        <span style={{ fontSize: 10, color: "var(--red)" }}>{patternError}</span>
      )}
    </div>
  );
}

// ── §9.3b Kernel selector pill row ───────────────────────────────────────

type KernelOption = "bicubic" | "bilinear" | "real-world";

interface KernelSelectorProps {
  value: KernelOption;
  onChange: (k: KernelOption) => void;
  disabled?: boolean;
  disabledReason?: string;
}

function KernelSelector({ value, onChange, disabled, disabledReason }: KernelSelectorProps) {
  const options: { id: KernelOption; label: string }[] = [
    { id: "bicubic", label: "Bicubic" },
    { id: "bilinear", label: "Bilinear" },
    { id: "real-world", label: "Real-World" },
  ];
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 8 }}
      title={disabled ? (disabledReason ?? "Disabled") : undefined}
    >
      <span style={{ fontSize: 10, color: disabled ? "var(--dim)" : "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>
        Kernel
      </span>
      {options.map((opt) => (
        <button
          key={opt.id}
          onClick={() => !disabled && onChange(opt.id)}
          disabled={disabled}
          style={{
            background: disabled ? "var(--bg2)" : value === opt.id ? "var(--green)" : "var(--bg3)",
            border: `1px solid ${disabled ? "var(--border)" : value === opt.id ? "var(--green)" : "var(--border)"}`,
            color: disabled ? "var(--dim)" : value === opt.id ? "#0d0f11" : "var(--muted)",
            fontSize: 11,
            fontWeight: !disabled && value === opt.id ? 600 : 400,
            padding: "3px 11px",
            borderRadius: 10,
            cursor: disabled ? "not-allowed" : "pointer",
            transition: "var(--transition-fast)",
            opacity: disabled ? 0.45 : 1,
          }}
        >
          {opt.label}
        </button>
      ))}
      {disabled && disabledReason && (
        <span style={{ fontSize: 10, color: "var(--amber)", fontStyle: "italic" }}>
          {disabledReason}
        </span>
      )}
    </div>
  );
}

// ── §9.4 Pre-extracted folders mode ──────────────────────────────────────

interface LrGenerationProgress {
  framesTotal: number;
  framesDone: number;
  done: boolean;
}

interface PreExtractedModeProps {
  hrPath: string;
  lrPath: string;
  onHrPath: (p: string) => void;
  onLrPath: (p: string) => void;
  lrProgress: LrGenerationProgress | null;
}

function PreExtractedMode({ hrPath, lrPath, onHrPath, onLrPath, lrProgress }: PreExtractedModeProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>
          HR Folder
        </label>
        <PathInput value={hrPath} onChange={onHrPath} browseTitle="Select HR image folder" mono />
        <span style={{ fontSize: 10, color: "var(--dim)" }}>
          High-resolution training images (PNG, JPEG, WebP)
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>
          LR Folder
        </label>
        <PathInput value={lrPath} onChange={onLrPath} browseTitle="Select LR image folder" mono />
        <span style={{ fontSize: 10, color: "var(--dim)" }}>
          Leave empty to auto-generate LR from HR using selected kernel
        </span>
      </div>
      {lrProgress && (
        <div
          style={{
            background: "var(--bg2)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 10px",
            display: "flex",
            flexDirection: "column",
            gap: 5,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 10, color: "var(--muted)" }}>
              Generating LR: hr/ → lr/
            </span>
            {lrProgress.done && (
              <span style={{ fontSize: 10, color: "var(--green)", fontWeight: 600 }}>✓ complete</span>
            )}
          </div>
          <PBar value={lrProgress.framesDone} max={lrProgress.framesTotal || 1} color="var(--amber)" />
          <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
            {lrProgress.framesDone} / {lrProgress.framesTotal} frames
          </span>
        </div>
      )}
    </div>
  );
}

// ── §9.5 Video extraction mode ────────────────────────────────────────────

interface VideoFile {
  name: string;
  path: string;
  status: "pending" | "extracting" | "done";
  duration?: string;
  resolution?: string;
  sizeDisplay?: string;
}

interface VideoExtractionModeProps {
  videoFiles: VideoFile[];
  onAddFiles: (paths: string[]) => void;
  extractionProgress: { frames_done: number; frames_total: number; fps: number; eta_sec: number } | null;
  lrScalingProgress: { done: number; total: number } | null;
}

function VideoExtractionMode({ videoFiles, extractionProgress, lrScalingProgress }: VideoExtractionModeProps) {
  const [dragOver, setDragOver] = useState(false);
  const dragOverRef = useRef(false);
  const extractFraction = extractionProgress
    ? extractionProgress.frames_done / Math.max(1, extractionProgress.frames_total)
    : 0;
  const lrFraction = lrScalingProgress
    ? lrScalingProgress.done / Math.max(1, lrScalingProgress.total)
    : 0;

  useEffect(() => {
    dragOverRef.current = dragOver;
  }, [dragOver]);

  // Drag-drop handling removed — was previously wired via tauri listen

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); }}
        style={{
          border: `2px dashed ${dragOver ? "var(--green)" : "var(--border)"}`,
          borderRadius: "var(--radius-md)",
          padding: "20px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 6,
          background: dragOver ? "var(--greenDim)" : "var(--bg2)",
          transition: "var(--transition-fast)",
          cursor: "pointer",
        }}
      >
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Drop video files here</span>
        <span style={{ fontSize: 10, color: "var(--dim)" }}>Supported: MKV, MP4, AVI, MOV</span>
      </div>

      {videoFiles.length > 0 && (
        <>
          <span style={{ fontSize: 10, color: "var(--muted)" }}>
            Frames will be saved to project dataset folder
          </span>

          <div style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 80px",
              padding: "5px 10px",
              background: "var(--bg2)",
              borderBottom: "1px solid var(--border)",
            }}
          >
            {["Filename", "Status"].map((h) => (
              <span key={h} style={{ fontSize: 9, color: "var(--dim)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.8 }}>
                {h}
              </span>
            ))}
          </div>
          {videoFiles.map((f, i) => (
            <div
              key={i}
              style={{
                display: "grid",
gridTemplateColumns: "1fr 80px",
                padding: "6px 10px",
                borderBottom: i < videoFiles.length - 1 ? "1px solid var(--border)" : undefined,
                alignItems: "center",
              }}
            >
              <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {f.name}
              </span>
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  color: f.status === "done" ? "var(--green)" : f.status === "extracting" ? "var(--amber)" : "var(--dim)",
                  background: f.status === "done" ? "var(--greenDim)" : f.status === "extracting" ? "color-mix(in srgb, var(--amber) 15%, transparent)" : "var(--bg3)",
                  border: `1px solid ${f.status === "done" ? "var(--green)" : f.status === "extracting" ? "var(--amber)" : "var(--border)"}`,
                  borderRadius: 8,
                  padding: "2px 7px",
                  display: "inline-block",
                }}
              >
                {f.status}
              </span>
            </div>
          ))}
        </div>
        </>
      )}

      {(extractionProgress || lrScalingProgress) && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontSize: 10, color: "var(--amber)" }}>Frame extraction</span>
              {extractionProgress && (
                <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
                  {extractionProgress.frames_done}/{extractionProgress.frames_total} · {extractionProgress.fps.toFixed(1)} fr/s · ETA {extractionProgress.eta_sec}s
                </span>
              )}
            </div>
            <PBar value={extractFraction * 100} color="var(--amber)" />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontSize: 10, color: "#4d9ef5" }}>LR downscaling</span>
              {lrScalingProgress && (
                <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
                  {lrScalingProgress.done}/{lrScalingProgress.total}
                  {lrScalingProgress.done === lrScalingProgress.total && lrScalingProgress.total > 0 ? " ✓" : ""}
                </span>
              )}
            </div>
            <PBar value={lrFraction * 100} color="#4d9ef5" />
          </div>
        </div>
      )}
    </div>
  );
}

// ── §9.6 On-the-fly video mode ────────────────────────────────────────────

interface OnTheFlyModeProps {
  hrPath: string;
  lrPath: string;
  onHrPath: (p: string) => void;
  onLrPath: (p: string) => void;
  encodingProgress: { frames_done: number; frames_total: number; fps: number; eta_sec: number } | null;
  sync: "frame" | "time";
  onSync: (s: "frame" | "time") => void;
  onPreprocess: () => void;
  onSkip: () => void;
}

function OnTheFlyMode({ hrPath, lrPath, onHrPath, onLrPath, encodingProgress, sync, onSync, onPreprocess, onSkip }: OnTheFlyModeProps) {
  const [codec, setCodec] = useState<"libx264" | "libx265" | "copy">("libx264");
  const [dragOver, setDragOver] = useState(false);
  const encodingFilename = hrPath ? hrPath.split("/").pop() ?? hrPath : "";
  const encFraction = encodingProgress
    ? encodingProgress.frames_done / Math.max(1, encodingProgress.frames_total)
    : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); }}
        style={{
          border: `2px dashed ${dragOver ? "var(--green)" : "var(--border)"}`,
          borderRadius: "var(--radius-md)",
          padding: "16px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 6,
          background: dragOver ? "var(--greenDim)" : "var(--bg2)",
          transition: "var(--transition-fast)",
        }}
      >
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Drop HR video files here</span>
        <span style={{ fontSize: 10, color: "var(--dim)" }}>MKV, MP4, AVI, MOV</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>
            HR Video
          </label>
          <PathInput value={hrPath} onChange={onHrPath} browseTitle="Select HR video" mono />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>
            LR Video (generated)
          </label>
          <PathInput value={lrPath} onChange={onLrPath} browseTitle="Select LR video output path" mono />
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
            <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1 }}>
              Downscale Codec
            </label>
            <select
              value={codec}
              onChange={(e) => setCodec(e.target.value as typeof codec)}
              style={{ background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11, padding: "5px 8px", borderRadius: "var(--radius-sm)", outline: "none" }}
            >
              <option value="libx264">libx264</option>
              <option value="libx265">libx265</option>
              <option value="copy">copy</option>
            </select>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
            <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1 }}>
              Sync Mode
            </label>
            <select
              value={sync}
              onChange={(e) => onSync(e.target.value as "frame" | "time")}
              style={{ background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11, padding: "5px 8px", borderRadius: "var(--radius-sm)", outline: "none" }}
            >
              <option value="frame">Frame</option>
              <option value="time">Time</option>
            </select>
          </div>
        </div>
        <div
          style={{
            background: "color-mix(in srgb, var(--amber) 10%, var(--bg2))",
            border: "1px solid color-mix(in srgb, var(--amber) 40%, transparent)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 10px",
            fontSize: 10,
            color: "var(--amber)",
            lineHeight: 1.5,
          }}
        >
          No frames extracted — ~90% lower disk usage vs frames mode
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Btn variant="solid" color="var(--amber)" onClick={onPreprocess}>
            Pre-process LR Video
          </Btn>
          <Btn onClick={onSkip}>
            Skip (LR video exists)
          </Btn>
        </div>
      </div>

      {/* §9.6a LR video encoding progress */}
      {encodingProgress && (
        <div
          style={{
            background: "var(--bg2)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 10px",
            display: "flex",
            flexDirection: "column",
            gap: 5,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: 10, color: "var(--amber)" }}>
              LR video encoding — {encodingFilename}
            </span>
            <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
              {encodingProgress.fps.toFixed(1)} fr/s · ETA {encodingProgress.eta_sec}s
            </span>
          </div>
          <PBar value={encFraction * 100} color="var(--amber)" />
          <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
            {encodingProgress.frames_done} / {encodingProgress.frames_total} frames
          </span>
        </div>
      )}
    </div>
  );
}

// ── §9.10–9.15 Validation Dataset section ─────────────────────────────────

interface ValidationSectionProps {
  strategy: ValidationStrategy;
  splitRatio: number;
  validationPath: string | null;
  splitResult: { trainingCount: number; validationCount: number } | null;
  onStrategy: (s: ValidationStrategy) => void;
  onSplitRatio: (r: number) => void;
  onValidationPath: (p: string) => void;
  onRequestSplit: () => void;
}

function ValidationSection({
  strategy,
  splitRatio,
  validationPath,
  splitResult,
  onStrategy,
  onSplitRatio,
  onValidationPath,
  onRequestSplit,
}: ValidationSectionProps) {
  const pillStyle = (active: boolean, color = "var(--green)"): React.CSSProperties => ({
    background: active ? color : "var(--bg3)",
    border: `1px solid ${active ? color : "var(--border)"}`,
    color: active ? "#0d0f11" : "var(--muted)",
    fontSize: 11,
    fontWeight: active ? 600 : 400,
    padding: "3px 12px",
    borderRadius: 10,
    cursor: "pointer",
    transition: "var(--transition-fast)",
  });

  return (
    <Panel title="Validation Dataset">
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <button style={pillStyle(strategy === "auto_split")} onClick={() => onStrategy("auto_split")}>
            Auto-split from training data
          </button>
          <button style={pillStyle(strategy === "separate_folder")} onClick={() => onStrategy("separate_folder")}>
            Use separate folder
          </button>
          <button style={pillStyle(strategy === "none", "var(--amber)")} onClick={() => onStrategy("none")}>
            Skip validation (not recommended)
          </button>
        </div>

        {strategy === "auto_split" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 10, color: "var(--muted)", minWidth: 60 }}>Split ratio</span>
              <input
                type="range"
                min={5}
                max={30}
                value={splitRatio}
                onChange={(e) => onSplitRatio(Number(e.target.value))}
                style={{ flex: 1, accentColor: "var(--green)" }}
              />
              <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", minWidth: 32, textAlign: "right" }}>
                {splitRatio}%
              </span>
              <Btn small onClick={onRequestSplit}>Apply split</Btn>
            </div>
            {splitResult && (
              <div style={{ display: "flex", gap: 16 }}>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  Training:{" "}
                  <span style={{ color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    {splitResult.trainingCount}
                  </span>
                </span>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  Validation:{" "}
                  <span style={{ color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    {splitResult.validationCount}
                  </span>
                </span>
              </div>
            )}
          </div>
        )}

        {strategy === "separate_folder" && (
          <PathInput value={validationPath ?? ""} onChange={onValidationPath} browseTitle="Select validation folder" mono />
        )}

        {strategy === "none" && (
          <div
            style={{
              background: "color-mix(in srgb, var(--amber) 10%, var(--bg2))",
              border: "1px solid color-mix(in srgb, var(--amber) 40%, transparent)",
              borderRadius: "var(--radius-sm)",
              padding: "8px 10px",
              fontSize: 10,
              color: "var(--amber)",
              lineHeight: 1.5,
            }}
          >
            Skipping validation is not recommended. PSNR/SSIM will not be tracked during training.
          </div>
        )}
      </div>
    </Panel>
  );
}

// ── §9.7 Summary right panel ──────────────────────────────────────────────

interface SummaryPanelProps {
  type: DatasetMode;
  scale: number;
  kernel: string;
  hrPath: string;
  lrPath: string;
}

function SummaryPanel({ type, scale, kernel, hrPath, lrPath }: SummaryPanelProps) {
  const modeLabel: Record<DatasetMode, string> = {
    image_folder: "Pre-extracted",
    video_extract: "Video Extract",
    on_the_fly: "On-the-fly",
  };
  const hrRes = 480;
  const lrRes = Math.round(hrRes / scale);

  const rows: { label: string; value: string }[] = [
    { label: "Mode", value: modeLabel[type] },
    { label: "Scale", value: `×${scale}` },
    { label: "HR res", value: `${hrRes}px` },
    { label: "LR res", value: `${lrRes}px` },
    { label: "Kernel", value: kernel },
    { label: "Output dir", value: hrPath ? hrPath.split("/").slice(0, -1).join("/") || hrPath : "—" },
  ];

  if (type === "video_extract" || type === "on_the_fly") {
    rows.push({ label: "LR output", value: lrPath || "—" });
  }

  return (
    <div style={{ width: 220, flexShrink: 0 }}>
      <Panel title="Dataset Summary">
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          {rows.map(({ label, value }) => (
            <div key={label} style={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <span style={{ fontSize: 9, color: "var(--dim)", textTransform: "uppercase", letterSpacing: 0.8, fontWeight: 600 }}>
                {label}
              </span>
              <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", wordBreak: "break-all" }}>
                {value}
              </span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

// ── §9.1 Main screen ──────────────────────────────────────────────────────

export function ScreenDatasetSetup() {
  const {
    type,
    scale,
    kernel,
    hrPath,
    lrPath,
    namingPattern,
    extractionProgress,
    strategy,
    validationPath,
    setType,
    setScale,
    setKernel,
    setHrPath,
    setLrPath,
    setNamingPattern,
    setStrategy,
    setValidationPath,
  } = useDatasetStore();

  const randomDegradation = useModelStore((s) => s.augmentations.random_degradation);

  const [splitRatio, setSplitRatio] = useState(20);
  const [splitResult] = useState<{ trainingCount: number; validationCount: number } | null>(null);
  const [videoFiles, setVideoFiles] = useState<VideoFile[]>([]);
  const [lrGenerationProgress] = useState<LrGenerationProgress | null>(null);
  const [encodingProgress, setEncodingProgress] = useState<{ frames_done: number; frames_total: number; fps: number; eta_sec: number } | null>(null);
  const [lrScalingProgress] = useState<{ done: number; total: number } | null>(null);
  const [onTheFlySync, setOnTheFlySync] = useState<"frame" | "time">("frame");

  const mode = type as DatasetMode;

  // Sidecar IPC listener removed — dataset progress is now polled via the API

  // §9.8 dataset.create IPC for video extraction
  const handleStartExtraction = async () => {
    if (validateNamingPattern(namingPattern)) return;
    const project = useProjectStore.getState().project;
    if (!project) return;
    const projectDir = project.filePath.replace(/\.srproj$/, "");
    const hrOut = projectDir + "/dataset/hr";
    const lrOut = projectDir + "/dataset/lr";
    setHrPath(hrOut);
    setLrPath(lrOut);
    // TODO: replace with api call
  };

  // §9.12 dataset.split.request IPC
  const handleRequestSplit = async () => {
    // TODO: replace with api call
  };

  const handlePreprocess = async () => {
    if (!hrPath) return;
    if (!lrPath) return;
    if (hrPath === lrPath) return;
    const hrOut = hrPath.replace(/\.[^.]+$/, "") + "_hr";
    if (hrOut === lrPath) return;
    // TODO: replace with api call
  };

  const typeCards: { id: DatasetMode; label: string; description: string }[] = [
    { id: "image_folder", label: "Pre-extracted", description: "Use existing HR/LR image folders. Fastest startup." },
    { id: "video_extract", label: "Video Extract", description: "Extract frames from video files. Best for video sources." },
    { id: "on_the_fly", label: "On-the-fly", description: "Decode video during training. ~90% less disk usage." },
  ];

  const showKernel = mode === "image_folder" || mode === "video_extract";

  return (
    <div style={{ display: "flex", gap: 12, padding: 16, height: "100%", overflow: "auto", boxSizing: "border-box" }}>
      {/* Main area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
        {/* §9.2 Type selector — 3-col grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
          {typeCards.map((c) => (
            <TypeCard
              key={c.id}
              id={c.id}
              label={c.label}
              description={c.description}
              active={mode === c.id}
              onClick={() => setType(c.id)}
            />
          ))}
        </div>

        {/* §9.3a Scale + Naming bar */}
        <ScaleNamingBar
          scale={scale}
          onScale={setScale}
          namingPattern={namingPattern}
          onNamingPattern={setNamingPattern}
        />

        {/* §9.3b Kernel selector — hidden in on_the_fly mode */}
        {showKernel && (
          <div style={{ padding: "7px 10px", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)" }}>
            <KernelSelector
              value={kernel as KernelOption}
              onChange={setKernel}
              disabled={randomDegradation}
              disabledReason="Kernel selection is overridden by Real-World Degradation"
            />
          </div>
        )}

        {/* Mode panels */}
        <Panel title={mode === "image_folder" ? "Training Dataset" : mode === "video_extract" ? "Video Files" : "On-the-fly Video"}>
          {mode === "image_folder" && (
            <PreExtractedMode
              hrPath={hrPath}
              lrPath={lrPath}
              onHrPath={setHrPath}
              onLrPath={setLrPath}
              lrProgress={lrGenerationProgress}
            />
          )}
          {mode === "video_extract" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <VideoExtractionMode
                videoFiles={videoFiles}
                onAddFiles={(paths) =>
                  setVideoFiles((prev) => {
                    const existing = new Set(prev.map((f) => f.path));
                    const toAdd = paths.filter((p) => !existing.has(p));
                    return [
                      ...prev,
                      ...toAdd.map((p) => ({ name: p.split("/").pop() ?? p, path: p, status: "pending" as const })),
                    ];
                  })
                }
                extractionProgress={
                  extractionProgress
                    ? { frames_done: extractionProgress.framesDone, frames_total: extractionProgress.framesTotal, fps: extractionProgress.fps, eta_sec: extractionProgress.etaSec }
                    : null
                }
                lrScalingProgress={lrScalingProgress}
              />
              {videoFiles.length > 0 && (
                <Btn variant="solid" onClick={handleStartExtraction}>Start Extraction</Btn>
              )}
            </div>
          )}
          {mode === "on_the_fly" && (
            <OnTheFlyMode
              hrPath={hrPath}
              lrPath={lrPath}
              onHrPath={setHrPath}
              onLrPath={setLrPath}
              encodingProgress={encodingProgress}
              sync={onTheFlySync}
              onSync={setOnTheFlySync}
              onPreprocess={handlePreprocess}
              onSkip={() => setEncodingProgress(null)}
            />
          )}
        </Panel>

        {/* §9.10–9.15 Validation Dataset section */}
        <ValidationSection
          strategy={strategy}
          splitRatio={splitRatio}
          validationPath={validationPath}
          splitResult={splitResult}
          onStrategy={setStrategy}
          onSplitRatio={setSplitRatio}
          onValidationPath={setValidationPath}
          onRequestSplit={handleRequestSplit}
        />
      </div>

      {/* §9.7 Summary right panel — 220px */}
      <SummaryPanel type={mode} scale={scale} kernel={kernel} hrPath={hrPath} lrPath={lrPath} />
    </div>
  );
}
