// §14 Inference Screen
// Tasks: 14.1–14.13

import { useState, useRef, useCallback, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useCheckpointStore } from "../../store/checkpointStore";
import { useInferenceStore } from "../../store/inferenceStore";
import { useProjectStore } from "../../store/projectStore";
import { useUiStore } from "../../store/uiStore";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Field } from "../../components/ui/Field";
import { Toggle } from "../../components/ui/Toggle";
import { PathInput } from "../../components/ui/PathInput";
import { Dropdown } from "../../components/ui/Dropdown";
import { open } from "@tauri-apps/plugin-dialog";
import { PBar } from "../../components/ui/PBar";


// ── Helpers ────────────────────────────────────────────────────────────────

function basename(filePath: string): string {
  return filePath.split(/[\\/]/).pop() ?? filePath;
}

// ── Cross-hatch background ─────────────────────────────────────────────────

const CROSSHATCH_BG: React.CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(45deg, var(--bg2) 0, var(--bg2) 4px, transparent 0, transparent 50%), " +
    "repeating-linear-gradient(-45deg, var(--bg2) 0, var(--bg2) 4px, transparent 0, transparent 50%)",
  backgroundSize: "12px 12px",
  backgroundColor: "var(--bg1)",
};

// ── §14.2 / §14.3 Image drop zone ─────────────────────────────────────────

interface DropZoneProps {
  label: string;
  path: string | null;
  accent?: string;
  onSelect: (path: string) => void;
  onClear?: () => void;
  browseTitle?: string;
}

function DropZone({ label, path, accent = "var(--border)", onSelect, onClear, browseTitle }: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      const filePath = (file as File & { path?: string }).path;
      if (filePath) onSelect(filePath);
    }
  };

  return (
    <>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          border: `1.5px dashed ${accent}`,
          borderRadius: "var(--radius-sm)",
          padding: "10px 8px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 6,
          background: dragOver ? "var(--bg2)" : "transparent",
          transition: "background 0.15s",
          cursor: "default",
          minHeight: 56,
        }}
      >
        {path ? (
          <div
            style={{
              fontSize: 10,
              color: "var(--text)",
              fontFamily: "var(--font-mono)",
              textAlign: "center",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              width: "100%",
            }}
            title={path}
          >
            {basename(path)}
          </div>
        ) : (
          <span style={{ fontSize: 10, color: "var(--dim)" }}>{label}</span>
        )}
        <div style={{ display: "flex", gap: 4 }}>
          <Btn small onClick={async () => {
            const selected = await open({
              directory: false,
              multiple: false,
              title: browseTitle,
              defaultPath: path ?? undefined,
              filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"] }],
            });
            if (selected) onSelect(selected);
          }}>
            Browse…
          </Btn>
          {onClear && path && (
            <Btn small onClick={onClear}>
              Clear
            </Btn>
          )}
        </div>
      </div>
    </>
  );
}

// ── §14.4 Model / settings options ────────────────────────────────────────

const ARCH_OPTIONS = ["rrdb_esrgan", "swinir", "HAT", "EDSR"];

const SCALE_OPTIONS = [
  { value: "2", label: "2×" },
  { value: "4", label: "4×" },
  { value: "8", label: "8×" },
];

const FORMAT_OPTIONS = ["png", "jpeg", "webp", "tiff"];

const TILE_OPTIONS = [
  { value: "0", label: "No tiling" },
  { value: "128", label: "128 px" },
  { value: "256", label: "256 px" },
  { value: "512", label: "512 px" },
];

// ── §14.1 Left settings panel ─────────────────────────────────────────────

function SettingsPanel({ onRun }: { onRun: () => void }) {
  const store = useInferenceStore();
  const checkpointsByRun = useCheckpointStore((s) => s.checkpointsByRun);
  const displayedRunId = useUiStore((s) => s.displayedRunId);
  const project = useProjectStore((s) => s.project);

  const allCheckpoints = Object.values(checkpointsByRun).flat();
  const checkpointOptions = allCheckpoints.map((c) => ({
    value: c.path,
    label: `Ep ${c.epoch} — ${basename(c.filename)}`,
  }));
  if (checkpointOptions.length === 0) {
    checkpointOptions.push({ value: "", label: "No checkpoints available" });
  }

  // §14.14 [Gap M] — On mount, send checkpoint.list.request for the active run
  useEffect(() => {
    const runId = displayedRunId;
    const run = project?.runs.find((r) => r.run_id === runId);
    if (!run?.paths.checkpoint_dir) return;

    // TODO: replace with api call
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // §14.4: Auto-select preselected checkpoint path from §13.9b, then clear
  useEffect(() => {
    const pre = store.preselectedCheckpointPath;
    if (pre && store.checkpointPath !== pre) {
      store.setCheckpointPath(pre);
      store.setPreselectedCheckpointPath(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.preselectedCheckpointPath]);

  const outputDirError = store.outputDir ? null : "Select an output directory";

  const canRun =
    !!store.inputPath && !!store.checkpointPath && !store.isRunning && !outputDirError;

  return (
    <div
      style={{
        flex: 1, minWidth: 180, maxWidth: 300,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        padding: "10px 8px",
        borderRight: "1px solid var(--border)",
        overflowY: "auto",
      }}
    >
      {/* §14.2 Input panel */}
      <Panel title="Input Image" style={{ flexShrink: 0 }}>
        <DropZone
          label="Drop image here"
          path={store.inputPath}
          onSelect={store.setInputPath}
          browseTitle="Select Input Image"
        />
      </Panel>

      {/* §14.3 GT image panel */}
      <Panel title="Ground Truth (optional)" style={{ flexShrink: 0 }}>
        <DropZone
          label="Drop GT image here"
          path={store.gtPath}
          accent="var(--blue)"
          onSelect={store.setGtPath}
          onClear={() => store.setGtPath(null)}
          browseTitle="Select Ground Truth Image"
        />
        {!store.gtPath && (
          <div style={{ fontSize: 10, color: "var(--dim)", marginTop: 4 }}>
            Required for quality metrics
          </div>
        )}
      </Panel>

      {/* §14.4 Model panel */}
      <Panel title="Model" style={{ flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Field label="Checkpoint">
            <Dropdown
              value={store.checkpointPath ?? ""}
              options={checkpointOptions}
              onChange={store.setCheckpointPath}
              placeholder="Select checkpoint…"
            />
          </Field>
          <Field label="Architecture">
            <Dropdown
              value={store.architecture}
              options={ARCH_OPTIONS}
              onChange={store.setArchitecture}
            />
          </Field>
          <Field label="Scale">
            <Dropdown
              value={String(store.scaleFactor)}
              options={SCALE_OPTIONS}
              onChange={(v) => store.setScaleFactor(Number(v))}
            />
          </Field>
          <Field label="Tile Size">
            <Dropdown
              value={String(store.tileSize)}
              options={TILE_OPTIONS}
              onChange={(v) => store.setTileSize(Number(v))}
            />
          </Field>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span
              style={{
                fontSize: 10,
                color: "var(--muted)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              FP16
            </span>
            <Toggle on={store.fp16} onChange={() => store.setFp16(!store.fp16)} />
          </div>
        </div>
      </Panel>

      {/* §14.5 Output panel */}
      <Panel title="Output" style={{ flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Field label="Save Directory">
            <PathInput
              value={store.outputDir}
              onChange={store.setOutputDir}
              browseTitle="Select Output Directory"
              compact
            />
            {/* §20.11 — Inline validation error; not a toast */}
            {outputDirError && (
              <div
                style={{
                  fontSize: 10,
                  color: "var(--red)",
                  marginTop: 3,
                  fontFamily: "var(--font-sans)",
                }}
              >
                {outputDirError}
              </div>
            )}
          </Field>
          <Field label="Format">
            <Dropdown
              value={store.outputFormat}
              options={FORMAT_OPTIONS}
              onChange={(v) =>
                store.setOutputFormat(v as "png" | "jpeg" | "webp" | "tiff")
              }
            />
          </Field>
        </div>
      </Panel>

      {/* Run button */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
        <Btn variant="solid" full disabled={!canRun} onClick={onRun}>
          {store.isRunning ? "Running…" : "Run Inference"}
        </Btn>

        {/* §14.11 Tile progress bar */}
        {store.isRunning && store.tilesTotal > 1 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <PBar value={store.tilesDone} max={store.tilesTotal} />
            <span
              style={{
                fontSize: 9,
                color: "var(--dim)",
                textAlign: "center",
                fontFamily: "var(--font-mono)",
              }}
            >
              {store.tilesDone} / {store.tilesTotal} tiles
            </span>
          </div>
        )}
        {store.isRunning && store.tilesTotal <= 1 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <PBar value={0} max={1} />
            <span
              style={{
                fontSize: 9,
                color: "var(--dim)",
                textAlign: "center",
                fontFamily: "var(--font-mono)",
              }}
            >
              Processing…
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── §14.6 / §14.7 Before/After comparison panel ───────────────────────────

interface ComparisonPanelProps {
  splitterPct: number;
  onSplitterPctChange: (pct: number) => void;
}

function ComparisonPanel({ splitterPct, onSplitterPctChange }: ComparisonPanelProps) {
  const result = useInferenceStore((s) => s.result);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  // §14.12 Use convertFileSrc for preview paths
  const lrSrc =
    result?.success && result.preview_input_path
      ? convertFileSrc(result.preview_input_path)
      : null;
  const srSrc =
    result?.success && result.preview_output_path
      ? convertFileSrc(result.preview_output_path)
      : null;

  const lrLabel = result?.success
    ? `${result.input_resolution!.width}×${result.input_resolution!.height}`
    : null;
  const srLabel = result?.success
    ? `${result.output_resolution!.width}×${result.output_resolution!.height}`
    : null;

  // §14.7 Drag handlers
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = Math.max(2, Math.min(98, (x / rect.width) * 100));
      onSplitterPctChange(pct);
    },
    [onSplitterPctChange]
  );

  const handleMouseUp = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  const startDrag = () => {
    dragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <div
      ref={containerRef}
      style={{ flex: 1, position: "relative", overflow: "hidden", ...CROSSHATCH_BG }}
    >
      {/* Left side — LR image */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          right: `${100 - splitterPct}%`,
          overflow: "hidden",
        }}
      >
        {lrSrc ? (
          <img
            src={lrSrc}
            alt="Input (LR)"
            style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
            draggable={false}
          />
        ) : (
          <div style={{ width: "100%", height: "100%", ...CROSSHATCH_BG }} />
        )}
        {lrLabel && (
          <div
            style={{
              position: "absolute",
              bottom: 8,
              left: 8,
              background: "rgba(0,0,0,0.7)",
              color: "var(--text)",
              fontSize: 10,
              fontFamily: "var(--font-mono)",
              padding: "2px 6px",
              borderRadius: "var(--radius-sm)",
            }}
          >
            LR {lrLabel}
          </div>
        )}
      </div>

      {/* Right side — SR image */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          left: `${splitterPct}%`,
          overflow: "hidden",
        }}
      >
        {srSrc ? (
          <img
            src={srSrc}
            alt="Output (SR)"
            style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
            draggable={false}
          />
        ) : (
          <div style={{ width: "100%", height: "100%", ...CROSSHATCH_BG }} />
        )}
        {srLabel && (
          <div
            style={{
              position: "absolute",
              bottom: 8,
              right: 8,
              background: "rgba(0,0,0,0.7)",
              color: "var(--text)",
              fontSize: 10,
              fontFamily: "var(--font-mono)",
              padding: "2px 6px",
              borderRadius: "var(--radius-sm)",
            }}
          >
            SR {srLabel}
          </div>
        )}
      </div>

      {/* §14.7 Draggable splitter — 2px green line */}
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${splitterPct}%`,
          transform: "translateX(-50%)",
          width: 2,
          background: "var(--green)",
          zIndex: 10,
          cursor: "col-resize",
          pointerEvents: "none",
        }}
      />

      {/* §14.7 20px circular handle */}
      <div
        style={{
          position: "absolute",
          left: `${splitterPct}%`,
          top: "50%",
          transform: "translate(-50%, -50%)",
          width: 20,
          height: 20,
          borderRadius: "50%",
          background: "var(--green)",
          border: "2px solid var(--bg0)",
          cursor: "col-resize",
          zIndex: 11,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 2px 8px rgba(0,0,0,0.5)",
        }}
        onMouseDown={startDrag}
      >
        <span style={{ fontSize: 7, color: "var(--bg0)", fontWeight: 700 }}>◂▸</span>
      </div>

      {/* Invisible hit target over splitter line */}
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: `${splitterPct}%`,
          transform: "translateX(-50%)",
          width: 16,
          zIndex: 10,
          cursor: "col-resize",
        }}
        onMouseDown={startDrag}
      />

      {/* Placeholder when no result */}
      {!result && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            pointerEvents: "none",
          }}
        >
          <span
            style={{
              fontSize: 12,
              color: "var(--dim)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Run inference to see comparison
          </span>
        </div>
      )}

      {/* §14.7 Range slider for keyboard / accessibility control */}
      <input
        type="range"
        min={2}
        max={98}
        value={splitterPct}
        onChange={(e) => onSplitterPctChange(Number(e.target.value))}
        style={{
          position: "absolute",
          bottom: 6,
          left: "50%",
          transform: "translateX(-50%)",
          width: "clamp(120px, 15vw, 250px)",
          opacity: 0.5,
          accentColor: "var(--green)",
          zIndex: 12,
        }}
      />
    </div>
  );
}

// ── §14.8 Quality metrics panel ────────────────────────────────────────────

interface MetricRowProps {
  label: string;
  value: number | null | undefined;
  color: string;
  dec?: number;
}

function MetricRow({ label, value, color, dec = 2 }: MetricRowProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "5px 0",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <span
        style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: 12,
          fontFamily: "var(--font-mono)",
          color: value != null ? color : "var(--dim)",
          fontWeight: value != null ? 600 : 400,
        }}
      >
        {value != null ? value.toFixed(dec) : "—"}
      </span>
    </div>
  );
}

function MetricsPanel() {
  const result = useInferenceStore((s) => s.result);
  const gtPath = useInferenceStore((s) => s.gtPath);
  const metrics = result?.success ? result.metrics : null;

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <MetricRow label="PSNR" value={metrics?.psnr} color="var(--green)" dec={2} />
      <MetricRow label="SSIM" value={metrics?.ssim} color="var(--blue)" dec={4} />
      <MetricRow label="LPIPS" value={metrics?.lpips} color="var(--muted)" dec={4} />
      <MetricRow label="MS-SSIM" value={metrics?.ms_ssim} color="var(--muted)" dec={4} />
      {!gtPath && !metrics && (
        <div
          style={{ fontSize: 9, color: "var(--dim)", marginTop: 6, textAlign: "center" }}
        >
          Add GT image to compute metrics
        </div>
      )}
    </div>
  );
}

// ── §14.9 Image info panel ─────────────────────────────────────────────────

interface InfoRowProps {
  label: string;
  value: string;
}

function InfoRow({ label, value }: InfoRowProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "5px 0",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <span style={{ fontSize: 10, color: "var(--muted)" }}>{label}</span>
      <span
        style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text)" }}
      >
        {value}
      </span>
    </div>
  );
}

function InfoPanel() {
  const result = useInferenceStore((s) => s.result);
  const scaleFactor = useInferenceStore((s) => s.scaleFactor);

  const inRes = result?.success
    ? `${result.input_resolution!.width}×${result.input_resolution!.height}`
    : "—";
  const outRes = result?.success
    ? `${result.output_resolution!.width}×${result.output_resolution!.height}`
    : "—";
  const inferTime = result?.success
    ? result.inference_time_ms! < 1000
      ? `${result.inference_time_ms!.toFixed(0)} ms`
      : `${(result.inference_time_ms! / 1000).toFixed(2)} s`
    : "—";

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <InfoRow label="Input" value={inRes} />
      <InfoRow label="Output" value={outRes} />
      <InfoRow label="Scale" value={result?.success ? `${scaleFactor}×` : "—"} />
      <InfoRow label="Time" value={inferTime} />
    </div>
  );
}

// ── §14.1 Right metrics + info column (180px) ─────────────────────────────

function RightColumn() {
  return (
    <div
      style={{
        flex: 1, minWidth: 160, maxWidth: 260,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        padding: "10px 8px",
        borderLeft: "1px solid var(--border)",
        overflowY: "auto",
      }}
    >
      <Panel title="Quality Metrics">
        <MetricsPanel />
      </Panel>
      <Panel title="Image Info">
        <InfoPanel />
      </Panel>
    </div>
  );
}

// ── §14.13 Error dialog ────────────────────────────────────────────────────

interface ErrorDialogProps {
  message: string;
  onClose: () => void;
}

function ErrorDialog({ message, onClose }: ErrorDialogProps) {
  return (
    <div
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.65)",
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--red)",
          borderRadius: "var(--radius-lg)",
          width: 400,
          padding: 24,
          display: "flex",
          flexDirection: "column",
          gap: 16,
          boxShadow: "0 24px 64px rgba(0,0,0,0.6)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 16, color: "var(--red)" }}>⚠</span>
          <span style={{ fontSize: 13, color: "var(--text)", fontWeight: 600 }}>
            Inference Failed
          </span>
        </div>
        <div
          style={{
            fontSize: 12,
            color: "var(--muted)",
            fontFamily: "var(--font-mono)",
            background: "var(--bg2)",
            padding: "10px 12px",
            borderRadius: "var(--radius-sm)",
            wordBreak: "break-word",
          }}
        >
          {message}
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <Btn variant="solid" color="var(--red)" onClick={onClose}>
            Dismiss
          </Btn>
        </div>
      </div>
    </div>
  );
}

// ── §14.1 Root screen ──────────────────────────────────────────────────────

export function ScreenInference() {
  const store = useInferenceStore();
  const [splitterPct, setSplitterPct] = useState(50);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // §14.10 Send inference.run IPC message
  const handleRun = useCallback(async () => {
    const s = useInferenceStore.getState();
    if (!s.inputPath || !s.checkpointPath) return;

    store.setRunning(true);
    store.setTileProgress(0, 0);
    store.setResult(null);

    try {
      // TODO: replace with api call
    } catch {
      store.setRunning(false);
    }
  }, [store]);

  // §14.13 Watch for inference.result and handle error branch
  const result = useInferenceStore((s) => s.result);
  useEffect(() => {
    if (!result) return;
    store.setRunning(false);
    // §14.13: Only show error dialog on failure; do NOT render preview images
    if (!result.success && result.error) {
      setErrorMsg(result.error);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "row",
        overflow: "hidden",
        background: "var(--bg0)",
      }}
    >
      {/* §14.1 Left settings (210px) */}
      <SettingsPanel onRun={handleRun} />

      {/* §14.1 Center comparison (flex-1) */}
      <ComparisonPanel
        splitterPct={splitterPct}
        onSplitterPctChange={setSplitterPct}
      />

      {/* §14.1 Right metrics + info (180px) */}
      <RightColumn />

      {/* §14.13 Error dialog — rendered separately; never shown alongside preview images */}
      {errorMsg && (
        <ErrorDialog message={errorMsg} onClose={() => setErrorMsg(null)} />
      )}
    </div>
  );
}
