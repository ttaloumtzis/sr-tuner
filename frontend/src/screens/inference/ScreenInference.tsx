// §14 Inference Screen — redesigned (settings drawer + result stage)
// Tasks: 14.1–14.13

import { useState, useRef, useCallback, useEffect } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useInferenceStore } from "../../store/inferenceStore";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Field } from "../../components/ui/Field";
import { PathInput } from "../../components/ui/PathInput";
import { Dropdown, type DropdownOption } from "../../components/ui/Dropdown";
import { PBar } from "../../components/ui/PBar";
import { InlineAlert } from "../../components/ui/InlineAlert";
import { useInferenceSSE } from "../../hooks/useInferenceSSE";
import { basename, join } from "../../lib/path";
import type { ModelVersion } from "../../lib/api-types";

// ── Cross-hatch background ─────────────────────────────────────────────────

const CROSSHATCH_BG: React.CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(45deg, var(--bg2) 0, var(--bg2) 4px, transparent 0, transparent 50%), " +
    "repeating-linear-gradient(-45deg, var(--bg2) 0, var(--bg2) 4px, transparent 0, transparent 50%)",
  backgroundSize: "12px 12px",
  backgroundColor: "var(--bg1)",
};

// ── Drop zone ─────────────────────────────────────────────────────────────

interface DropZoneProps {
  label: string;
  path: string | null;
  accent?: string;
  onSelect: (path: string) => void;
  onClear?: () => void;
  browseTitle?: string;
  fileFilters?: { name: string; extensions: string[] }[];
}

function DropZone({ label, path, accent = "var(--border)", onSelect, onClear, browseTitle, fileFilters }: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false);

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
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      style={{
        border: `1.5px dashed ${dragOver ? accent : accent}`,
        borderRadius: "var(--radius-sm)",
        padding: "8px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        background: dragOver ? "var(--bg2)" : "transparent",
        transition: "background 0.15s",
        minHeight: 52,
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
            filters: fileFilters ?? [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"] }],
          });
          if (selected) onSelect(selected);
        }}>
          Browse…
        </Btn>
        {onClear && path && (
          <Btn small onClick={onClear}>Clear</Btn>
        )}
      </div>
    </div>
  );
}

// ── Small number input ────────────────────────────────────────────────────

function NumberInput({ value, onChange, min, max, title }: { value: number; onChange: (v: number) => void; min: number; max: number; title?: string }) {
  return (
    <input
      type="number"
      min={min}
      max={max}
      value={Number.isFinite(value) ? value : ""}
      title={title}
      onChange={(e) => {
        const n = parseInt(e.target.value, 10);
        onChange(Number.isNaN(n) ? 0 : Math.max(min, Math.min(max, n)));
      }}
      style={{
        background: "var(--bg3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
        padding: "5px 8px",
        fontSize: 12,
        color: "var(--text)",
        fontFamily: "var(--font-mono)",
        width: "100%",
        outline: "none",
      }}
    />
  );
}

// ── Model panel (instance+version OR raw checkpoint path) ─────────────────

const TILE_OPTIONS = [
  { value: "0", label: "No tiling" },
  { value: "128", label: "128 px" },
  { value: "256", label: "256 px" },
  { value: "512", label: "512 px" },
];

const FORMAT_OPTIONS = ["png", "jpeg", "webp", "tiff"];

const DEVICE_OPTIONS = ["auto", "cuda", "cpu"];

function ModelPanel() {
  const store = useInferenceStore();
  const [instances, setInstances] = useState<DropdownOption[]>([]);
  const [instanceMeta, setInstanceMeta] = useState<Record<string, { architecture: string | null; scale: number | null }>>({});
  const [selInstance, setSelInstance] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);

  // Fetch model instances on mount, then sync store state.
  useEffect(() => {
    (async () => {
      try {
        const { listInstances } = await import("../../lib/api");
        const list = await listInstances();
        setInstances(list.map((i) => ({ value: i.name, label: i.name })));
        const meta: Record<string, { architecture: string | null; scale: number | null }> = {};
        for (const i of list) meta[i.name] = { architecture: i.architecture, scale: i.scale };
        setInstanceMeta(meta);

        // If the store already has an instance selected, seed the local state
        // and fetch versions so stale selections are cleared.
        const storedInstance = store.instance;
        if (storedInstance && list.some((i) => i.name === storedInstance)) {
          setSelInstance(storedInstance);
          try {
            const { getInstanceVersions } = await import("../../lib/api");
            const vlist = await getInstanceVersions(storedInstance);
            setVersions(vlist);
            if (vlist.length > 0 && store.version) {
              const stillExists = vlist.some((v) => v.tag === store.version);
              if (!stillExists) {
                store.setVersion(null);
                store.setModelPath(null);
              }
            }
          } catch {
            setVersions([]);
          }
        }
      } catch {
        setInstances([]);
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // §13.9b — a checkpoint preselected from the Checkpoints tab lands as a raw model file.
  useEffect(() => {
    const pre = store.preselectedCheckpointPath;
    if (pre) {
      store.setInstance(null);
      store.setVersion(null);
      store.setModelPath(pre);
      store.setPreselectedCheckpointPath(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.preselectedCheckpointPath]);

  const handleInstanceChange = async (name: string) => {
    setSelInstance(name || null);
    setVersions([]);
    if (!name) {
      store.setInstance(null);
      store.setVersion(null);
      store.setModelPath(null);
      return;
    }
    store.setInstance(name);
    store.setVersion(null);
    store.setModelPath(null);
    try {
      const { getInstanceVersions } = await import("../../lib/api");
      const list = await getInstanceVersions(name);
      setVersions(list);
      // Auto-select the latest *available* version (has_weights !== false)
      const available = list.filter((v) => v.has_weights !== false);
      if (available.length > 0) {
        const latest = available[available.length - 1];
        store.setVersion(latest.tag);
        store.setModelPath(join(latest.path, "model.pt"));
      }
    } catch {
      setVersions([]);
    }
  };

  const handleVersionChange = (tag: string) => {
    const v = versions.find((x) => x.tag === tag);
    if (!v) {
      store.setVersion(null);
      store.setModelPath(null);
      return;
    }
    // Don't select versions missing weights
    if (v.has_weights === false) {
      store.setVersion(null);
      store.setModelPath(null);
      return;
    }
    store.setVersion(v.tag);
    store.setModelPath(join(v.path, "model.pt"));
  };

  const missingAny = versions.some((v) => v.has_weights === false);
  const versionOptions: DropdownOption[] = versions
    .filter((v) => v.has_weights !== false)
    .map((v) => ({
      value: v.tag,
      label: v.tag,
    }));

  const meta = selInstance ? instanceMeta[selInstance] : undefined;
  const rawFile = !store.instance && store.modelPath;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <Field label="Model">
        <Dropdown
          value={selInstance ?? ""}
          options={instances}
          onChange={handleInstanceChange}
          placeholder="Select model…"
        />
      </Field>
      {meta && (
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {meta.architecture && (
            <span style={{ fontSize: 9, color: "var(--text)", background: "var(--green-dim)", border: "1px solid var(--green)44", borderRadius: "var(--radius-sm)", padding: "1px 6px", fontFamily: "var(--font-mono)" }}>
              {meta.architecture}
            </span>
          )}
          {meta.scale != null && (
            <span style={{ fontSize: 9, color: "var(--text)", background: "var(--blue-dim)", border: "1px solid var(--blue)44", borderRadius: "var(--radius-sm)", padding: "1px 6px", fontFamily: "var(--font-mono)" }}>
              {meta.scale}×
            </span>
          )}
        </div>
      )}
      {rawFile ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: 9, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>Model file</span>
          <div
            style={{
              fontSize: 10,
              color: "var(--text)",
              fontFamily: "var(--font-mono)",
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              padding: "5px 8px",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
            title={store.modelPath ?? ""}
          >
            {basename(store.modelPath ?? "")}
          </div>
          <div>
            <Btn small onClick={() => store.setModelPath(null)}>Clear</Btn>
          </div>
        </div>
      ) : (
        <div>
          {missingAny && (
            <div style={{ marginBottom: 6 }}>
              <InlineAlert tone="muted">
                Some versions are missing weights and are not selectable.
              </InlineAlert>
            </div>
          )}
          <Field label="Version">
            <Dropdown
              value={store.version ?? ""}
              options={versionOptions}
              onChange={handleVersionChange}
              placeholder={selInstance ? (versionOptions.length ? "Select version…" : "No versions yet — save a checkpoint first") : "Select a model first"}
              mono
            />
          </Field>
        </div>
      )}
    </div>
  );
}

// ── Left settings rail ────────────────────────────────────────────────────

function SettingsRail({ onRun, onCancel }: { onRun: () => void; onCancel: () => void }) {
  const store = useInferenceStore();

  const outputDirError = store.outputDir ? null : "Select an output directory";
  const modelReady = !!store.modelPath;
  const canRun =
    !!store.inputPath && modelReady && !outputDirError && store.status !== "running";

  return (
    <div
      style={{
        width: 264,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        padding: 10,
        borderRight: "1px solid var(--border)",
        overflowY: "auto",
        background: "var(--bg0)",
      }}
    >
      {/* Input image */}
      <Panel title="Input Image" style={{ flexShrink: 0 }}>
        <DropZone
          label="Drop image here"
          path={store.inputPath}
          onSelect={store.setInputPath}
          browseTitle="Select Input Image"
        />
      </Panel>

      {/* Ground truth */}
      <Panel title="Ground Truth" subtitle="for quality metrics" style={{ flexShrink: 0 }}>
        <DropZone
          label="Drop GT image here"
          path={store.gtPath}
          accent="var(--blue)"
          onSelect={store.setGtPath}
          onClear={() => store.setGtPath(null)}
          browseTitle="Select Ground Truth Image"
        />
      </Panel>

      {/* Model */}
      <Panel title="Model" style={{ flexShrink: 0 }}>
        <ModelPanel />
      </Panel>

      {/* Output */}
      <Panel title="Output" style={{ flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Field label="Save Directory">
            <PathInput
              value={store.outputDir}
              onChange={store.setOutputDir}
              browseTitle="Select Output Directory"
              compact
            />
            {outputDirError && (
              <div style={{ fontSize: 10, color: "var(--red)", marginTop: 3 }}>
                {outputDirError}
              </div>
            )}
          </Field>
          <Field label="Format">
            <Dropdown
              value={store.outputFormat}
              options={FORMAT_OPTIONS}
              onChange={(v) => store.setOutputFormat(v as "png" | "jpeg" | "webp" | "tiff")}
            />
          </Field>
        </div>
      </Panel>

      {/* Tiling + device */}
      <Panel title="Advanced" style={{ flexShrink: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Field label="Tile Size">
            <Dropdown
              value={String(store.tileSize)}
              options={TILE_OPTIONS}
              onChange={(v) => store.setTileSize(Number(v))}
            />
          </Field>
          <Field label="Tile Overlap" hint="px">
            <NumberInput
              value={store.overlap}
              min={0}
              max={store.tileSize > 0 ? store.tileSize - 1 : 512}
              onChange={store.setOverlap}
              title="Overlap must be less than tile size"
            />
          </Field>
          <Field label="Device">
            <Dropdown
              value={store.device}
              options={DEVICE_OPTIONS}
              onChange={(v) => store.setDevice(v as "auto" | "cuda" | "cpu")}
            />
          </Field>
        </div>
      </Panel>

      {/* Run / cancel + progress */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
        {store.status === "running" ? (
          <Btn variant="solid" color="var(--red)" full onClick={onCancel}>
            Cancel
          </Btn>
        ) : (
          <Btn variant="solid" full disabled={!canRun} onClick={onRun}>
            Run Inference
          </Btn>
        )}

        {store.status === "running" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <PBar value={store.tilesDone} max={Math.max(store.tilesTotal, 1)} />
            <span style={{ fontSize: 9, color: "var(--dim)", textAlign: "center", fontFamily: "var(--font-mono)" }}>
              {store.tilesTotal > 1
                ? `${store.tilesDone} / ${store.tilesTotal} tiles`
                : "Processing…"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Before/after comparison stage ─────────────────────────────────────────

function ComparisonPanel({ splitterPct, onSplitterPctChange }: { splitterPct: number; onSplitterPctChange: (pct: number) => void }) {
  const result = useInferenceStore((s) => s.result);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const lrSrc = result?.success && result.preview_input_path ? convertFileSrc(result.preview_input_path) : null;
  const srSrc = result?.success && result.preview_output_path ? convertFileSrc(result.preview_output_path) : null;

  const lrLabel = result?.success && result.input_resolution ? `${result.input_resolution.width}×${result.input_resolution.height}` : null;
  const srLabel = result?.success && result.output_resolution ? `${result.output_resolution.width}×${result.output_resolution.height}` : null;

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    onSplitterPctChange(Math.max(2, Math.min(98, (x / rect.width) * 100)));
  }, [onSplitterPctChange]);

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
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";
  };

  const layerStyle: React.CSSProperties = {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    pointerEvents: "none",
    overflow: "hidden",
  };

  const badgeStyle: React.CSSProperties = {
    position: "absolute",
    top: 12,
    padding: "4px 10px",
    borderRadius: "var(--radius-md)",
    fontSize: 11,
    fontWeight: 700,
    backdropFilter: "blur(8px)",
    color: "var(--bg0)",
    zIndex: 30,
    pointerEvents: "none",
  };

  return (
    <div
      ref={containerRef}
      onMouseDown={startDrag}
      style={{ flex: 1, position: "relative", overflow: "hidden", cursor: "ew-resize", userSelect: "none", ...CROSSHATCH_BG }}
    >
      {/* Base layer — SR */}
      <div style={{ ...layerStyle }}>
        {srSrc ? (
          <img src={srSrc} alt="Output (SR)" style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }} draggable={false} />
        ) : (
          <div style={{ width: "100%", height: "100%", ...CROSSHATCH_BG }} />
        )}
      </div>

      {/* Top layer — LR (upscaled to match SR scale), clipped at the handle */}
      <div style={{ ...layerStyle, clipPath: `polygon(0 0, ${splitterPct}% 0, ${splitterPct}% 100%, 0 100%)` }}>
        {lrSrc ? (
          <img src={lrSrc} alt="Input (LR)" style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }} draggable={false} />
        ) : (
          <div style={{ width: "100%", height: "100%", ...CROSSHATCH_BG }} />
        )}
      </div>

      {/* Badges */}
      {lrLabel && (
        <div style={{ ...badgeStyle, left: 12, background: "color-mix(in srgb, var(--amber) 85%, transparent)" }}>
          LR {lrLabel}
        </div>
      )}
      {srLabel && (
        <div style={{ ...badgeStyle, right: 12, background: "color-mix(in srgb, var(--green) 85%, transparent)" }}>
          SR {srLabel}
        </div>
      )}

      {/* Handle line + knob */}
      <div style={{ position: "absolute", top: 0, bottom: 0, left: `${splitterPct}%`, transform: "translateX(-50%)", width: 2, background: "var(--green)", boxShadow: "0 0 10px color-mix(in srgb, var(--green) 60%, transparent)", zIndex: 20, pointerEvents: "none" }} />
      <div style={{ position: "absolute", left: `${splitterPct}%`, top: "50%", transform: "translate(-50%, -50%)", width: 28, height: 28, borderRadius: "50%", background: "var(--green)", color: "var(--bg0)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 2px 8px rgba(0,0,0,0.5)", zIndex: 21, pointerEvents: "none" }}>
        <span style={{ fontSize: 12, fontWeight: 700 }}>◂▸</span>
      </div>

      {/* Placeholder */}
      {!result && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
          <span style={{ fontSize: 12, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
            Run inference to see comparison
          </span>
        </div>
      )}

      <input
        type="range"
        min={2}
        max={98}
        value={splitterPct}
        onChange={(e) => onSplitterPctChange(Number(e.target.value))}
        style={{ position: "absolute", bottom: 6, left: "50%", transform: "translateX(-50%)", width: "clamp(120px, 15vw, 250px)", opacity: 0.5, accentColor: "var(--green)", zIndex: 30 }}
      />
    </div>
  );
}

// ── Result footer (metrics + info) ────────────────────────────────────────

function MetricRow({ label, value, color, dec = 2 }: { label: string; value: number | null | undefined; color: string; dec?: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>{label}</span>
      <span style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: value != null ? color : "var(--dim)", fontWeight: value != null ? 600 : 400 }}>
        {value != null ? value.toFixed(dec) : "—"}
      </span>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>{label}</span>
      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)" }}>{value}</span>
    </div>
  );
}

function ResultFooter() {
  const result = useInferenceStore((s) => s.result);
  const status = useInferenceStore((s) => s.status);
  const errorMsg = useInferenceStore((s) => s.errorMsg);
  const gtPath = useInferenceStore((s) => s.gtPath);

  if (status === "error") {
    return (
      <div style={{ flexShrink: 0, borderTop: "1px solid var(--red)66", background: "#1a1111", padding: "10px 16px", display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 12, color: "var(--red)" }}>⚠</span>
        <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)", wordBreak: "break-word", flex: 1 }}>
          {errorMsg || "Inference failed"}
        </span>
        <Btn small color="var(--red)" onClick={() => useInferenceStore.getState().resetRun()}>Dismiss</Btn>
      </div>
    );
  }

  if (status !== "done" || !result?.success) return null;

  const metrics = result.metrics;
  const outRes = result.output_resolution;
  const inRes = result.input_resolution;

  let scaleLabel = "—";
  if (inRes && outRes && inRes.width > 0) {
    const sx = outRes.width / inRes.width;
    const sy = outRes.height / inRes.height;
    if (sx === sy) scaleLabel = `${sx}×`;
  }
  const timeMs = result.inference_time_ms;
  const timeLabel = timeMs != null
    ? timeMs < 1000 ? `${timeMs.toFixed(0)} ms` : `${(timeMs / 1000).toFixed(2)} s`
    : "—";

  return (
    <div style={{ flexShrink: 0, borderTop: "1px solid var(--border)", background: "var(--bg1)", display: "flex", gap: 24, padding: "10px 16px", overflowX: "auto" }}>
      {/* Metrics */}
      <div style={{ minWidth: 200 }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>Quality Metrics</div>
        <MetricRow label="PSNR" value={metrics?.psnr} color="var(--green)" dec={2} />
        <MetricRow label="SSIM" value={metrics?.ssim} color="var(--blue)" dec={4} />
        <MetricRow label="LPIPS" value={metrics?.lpips} color="var(--muted)" dec={4} />
        <MetricRow label="MS-SSIM" value={metrics?.ms_ssim} color="var(--muted)" dec={4} />
        {!gtPath && !metrics && (
          <div style={{ fontSize: 9, color: "var(--dim)", marginTop: 6 }}>Add a GT image to compute metrics</div>
        )}
      </div>

      {/* Info */}
      <div style={{ minWidth: 180 }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>Image Info</div>
        <InfoRow label="Input" value={inRes ? `${inRes.width}×${inRes.height}` : "—"} />
        <InfoRow label="Output" value={outRes ? `${outRes.width}×${outRes.height}` : "—"} />
        <InfoRow label="Scale" value={scaleLabel} />
        <InfoRow label="Time" value={timeLabel} />
      </div>

      {/* Output path */}
      {result.output && (
        <div style={{ flex: 1, minWidth: 200, display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Output</div>
          <div style={{ fontSize: 10, color: "var(--text)", fontFamily: "var(--font-mono)", wordBreak: "break-all", lineHeight: 1.5 }}>{result.output}</div>
          <div>
            <Btn small onClick={() => { try { invoke("open_in_file_manager", { path: result.output }); } catch { /* browser mode */ } }}>
              Open in file manager
            </Btn>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Root screen ───────────────────────────────────────────────────────────

export function ScreenInference() {
  const store = useInferenceStore();
  const [splitterPct, setSplitterPct] = useState(50);
  useInferenceSSE();

  // Default the save directory to the user's Pictures folder.
  useEffect(() => {
    const s = useInferenceStore.getState();
    if (s.outputDir) return;
    (async () => {
      const { defaultOutputDir } = await import("../../lib/api");
      const dir = await defaultOutputDir();
      if (dir) s.setOutputDir(dir);
    })();
  }, []);

  const handleRun = useCallback(async () => {
    const s = useInferenceStore.getState();
    if (!s.inputPath || !s.outputDir) return;
    if (!s.instance && !s.modelPath) return;

    const stem = basename(s.inputPath).replace(/\.[^.]+$/, "");
    const output = join(s.outputDir, `${stem}_sr.${s.outputFormat}`);

    store.resetRun();

    try {
      const { startInference } = await import("../../lib/api");
      const res = await startInference({
        input: s.inputPath,
        output,
        gt: s.gtPath ?? undefined,
        format: s.outputFormat,
        tile: s.tileSize,
        overlap: s.overlap,
        device: s.device,
        ...(s.instance
          ? { instance: s.instance, version: s.version ?? undefined }
          : { model: s.modelPath ?? undefined }),
      });
      store.setActiveJobId(res.job_id);
      store.setStatus("running");
      store.setTileProgress(0, 0);
    } catch (err) {
      store.setErrorMsg(err instanceof Error ? err.message : String(err));
      store.setStatus("error");
    }
  }, [store]);

  const handleCancel = useCallback(async () => {
    const s = useInferenceStore.getState();
    if (s.activeJobId) {
      try {
        const { cancelJob } = await import("../../lib/api");
        await cancelJob(s.activeJobId);
      } catch {
        // backend may have already finished
      }
    }
    s.resetRun();
  }, []);

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "row", overflow: "hidden", background: "var(--bg0)" }}>
      <SettingsRail onRun={handleRun} onCancel={handleCancel} />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
        <ComparisonPanel splitterPct={splitterPct} onSplitterPctChange={setSplitterPct} />
        <ResultFooter />
      </div>
    </div>
  );
}
