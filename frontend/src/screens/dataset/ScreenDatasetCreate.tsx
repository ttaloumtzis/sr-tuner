import { useState, useEffect, useCallback, useRef } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { PathInput } from "../../components/ui/PathInput";
import { PBar } from "../../components/ui/PBar";
import { useDatasetStore, type DatasetMode, type DownscaleKernel } from "../../store/datasetStore";
import { useProjectStore } from "../../store/projectStore";
import { DegradationPanel } from "./DegradationPanel";
import { basename, join, parentFromProjFile } from "../../lib/path";
import { inspectDataset, finalizeDataset } from "../../lib/api";
import type { DatasetInspectInfo } from "../../lib/api-types";
import { useToast } from "../../components/shell/ToastProvider";


function TypeCard({ id: _id, label, description, active, disabled, onClick }: {
  id: DatasetMode; label: string; description: string; active: boolean; disabled?: boolean; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <div onClick={disabled ? undefined : onClick} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
      style={{ background: active ? "var(--greenDim)" : hovered && !disabled ? "var(--bg2)" : "var(--bg1)", border: `1px solid ${active ? "var(--green)" : hovered && !disabled ? "var(--muted)" : "var(--border)"}`, borderRadius: "var(--radius-md)", padding: "12px 14px", cursor: disabled ? "not-allowed" : "pointer", transition: "var(--transition-fast)", display: "flex", flexDirection: "column", gap: 4, opacity: disabled ? 0.5 : 1 }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: active ? "var(--green)" : "var(--text)" }}>{label}{disabled && <span style={{ color: "var(--dim)", fontWeight: 400, marginLeft: 6 }}>(soon)</span>}</span>
      <span style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>{description}</span>
    </div>
  );
}

function ScaleBar() {
  const s = useDatasetStore();
  const presets = [1, 2, 4, 8];

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 10px", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", flexWrap: "wrap" }}>
      <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Scale</span>
      {presets.map((p) => (
        <button key={p} onClick={() => { s.setScale(p); }}
          style={{ background: s.scale === p ? "var(--green)" : "var(--bg3)", border: `1px solid ${s.scale === p ? "var(--green)" : "var(--border)"}`, color: s.scale === p ? "#0d0f11" : "var(--muted)", fontSize: 11, fontWeight: 600, padding: "2px 9px", borderRadius: "var(--radius-sm)", cursor: "pointer", transition: "var(--transition-fast)" }}>
          ×{p}
        </button>
      ))}
    </div>
  );
}

function DownsampleMethodSelector() {
  const s = useDatasetStore();
  const options: { id: DownscaleKernel; label: string }[] = [
    { id: "area", label: "Area" }, { id: "bicubic", label: "Bicubic" }, { id: "bilinear", label: "Bilinear" },
    { id: "lanczos", label: "Lanczos" }, { id: "nearest", label: "Nearest" },
  ];
  return (
    <div style={{ padding: "7px 10px", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Downsample Method</span>
        {options.map((opt) => (
          <button key={opt.id} onClick={() => s.setKernel(opt.id)}
            style={{ background: s.kernel === opt.id ? "var(--green)" : "var(--bg3)", border: `1px solid ${s.kernel === opt.id ? "var(--green)" : "var(--border)"}`, color: s.kernel === opt.id ? "#0d0f11" : "var(--muted)", fontSize: 11, fontWeight: s.kernel === opt.id ? 600 : 400, padding: "3px 11px", borderRadius: 10, cursor: "pointer", transition: "var(--transition-fast)" }}>
            {opt.label}
          </button>
        ))}
      </div>
      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--muted)", cursor: "pointer" }}>
        <input type="checkbox" checked={s.antialias} onChange={(e) => s.setAntialias(e.target.checked)} style={{ accentColor: "var(--green)" }} />
        Antialias pre-filter
      </label>
    </div>
  );
}

function PreExistingMode() {
  const s = useDatasetStore();
  const project = useProjectStore((s) => s.project);
  const { show: toast } = useToast();
  const [inspectInfo, setInspectInfo] = useState<DatasetInspectInfo | null>(null);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const inspectSeq = useRef(0);

  useEffect(() => {
    const root = s.rootPath;
    inspectSeq.current += 1;
    const seq = inspectSeq.current;
    if (!root) {
      setInspectInfo(null);
      setInspectError(null);
      return;
    }
    setInspectInfo(null);
    setInspectError(null);
    const timer = setTimeout(async () => {
      try {
        const info = await inspectDataset({ path: root });
        if (inspectSeq.current !== seq) return;
        setInspectInfo(info);
      } catch (err) {
        if (inspectSeq.current !== seq) return;
        setInspectError(err instanceof Error ? err.message : String(err));
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [s.rootPath]);

  const detectedScale = inspectInfo?.scale_ratio != null ? Math.round(inspectInfo.scale_ratio) : null;
  const scaleUsable =
    inspectInfo?.scale_ratio != null && inspectInfo.scale_exact && (detectedScale ?? 0) > 0;
  const canImport = (inspectInfo?.pair_count ?? 0) > 0 && scaleUsable && !importing;

  const handleImport = async () => {
    if (!s.rootPath || !project) return;
    const { invoke } = await import("@tauri-apps/api/core");
    const projectDir = parentFromProjFile(project.filePath);
    const cleanRoot = s.rootPath.replace(/[/\\]+$/, "");
    const name = basename(cleanRoot) || "imported";
    const dst = join(projectDir, "datasets", name);
    setImporting(true);
    try {
      if (await invoke<boolean>("path_exists", { path: dst })) {
        throw new Error(`A dataset named "${name}" already exists in this project`);
      }
      await invoke("copy_directory", { src: cleanRoot, dst });
      let result;
      try {
        // canImport guarantees detectedScale is non-null here (scale detected & exact)
        const finalizeParams: { path: string; scale: number; config_overrides?: Record<string, unknown> } = {
          path: dst,
          scale: detectedScale!,
        };
        result = await finalizeDataset(finalizeParams);
      } catch (err) {
        await invoke("delete_directory", { path: dst }).catch(() => {});
        throw err;
      }
      toast("success", `Imported "${name}" — ${result.num_pairs.toLocaleString()} pairs at ×${result.scale}`);
      s.setRootPath("");
    } catch (err) {
      toast("error", `Import failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setImporting(false);
    }
  };

  const dimsText = (size: { width: number; height: number } | null) =>
    size ? `${size.width}×${size.height}` : "—";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Dataset root folder</label>
        <PathInput value={s.rootPath} onChange={s.setRootPath} browseTitle="Select dataset root folder (containing HR/ and LR/)" mono />
        <span style={{ fontSize: 10, color: "var(--dim)" }}>Select the root folder containing HR/ and LR/ subdirectories</span>
      </div>
      {s.rootPath && (
        <div style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8 }}>
          {inspectError && (
            <div style={{ fontSize: 10, color: "var(--red)", lineHeight: 1.4 }}>Could not inspect folder: {inspectError}</div>
          )}
          {inspectInfo === null && !inspectError && (
            <div style={{ fontSize: 10, color: "var(--muted)" }}>Inspecting folder…</div>
          )}
          {inspectInfo && (
            <>
              <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  HR: <span style={{ color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>{inspectInfo.hr_count}</span>
                  {" "}({dimsText(inspectInfo.hr_size)})
                </span>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  LR: <span style={{ color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>{inspectInfo.lr_count}</span>
                  {" "}({dimsText(inspectInfo.lr_size)})
                </span>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  Pairs: <span style={{ color: "var(--text)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>{inspectInfo.pair_count}</span>
                </span>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  Scale:
                  {inspectInfo.scale_ratio != null ? (
                    <>
                      <span style={{ color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}> ×{detectedScale}</span>
                      {" "}
                      <span style={{ fontSize: 9, color: "var(--dim)" }}>
                        detected (×{inspectInfo.scale_ratio.toFixed(2)})
                      </span>
                    </>
                  ) : (
                    <span style={{ fontSize: 9, color: "var(--amber)" }}> unknown</span>
                  )}
                </span>
              </div>
              {inspectInfo.warnings.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {inspectInfo.warnings.map((w, i) => (
                    <span key={i} style={{ fontSize: 10, color: "var(--amber)", lineHeight: 1.4 }}>⚠ {w}</span>
                  ))}
                </div>
              )}
              {!scaleUsable && inspectInfo.scale_ratio != null && (
                <div style={{ fontSize: 10, color: "var(--red)", lineHeight: 1.4 }}>
                  ⚠ Detected scale ×{inspectInfo.scale_ratio.toFixed(2)} is not a whole number — the HR/LR
                  dimensions don't match a clean scale factor. The images cannot be rescaled without
                  destroying data, so import is disabled.
                </div>
              )}
              {!scaleUsable && inspectInfo.scale_ratio == null && inspectInfo.pair_count > 0 && (
                <div style={{ fontSize: 10, color: "var(--red)", lineHeight: 1.4 }}>
                  ⚠ Could not determine the scale from the images — import is disabled.
                </div>
              )}
              <div>
                <Btn small variant="solid" onClick={handleImport} disabled={!project || !canImport}>
                  {importing ? "Importing…" : "Import into project"}
                </Btn>
                {!canImport && !importing && inspectInfo.pair_count === 0 && (
                  <span style={{ fontSize: 10, color: "var(--red)", marginLeft: 8 }}>No matching HR/LR pairs found</span>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function VideoExtractMode() {
  const s = useDatasetStore();
  const s_error = useDatasetStore((s) => s.jobError);
  const s_status = useDatasetStore((s) => s.jobStatus);
  const progressSteps = useDatasetStore((s) => s.progressSteps);
  const project = useProjectStore((s) => s.project);
  const [dragOver, setDragOver] = useState(false);
  const [starting, setStarting] = useState(false);
  const [extractError, setExtractError] = useState<string | null>(null);

  const handleBrowse = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        filters: [{ name: "Video", extensions: ["mkv", "mp4", "avi", "mov"] }],
      });
      if (selected) {
        const path = Array.isArray(selected) ? selected[0] : selected;
        if (path) s.setVideoFile({ name: basename(path) ?? path, path });
      }
    } catch {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".mkv,.mp4,.avi,.mov";
      input.multiple = false;
      input.onchange = () => {
        const file = input.files?.[0];
        const path = (file as File & { path?: string }).path || file?.name;
        if (path) s.setVideoFile({ name: basename(path) ?? path, path });
      };
      input.click();
    }
  }, [s.setVideoFile]);

  useEffect(() => {
    if (s_status === "error" && s_error) setExtractError(s_error);
    else if (s_status === "idle") setExtractError(null);
  }, [s_status, s_error]);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    (async () => {
      try {
        const { getCurrentWebview } = await import("@tauri-apps/api/webview");
        unlisten = await getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type === "over" || event.payload.type === "enter") {
            setDragOver(true);
          } else if (event.payload.type === "leave") {
            setDragOver(false);
          } else if (event.payload.type === "drop") {
            setDragOver(false);
            const paths = event.payload.paths as string[];
            if (paths.length > 0) {
              const path = paths[0];
              s.setVideoFile({ name: basename(path) ?? path, path });
            }
          }
        });
      } catch {
        // running in browser dev mode — no Tauri drag-drop events
      }
    })();
    return () => { unlisten?.(); };
  }, []);

  const handleStartExtraction = async () => {
    try {
      if (!project) { s.setJobError("No project loaded"); s.setJobStatus("error"); return; }
      const projectDir = parentFromProjFile(project.filePath);
      const firstVideo = s.videoFile?.path;
      if (!firstVideo) { s.setJobError("No video file selected"); s.setJobStatus("error"); return; }
      const videoName = basename(firstVideo).replace(/\.[^/.]+$/, "") || "extracted";
      const out = join(projectDir, "datasets", videoName);
      const degParts: string[] = [];
      if (s.degBlur) degParts.push("blur");
      if (s.degNoise) degParts.push("noise");
      if (s.degJpeg) degParts.push("jpeg");
      if (s.degJpeg2000) degParts.push("jpeg2000");
      if (s.degColorJitter) degParts.push("color-jitter");

      const configOverrides: Record<string, unknown> = {};
      configOverrides["scale"] = s.scale;
      configOverrides["frame_rate"] = s.frameRate;
      if (s.startTime > 0) configOverrides["start_time"] = s.startTime;
      if (s.duration !== null) configOverrides["duration"] = s.duration;

      const degCfg: Record<string, unknown> = {};
      if (s.degBlur) {
        degCfg["blur"] = {
          enabled: true,
          gaussian: { kernel_size: s.blurKernelSize, sigma: [s.blurSigmaMin, s.blurSigmaMax], prob: s.blurGaussianProb },
          motion: { enabled: s.motionBlurEnabled, max_kernel_size: s.motionBlurMaxKernel, prob: s.blurMotionProb },
        };
      }
      if (s.degNoise) {
        degCfg["noise"] = {
          enabled: true,
          gaussian: { sigma_range: [s.noiseSigmaMin, s.noiseSigmaMax], prob: s.noiseGaussianProb },
          poisson: { scale_range: [s.poissonScaleMin, s.poissonScaleMax], prob: s.noisePoissonProb },
          salt_pepper: { amount: s.saltPepperAmount, prob: s.noiseSaltPepperProb },
        };
      }
      if (s.degJpeg) {
        degCfg["jpeg"] = { enabled: true, quality_range: [s.jpegQualityMin, s.jpegQualityMax], prob: s.jpegProb };
      }
      if (s.degJpeg2000) {
        degCfg["jpeg2000"] = { enabled: true, quality_range: [s.jpeg2000QualityMin, s.jpeg2000QualityMax], prob: s.jpeg2000Prob };
      }
      if (s.degColorJitter) {
        degCfg["color_jitter"] = {
          enabled: true,
          hue_range: [-s.jitterHueRange, s.jitterHueRange],
          saturation_range: [-s.jitterSaturationRange, s.jitterSaturationRange],
          value_range: [-s.jitterValueRange, s.jitterValueRange],
          prob: s.jitterProb,
        };
      }
      const resizeMethod = s.kernel;
      degCfg["resize"] = { method: resizeMethod, antialias: s.antialias };
      configOverrides["degradation"] = degCfg;

      setStarting(true);
      s.clearJob();
      s.setJobType("build");
      s.setJobStatus("running");
      const { buildDataset } = await import("../../lib/api");
      const result = await buildDataset({
        input: firstVideo,
        out,
        degradations: degParts.join(",") || undefined,
        config_overrides: configOverrides,
      });
      s.setJobId(result.job_id);
      s.clearVideoFile();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      s.setJobError(msg);
      s.setJobStatus("error");
    } finally {
      setStarting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); try { const file = e.dataTransfer?.files?.[0]; const path = (file as File & { path?: string } | undefined)?.path || file?.name; if (path) s.setVideoFile({ name: basename(path) ?? path, path }); } catch {} }}
        style={{ border: `2px dashed ${dragOver ? "var(--green)" : "var(--border)"}`, borderRadius: "var(--radius-md)", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 6, background: dragOver ? "var(--greenDim)" : "var(--bg2)", transition: "var(--transition-fast)", cursor: "pointer" }}>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Drop a video file here</span>
        <span style={{ fontSize: 10, color: "var(--dim)" }}>Supported: MKV, MP4, AVI, MOV</span>
        <Btn small variant="ghost" onClick={() => handleBrowse()} style={{ marginTop: 4 }}>
          Browse Files
        </Btn>
      </div>

      {s.videoFile && (
          <div style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 30px", padding: "6px 10px", alignItems: "center" }}>
              <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.videoFile.name}</span>
              <button onClick={() => s.clearVideoFile()} title="Remove"
                style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", fontSize: 14, lineHeight: 1, padding: 2, opacity: 0.4, transition: "var(--transition-fast)" }}
                onMouseEnter={(e) => { (e.target as HTMLButtonElement).style.opacity = "1"; }}
                onMouseLeave={(e) => { (e.target as HTMLButtonElement).style.opacity = "0.4"; }}>
                ✕
              </button>
            </div>
          </div>
      )}

      <ScaleBar />
      <DownsampleMethodSelector />
      <DegradationPanel />

      {s_status === "running" && (() => {
        const active = [...progressSteps].reverse().find((st) => st.status === "active");
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 4, padding: "8px 10px", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
              <span style={{ fontSize: 11, color: "var(--text)", fontWeight: 600 }}>
                {active ? active.desc : "Extracting frames…"}
              </span>
              {active && active.total != null && active.total > 0 && (
                <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                  {Math.round((active.current / active.total) * 100)}%
                </span>
              )}
            </div>
            <PBar value={active?.current ?? 0} max={active?.total ?? (active?.current || 1)} color="var(--amber)" height={5} />
          </div>
        );
      })()}

      {extractError && (
        <div style={{ border: "1px solid color-mix(in srgb, var(--red) 40%, transparent)", borderRadius: "var(--radius-sm)", padding: "7px 10px", fontSize: 10, color: "var(--red)", background: "color-mix(in srgb, var(--red) 10%, transparent)", lineHeight: 1.4 }}>
          {extractError}
        </div>
      )}

      {s.videoFile && s.jobStatus !== "running" && (
        <Btn variant="solid" onClick={handleStartExtraction} disabled={starting}>
          {starting ? "Starting..." : s.jobStatus === "done" ? "Start Another" : "Start Extraction"}
        </Btn>
      )}
    </div>
  );
}

function OnTheFlyMode() {
  return (
    <div style={{ background: "color-mix(in srgb, var(--amber) 10%, var(--bg2))", border: "1px solid color-mix(in srgb, var(--amber) 40%, transparent)", borderRadius: "var(--radius-md)", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, textAlign: "center" }}>
      <span style={{ fontSize: 14, color: "var(--amber)", fontWeight: 600 }}>On-the-fly</span>
      <span style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>Decode video directly during training — ~90% less disk usage.<br />Coming soon.</span>
    </div>
  );
}

function SummaryPanel() {
  const s = useDatasetStore();
  const modeLabel: Record<string, string> = { image_folder: "Pre-extracted", video_extract: "Video Extract", on_the_fly: "On-the-fly" };

  const rows: { label: string; value: string }[] = [{ label: "Mode", value: modeLabel[s.mode] || s.mode }];

  if (s.mode === "video_extract") {
    rows.push({ label: "Scale", value: `×${s.scale}` });
    rows.push({ label: "Downsample", value: s.kernel });
    rows.push({ label: "FPS", value: String(s.frameRate) });
    const activeDegs = [s.degBlur && "blur", s.degNoise && "noise", s.degJpeg && "jpeg", s.degJpeg2000 && "jpeg2000", s.degColorJitter && "color-jitter"].filter(Boolean);
    rows.push({ label: "Degradations", value: activeDegs.length ? activeDegs.join(", ") : "none" });
  }
  if (s.mode === "image_folder" && s.rootPath) {
    rows.push({ label: "Source", value: s.rootPath });
  }

  return (
    <div style={{ flex: 1, minWidth: 180, maxWidth: 300 }}>
      <Panel title="Dataset Summary">
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          {rows.map(({ label, value }) => (
            <div key={label} style={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <span style={{ fontSize: 9, color: "var(--dim)", textTransform: "uppercase", letterSpacing: 0.8, fontWeight: 600 }}>{label}</span>
              <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", wordBreak: "break-all" }}>{value}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

export function ScreenDatasetCreate() {
  const s = useDatasetStore();

  const typeCards: { id: DatasetMode; label: string; description: string }[] = [
    { id: "image_folder", label: "Pre-existing", description: "Import existing HR/LR dataset folders into the project." },
    { id: "video_extract", label: "Video Extract", description: "Extract frames from video files. Full degradation pipeline." },
    { id: "on_the_fly", label: "On-the-fly", description: "Decode video during training. ~90% less disk usage. (Coming soon)" },
  ];

  return (
    <div style={{ display: "flex", gap: 12, height: "100%", overflow: "auto", boxSizing: "border-box" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
          {typeCards.map((c) => (
            <TypeCard key={c.id} id={c.id} label={c.label} description={c.description} active={s.mode === c.id} disabled={c.id === "on_the_fly"} onClick={() => s.setMode(c.id)} />
          ))}
        </div>

        <Panel title={s.mode === "image_folder" ? "Pre-existing Dataset" : s.mode === "video_extract" ? "Video Extraction" : "On-the-fly"}>
          {s.mode === "image_folder" && <PreExistingMode />}
          {s.mode === "video_extract" && <VideoExtractMode />}
          {s.mode === "on_the_fly" && <OnTheFlyMode />}
        </Panel>
      </div>
      <SummaryPanel />
    </div>
  );
}