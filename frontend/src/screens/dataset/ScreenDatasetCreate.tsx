import { useState, useEffect, useCallback } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { PathInput } from "../../components/ui/PathInput";
import { useDatasetStore, type DatasetMode, type DownscaleKernel } from "../../store/datasetStore";
import { useProjectStore } from "../../store/projectStore";
import { DegradationPanel } from "./DegradationPanel";
import { validateNamingPattern, previewFilename } from "../../lib/namingPattern";


function TypeCard({ id: _id, label, description, active, onClick }: {
  id: DatasetMode; label: string; description: string; active: boolean; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <div onClick={onClick} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
      style={{ background: active ? "var(--greenDim)" : hovered ? "var(--bg2)" : "var(--bg1)", border: `1px solid ${active ? "var(--green)" : hovered ? "var(--muted)" : "var(--border)"}`, borderRadius: "var(--radius-md)", padding: "12px 14px", cursor: "pointer", transition: "var(--transition-fast)", display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: active ? "var(--green)" : "var(--text)" }}>{label}</span>
      <span style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>{description}</span>
    </div>
  );
}

function ScaleNamingBar() {
  const s = useDatasetStore();
  const presets = [1, 2, 4, 8];
  const [customVal, setCustomVal] = useState(presets.includes(s.scale) ? "" : String(s.scale));
  const [customActive, setCustomActive] = useState(!presets.includes(s.scale));
  const patternError = validateNamingPattern(s.namingPattern);
  const isValid = !patternError;
  const preview = isValid ? previewFilename(s.namingPattern) : "invalid";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 10px", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", flexWrap: "wrap" }}>
      <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Scale</span>
      {presets.map((p) => (
        <button key={p} onClick={() => { s.setScale(p); setCustomActive(false); setCustomVal(""); }}
          style={{ background: s.scale === p && !customActive ? "var(--green)" : "var(--bg3)", border: `1px solid ${s.scale === p && !customActive ? "var(--green)" : "var(--border)"}`, color: s.scale === p && !customActive ? "#0d0f11" : "var(--muted)", fontSize: 11, fontWeight: 600, padding: "2px 9px", borderRadius: "var(--radius-sm)", cursor: "pointer", transition: "var(--transition-fast)" }}>
          ×{p}
        </button>
      ))}
      <input value={customVal} onChange={(e) => { setCustomActive(true); const n = parseInt(e.target.value, 10); if (!isNaN(n) && n > 0) s.setScale(n); setCustomVal(e.target.value); }}
        onFocus={() => setCustomActive(true)} placeholder="custom"
        style={{ width: 68, background: customActive ? "var(--greenDim)" : "var(--bg3)", border: `1px solid ${customActive ? "var(--green)" : "var(--border)"}`, color: "var(--text)", fontSize: 11, padding: "2px 7px", borderRadius: "var(--radius-sm)", outline: "none", fontFamily: "var(--font-mono)" }} />
      <div style={{ width: 1, height: 16, background: "var(--border)" }} />
      <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Naming</span>
      <input value={s.namingPattern} onChange={(e) => s.setNamingPattern(e.target.value)} placeholder="%06d"
        style={{ width: 68, background: patternError ? "color-mix(in srgb, var(--red) 15%, var(--bg3))" : "var(--bg3)", border: `1px solid ${patternError ? "var(--red)" : "var(--border)"}`, color: "var(--text)", fontSize: 11, padding: "2px 7px", borderRadius: "var(--radius-sm)", outline: "none", fontFamily: "var(--font-mono)" }} />
      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: isValid ? "var(--green)" : "var(--red)" }}>→ {preview}</span>
      {patternError && <span style={{ fontSize: 10, color: "var(--red)" }}>{patternError}</span>}
    </div>
  );
}

function DownsampleMethodSelector() {
  const s = useDatasetStore();
  const options: { id: DownscaleKernel; label: string }[] = [
    { id: "area", label: "Area" }, { id: "bicubic", label: "Bicubic" }, { id: "bilinear", label: "Bilinear" },
    { id: "lanczos", label: "Lanczos" }, { id: "nearest", label: "Nearest" }, { id: "real-world", label: "Real-World" },
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
  const [detectedHr, setDetectedHr] = useState(0);
  const [detectedLr, setDetectedLr] = useState(0);

  useEffect(() => {
    if (!s.rootPath) { setDetectedHr(0); setDetectedLr(0); return; }
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const hrFiles: string[] = await invoke("list_image_files", { path: s.rootPath + "/HR" });
        const lrFiles: string[] = await invoke("list_image_files", { path: s.rootPath + "/LR" });
        setDetectedHr(hrFiles.length);
        setDetectedLr(lrFiles.length);
      } catch { setDetectedHr(0); setDetectedLr(0); }
    })();
  }, [s.rootPath]);

  const handleImport = async () => {
    if (!s.rootPath || !project) return;
    const { invoke } = await import("@tauri-apps/api/core");
    const projectDir = project.filePath.replace(/\/[^/]+\.srproj$/, "");
    const name = s.rootPath.split("/").pop() || "imported";
    const dst = projectDir + "/datasets/" + name;
    await invoke("copy_directory", { src: s.rootPath, dst });
  };

  const handleValidate = async () => {
    const { validateDatasetPath } = await import("../../lib/api");
    const res = await validateDatasetPath({ path: s.rootPath });
    alert(`Valid: ${res.valid}\nProblems: ${res.problems.join("\n")}`);
  };

  const handleHealth = async () => {
    s.clearJob();
    s.setJobType("health");
    const { healthCheck } = await import("../../lib/api");
    const res = await healthCheck({ path: s.rootPath });
    s.setJobId(res.job_id);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Dataset root folder</label>
        <PathInput value={s.rootPath} onChange={s.setRootPath} browseTitle="Select dataset root folder (containing HR/ and LR/)" mono />
        <span style={{ fontSize: 10, color: "var(--dim)" }}>Select the root folder containing HR/ and LR/ subdirectories</span>
      </div>
      {s.rootPath && (
        <div style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "8px 10px", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 11, color: "var(--muted)" }}>HR: <span style={{ color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>{detectedHr}</span></span>
          <span style={{ fontSize: 11, color: "var(--muted)" }}>LR: <span style={{ color: "var(--green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>{detectedLr}</span></span>
          <Btn small onClick={handleValidate}>Validate</Btn>
          <Btn small onClick={handleHealth}>Health Check</Btn>
          <Btn small variant="solid" onClick={handleImport} disabled={!project}>Import into project</Btn>
        </div>
      )}
    </div>
  );
}

function VideoExtractMode() {
  const s = useDatasetStore();
  const s_error = useDatasetStore((s) => s.jobError);
  const s_status = useDatasetStore((s) => s.jobStatus);
  const project = useProjectStore((s) => s.project);
  const [dragOver, setDragOver] = useState(false);
  const [starting, setStarting] = useState(false);
  const [extractError, setExtractError] = useState<string | null>(null);

  const handleBrowse = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: true,
        filters: [{ name: "Video", extensions: ["mkv", "mp4", "avi", "mov"] }],
      });
      if (selected) {
        const paths = Array.isArray(selected) ? selected : [selected];
        s.addVideoFiles(paths.filter(Boolean));
      }
    } catch {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".mkv,.mp4,.avi,.mov";
      input.multiple = true;
      input.onchange = () => {
        const paths = Array.from(input.files ?? []).map((f) => (f as File & { path?: string }).path || f.name).filter(Boolean);
        if (paths.length > 0) s.addVideoFiles(paths);
      };
      input.click();
    }
  }, [s.addVideoFiles]);

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
            if (event.payload.paths && event.payload.paths.length > 0) {
              s.addVideoFiles(event.payload.paths as string[]);
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
      const patternErr = validateNamingPattern(s.namingPattern);
      if (patternErr) { s.setJobError(patternErr); s.setJobStatus("error"); return; }
      if (!project) { s.setJobError("No project loaded"); s.setJobStatus("error"); return; }
      const projectDir = project.filePath.replace(/\/[^/]+\.srproj$/, "");
      const firstVideo = s.videoFiles[0]?.path;
      if (!firstVideo) { s.setJobError("No video file selected"); s.setJobStatus("error"); return; }
      const videoName = firstVideo.split("/").pop()?.replace(/\.[^/.]+$/, "") || "extracted";
      const out = projectDir + "/datasets/" + videoName;
      const degParts: string[] = [];
      if (s.degBlur) degParts.push("blur");
      if (s.degNoise) degParts.push("noise");
      if (s.degJpeg) degParts.push("jpeg");
      if (s.degJpeg2000) degParts.push("jpeg2000");
      if (s.degColorJitter) degParts.push("color-jitter");

      const configOverrides: Record<string, unknown> = {};
      configOverrides["scale"] = s.scale;
      configOverrides["frame_rate"] = s.frameRate;
      configOverrides["frame_format"] = s.frameFormat;
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
      const resizeMethod = s.kernel === "real-world" ? "area" : s.kernel;
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
        onDrop={(e) => { e.preventDefault(); setDragOver(false); try { const paths = Array.from(e.dataTransfer?.files ?? []).map((f) => (f as File & { path?: string }).path || f.name).filter(Boolean); if (paths.length > 0) s.addVideoFiles(paths); } catch {} }}
        style={{ border: `2px dashed ${dragOver ? "var(--green)" : "var(--border)"}`, borderRadius: "var(--radius-md)", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 6, background: dragOver ? "var(--greenDim)" : "var(--bg2)", transition: "var(--transition-fast)", cursor: "pointer" }}>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Drop video files here</span>
        <span style={{ fontSize: 10, color: "var(--dim)" }}>Supported: MKV, MP4, AVI, MOV</span>
        <Btn small variant="ghost" onClick={() => handleBrowse()} style={{ marginTop: 4 }}>
          Browse Files
        </Btn>
      </div>

      {s.videoFiles.length > 0 && (
        <div style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 80px", padding: "5px 10px", background: "var(--bg2)", borderBottom: "1px solid var(--border)" }}>
            {["Filename", "Status"].map((h) => (
              <span key={h} style={{ fontSize: 9, color: "var(--dim)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.8 }}>{h}</span>
            ))}
          </div>
          {s.videoFiles.map((f, i) => (
            <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 80px", padding: "6px 10px", borderBottom: i < s.videoFiles.length - 1 ? "1px solid var(--border)" : undefined, alignItems: "center" }}>
              <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name}</span>
              <span style={{ fontSize: 10, fontWeight: 600, color: f.status === "done" ? "var(--green)" : f.status === "extracting" ? "var(--amber)" : "var(--dim)", background: f.status === "done" ? "var(--greenDim)" : f.status === "extracting" ? "color-mix(in srgb, var(--amber) 15%, transparent)" : "var(--bg3)", border: `1px solid ${f.status === "done" ? "var(--green)" : f.status === "extracting" ? "var(--amber)" : "var(--border)"}`, borderRadius: 8, padding: "2px 7px", display: "inline-block" }}>{f.status}</span>
            </div>
          ))}
        </div>
      )}

      <ScaleNamingBar />
      <DownsampleMethodSelector />
      <DegradationPanel />

      {extractError && (
        <div style={{ border: "1px solid color-mix(in srgb, var(--red) 40%, transparent)", borderRadius: "var(--radius-sm)", padding: "7px 10px", fontSize: 10, color: "var(--red)", background: "color-mix(in srgb, var(--red) 10%, transparent)", lineHeight: 1.4 }}>
          {extractError}
        </div>
      )}

      {s.videoFiles.length > 0 && s.jobStatus !== "running" && (
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

  const rows: { label: string; value: string }[] = [
    { label: "Mode", value: modeLabel[s.mode] || s.mode },
    { label: "Scale", value: `×${s.scale}` },
    { label: "Downsample", value: s.kernel === "real-world" ? "Real-World" : s.kernel },
  ];

  if (s.mode === "video_extract") {
    rows.push({ label: "FPS", value: String(s.frameRate) });
    const activeDegs = [s.degBlur && "blur", s.degNoise && "noise", s.degJpeg && "jpeg", s.degJpeg2000 && "jpeg2000", s.degColorJitter && "color-jitter"].filter(Boolean);
    rows.push({ label: "Degradations", value: activeDegs.length ? activeDegs.join(", ") : "none" });
  }
  if (s.mode === "image_folder" && s.rootPath) {
    rows.push({ label: "Source", value: s.rootPath });
  }

  return (
    <div style={{ width: 220, flexShrink: 0 }}>
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
    { id: "image_folder", label: "Pre-existing", description: "Import existing HR/LR dataset folders. Validate and add to project." },
    { id: "video_extract", label: "Video Extract", description: "Extract frames from video files. Full degradation pipeline." },
    { id: "on_the_fly", label: "On-the-fly", description: "Decode video during training. ~90% less disk usage. (Coming soon)" },
  ];

  return (
    <div style={{ display: "flex", gap: 12, height: "100%", overflow: "auto", boxSizing: "border-box" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
          {typeCards.map((c) => (
            <TypeCard key={c.id} id={c.id} label={c.label} description={c.description} active={s.mode === c.id} onClick={() => s.setMode(c.id)} />
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