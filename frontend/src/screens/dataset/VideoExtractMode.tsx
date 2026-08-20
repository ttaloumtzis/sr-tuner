import { useState, useEffect } from "react";
import { Btn } from "../../components/ui/Btn";
import { PBar } from "../../components/ui/PBar";
import { DropZone } from "../../components/ui/DropZone";
import { useDatasetStore } from "../../store/datasetStore";
import { useProjectStore } from "../../store/projectStore";
import { basename, join, parentFromProjFile } from "../../lib/path";
import { ScaleBar } from "./ScaleBar";
import { DownsampleMethodSelector } from "./DownsampleMethodSelector";
import { DegradationPanel } from "./DegradationPanel";

export function VideoExtractMode() {
  const s = useDatasetStore();
  const s_error = useDatasetStore((s) => s.jobError);
  const s_status = useDatasetStore((s) => s.jobStatus);
  const progressSteps = useDatasetStore((s) => s.progressSteps);
  const project = useProjectStore((s) => s.project);
  const [starting, setStarting] = useState(false);
  const [extractError, setExtractError] = useState<string | null>(null);

  useEffect(() => {
    if (s_status === "error" && s_error) setExtractError(s_error);
    else if (s_status === "idle") setExtractError(null);
  }, [s_status, s_error]);

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
      <DropZone
        large
        selectedAsRow
        osDrag
        label="Drop a video file here"
        sublabel="Supported: MKV, MP4, AVI, MOV"
        browseLabel="Browse Files"
        path={s.videoFile?.path ?? null}
        name={s.videoFile?.name}
        fileFilters={[{ name: "Video", extensions: ["mkv", "mp4", "avi", "mov"] }]}
        onSelect={(p) => s.setVideoFile({ name: basename(p) ?? p, path: p })}
        onClear={() => s.clearVideoFile()}
      />

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