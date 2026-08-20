// §14 Inference Screen — redesigned (settings drawer + result stage)
// Tasks: 14.1–14.13

import { useState, useCallback, useEffect } from "react";
import "./ScreenInference.css";
import { useInferenceStore } from "../../store/inferenceStore";
import { useInferenceSSE } from "../../hooks/useInferenceSSE";
import { basename, join } from "../../lib/path";
import { SettingsRail } from "./SettingsRail";
import { ComparisonPanel } from "./ComparisonPanel";
import { ResultFooter } from "./ResultFooter";

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
        ...(s.version
          ? { instance: s.instance ?? undefined, version: s.version }
          : s.modelPath
            ? { model: s.modelPath, ...(s.instance ? { instance: s.instance } : {}) }
            : {}),
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
    <div className="si-layout">
      <SettingsRail onRun={handleRun} onCancel={handleCancel} />

      <div className="si-stage">
        <ComparisonPanel splitterPct={splitterPct} onSplitterPctChange={setSplitterPct} />
        <ResultFooter />
      </div>
    </div>
  );
}