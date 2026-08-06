import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useDatasetStore } from "../../../store/datasetStore";
import { useProjectStore } from "../../../store/projectStore";
import { scanDatasets, type ScannedDataset } from "../../../lib/scanDatasets";
import { join, parentFromProjFile } from "../../../lib/path";
import {
  buildPlans,
  excludeMerged,
  type MergePlan,
} from "./mergePlanner";
import { DatasetPicker } from "./DatasetPicker";
import { MergeSettings } from "./MergeSettings";
import { MergePreview } from "./MergePreview";
import { MergeProgress } from "./MergeProgress";
import { MergeResults } from "./MergeResults";

export function ScreenMergeDatasets() {
  const s = useDatasetStore();
  const project = useProjectStore((p) => p.project);
  const projectDir = project ? parentFromProjFile(project.filePath) : "";
  const datasetsDir = projectDir ? join(projectDir, "datasets") : "";

  const [scanned, setScanned] = useState<ScannedDataset[]>([]);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<MergePlan | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scanSeq = useRef(0);

  const refreshScan = useCallback(async () => {
    const seq = ++scanSeq.current;
    if (!datasetsDir) {
      setScanned([]);
      setSelectedPaths(new Set());
      return;
    }
    setScanning(true);
    setError(null);
    try {
      const exists = await invoke<boolean>("path_exists", { path: datasetsDir });
      if (scanSeq.current !== seq) return;
      if (!exists) {
        setScanned([]);
        setSelectedPaths(new Set());
        return;
      }
      const raw = await scanDatasets(datasetsDir);
      if (scanSeq.current !== seq) return;
      const list = excludeMerged(raw);
      setScanned(list);
      setSelectedPaths(new Set(list.map((d) => d.path)));
    } catch (e) {
      if (scanSeq.current === seq) setError(String(e));
    } finally {
      if (scanSeq.current === seq) setScanning(false);
    }
  }, [datasetsDir]);

  useEffect(() => {
    void refreshScan();
    return () => {
      scanSeq.current += 1;
    };
  }, [refreshScan]);

  useEffect(() => {
    if (s.jobType === "merge" && s.jobStatus === "done") {
      void refreshScan();
    }
  }, [s.jobType, s.jobStatus, refreshScan]);

  const toggleDataset = (path: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
    setPreview(null);
  };

  const selectVisible = (paths: string[]) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      for (const p of paths) next.add(p);
      return next;
    });
    setPreview(null);
  };

  const clearVisible = (paths: string[]) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      for (const p of paths) next.delete(p);
      return next;
    });
    setPreview(null);
  };

  const selected = scanned.filter((d) => selectedPaths.has(d.path));
  const livePlan = buildPlans({
    selected,
    outputPath: datasetsDir,
    customName: s.mergeCustomName,
  });
  const scaleOptions = [...new Set(scanned.map((d) => d.scale))].sort();

  const handlePreview = async () => {
    setError(null);
    if (selected.length === 0) {
      setError("Select at least one dataset");
      return;
    }
    const base = buildPlans({
      selected,
      outputPath: datasetsDir,
      customName: s.mergeCustomName,
    });
    const exists = await Promise.all(
      base.groups.map((g) => invoke<boolean>("path_exists", { path: g.outputPath })),
    );
    const existingTargets = base.groups
      .filter((_, i) => exists[i])
      .map((g) => g.outputPath);
    setPreview({ ...base, warnings: { ...base.warnings, existingTargets } });
  };

  const handleMerge = async () => {
    setError(null);
    if (!datasetsDir) {
      setError("No project datasets directory");
      return;
    }
    if (selected.length === 0) {
      setError("Select at least one dataset");
      return;
    }
    try {
      const { mergeDatasets } = await import("../../../lib/api");
      const res = await mergeDatasets({
        input: datasetsDir,
        name: s.mergeCustomName.trim() || undefined,
        keep_sources: s.mergeKeepSources,
        input_datasets: selected.map((d) => d.path),
      });
      s.setMergeResults(null);
      s.setJobType("merge");
      s.setJobId(res.job_id);
      s.setJobStatus("running");
    } catch (e) {
      setError(String(e));
    }
  };

  const isMerging = s.jobType === "merge" && s.jobStatus === "running";
  const mergeFailed = s.jobType === "merge" && s.jobStatus === "error";

  if (!project) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "var(--dim)",
          fontSize: 12,
        }}
      >
        Load a project first to access dataset merge.
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        height: "100%",
        overflow: "auto",
        padding: "0 4px",
        boxSizing: "border-box",
      }}
    >
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
        <div style={{ flex: "0 0 340px", minWidth: 280 }}>
          <DatasetPicker
            scanned={scanned}
            selectedPaths={selectedPaths}
            scanning={scanning}
            scaleOptions={scaleOptions}
            scaleFilter={s.mergeScaleFilter}
            onScaleFilterChange={s.setMergeScaleFilter}
            onToggle={toggleDataset}
            onSelectVisible={selectVisible}
            onClearVisible={clearVisible}
          />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <MergeSettings
            plan={livePlan}
            datasetsDir={datasetsDir}
            customName={s.mergeCustomName}
            onCustomNameChange={s.setMergeCustomName}
            keepSources={s.mergeKeepSources}
            onKeepSourcesChange={s.setMergeKeepSources}
            scanning={scanning}
            isMerging={isMerging}
            hasSelection={selected.length > 0}
            onPreview={handlePreview}
            onMerge={handleMerge}
          />
        </div>
      </div>

      {error && (
        <div
          style={{
            background: "color-mix(in srgb, var(--red) 15%, var(--bg2))",
            border: "1px solid color-mix(in srgb, var(--red) 40%, transparent)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 10px",
            fontSize: 10,
            color: "var(--red)",
            lineHeight: 1.5,
          }}
        >
          {error}
        </div>
      )}

      {isMerging && <MergeProgress />}

      {mergeFailed && s.jobError && (
        <div
          style={{
            background: "color-mix(in srgb, var(--red) 15%, var(--bg2))",
            border: "1px solid color-mix(in srgb, var(--red) 40%, transparent)",
            borderRadius: "var(--radius-sm)",
            padding: "8px 10px",
            fontSize: 10,
            color: "var(--red)",
            lineHeight: 1.5,
          }}
        >
          Merge failed: {s.jobError}
        </div>
      )}

      {preview && preview.groups.length > 0 && <MergePreview plan={preview} />}

      {s.mergeResults && s.mergeResults.length > 0 && (
        <MergeResults results={s.mergeResults} keepSources={s.mergeKeepSources} />
      )}
    </div>
  );
}
