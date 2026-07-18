import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { PathInput } from "../../components/ui/PathInput";
import { Dropdown } from "../../components/ui/Dropdown";
import { useDatasetStore } from "../../store/datasetStore";
import { useProjectStore } from "../../store/projectStore";
import { scanDatasets, type ScannedDataset } from "../../lib/scanDatasets";


interface MergePreview {
  scale: number;
  sourceDatasets: ScannedDataset[];
  totalPairs: number;
  outputPath: string;
}

export function ScreenMergeDatasets() {
  const s = useDatasetStore();
  const project = useProjectStore((s) => s.project);
  const projectDir = project ? project.filePath.replace(/\/[^/]+\.srproj$/, "") : "";
  const datasetsDir = projectDir ? projectDir + "/datasets" : "";
  const defaultOutput = datasetsDir ? datasetsDir + "/merged" : "";

  const [scanned, setScanned] = useState<ScannedDataset[]>([]);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<MergePreview[] | null>(null);
  const [mergeResults, setMergeResults] = useState<{ scale: number; output_path: string }[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!s.mergeOutputPath && defaultOutput) {
      s.setMergeOutputPath(defaultOutput);
    }
  }, [defaultOutput]);

  useEffect(() => {
    if (!datasetsDir) { setScanned([]); return; }
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const exists = await invoke<boolean>("path_exists", { path: datasetsDir });
        if (!exists) { setScanned([]); setLoading(false); return; }
        const ds = await scanDatasets(datasetsDir);
        setScanned(ds);
        setSelectedPaths(new Set(ds.map((d) => d.path)));
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, [datasetsDir]);

  const toggleDataset = (path: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
    setPreview(null);
    setMergeResults(null);
  };

  const filteredScanned = s.mergeScaleFilter
    ? scanned.filter((d) => d.scale === s.mergeScaleFilter)
    : scanned;

  const handlePreview = () => {
    setError(null);
    const selected = scanned.filter((d) => selectedPaths.has(d.path));
    if (selected.length === 0) { setError("Select at least one dataset"); return; }

    const groups: Record<number, ScannedDataset[]> = {};
    for (const ds of selected) {
      (groups[ds.scale] ??= []).push(ds);
    }

    const p: MergePreview[] = [];
    for (const [scaleStr, dsList] of Object.entries(groups)) {
      const scaleNum = Number(scaleStr);
      const totalPairs = dsList.reduce((sum, d) => sum + d.pairCount, 0);
      const dirName = s.mergeCustomName || `scale_${scaleNum}`;
      const outputPath = (s.mergeOutputPath || defaultOutput) + "/" + dirName;
      p.push({ scale: scaleNum, sourceDatasets: dsList, totalPairs, outputPath });
    }
    setPreview(p);
  };

  const handleMerge = async () => {
    setError(null);
    if (!datasetsDir) { setError("No project datasets directory"); return; }
    const selected = scanned.filter((d) => selectedPaths.has(d.path));
    if (selected.length === 0) { setError("Select at least one dataset"); return; }

    setLoading(true);
    try {
      const { mergeDatasets } = await import("../../lib/api");
      const res = await mergeDatasets({
        input: datasetsDir,
        out: s.mergeOutputPath || defaultOutput,
        scale: s.mergeScaleFilter ?? undefined,
        name: s.mergeCustomName || undefined,
        keep_sources: s.mergeKeepSources,
        input_datasets: selected.map((d) => d.path),
      });
      s.setJobType("merge");
      s.setJobId(res.job_id);
      s.setJobStatus("running");
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const scaleOptions = [...new Set(scanned.map((d) => d.scale))].sort();
  const scaleGroupsDetected = new Set(filteredScanned.map((d) => d.scale)).size;
  const showNameWarning = s.mergeCustomName.trim() !== "" && scaleGroupsDetected > 1;
  const isMerging = loading || (s.jobType === "merge" && s.jobStatus === "running");

  if (!project) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--dim)", fontSize: 12 }}>
        Load a project first to access dataset merge.
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", overflow: "auto", padding: "0 4px" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Source</label>
        <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", padding: "6px 10px", background: "var(--bg3)", borderRadius: "var(--radius-sm)" }}>{datasetsDir}</span>
        <span style={{ fontSize: 10, color: "var(--dim)" }}>Datasets are auto-detected from the project's datasets folder.</span>
      </div>

      {scaleOptions.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 10, color: "var(--muted)" }}>Filter scale:</span>
          <Dropdown value={String(s.mergeScaleFilter ?? "")} options={[{ value: "", label: "All" }, ...scaleOptions.map((o) => ({ value: String(o), label: `×${o}` }))]} onChange={(v) => s.setMergeScaleFilter(v ? Number(v) : null)} />
        </div>
      )}

      {scanned.length === 0 && !loading && (
        <div style={{ padding: "12px 0", fontSize: 11, color: "var(--dim)", textAlign: "center" }}>
          No datasets found in the project. Use the Create Dataset tab to add one.
        </div>
      )}

      {scanned.length > 0 && (
        <Panel title="Detected datasets">
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {filteredScanned.map((ds) => (
              <label key={ds.path} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 6px", borderRadius: "var(--radius-sm)", cursor: "pointer", fontSize: 11, color: "var(--text)" }}>
                <input type="checkbox" checked={selectedPaths.has(ds.path)} onChange={() => toggleDataset(ds.path)} style={{ accentColor: "var(--green)" }} />
                <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
                <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>×{ds.scale}</span>
                <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>{ds.pairCount} pairs</span>
                <span style={{ fontSize: 10, flexShrink: 0 }}>{ds.hasManifest ? "✅" : "⚠"}</span>
              </label>
            ))}
          </div>
        </Panel>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Output directory</label>
        <PathInput value={s.mergeOutputPath} onChange={s.setMergeOutputPath} browseTitle="Select output directory" mono />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>Custom name (optional)</label>
        <input value={s.mergeCustomName} onChange={(e) => s.setMergeCustomName(e.target.value)} placeholder="Leave empty for auto-naming (scale_N)"
          style={{ width: "100%", maxWidth: 300, background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 11, padding: "5px 10px", fontFamily: "var(--font-mono)", outline: "none" }} />
        {showNameWarning && (
          <span style={{ fontSize: 10, color: "var(--amber)", fontStyle: "italic" }}>
            Custom name with multiple scale groups will raise an error. Use the scale filter or leave name empty.
          </span>
        )}
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--muted)", cursor: "pointer" }}>
        <input type="checkbox" checked={s.mergeKeepSources} onChange={(e) => s.setMergeKeepSources(e.target.checked)} style={{ accentColor: "var(--green)" }} />
        Keep source datasets (don't delete after merge)
      </label>

      <div style={{ display: "flex", gap: 8 }}>
        <Btn onClick={handlePreview} disabled={scanned.length === 0 || isMerging}>Preview Merge</Btn>
        <Btn variant="solid" onClick={handleMerge} disabled={scanned.length === 0 || isMerging}>
          {isMerging ? "Merging..." : "Execute Merge"}
        </Btn>
      </div>

      {error && (
        <div style={{ background: "color-mix(in srgb, var(--red) 15%, var(--bg2))", border: "1px solid color-mix(in srgb, var(--red) 40%, transparent)", borderRadius: "var(--radius-sm)", padding: "8px 10px", fontSize: 10, color: "var(--red)", lineHeight: 1.5 }}>
          {error}
        </div>
      )}

      {preview && preview.length > 0 && (
        <Panel title="Merge Preview">
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {preview.map((p) => (
              <div key={p.scale} style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "8px 10px", display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "var(--green)" }}>Merge ×{p.scale}</span>
                  <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>{p.totalPairs} pairs total</span>
                </div>
                <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>→ {p.outputPath}</span>
                <span style={{ fontSize: 10, color: "var(--muted)" }}>Sources: {p.sourceDatasets.map((d) => d.name).join(", ")}</span>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {(mergeResults && mergeResults.length > 0) || (s.mergeResults && s.mergeResults.length > 0) ? (
        <Panel title="Merge Results">
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {(mergeResults || s.mergeResults)?.map((r) => (
              <div key={r.scale} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--green)" }}>
                <span>✓ Merged ×{r.scale}</span>
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--muted)", fontSize: 10 }}>→ {r.output_path}</span>
              </div>
            ))}
            {!s.mergeKeepSources && (
              <span style={{ fontSize: 10, color: "var(--amber)", fontStyle: "italic" }}>Source datasets have been removed.</span>
            )}
          </div>
        </Panel>
      ) : null}
    </div>
  );
}