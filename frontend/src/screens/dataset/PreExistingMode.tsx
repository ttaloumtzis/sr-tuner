import { useState, useEffect, useRef } from "react";
import { PathInput } from "../../components/ui/PathInput";
import { Btn } from "../../components/ui/Btn";
import { useDatasetStore } from "../../store/datasetStore";
import { useProjectStore } from "../../store/projectStore";
import { basename, join, parentFromProjFile } from "../../lib/path";
import { inspectDataset, finalizeDataset } from "../../lib/api";
import type { DatasetInspectInfo } from "../../lib/api-types";
import { useToast } from "../../components/shell/ToastProvider";

export function PreExistingMode() {
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