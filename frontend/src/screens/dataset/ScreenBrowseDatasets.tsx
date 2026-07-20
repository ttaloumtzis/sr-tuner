import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { convertFileSrc } from "@tauri-apps/api/core";
import { Btn } from "../../components/ui/Btn";
import { Panel } from "../../components/ui/Panel";
import { useProjectStore } from "../../store/projectStore";
import { useDatasetStore } from "../../store/datasetStore";
import { scanDatasets, listDatasetPairs, type ScannedDataset } from "../../lib/scanDatasets";
import { join, parentFromProjFile } from "../../lib/path";

function DatasetListItem({ ds, active, onClick }: { ds: ScannedDataset; active: boolean; onClick: () => void }) {
  const [hovered, setHovered] = useState(false);
  const status = ds.hasManifest ? "✅" : "⚠";
  return (
    <div onClick={onClick} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
      style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", cursor: "pointer", background: active ? "var(--greenDim)" : hovered ? "var(--bg2)" : "transparent", borderRadius: "var(--radius-sm)", transition: "var(--transition-fast)" }}>
      <span style={{ fontSize: 14, flexShrink: 0 }}>📁</span>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 1 }}>
        <span style={{ fontSize: 11, color: "var(--text)", fontWeight: active ? 600 : 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
        <span style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>×{ds.scale} · {ds.pairCount} pairs</span>
      </div>
      <span style={{ fontSize: 10, flexShrink: 0 }} title={ds.hasManifest ? "Has manifest" : "Needs validation"}>{status}</span>
    </div>
  );
}

function ThumbStrip({ pairs, current, onSelect }: { pairs: { hr: string; lr: string }[]; current: number; onSelect: (i: number) => void }) {
  const total = pairs.length;
  const half = 10;
  const start = Math.max(0, current - half);
  const end = Math.min(total, current + half + 1);
  const visible = [];

  for (let i = start; i < end; i++) {
    visible.push(i);
  }

  return (
    <div style={{ display: "flex", gap: 4, overflow: "hidden", alignItems: "center", justifyContent: "center", padding: "4px 0" }}>
      <button onClick={() => onSelect(Math.max(0, current - 20))} disabled={current === 0}
        style={{ background: "none", border: "none", color: "var(--muted)", fontSize: 12, cursor: current === 0 ? "default" : "pointer", padding: "2px 4px", opacity: current === 0 ? 0.3 : 1 }}>
        ◄
      </button>
      {visible.map((i) => (
        <div key={i} onClick={() => onSelect(i)}
          style={{ width: 40, height: 30, borderRadius: "var(--radius-sm)", overflow: "hidden", border: i === current ? "2px solid var(--green)" : "2px solid transparent", cursor: "pointer", opacity: i === current ? 1 : 0.6, flexShrink: 0, transition: "var(--transition-fast)" }}>
          <img src={convertFileSrc(pairs[i].lr)} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </div>
      ))}
      <button onClick={() => onSelect(Math.min(total - 1, current + 20))} disabled={current >= total - 1}
        style={{ background: "none", border: "none", color: "var(--muted)", fontSize: 12, cursor: current >= total - 1 ? "default" : "pointer", padding: "2px 4px", opacity: current >= total - 1 ? 0.3 : 1 }}>
        ►
      </button>
    </div>
  );
}

export function ScreenBrowseDatasets() {
  const project = useProjectStore((s) => s.project);
  const store = useDatasetStore();
  const [datasets, setDatasets] = useState<ScannedDataset[]>([]);
  const [selected, setSelected] = useState<ScannedDataset | null>(null);
  const [pairs, setPairs] = useState<{ hr: string; lr: string }[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [pruneResult, setPruneResult] = useState<{ pruned: number; valid: boolean; problems: string[] } | null>(null);
  const [showBlackFrames, setShowBlackFrames] = useState(false);
  const [healthReport, setHealthReport] = useState<Record<string, unknown> | null>(null);
  const prevJobStatusRef = useRef(store.jobStatus);

  const projectDir = project ? parentFromProjFile(project.filePath) : "";
  const datasetsDir = projectDir ? join(projectDir, "datasets") : "";

  const blackFrames = (healthReport?.black_frames as string[]) ?? [];

  const refresh = useCallback(async () => {
    if (!datasetsDir) { setDatasets([]); return; }
    setLoading(true);
    try {
      const exists = await invoke<boolean>("path_exists", { path: datasetsDir });
      if (!exists) { setDatasets([]); return; }
      const ds = await scanDatasets(datasetsDir);
      setDatasets(ds);
      if (ds.length > 0 && !selected) {
        setSelected(ds[0]);
      }
    } finally {
      setLoading(false);
    }
  }, [datasetsDir]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!selected) { setPairs([]); return; }
    (async () => {
      const p = await listDatasetPairs(selected.path);
      setPairs(p);
      setCurrentIndex(0);
    })();
  }, [selected]);

  // Fetch cached health report from backend when dataset changes
  useEffect(() => {
    if (!selected) { setHealthReport(null); return; }
    (async () => {
      try {
        const { getDatasetHealth } = await import("../../lib/api");
        const report = await getDatasetHealth(selected.path);
        setHealthReport(report);
      } catch {
        setHealthReport(null);
      }
    })();
  }, [selected]);

  useEffect(() => {
    const prev = prevJobStatusRef.current;
    prevJobStatusRef.current = store.jobStatus;
    if (store.jobStatus === "done" && prev === "running") {
      if (store.jobType === "prune") {
        setPruneResult({ pruned: blackFrames.length, valid: true, problems: [] });
        refresh();
      } else if (store.jobType === "build") {
        refresh();
      } else if (store.jobType === "validate") {
        refresh();
      } else if (store.jobType === "health") {
        (async () => {
          try {
            const { getDatasetHealth } = await import("../../lib/api");
            const report = await getDatasetHealth(selected?.path ?? "");
            setHealthReport(report);
          } catch {
            setHealthReport(null);
          }
        })();
      }
    }
  }, [store.jobStatus]);

  const handleSelect = (ds: ScannedDataset) => {
    setSelected(ds);
    setPruneResult(null);
  };

  const handleValidate = async () => {
    if (!selected) return;
    store.clearJob();
    store.setJobType("validate");
    store.setJobStatus("running");
    try {
      const { startValidateDataset } = await import("../../lib/api");
      const res = await startValidateDataset({ path: selected.path });
      store.setJobId(res.job_id);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      store.setJobError(msg);
      store.setJobStatus("error");
    }
  };

  const handleHealth = async () => {
    if (!selected) return;
    try {
      store.clearJob();
      store.setJobType("health");
      const { healthCheck } = await import("../../lib/api");
      const res = await healthCheck({ path: selected.path, yes: false });
      store.setJobId(res.job_id);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      store.setJobError(msg);
      store.setJobStatus("error");
    }
  };

  const handlePrune = async () => {
    if (!selected) return;
    store.clearJob();
    store.setJobType("prune");
    store.setJobStatus("running");
    try {
      const { pruneBlackFrames } = await import("../../lib/api");
      const res = await pruneBlackFrames({
        path: selected.path,
        black_frames: blackFrames,
      });
      store.setJobId(res.job_id);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      store.setJobError(msg);
      store.setJobStatus("error");
    }
  };

  const handleOpen = async () => {
    if (!selected) return;
    await invoke("open_in_file_manager", { path: selected.path });
  };

  const handleDelete = async () => {
    if (!selected) return;
    if (!confirm(`Delete dataset "${selected.name}"? This cannot be undone.`)) return;
    await invoke("delete_directory", { path: selected.path });
    refresh();
    setSelected(null);
  };

  const currentPair = pairs[currentIndex];

  const resolutions = healthReport?.resolutions as Record<string, number> | undefined;
  const channels = healthReport?.channels as Record<string, number> | undefined;

  return (
    <div style={{ display: "flex", gap: 0, height: "100%", overflow: "hidden" }}>
      <div style={{ flex: 1, minWidth: 180, maxWidth: 280, display: "flex", flexDirection: "column", borderRight: "1px solid var(--border)", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 10px", borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)" }}>Datasets</span>
          <Btn small onClick={refresh} title="Refresh">↻</Btn>
        </div>
        <div style={{ flex: 1, overflow: "auto", padding: "4px 6px" }}>
          {loading && <div style={{ padding: 10, textAlign: "center", fontSize: 10, color: "var(--dim)" }}>Scanning...</div>}
          {!loading && datasets.length === 0 && (
            <div style={{ padding: 16, textAlign: "center", color: "var(--dim)", fontSize: 11, lineHeight: 1.5 }}>
              No datasets found.<br />Go to Create Dataset tab to add one.
            </div>
          )}
          {datasets.map((ds) => (
            <DatasetListItem key={ds.path} ds={ds} active={selected?.path === ds.path} onClick={() => handleSelect(ds)} />
          ))}
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {!selected && (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--dim)", fontSize: 12 }}>
            Select a dataset from the left panel
          </div>
        )}

        {selected && (
          <>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 12px", borderBottom: "1px solid var(--border)", background: "var(--bg1)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{selected.name}</span>
                <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>×{selected.scale} · {pairs.length} pairs</span>
              </div>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <Btn small color="var(--green)" onClick={handleValidate}>Validate</Btn>
                <Btn small color="var(--amber)" onClick={handleHealth} disabled={store.jobStatus === "running"}>Health</Btn>
                <Btn small onClick={handleOpen}>Open</Btn>
                <div style={{ flex: 1 }} />
                <Btn small color="var(--red)" onClick={handleDelete}>Delete</Btn>
              </div>
            </div>

            {healthReport && (
              <div style={{ padding: "6px 12px" }}>
                <Panel title="Health Report" actions={
                  <button onClick={() => setHealthReport(null)}
                    style={{ background: "none", border: "none", color: "var(--muted)", fontSize: 12, cursor: "pointer", padding: "0 4px", lineHeight: 1 }}>
                    ✕
                  </button>
                }>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 11 }}>
                    <div>Total images: <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--green)" }}>{String(healthReport.total_images ?? "?")}</span></div>
                    <div>Computed threshold: <span style={{ fontFamily: "var(--font-mono)" }}>{String(healthReport.computed_threshold ?? "?")}</span></div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      Black frames: <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: blackFrames.length ? "var(--red)" : "var(--green)" }}>
                        {blackFrames.length}
                      </span>
                      {blackFrames.length > 0 && (
                        <>
                          <Btn small onClick={handlePrune} disabled={store.jobStatus === "running"}>
                            Prune {blackFrames.length}
                          </Btn>
                          <button onClick={() => setShowBlackFrames(!showBlackFrames)}
                            style={{ background: "none", border: "none", color: "var(--muted)", fontSize: 11, cursor: "pointer", padding: 2 }}>
                            {showBlackFrames ? "▲" : "▼"}
                          </button>
                        </>
                      )}
                    </div>
                    {showBlackFrames && blackFrames.length > 0 && (
                      <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--dim)", maxHeight: 100, overflow: "auto", padding: "4px 0" }}>
                        {blackFrames.map((f) => <div key={f}>{f}</div>)}
                      </div>
                    )}
                    {resolutions && (
                      <div style={{ fontSize: 10, color: "var(--muted)" }}>
                        Resolutions: {Object.entries(resolutions).map(([k, v]) => `${k} (${v})`).join(", ")}
                      </div>
                    )}
                    {channels && (
                      <div style={{ fontSize: 10, color: "var(--muted)" }}>
                        Channels: {Object.entries(channels).map(([k, v]) => `${k} (${v})`).join(", ")}
                      </div>
                    )}
                  </div>
                </Panel>
              </div>
            )}

            {!healthReport && selected && (
              <div style={{ padding: "6px 12px" }}>
                <Panel title="Dataset Info">
                  <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 11 }}>
                    <div>Pairs: <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--green)" }}>{selected.pairCount}</span></div>
                    <div>Scale: <span style={{ fontFamily: "var(--font-mono)" }}>×{selected.scale}</span></div>
                    <div>Manifest: <span style={{ fontFamily: "var(--font-mono)", color: selected.hasManifest ? "var(--green)" : "var(--amber)" }}>{selected.hasManifest ? "Yes" : "No"}</span></div>
                    <div style={{ fontSize: 10, color: "var(--dim)", fontStyle: "italic" }}>Run Health Check for resolution and quality data</div>
                  </div>
                </Panel>
              </div>
            )}

            {store.validationResult && (
              <div style={{
                margin: "0 12px 6px",
                padding: "7px 10px",
                borderRadius: "var(--radius-sm)",
                fontSize: 10,
                background: store.validationResult.valid ? "var(--greenDim)" : "color-mix(in srgb, var(--red) 15%, var(--bg2))",
                border: `1px solid ${store.validationResult.valid ? "var(--green)" : "color-mix(in srgb, var(--red) 40%, transparent)"}`,
                color: store.validationResult.valid ? "var(--green)" : "var(--red)",
                lineHeight: 1.4,
              }}>
                Validation: {store.validationResult.valid ? "OK" : "FAILED"} — {store.validationResult.num_pairs} pair(s)
                {store.validationResult.problems.length > 0 && (
                  <div style={{ marginTop: 4, color: "var(--muted)" }}>{store.validationResult.problems.join("; ")}</div>
                )}
              </div>
            )}

            {pruneResult && (
              <div style={{
                margin: "0 12px 6px",
                padding: "7px 10px",
                borderRadius: "var(--radius-sm)",
                fontSize: 10,
                background: pruneResult.valid ? "var(--greenDim)" : "color-mix(in srgb, var(--red) 15%, var(--bg2))",
                border: `1px solid ${pruneResult.valid ? "var(--green)" : "color-mix(in srgb, var(--red) 40%, transparent)"}`,
                color: pruneResult.valid ? "var(--green)" : "var(--red)",
                lineHeight: 1.4,
              }}>
                Pruned {pruneResult.pruned} frame(s)
                {pruneResult.problems.length > 0 && (
                  <div style={{ marginTop: 4, color: "var(--muted)" }}>{pruneResult.problems.join("; ")}</div>
                )}
              </div>
            )}

            {pairs.length > 0 && currentPair && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", padding: "6px 12px", gap: 6 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, flexShrink: 0 }}>
                  <Btn small onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))} disabled={currentIndex === 0}>◀ Prev</Btn>
                  <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", minWidth: 60, textAlign: "center" }}>
                    {currentIndex + 1} / {pairs.length}
                  </span>
                  <Btn small onClick={() => setCurrentIndex(Math.min(pairs.length - 1, currentIndex + 1))} disabled={currentIndex >= pairs.length - 1}>Next ▶</Btn>
                </div>

                <div style={{ flex: 1, display: "flex", gap: 12, overflow: "hidden", alignItems: "center", justifyContent: "center" }}>
                  <div style={{ flex: "0 0 auto", width: "38%", maxHeight: "32vh", display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
                    <span style={{ fontSize: 9, color: "var(--green)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, textAlign: "center" }}>HR</span>
                    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
                      <img src={convertFileSrc(currentPair.hr)} alt="HR" style={{ maxWidth: "100%", maxHeight: "28vh", objectFit: "contain" }} />
                    </div>
                  </div>
                  <div style={{ flex: "0 0 auto", width: "38%", maxHeight: "32vh", display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
                    <span style={{ fontSize: 9, color: "var(--amber)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, textAlign: "center" }}>LR</span>
                    <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
                      <img src={convertFileSrc(currentPair.lr)} alt="LR" style={{ maxWidth: "100%", maxHeight: "28vh", objectFit: "contain" }} />
                    </div>
                  </div>
                </div>

                <div style={{ flexShrink: 0 }}>
                  <ThumbStrip pairs={pairs} current={currentIndex} onSelect={setCurrentIndex} />
                </div>
              </div>
            )}

            {pairs.length === 0 && (
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--dim)", fontSize: 12 }}>
                No image pairs found in this dataset
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
