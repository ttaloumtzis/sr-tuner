import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Search,
  Sliders,
  ZoomIn,
  ZoomOut,
  FolderOpen,
  Activity,
  CheckCircle,
  AlertCircle,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Eye,
  Columns,
  Layers,
  RotateCcw,
  Loader2,
} from "lucide-react";
import "./ScreenBrowseDatasets.css";
import { listDatasets, getDatasetImageUrl, validateDatasetPath, healthCheck, getDatasetHealth, deleteDataset, pruneBlackFrames } from "../../lib/api";
import type { DatasetInfo, HealthReport } from "../../lib/api-types";
import { useToast } from "../../components/shell/ToastProvider";
import { useDatasetStore } from "../../store/datasetStore";
import { useDatasetSSE } from "../../hooks/useDatasetSSE";
import { JobOverlay } from "../../components/dataset/JobOverlay";

const FILMSTRIP_WINDOW = 25;
const FALLBACK_IMG =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='80'%3E%3Crect fill='%231c1f23' width='120' height='80'/%3E%3Ctext x='60' y='42' text-anchor='middle' fill='%236b7583' font-size='10' font-family='sans-serif'%3ENo image%3C/text%3E%3C/svg%3E";

export const ScreenBrowseDatasets: React.FC = () => {
  const { show: toast } = useToast();

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedScaleFilter, setSelectedScaleFilter] = useState("all");
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [currentPairIndex, setCurrentPairIndex] = useState(1);
  const [viewMode, setViewMode] = useState<"slider" | "split" | "diff">("slider");
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  const sliderContainerRef = useRef<HTMLDivElement>(null);
  const thumbScrollRef = useRef<HTMLDivElement>(null);

  useDatasetSSE();
  const jobStatus = useDatasetStore((s) => s.jobStatus);
  const jobType = useDatasetStore((s) => s.jobType);
  const setJobId = useDatasetStore((s) => s.setJobId);
  const setJobStatus = useDatasetStore((s) => s.setJobStatus);
  const setJobType = useDatasetStore((s) => s.setJobType);

  const [healthReport, setHealthReport] = useState<HealthReport | null>(null);
  const [healthReportLoading, setHealthReportLoading] = useState(false);
  const [selectedBlackFrames, setSelectedBlackFrames] = useState<Set<string>>(new Set());

  const currentDataset = datasets.find((d) => d.name === selectedName) ?? null;
  const pairsCount = currentDataset?.num_pairs ?? 0;

  const setPair = useCallback(
    (n: number) => setCurrentPairIndex(Math.max(1, Math.min(pairsCount, n))),
    [pairsCount],
  );

  const handleValidate = useCallback(async () => {
    if (!currentDataset) return;
    try {
      const res = await validateDatasetPath({ path: currentDataset.path });
      if (res.valid) {
        toast("success", `Dataset validated — ${res.num_pairs} pairs, no problems`);
      } else {
        toast("warning", `Validation found ${res.problems.length} problem(s): ${res.problems.join(", ")}`);
      }
    } catch (err) {
      toast("error", `Validation failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, toast]);

  const handleHealthReport = useCallback(async () => {
    if (!currentDataset) return;
    try {
      let report = await getDatasetHealth(currentDataset.path);
      if (report === null) {
        const result = await healthCheck({ path: currentDataset.path, yes: false });
        setJobId(result.job_id);
        setJobType("health");
        setJobStatus("running");
        toast("info", "Health check started");
      } else {
        setHealthReport(report);
        toast("success", `Health report loaded — ${report.black_frames.length} black frames`);
      }
    } catch (err) {
      setJobId(null);
      setJobType(null);
      setJobStatus("idle");
      toast("error", `Health check failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, toast]);

  const toggleBlackFrame = useCallback((filename: string) => {
    setSelectedBlackFrames((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
  }, []);

  const selectAllBlackFrames = useCallback(() => {
    if (!healthReport) return;
    setSelectedBlackFrames(new Set(healthReport.black_frames));
  }, [healthReport]);

  const deselectAllBlackFrames = useCallback(() => {
    setSelectedBlackFrames(new Set());
  }, []);

  const handlePrune = useCallback(async () => {
    if (!currentDataset || selectedBlackFrames.size === 0) return;
    try {
      const result = await pruneBlackFrames({
        path: currentDataset.path,
        black_frames: Array.from(selectedBlackFrames),
      });
      setJobId(result.job_id);
      setJobType("prune");
      setJobStatus("running");
      toast("info", `Pruning ${selectedBlackFrames.size} black frames...`);
      setSelectedBlackFrames(new Set());
    } catch (err) {
      toast("error", `Prune failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, selectedBlackFrames, toast]);

  const handleOpenDirectory = useCallback(async () => {
    if (!currentDataset) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("open_in_file_manager", { path: currentDataset.path });
    } catch {
      toast("info", `Dataset path: ${currentDataset.path}`);
    }
  }, [currentDataset, toast]);

  const handleDelete = useCallback(async () => {
    if (!currentDataset) return;
    if (!window.confirm(`Delete dataset "${currentDataset.name}" and all its files?`)) return;
    try {
      await deleteDataset(currentDataset.name);
      toast("success", `Dataset "${currentDataset.name}" deleted`);
      setCurrentPairIndex(1);
      listDatasets().then((data) => {
        setDatasets(data);
        setSelectedName((prev) => {
          if (prev && data.some((d) => d.name === prev)) return prev;
          return data.length > 0 ? data[0].name : null;
        });
      }).catch(() => {});
    } catch (err) {
      toast("error", `Delete failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, toast]);

  const fetchDatasets = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listDatasets()
      .then((data) => {
        if (cancelled) return;
        setDatasets(data);
        setSelectedName((prev) => {
          if (prev && data.some((d) => d.name === prev)) return prev;
          return data.length > 0 ? data[0].name : null;
        });
        setCurrentPairIndex(1);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => fetchDatasets(), [fetchDatasets]);

  useEffect(() => {
    if (!currentDataset) return;
    let cancelled = false;
    setHealthReportLoading(true);
    setHealthReport(null);
    getDatasetHealth(currentDataset.path)
      .then((report) => {
        if (!cancelled) {
          setHealthReport(report);
          setHealthReportLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHealthReport(null);
          setHealthReportLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [currentDataset?.path]);

  useEffect(() => {
    if (jobStatus !== "done" || jobType !== "health" || !currentDataset) return;
    getDatasetHealth(currentDataset.path)
      .then((report) => setHealthReport(report))
      .catch(() => setHealthReport(null));
  }, [jobStatus, jobType, currentDataset?.path]);

  useEffect(() => {
    if (jobStatus === "done" && jobType === "prune") {
      fetchDatasets();
      setHealthReport(null);
      setSelectedBlackFrames(new Set());
    }
  }, [jobStatus, jobType, fetchDatasets]);

  const filteredDatasets = datasets.filter((ds) => {
    const matchesSearch = ds.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesScale = selectedScaleFilter === "all" || `x${ds.scale}` === selectedScaleFilter;
    return matchesSearch && matchesScale;
  });

  const imgTransform = useCallback(
    (includePan = true): React.CSSProperties => ({
      width: "100%",
      height: "100%",
      transform: `scale(${zoomLevel})${includePan ? ` translate(${panOffset.x}px, ${panOffset.y}px)` : ""}`,
      maxWidth: zoomLevel > 1 ? "none" : "100%",
      maxHeight: zoomLevel > 1 ? "none" : "100%",
      objectFit: "contain",
    }),
    [zoomLevel, panOffset],
  );

  const currentHrUrl =
    currentDataset ? getDatasetImageUrl(currentDataset.name, "hr", currentPairIndex - 1) : "";
  const currentLrUrl =
    currentDataset ? getDatasetImageUrl(currentDataset.name, "lr", currentPairIndex - 1) : "";

  const startThumb = Math.max(1, currentPairIndex - FILMSTRIP_WINDOW);
  const endThumb = Math.min(pairsCount, currentPairIndex + FILMSTRIP_WINDOW);
  const thumbIndices: number[] = [];
  if (pairsCount > 0) {
    for (let i = startThumb; i <= endThumb; i++) thumbIndices.push(i);
  }

  const scrollThumbs = useCallback((dir: "left" | "right") => {
    thumbScrollRef.current?.scrollBy({
      left: dir === "left" ? -200 : 200,
      behavior: "smooth",
    });
  }, []);

  const handleMove = useCallback((clientX: number) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    setSliderPosition(Math.max(0, Math.min(100, (x / rect.width) * 100)));
  }, []);

  useEffect(() => {
    if (!isDraggingSlider) return;
    const onMove = (e: MouseEvent) => handleMove(e.clientX);
    const onUp = () => setIsDraggingSlider(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [isDraggingSlider, handleMove]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "ArrowRight") setPair(currentPairIndex + 1);
      else if (e.key === "ArrowLeft") setPair(currentPairIndex - 1);
      else if (e.key === "1") setViewMode("slider");
      else if (e.key === "2") setViewMode("split");
      else if (e.key === "3") setViewMode("diff");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPair, currentPairIndex]);

  if (loading) {
    return (
      <>
        <div className="sr-browse-container">
          <div className="loading-spinner">
            <Loader2 size={18} className="spin" />
            Loading datasets…
          </div>
        </div>
        <JobOverlay />
      </>
    );
  }

  if (error) {
    return (
      <>
        <div className="sr-browse-container">
          <div className="error-banner">
            <AlertCircle size={20} />
            <span>Failed to load datasets: {error}</span>
            <button onClick={fetchDatasets}>Retry</button>
          </div>
        </div>
        <JobOverlay />
      </>
    );
  }

  if (datasets.length === 0) {
    return (
      <>
        <div className="sr-browse-container">
          <div className="empty-state">No datasets found. Create one in the "Create Dataset" tab.</div>
        </div>
        <JobOverlay />
      </>
    );
  }

  return (
    <>
    <div className="sr-browse-container">
      <aside className="sr-sidebar">
        <div className="sidebar-header">
          <h3>Datasets</h3>
          <span className="dataset-count-badge">{filteredDatasets.length}</span>
        </div>

        <div className="sidebar-controls">
          <div className="search-input-wrapper">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              placeholder="Search datasets…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div className="scale-filter-pills">
            {["all", "x2", "x4", "x8"].map((scale) => (
              <button
                key={scale}
                className={`scale-pill ${selectedScaleFilter === scale ? "active" : ""}`}
                onClick={() => setSelectedScaleFilter(scale)}
              >
                {scale.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="dataset-list">
          {filteredDatasets.map((ds) => {
            const isSelected = ds.name === selectedName;
            return (
              <div
                key={ds.name}
                className={`dataset-card ${isSelected ? "selected" : ""}`}
                onClick={() => {
                  setSelectedName(ds.name);
                  setCurrentPairIndex(1);
                  setZoomLevel(1);
                  setPanOffset({ x: 0, y: 0 });
                }}
              >
                <div className="card-top-row">
                  <span className="dataset-name" title={ds.name}>
                    {ds.name}
                  </span>
                  <CheckCircle size={14} className="manifest-check" />
                </div>
                <div className="card-bottom-row">
                  <span className="scale-tag">x{ds.scale}</span>
                  <span className="pairs-count">{ds.num_pairs.toLocaleString()} pairs</span>
                </div>
              </div>
            );
          })}
        </div>
      </aside>

      <main className="sr-main-studio">
        <header className="studio-header">
          <div className="dataset-meta-group">
            <h2 className="selected-title">{currentDataset?.name ?? "—"}</h2>
            <div className="meta-pill-group">
              <span className="meta-badge scale">
                {currentDataset ? `x${currentDataset.scale}` : "—"}
              </span>
              <span className="meta-badge pairs">
                {pairsCount.toLocaleString()} pairs
              </span>
              <span className="meta-badge manifest">Manifest OK</span>
              {healthReportLoading
  ? <span className="meta-badge health unverified">Health: Checking…</span>
  : jobStatus === "running" && jobType === "health"
    ? <span className="meta-badge health unverified">Health: Running…</span>
    : healthReport === null
      ? <span className="meta-badge health unverified">Health: Unchecked</span>
      : healthReport.black_frames.length === 0
        ? <span className="meta-badge health healthy">Health: OK</span>
        : <span className="meta-badge health warning">Health: {healthReport.black_frames.length} issue{healthReport.black_frames.length !== 1 ? "s" : ""}</span>
}
            </div>
          </div>

          <div className="studio-actions">
            <button className="btn-secondary" onClick={handleValidate}>
              <Activity size={14} /> Validate
            </button>
            <button className="btn-secondary" onClick={handleHealthReport}>
              <Eye size={14} /> Health Report
            </button>
            <button className="btn-secondary" onClick={handleOpenDirectory}>
              <FolderOpen size={14} /> Open Directory
            </button>
            <button className="btn-danger" onClick={handleDelete}>
              <Trash2 size={14} /> Delete
            </button>
          </div>
        </header>

        <div className="studio-sub-bar">
          <div className="view-mode-toggle">
            <button
              className={`mode-btn ${viewMode === "slider" ? "active" : ""}`}
              onClick={() => setViewMode("slider")}
              title="Split Slider (1)"
            >
              <Sliders size={15} /> Split Slider
            </button>
            <button
              className={`mode-btn ${viewMode === "split" ? "active" : ""}`}
              onClick={() => setViewMode("split")}
              title="Side-by-Side (2)"
            >
              <Columns size={15} /> Side-by-Side
            </button>
            <button
              className={`mode-btn ${viewMode === "diff" ? "active" : ""}`}
              onClick={() => setViewMode("diff")}
              title="Difference Layer (3)"
            >
              <Layers size={15} /> Overlay
            </button>
          </div>

          <div className="zoom-controls">
            <button onClick={() => setZoomLevel((z) => Math.max(0.5, z - 0.5))} title="Zoom Out">
              <ZoomOut size={15} />
            </button>
            <span className="zoom-indicator">{Math.round(zoomLevel * 100)}%</span>
            <button onClick={() => setZoomLevel((z) => Math.min(4, z + 0.5))} title="Zoom In">
              <ZoomIn size={15} />
            </button>
            <button
              onClick={() => {
                setZoomLevel(1);
                setPanOffset({ x: 0, y: 0 });
              }}
              title="Reset View"
            >
              <RotateCcw size={14} />
            </button>
          </div>

          <div className="pair-pagination">
            <button
              disabled={currentPairIndex <= 1}
              onClick={() => setPair(currentPairIndex - 1)}
              className="page-nav-btn"
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <div className="pair-counter">
              <input
                type="number"
                min={1}
                max={pairsCount}
                value={currentPairIndex}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  if (val >= 1 && val <= pairsCount) setCurrentPairIndex(val);
                }}
              />
              <span>/ {pairsCount}</span>
            </div>
            <button
              disabled={currentPairIndex >= pairsCount}
              onClick={() => setPair(currentPairIndex + 1)}
              className="page-nav-btn"
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>

        {viewMode === "slider" && (
          <div className="canvas-wrapper">
            <div
              className="comparison-slider-container"
              ref={sliderContainerRef}
              onMouseDown={() => setIsDraggingSlider(true)}
            >
              <div className="image-layer hr-layer">
                <img
                  src={currentHrUrl}
                  alt="HR Ground Truth"
                  style={imgTransform(true)}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                  }}
                />
                <span className="badge-tag tag-hr">HR (GT)</span>
              </div>

              <div
                className="image-layer lr-layer"
                style={{
                  clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`,
                }}
              >
                <img
                  src={currentLrUrl}
                  alt="LR Degradation"
                  style={imgTransform(true)}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                  }}
                />
                <span className="badge-tag tag-lr">LR Input</span>
              </div>

              <div className="slider-handle-line" style={{ left: `${sliderPosition}%` }}>
                <div className="slider-handle-knob">
                  <Sliders size={14} />
                </div>
              </div>
            </div>
          </div>
        )}

        {viewMode === "split" && (
          <div className="canvas-wrapper">
            <div className="side-by-side-container">
              <div className="split-pane">
                <span className="badge-tag tag-hr">HR (Ground Truth)</span>
                <img
                  src={currentHrUrl}
                  alt="HR Ground Truth"
                  style={imgTransform(true)}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                  }}
                />
              </div>
              <div className="split-pane">
                <span className="badge-tag tag-lr">LR (Degraded)</span>
                <img
                  src={currentLrUrl}
                  alt="LR Degraded"
                  style={imgTransform(true)}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {viewMode === "diff" && (
          <div className="canvas-wrapper">
            <div className="diff-container">
              <span className="badge-tag tag-diff">Difference / Residual Map</span>
              <img
                src={currentHrUrl}
                alt="Residual Difference"
                style={{
                  ...imgTransform(true),
                  filter: "invert(0.8) contrast(200%)",
                }}
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                }}
              />
            </div>
          </div>
        )}

        {currentDataset && (
          <div className="health-report-panel">
            <div className="health-report-header">
              <span className="health-report-title">Health Report</span>
              {healthReportLoading && (
                <span className="health-report-status loading">
                  <Loader2 size={12} className="spin" /> Loading...
                </span>
              )}
              {!healthReportLoading && healthReport === null && (
                <span className="health-report-status unchecked">No report</span>
              )}
              {!healthReportLoading && healthReport !== null && healthReport.black_frames.length === 0 && (
                <span className="health-report-status ok">OK</span>
              )}
              {!healthReportLoading && healthReport !== null && healthReport.black_frames.length > 0 && (
                <span className="health-report-status issues">
                  {healthReport.black_frames.length} black frame{healthReport.black_frames.length !== 1 ? "s" : ""}
                </span>
              )}
              <button className="health-report-run-btn" onClick={handleHealthReport} disabled={jobStatus === "running"}>
                {jobStatus === "running" && jobType === "health" ? "Running..." : "Run Health Check"}
              </button>
            </div>
            {healthReport !== null && !healthReportLoading && (
              <div className="health-report-body">
                <div className="health-report-summary">
                  <span>Total images: {healthReport.total_images.toLocaleString()}</span>
                  <span>Threshold: {healthReport.computed_threshold}</span>
                  <span>Black frames: {healthReport.black_frames.length}</span>
                </div>
                {healthReport.black_frames.length > 0 && (
                  <div className="health-report-blackframes">
                    <div className="health-report-blackframes-toolbar">
                      <span className="health-report-blackframes-label">
                        {selectedBlackFrames.size} of {healthReport.black_frames.length} selected
                      </span>
                      <div className="health-report-blackframes-actions">
                        <button className="btn-text" onClick={selectAllBlackFrames}>Select All</button>
                        <button className="btn-text" onClick={deselectAllBlackFrames}>Deselect All</button>
                        <button
                          className="btn-danger btn-sm"
                          onClick={handlePrune}
                          disabled={selectedBlackFrames.size === 0 || jobStatus === "running"}
                        >
                          <Trash2 size={12} /> Prune Selected ({selectedBlackFrames.size})
                        </button>
                      </div>
                    </div>
                    <div className="health-report-blackframes-list">
                      {healthReport.black_frames.map((filename) => (
                        <label key={filename} className="health-report-blackframe-item">
                          <input
                            type="checkbox"
                            checked={selectedBlackFrames.has(filename)}
                            onChange={() => toggleBlackFrame(filename)}
                          />
                          <span className="health-report-blackframe-name">{filename}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {pairsCount > 0 && (
          <div className="thumbnail-filmstrip">
            <button className="strip-arrow" onClick={() => scrollThumbs("left")}>
              <ChevronLeft size={16} />
            </button>
            <div className="thumb-scroll" ref={thumbScrollRef}>
              {thumbIndices.map((idx) => (
                <div
                  key={idx}
                  className={`thumb-item ${currentPairIndex === idx ? "active" : ""}`}
                  onClick={() => setCurrentPairIndex(idx)}
                >
                  <img
                    src={getDatasetImageUrl(currentDataset!.name, "lr", idx - 1)}
                    alt={`Pair ${idx}`}
                    loading="lazy"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                    }}
                  />
                  <span className="thumb-number">#{idx}</span>
                </div>
              ))}
            </div>
            <button className="strip-arrow" onClick={() => scrollThumbs("right")}>
              <ChevronRight size={16} />
            </button>
          </div>
        )}
      </main>
    </div>
    <JobOverlay />
    </>
  );
};
