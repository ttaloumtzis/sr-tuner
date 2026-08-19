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
  RotateCcw,
  Loader2,
} from "lucide-react";
import "./ScreenBrowseDatasets.css";
import { listDatasets, startValidateDataset, healthCheck, getDatasetHealth, deleteDataset, pruneDatasetFiles } from "../../lib/api";
import type { DatasetInfo, HealthReport, ImagePairInfo } from "../../lib/api-types";
import { getDatasetPairUrls, getDatasetPairInfo } from "../../lib/scanDatasets";
import { useToast } from "../../components/shell/ToastProvider";
import { useDatasetStore } from "../../store/datasetStore";
import { PBar } from "../../components/ui/PBar";

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
  const [viewMode, setViewMode] = useState<"slider" | "split">("slider");
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [pairUrls, setPairUrls] = useState<{ hr: string; lr: string }[]>([]);
  const [pairInfo, setPairInfo] = useState<ImagePairInfo[]>([]);
  const [pairLoading, setPairLoading] = useState(false);
  const isPanningRef = useRef(false);
  const panStartRef = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const pairPreloadRef = useRef(0);

  const sliderContainerRef = useRef<HTMLDivElement>(null);
  const thumbScrollRef = useRef<HTMLDivElement>(null);

  const jobStatus = useDatasetStore((s) => s.jobStatus);
  const jobType = useDatasetStore((s) => s.jobType);
  const jobDatasetPath = useDatasetStore((s) => s.jobDatasetPath);
  const jobHealthReport = useDatasetStore((s) => s.jobHealthReport);
  const validationResult = useDatasetStore((s) => s.validationResult);
  const progressSteps = useDatasetStore((s) => s.progressSteps);
  const setJobId = useDatasetStore((s) => s.setJobId);
  const setJobStatus = useDatasetStore((s) => s.setJobStatus);
  const setJobType = useDatasetStore((s) => s.setJobType);
  const setJobDatasetPath = useDatasetStore((s) => s.setJobDatasetPath);
  const setValidationResult = useDatasetStore((s) => s.setValidationResult);

  const [healthForPath, setHealthForPath] = useState<string | null>(null);
  const [healthReport, setHealthReport] = useState<HealthReport | null>(null);
  const [healthReportLoading, setHealthReportLoading] = useState(false);
  const healthReqIdRef = useRef(0);
  const [selectedBlackFrames, setSelectedBlackFrames] = useState<Set<string>>(new Set());
  const [selectedUnreadable, setSelectedUnreadable] = useState<Set<string>>(new Set());

  const currentDataset = datasets.find((d) => d.name === selectedName) ?? null;
  const pairsCount = currentDataset?.num_pairs ?? 0;

  const loadHealth = useCallback((path: string) => {
    const reqId = ++healthReqIdRef.current;
    setHealthForPath(path);
    setHealthReportLoading(true);
    setHealthReport(null);
    getDatasetHealth(path)
      .then((report) => {
        if (healthReqIdRef.current !== reqId) return;
        setHealthReport(report);
        setHealthReportLoading(false);
      })
      .catch(() => {
        if (healthReqIdRef.current !== reqId) return;
        setHealthReport(null);
        setHealthReportLoading(false);
      });
  }, []);

  const applyHealth = useCallback((path: string, report: HealthReport | null) => {
    healthReqIdRef.current += 1;
    setHealthForPath(path);
    setHealthReport(report);
    setHealthReportLoading(false);
  }, []);

  const showHealthReport = healthForPath === currentDataset?.path ? healthReport : null;
  const showHealthLoading = healthForPath === currentDataset?.path ? healthReportLoading : false;
  const isJobOnCurrent = jobStatus === "running" && jobDatasetPath === currentDataset?.path;
  const isHealthJobOnCurrent = isJobOnCurrent && jobType === "health";

  const setPair = useCallback(
    (n: number) => setCurrentPairIndex(Math.max(1, Math.min(pairsCount, n))),
    [pairsCount],
  );

  const handleValidate = useCallback(async () => {
    if (!currentDataset) return;
    try {
      const res = await startValidateDataset({ path: currentDataset.path });
      setJobId(res.job_id);
      setJobDatasetPath(currentDataset.path);
      setJobType("validate");
      setJobStatus("running");
      toast("info", "Validation started");
    } catch (err) {
      toast("error", `Validation failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, toast, setJobId, setJobDatasetPath, setJobType, setJobStatus]);

  useEffect(() => {
    if (validationResult === null) return;
    if (validationResult.valid) {
      toast("success", `Dataset validated — ${validationResult.num_pairs} pairs, no problems`);
    } else {
      const problems = validationResult.problems;
      const shown = problems.slice(0, 8).join(", ");
      const tail = problems.length > 8 ? ` (+${problems.length - 8} more)` : "";
      toast("warning", `Validation found ${problems.length} problem(s): ${shown}${tail}`);
    }
  }, [validationResult, toast]);

  const handleHealthReport = useCallback(async () => {
    if (!currentDataset) return;
    try {
      let report = await getDatasetHealth(currentDataset.path);
      if (report === null) {
        const result = await healthCheck({ path: currentDataset.path, yes: false });
        setJobId(result.job_id);
        setJobDatasetPath(currentDataset.path);
        setJobType("health");
        setJobStatus("running");
        toast("info", "Health check started");
      } else {
        applyHealth(currentDataset.path, report);
        const n = report.unreadable?.length ?? 0;
        toast("success", `Health report loaded — ${report.black_frames.length} black frames${n > 0 ? `, ${n} unreadable` : ""}`);
      }
    } catch (err) {
      setJobId(null);
      setJobDatasetPath(null);
      setJobType(null);
      setJobStatus("idle");
      toast("error", `Health check failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, toast, applyHealth, setJobId, setJobDatasetPath, setJobType, setJobStatus]);

  const toggleBlackFrame = useCallback((filename: string) => {
    setSelectedBlackFrames((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
  }, []);

  const selectAllBlackFrames = useCallback(() => {
    if (!showHealthReport) return;
    setSelectedBlackFrames(new Set(showHealthReport.black_frames));
  }, [showHealthReport]);

  const deselectAllBlackFrames = useCallback(() => {
    setSelectedBlackFrames(new Set());
  }, []);

  const handlePrune = useCallback(async () => {
    if (!currentDataset || selectedBlackFrames.size === 0) return;
    try {
      const files = Array.from(selectedBlackFrames).map((name) =>
        name.startsWith("HR/") || name.startsWith("LR/") ? name : `HR/${name}`,
      );
      const result = await pruneDatasetFiles({
        path: currentDataset.path,
        files,
      });
      setJobId(result.job_id);
      setJobDatasetPath(currentDataset.path);
      setJobType("prune");
      setJobStatus("running");
      toast("info", `Pruning ${selectedBlackFrames.size} black frames...`);
      setSelectedBlackFrames(new Set());
    } catch (err) {
      toast("error", `Prune failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, selectedBlackFrames, toast, setJobId, setJobDatasetPath, setJobType, setJobStatus]);

  const toggleUnreadable = useCallback((rel: string) => {
    setSelectedUnreadable((prev) => {
      const next = new Set(prev);
      if (next.has(rel)) next.delete(rel);
      else next.add(rel);
      return next;
    });
  }, []);

  const selectAllUnreadable = useCallback(() => {
    if (!showHealthReport) return;
    setSelectedUnreadable(new Set(showHealthReport.unreadable ?? []));
  }, [showHealthReport]);

  const deselectAllUnreadable = useCallback(() => {
    setSelectedUnreadable(new Set());
  }, []);

  const handleRemoveUnreadable = useCallback(async () => {
    if (!currentDataset || selectedUnreadable.size === 0) return;
    try {
      const result = await pruneDatasetFiles({
        path: currentDataset.path,
        files: Array.from(selectedUnreadable),
      });
      setJobId(result.job_id);
      setJobDatasetPath(currentDataset.path);
      setJobType("prune");
      setJobStatus("running");
      toast("info", `Removing ${selectedUnreadable.size} corrupt pair(s)...`);
      setSelectedUnreadable(new Set());
    } catch (err) {
      toast("error", `Remove failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [currentDataset, selectedUnreadable, toast, setJobId, setJobDatasetPath, setJobType, setJobStatus]);

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
    setSelectedBlackFrames(new Set());
    setSelectedUnreadable(new Set());
    setValidationResult(null);
    loadHealth(currentDataset.path);
  }, [currentDataset?.path, loadHealth, setValidationResult]);

  useEffect(() => {
    if (!currentDataset) {
      setPairUrls([]);
      setPairInfo([]);
      return;
    }
    let cancelled = false;
    const reqId = ++pairPreloadRef.current;
    const abort = new AbortController();
    setPairUrls([]);
    setPairInfo([]);
    setPairLoading(true);

    getDatasetPairUrls(currentDataset.path, currentDataset.name, currentDataset.num_pairs)
      .then((urls) => {
        if (!cancelled) setPairUrls(urls);
      })
      .catch(() => {
        if (!cancelled) setPairUrls([]);
      });

    getDatasetPairInfo(currentDataset.path, abort.signal)
      .then((info) => {
        if (!cancelled && reqId === pairPreloadRef.current) {
          setPairInfo(info ?? []);
        }
        if (!cancelled) setPairLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          setPairInfo([]);
          setPairLoading(false);
        }
      });

    return () => { cancelled = true; abort.abort(); };
  }, [currentDataset?.path, currentDataset?.name, currentDataset?.num_pairs]);

  useEffect(() => {
    if (jobStatus !== "done" || jobType !== "health") return;
    const jobPath = jobDatasetPath ?? currentDataset?.path;
    if (!jobPath || jobPath !== currentDataset?.path) return;
    if (jobHealthReport) {
      applyHealth(jobPath, jobHealthReport);
      return;
    }
    getDatasetHealth(jobPath)
      .then((report) => applyHealth(jobPath, report))
      .catch(() => applyHealth(jobPath, null));
  }, [jobStatus, jobType, jobDatasetPath, jobHealthReport, currentDataset?.path, applyHealth]);

  useEffect(() => {
    if (jobStatus === "done" && jobType === "prune" && currentDataset) {
      fetchDatasets();
      setSelectedBlackFrames(new Set());
      setSelectedUnreadable(new Set());
      loadHealth(currentDataset.path);
    }
  }, [jobStatus, jobType, fetchDatasets, currentDataset?.path, loadHealth]);

  const filteredDatasets = datasets.filter((ds) => {
    const matchesSearch = ds.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesScale = selectedScaleFilter === "all" || `x${ds.scale}` === selectedScaleFilter;
    return matchesSearch && matchesScale;
  });

  const currentPair = pairInfo[currentPairIndex - 1];
  const hrUrl = currentPair?.hr.url ?? pairUrls[currentPairIndex - 1]?.hr ?? "";
  const lrUrl = currentPair?.lr.url ?? pairUrls[currentPairIndex - 1]?.lr ?? "";

  const alignStyle: React.CSSProperties = {
    width: "100%",
    height: "100%",
    objectFit: "contain",
  };

  const imgZoomStyle: React.CSSProperties = {
    ...alignStyle,
    transform: `scale(${zoomLevel}) translate(${panOffset.x}px, ${panOffset.y}px)`,
  };

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

  const onCanvasMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 1) return;
    e.preventDefault();
    isPanningRef.current = true;
    panStartRef.current = { x: e.clientX, y: e.clientY, px: panOffset.x, py: panOffset.y };
  }, [panOffset]);

  const onCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPanningRef.current || !panStartRef.current) return;
    setPanOffset({
      x: panStartRef.current.px + (e.clientX - panStartRef.current.x),
      y: panStartRef.current.py + (e.clientY - panStartRef.current.y),
    });
  }, []);

  const stopPan = useCallback(() => {
    isPanningRef.current = false;
    panStartRef.current = null;
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
      </>
    );
  }

  if (datasets.length === 0) {
    return (
      <>
        <div className="sr-browse-container">
          <div className="empty-state">No datasets found. Create one in the "Create Dataset" tab.</div>
        </div>
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
            {["all", ...Array.from(new Set(datasets.map((d) => `x${d.scale}`))).sort()].map((scale) => (
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
            const validatingThis = jobStatus === "running" && jobType === "validate" && jobDatasetPath === ds.path;
            const activeStep = [...progressSteps].reverse().find((st) => st.status === "active");
            const valPct = validatingThis && activeStep && activeStep.total != null && activeStep.total > 0
              ? Math.round((activeStep.current / activeStep.total) * 100)
              : null;
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
                  {validatingThis ? (
                    <span className="validation-progress-badge" title={activeStep?.desc}>
                      {valPct != null ? `${valPct}%` : "…"}
                    </span>
                  ) : (
                    <CheckCircle size={14} className="manifest-check" />
                  )}
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
              {showHealthLoading ? (
                <span className="meta-badge health unverified">Health: Checking…</span>
              ) : jobStatus === "running" && isHealthJobOnCurrent ? (
                <span className="meta-badge health unverified">Health: Running…</span>
              ) : showHealthReport === null ? (
                <span className="meta-badge health unverified">Health: Unchecked</span>
              ) : showHealthReport.black_frames.length === 0 && (showHealthReport.unreadable?.length ?? 0) === 0 && (showHealthReport.suspicious_frames?.length ?? 0) === 0 && (showHealthReport.scale_mismatches?.length ?? 0) === 0 ? (
                <span className="meta-badge health healthy">Health: OK</span>
              ) : (
                <span className="meta-badge health warning">{
                  "Health: " + (showHealthReport.black_frames.length + (showHealthReport.unreadable?.length ?? 0) + (showHealthReport.suspicious_frames?.length ?? 0) + (showHealthReport.scale_mismatches?.length ?? 0) === 1 ? "1 issue" : (showHealthReport.black_frames.length + (showHealthReport.unreadable?.length ?? 0) + (showHealthReport.suspicious_frames?.length ?? 0) + (showHealthReport.scale_mismatches?.length ?? 0)) + " issues")
                }</span>
              )}
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

        {pairLoading && !hrUrl && !lrUrl && (
          <div className="canvas-wrapper">
            <div className="loading-spinner">
              <Loader2 size={18} className="spin" />
              Loading preview…
            </div>
          </div>
        )}

        {(!pairLoading || hrUrl || lrUrl) && viewMode === "slider" && (
          <div className="canvas-wrapper">
            <div
              className="comparison-slider-container"
              ref={sliderContainerRef}
              onMouseDown={() => setIsDraggingSlider(true)}
            >
              <div className="image-layer hr-layer">
                <img
                  src={hrUrl}
                  alt="HR Ground Truth"
                  style={alignStyle}
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
                  src={lrUrl}
                  alt="LR Degradation"
                  style={alignStyle}
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

        {(!pairLoading || hrUrl || lrUrl) && viewMode === "split" && (
          <div className="canvas-wrapper" onMouseDown={onCanvasMouseDown} onMouseMove={onCanvasMouseMove} onMouseUp={stopPan} onMouseLeave={stopPan}>
            <div className="side-by-side-container">
              <div className="split-pane">
                <span className="badge-tag tag-lr">LR (Degraded)</span>
                <img
                  src={lrUrl}
                  alt="LR Degraded"
                  style={imgZoomStyle}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                  }}
                />
              </div>
              <div className="split-pane">
                <span className="badge-tag tag-hr">HR (Ground Truth)</span>
                <img
                  src={hrUrl}
                  alt="HR Ground Truth"
                  style={imgZoomStyle}
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).src = FALLBACK_IMG;
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {currentDataset && (
          <div className="health-report-panel">
            <div className="health-report-header">
              <span className="health-report-title">Health Report</span>
              {showHealthLoading && (
                <span className="health-report-status loading">
                  <Loader2 size={12} className="spin" /> Loading...
                </span>
              )}
              {!showHealthLoading && showHealthReport === null && (
                <span className="health-report-status unchecked">No report</span>
              )}
              {!showHealthLoading && showHealthReport !== null && showHealthReport.black_frames.length === 0 && (showHealthReport.unreadable?.length ?? 0) === 0 && (showHealthReport.suspicious_frames?.length ?? 0) === 0 && (showHealthReport.scale_mismatches?.length ?? 0) === 0 && (
                <span className="health-report-status ok">OK</span>
              )}
              {!showHealthLoading && showHealthReport !== null && (showHealthReport.black_frames.length > 0 || (showHealthReport.unreadable?.length ?? 0) > 0 || (showHealthReport.suspicious_frames?.length ?? 0) > 0 || (showHealthReport.scale_mismatches?.length ?? 0) > 0) && (
                <span className="health-report-status issues">
                  {[
                    showHealthReport.black_frames.length > 0 && `${showHealthReport.black_frames.length} black`,
                    (showHealthReport.suspicious_frames?.length ?? 0) > 0 && `${showHealthReport.suspicious_frames.length} suspect`,
                    (showHealthReport.scale_mismatches?.length ?? 0) > 0 && `${showHealthReport.scale_mismatches.length} scale`,
                    (showHealthReport.unreadable?.length ?? 0) > 0 && `${showHealthReport.unreadable.length} corrupt`,
                  ].filter(Boolean).join(", ")}
                </span>
              )}
              <button className="health-report-run-btn" onClick={handleHealthReport} disabled={isJobOnCurrent}>
                {jobStatus === "running" && isHealthJobOnCurrent ? "Running..." : "Run Health Check"}
              </button>
            </div>
            {jobStatus === "running" && isHealthJobOnCurrent && (() => {
              const active = [...progressSteps].reverse().find((st) => st.status === "active");
              const pct = active && active.total != null && active.total > 0
                ? Math.round((active.current / active.total) * 100)
                : null;
              return (
                <div className="health-report-progress">
                  <div className="health-report-progress-label">
                    <span>{active ? active.desc : "Scanning frames…"}</span>
                    {pct != null && <span className="health-report-progress-pct">{pct}%</span>}
                  </div>
                  <PBar value={active?.current ?? 0} max={active?.total ?? (active?.current || 1)} color="var(--amber)" height={5} />
                </div>
              );
            })()}
            {showHealthReport !== null && !showHealthLoading && (
              <div className="health-report-body">
                <div className="health-report-summary">
                  <span>Pairs: {(showHealthReport.total_pairs ?? showHealthReport.total_hr_images ?? 0).toLocaleString()}</span>
                  <span>HR: {showHealthReport.total_hr_images?.toLocaleString() ?? "—"}</span>
                  <span>LR: {showHealthReport.total_lr_images?.toLocaleString() ?? "—"}</span>
                  <span>Threshold: {showHealthReport.computed_threshold}</span>
                  <span>Black: {showHealthReport.black_frames.length}</span>
                  {(showHealthReport.suspicious_frames?.length ?? 0) > 0 && (
                    <span className="health-report-summary-unreadable">Suspicious: {showHealthReport.suspicious_frames.length}</span>
                  )}
                  {(showHealthReport.scale_mismatches?.length ?? 0) > 0 && (
                    <span className="health-report-summary-unreadable">Scale mismatches: {showHealthReport.scale_mismatches.length}</span>
                  )}
                  {(showHealthReport.unreadable?.length ?? 0) > 0 && (
                    <span className="health-report-summary-unreadable">Unreadable: {showHealthReport.unreadable.length}</span>
                  )}
                </div>
                {showHealthReport.suspicious_frames?.length > 0 && (
                  <div className="health-report-blackframes">
                    <div className="health-report-blackframes-toolbar">
                      <span className="health-report-blackframes-label">
                        {showHealthReport.suspicious_frames.length} suspicious {"("}borderline low brightness{")"}
                      </span>
                    </div>
                    <div className="health-report-blackframes-list">
                      {showHealthReport.suspicious_frames.map((rel) => {
                        const mean = showHealthReport.frame_means?.[rel];
                        return (
                          <div key={rel} className="health-report-blackframe-item" style={{ cursor: "default", opacity: 0.75 }}>
                            <span className="health-report-blackframe-name">
                              {rel}{mean != null ? ` (mean=${mean})` : ""}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                {showHealthReport.scale_mismatches?.length > 0 && (
                  <div className="health-report-blackframes">
                    <div className="health-report-blackframes-toolbar">
                      <span className="health-report-blackframes-label">
                        {showHealthReport.scale_mismatches.length} scale mismatch(es)
                      </span>
                    </div>
                    <div className="health-report-blackframes-list">
                      {showHealthReport.scale_mismatches.map((msg) => (
                        <div key={msg} className="health-report-blackframe-item" style={{ cursor: "default", color: "var(--red)" }}>
                          <span className="health-report-blackframe-name">{msg}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {(showHealthReport.unreadable?.length ?? 0) > 0 && (
                  <div className="health-report-blackframes">
                    <div className="health-report-blackframes-toolbar">
                      <span className="health-report-blackframes-label">
                        {selectedUnreadable.size} of {showHealthReport.unreadable.length} selected
                      </span>
                      <div className="health-report-blackframes-actions">
                        <button className="btn-text" onClick={selectAllUnreadable}>Select All</button>
                        <button className="btn-text" onClick={deselectAllUnreadable}>Deselect All</button>
                        <button
                          className="btn-danger btn-sm"
                          onClick={handleRemoveUnreadable}
                          disabled={selectedUnreadable.size === 0 || isJobOnCurrent}
                          title="Deletes the HR and LR files of each selected pair and removes them from the manifest"
                        >
                          <Trash2 size={12} /> Remove Selected ({selectedUnreadable.size})
                        </button>
                      </div>
                    </div>
                    <div className="health-report-blackframes-list">
                      {showHealthReport.unreadable.map((rel) => (
                        <label key={rel} className="health-report-blackframe-item">
                          <input
                            type="checkbox"
                            checked={selectedUnreadable.has(rel)}
                            onChange={() => toggleUnreadable(rel)}
                          />
                          <span className="health-report-blackframe-name">{rel}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}
                {showHealthReport.black_frames.length > 0 && (
                  <div className="health-report-blackframes">
                    <div className="health-report-blackframes-toolbar">
                      <span className="health-report-blackframes-label">
                        {selectedBlackFrames.size} of {showHealthReport.black_frames.length} selected
                      </span>
                      <div className="health-report-blackframes-actions">
                        <button className="btn-text" onClick={selectAllBlackFrames}>Select All</button>
                        <button className="btn-text" onClick={deselectAllBlackFrames}>Deselect All</button>
                        <button
                          className="btn-danger btn-sm"
                          onClick={handlePrune}
                          disabled={selectedBlackFrames.size === 0 || isJobOnCurrent}
                        >
                          <Trash2 size={12} /> Prune Selected ({selectedBlackFrames.size})
                        </button>
                      </div>
                    </div>
                    <div className="health-report-blackframes-list">
                      {showHealthReport.black_frames.map((filename) => {
                        const mean = showHealthReport.frame_means?.[filename];
                        return (
                          <label key={filename} className="health-report-blackframe-item">
                            <input
                              type="checkbox"
                              checked={selectedBlackFrames.has(filename)}
                              onChange={() => toggleBlackFrame(filename)}
                            />
                            <span className="health-report-blackframe-name">
                              {filename}{mean != null ? ` (mean=${mean})` : ""}
                            </span>
                          </label>
                        );
                      })}
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
                    src={pairUrls[idx - 1]?.lr ?? ""}
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
    </>
  );
};
