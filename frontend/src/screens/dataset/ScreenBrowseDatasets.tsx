import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  AlertCircle,
  Loader2,
} from "lucide-react";
import "./ScreenBrowseDatasets.css";
import { listDatasets, startValidateDataset, healthCheck, getDatasetHealth, deleteDataset, pruneDatasetFiles } from "../../lib/api";
import type { DatasetInfo, HealthReport, ImagePairInfo } from "../../lib/api-types";
import { getDatasetPairUrls, getDatasetPairInfo } from "../../lib/scanDatasets";
import { useToast } from "../../components/shell/ToastProvider";
import { useDatasetStore } from "../../store/datasetStore";
import { DatasetList } from "./DatasetList";
import { DatasetDetail } from "./DatasetDetail";

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
      <DatasetList
        datasets={datasets}
        filteredDatasets={filteredDatasets}
        selectedName={selectedName}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        selectedScaleFilter={selectedScaleFilter}
        onScaleFilterChange={setSelectedScaleFilter}
        jobStatus={jobStatus}
        jobType={jobType}
        jobDatasetPath={jobDatasetPath}
        progressSteps={progressSteps}
        onSelect={(name) => {
          setSelectedName(name);
          setCurrentPairIndex(1);
          setZoomLevel(1);
          setPanOffset({ x: 0, y: 0 });
        }}
      />

      <DatasetDetail
        currentDataset={currentDataset}
        pairsCount={pairsCount}
        pairInfo={pairInfo}
        pairUrls={pairUrls}
        currentPairIndex={currentPairIndex}
        setCurrentPairIndex={setCurrentPairIndex}
        setPair={setPair}
        viewMode={viewMode}
        setViewMode={setViewMode}
        sliderPosition={sliderPosition}
        setIsDraggingSlider={setIsDraggingSlider}
        sliderContainerRef={sliderContainerRef}
        pairLoading={pairLoading}
        zoomLevel={zoomLevel}
        setZoomLevel={setZoomLevel}
        panOffset={panOffset}
        setPanOffset={setPanOffset}
        onCanvasMouseDown={onCanvasMouseDown}
        onCanvasMouseMove={onCanvasMouseMove}
        stopPan={stopPan}
        handleValidate={handleValidate}
        handleHealthReport={handleHealthReport}
        handleOpenDirectory={handleOpenDirectory}
        handleDelete={handleDelete}
        showHealthLoading={showHealthLoading}
        showHealthReport={showHealthReport}
        isJobOnCurrent={isJobOnCurrent}
        isHealthJobOnCurrent={isHealthJobOnCurrent}
        jobStatus={jobStatus}
        progressSteps={progressSteps}
        selectedBlackFrames={selectedBlackFrames}
        toggleBlackFrame={toggleBlackFrame}
        selectAllBlackFrames={selectAllBlackFrames}
        deselectAllBlackFrames={deselectAllBlackFrames}
        handlePrune={handlePrune}
        selectedUnreadable={selectedUnreadable}
        toggleUnreadable={toggleUnreadable}
        selectAllUnreadable={selectAllUnreadable}
        deselectAllUnreadable={deselectAllUnreadable}
        handleRemoveUnreadable={handleRemoveUnreadable}
        thumbScrollRef={thumbScrollRef}
        scrollThumbs={scrollThumbs}
      />
    </div>
    </>
  );
};