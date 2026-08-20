import { type CSSProperties, type RefObject, type MouseEvent, type Dispatch, type SetStateAction } from "react";
import {
  Sliders,
  Columns,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  Eye,
  Activity,
  FolderOpen,
  Trash2,
  Loader2,
} from "lucide-react";
import { PBar } from "../../components/ui/PBar";
import type { DatasetInfo, HealthReport, ImagePairInfo } from "../../lib/api-types";
import type { JobStatus, ProgressStep } from "../../store/datasetStore";

const FILMSTRIP_WINDOW = 25;
const FALLBACK_IMG =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='80'%3E%3Crect fill='%231c1f23' width='120' height='80'/%3E%3Ctext x='60' y='42' text-anchor='middle' fill='%236b7583' font-size='10' font-family='sans-serif'%3ENo image%3C/text%3E%3C/svg%3E";

interface DatasetDetailProps {
  currentDataset: DatasetInfo | null;
  pairsCount: number;
  pairInfo: ImagePairInfo[];
  pairUrls: { hr: string; lr: string }[];
  currentPairIndex: number;
  setCurrentPairIndex: Dispatch<SetStateAction<number>>;
  setPair: (n: number) => void;
  viewMode: "slider" | "split";
  setViewMode: Dispatch<SetStateAction<"slider" | "split">>;
  sliderPosition: number;
  setIsDraggingSlider: Dispatch<SetStateAction<boolean>>;
  sliderContainerRef: RefObject<HTMLDivElement>;
  pairLoading: boolean;
  zoomLevel: number;
  setZoomLevel: Dispatch<SetStateAction<number>>;
  panOffset: { x: number; y: number };
  setPanOffset: Dispatch<SetStateAction<{ x: number; y: number }>>;
  onCanvasMouseDown: (e: MouseEvent<HTMLDivElement>) => void;
  onCanvasMouseMove: (e: MouseEvent<HTMLDivElement>) => void;
  stopPan: () => void;
  handleValidate: () => void;
  handleHealthReport: () => void;
  handleOpenDirectory: () => void;
  handleDelete: () => void;
  showHealthLoading: boolean;
  showHealthReport: HealthReport | null;
  isJobOnCurrent: boolean;
  isHealthJobOnCurrent: boolean;
  jobStatus: JobStatus;
  progressSteps: ProgressStep[];
  selectedBlackFrames: Set<string>;
  toggleBlackFrame: (f: string) => void;
  selectAllBlackFrames: () => void;
  deselectAllBlackFrames: () => void;
  handlePrune: () => void;
  selectedUnreadable: Set<string>;
  toggleUnreadable: (r: string) => void;
  selectAllUnreadable: () => void;
  deselectAllUnreadable: () => void;
  handleRemoveUnreadable: () => void;
  thumbScrollRef: RefObject<HTMLDivElement>;
  scrollThumbs: (dir: "left" | "right") => void;
}

export function DatasetDetail(props: DatasetDetailProps) {
  const {
    currentDataset,
    pairsCount,
    pairInfo,
    pairUrls,
    currentPairIndex,
    setCurrentPairIndex,
    setPair,
    viewMode,
    setViewMode,
    sliderPosition,
    setIsDraggingSlider,
    sliderContainerRef,
    pairLoading,
    zoomLevel,
    setZoomLevel,
    panOffset,
    setPanOffset,
    onCanvasMouseDown,
    onCanvasMouseMove,
    stopPan,
    handleValidate,
    handleHealthReport,
    handleOpenDirectory,
    handleDelete,
    showHealthLoading,
    showHealthReport,
    isJobOnCurrent,
    isHealthJobOnCurrent,
    jobStatus,
    progressSteps,
    selectedBlackFrames,
    toggleBlackFrame,
    selectAllBlackFrames,
    deselectAllBlackFrames,
    handlePrune,
    selectedUnreadable,
    toggleUnreadable,
    selectAllUnreadable,
    deselectAllUnreadable,
    handleRemoveUnreadable,
    thumbScrollRef,
    scrollThumbs,
  } = props;

  const currentPair = pairInfo[currentPairIndex - 1];
  const hrUrl = currentPair?.hr.url ?? pairUrls[currentPairIndex - 1]?.hr ?? "";
  const lrUrl = currentPair?.lr.url ?? pairUrls[currentPairIndex - 1]?.lr ?? "";

  const imgZoomStyle: CSSProperties = {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    transform: `scale(${zoomLevel}) translate(${panOffset.x}px, ${panOffset.y}px)`,
  };

  const startThumb = Math.max(1, currentPairIndex - FILMSTRIP_WINDOW);
  const endThumb = Math.min(pairsCount, currentPairIndex + FILMSTRIP_WINDOW);
  const thumbIndices: number[] = [];
  if (pairsCount > 0) {
    for (let i = startThumb; i <= endThumb; i++) thumbIndices.push(i);
  }

  return (
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
                        <div key={rel} className="health-report-blackframe-item readonly">
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
                      <div key={msg} className="health-report-blackframe-item issue">
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
  );
}