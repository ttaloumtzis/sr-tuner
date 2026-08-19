import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useTrainingStore } from "../../store/trainingStore";
import type { ValidationFrames, ValidationHistoryEntry } from "../../store/trainingStore";
import { CHECKERBOARD_BG } from "../../lib/checkerboardBg";

type FrameKind = "lr" | "sr" | "gt" | "diff";

const FRAME_META: Record<FrameKind, { label: string; key: keyof ValidationFrames }> = {
  lr:   { label: "LR",   key: "lrPath" },
  sr:   { label: "SR",   key: "srPath" },
  gt:   { label: "GT",   key: "gtPath" },
  diff: { label: "Diff", key: "diffPath" },
};
const FRAME_ORDER: FrameKind[] = ["lr", "sr", "gt", "diff"];

function pathFor(frames: ValidationFrames | null, kind: FrameKind): string | null {
  if (!frames) return null;
  const v = frames[FRAME_META[kind].key];
  return typeof v === "string" ? v : null;
}

function fmtMetric(n: number | null | undefined, decimals = 2): string {
  if (n == null) return "—";
  return n.toFixed(decimals);
}

function PanelHeader({ label, right }: { label: string; right?: ReactNode }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "9px 14px 7px", flexShrink: 0, gap: 8,
    }}>
      <span style={{
        fontSize: 10, letterSpacing: "0.06em", color: "var(--muted)",
        fontFamily: "var(--font-mono)", fontWeight: 600, textTransform: "uppercase",
      }}>
        {label}
      </span>
      {right}
    </div>
  );
}

function entryPsnr(entry: ValidationHistoryEntry | null): number | null {
  return entry?.fullPsnr ?? entry?.psnr ?? null;
}

function MetricBadge({ color, children }: { color: string; children: ReactNode }) {
  return (
    <span style={{
      fontSize: 9.5, fontFamily: "var(--font-mono)", color,
      padding: "1px 7px", background: "var(--bg2)",
      border: `1px solid color-mix(in srgb, ${color} 32%, var(--border))`,
      borderRadius: 20, whiteSpace: "nowrap",
    }}>
      {children}
    </span>
  );
}

// Inline-style button helper for the compact icon toolbar (disabled greys out).
function toolBtn(disabled: boolean): CSSProperties {
  return {
    fontSize: 10, fontFamily: "var(--font-mono)", padding: "3px 8px", borderRadius: 20,
    cursor: disabled ? "default" : "pointer",
    border: "1px solid var(--border)", background: "var(--bg2)",
    color: disabled ? "var(--dim)" : "var(--muted)",
  };
}

function Filmstrip({
  history, selectedEpoch, isLive, onSelect, onResumeLive,
}: {
  history: ValidationHistoryEntry[];
  selectedEpoch: number | null;
  isLive: boolean;
  onSelect: (epoch: number) => void;
  onResumeLive: () => void;
}) {
  const activeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", inline: "nearest", block: "nearest" });
  }, [selectedEpoch, history.length]);

  if (history.length === 0) return null;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 14px 8px" }}>
      <div style={{
        display: "flex", gap: 5, overflowX: "auto", overflowY: "hidden",
        paddingBottom: 2, flex: 1, scrollbarWidth: "thin",
      }}>
        {history.map((entry) => {
          const active = entry.epoch === selectedEpoch;
          const thumb = pathFor(entry, "sr") ?? pathFor(entry, "lr");
          return (
            <button
              key={entry.epoch}
              ref={active ? activeRef : undefined}
              onClick={() => onSelect(entry.epoch)}
              title={`Epoch ${entry.epoch}${entryPsnr(entry) != null ? ` · PSNR ${fmtMetric(entryPsnr(entry))} dB` : ""}`}
              style={{
                flexShrink: 0, width: 42, height: 42, borderRadius: "var(--radius-sm)",
                border: active ? "1.5px solid var(--blue)" : "1.5px solid var(--border)",
                background: "var(--bg2)", padding: 0, cursor: "pointer", overflow: "hidden",
                position: "relative", transition: "border-color 0.12s ease",
              }}
            >
              {thumb ? (
                <img src={convertFileSrc(thumb)} alt={`epoch ${entry.epoch}`}
                  style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} loading="lazy" />
              ) : (
                <span style={{ fontSize: 8, color: "var(--dim)" }}>—</span>
              )}
              <span style={{
                position: "absolute", bottom: 0, left: 0, right: 0, textAlign: "center",
                fontSize: 7.5, fontFamily: "var(--font-mono)", color: "var(--text)",
                background: "rgba(13,15,17,0.72)", lineHeight: "11px",
              }}>
                {entry.epoch}
              </span>
            </button>
          );
        })}
      </div>
      {!isLive && (
        <button
          onClick={onResumeLive}
          style={{
            flexShrink: 0, fontSize: 9.5, fontFamily: "var(--font-mono)", cursor: "pointer",
            color: "var(--green)", background: "var(--green-dim)", border: "1px solid rgba(77,186,127,0.3)",
            borderRadius: 20, padding: "3px 9px", display: "flex", alignItems: "center", gap: 4,
          }}
        >
          ● jump to latest
        </button>
      )}
    </div>
  );
}

function Lightbox({
  history, epoch, kind, onClose, onNavigate,
}: {
  history: ValidationHistoryEntry[];
  epoch: number;
  kind: FrameKind;
  onClose: () => void;
  onNavigate: (epoch: number, kind: FrameKind) => void;
}) {
  const idx = history.findIndex((e) => e.epoch === epoch);
  const entry = idx >= 0 ? history[idx] : null;
  const path = pathFor(entry, kind);

  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [fitZoom, setFitZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  // Keyed per-frame so navigation can't strand the loading overlay: `loaded`
  // is only true once the *current* path's load event has fired, so switching
  // to a new frame immediately flips back to loading without an effect reset.
  const [loadedPath, setLoadedPath] = useState<string | null>(null);
  const [, setImgNatural] = useState({ w: 0, h: 0 });
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);
  const fitPanRef = useRef({ x: 0, y: 0 });

  const isZoomed = zoom > 1;
  const effectiveZoom = fitZoom * zoom;
  const loaded = path != null && loadedPath === path;

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setFitZoom(1);
    setImgNatural({ w: 0, h: 0 });
  }, [epoch, kind]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && idx > 0) onNavigate(history[idx - 1].epoch, kind);
      else if (e.key === "ArrowRight" && idx < history.length - 1) onNavigate(history[idx + 1].epoch, kind);
      else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const dir = e.key === "ArrowUp" ? -1 : 1;
        const nextIdx = (FRAME_ORDER.indexOf(kind) + dir + FRAME_ORDER.length) % FRAME_ORDER.length;
        onNavigate(epoch, FRAME_ORDER[nextIdx]);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [idx, epoch, kind, history, onClose, onNavigate]);

  const handleImgLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (w === 0 || h === 0) return;
    setImgNatural({ w, h });
    setLoadedPath(path);
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      const fz = Math.min(rect.width / w, rect.height / h);
      setFitZoom(fz);
      const newPan = {
        x: (rect.width - w * fz) / 2,
        y: (rect.height - h * fz) / 2,
      };
      fitPanRef.current = newPan;
      setPan(newPan);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!path) return;
    e.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const imgX = (mouseX - pan.x) / effectiveZoom;
    const imgY = (mouseY - pan.y) / effectiveZoom;

    const delta = e.deltaY > 0 ? -0.25 : 0.25;
    const nextZoom = Math.max(1, Math.min(10, zoom + delta));
    const nextEff = fitZoom * nextZoom;

    const newPan = nextZoom <= 1
      ? fitPanRef.current
      : { x: mouseX - imgX * nextEff, y: mouseY - imgY * nextEff };

    setZoom(nextZoom);
    setPan(newPan);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isZoomed) return;
    e.preventDefault();
    setDragging(true);
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return;
      setPan({
        x: dragRef.current.panX + (ev.clientX - dragRef.current.startX),
        y: dragRef.current.panY + (ev.clientY - dragRef.current.startY),
      });
    };
    const onUp = () => {
      dragRef.current = null;
      setDragging(false);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const zoomLabel = zoom === 1 ? "fit" : `${Math.round(zoom * 100)}%`;

  const zoomAroundPoint = (nextZoom: number) => {
    if (nextZoom <= 1) {
      setZoom(1);
      setPan(fitPanRef.current);
      return;
    }
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) { setZoom(nextZoom); return; }
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const imgX = (cx - pan.x) / effectiveZoom;
    const imgY = (cy - pan.y) / effectiveZoom;
    const nextEff = fitZoom * nextZoom;
    setZoom(nextZoom);
    setPan({ x: cx - imgX * nextEff, y: cy - imgY * nextEff });
  };

  // Double-click toggles between fit and 2× — an expected gesture in image
  // preview tools. Pans back to the centered fit position when leaving zoom.
  const handleDoubleClick = () => {
    zoomAroundPoint(zoom > 1 ? 1 : 2);
  };

  return (
    <div
      onClick={onClose}
      role="dialog"
      aria-label={`Epoch ${epoch} — ${FRAME_META[kind].label} preview`}
      style={{
        position: "fixed", inset: 0, zIndex: 200, background: "rgba(8,9,11,0.86)",
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        padding: 12, backdropFilter: "blur(3px)",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          display: "flex", flexDirection: "column",
          width: "88vw", height: "80vh",
          maxWidth: 1200, maxHeight: 900,
          minWidth: 500, minHeight: 350,
          gap: 8,
        }}
      >
        {/* ── Toolbar: left metrics · center frame tabs · right zoom/close ── */}
        <div style={{
          display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
          justifyContent: "space-between", flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", flex: "1 1 260px", minWidth: 0 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", fontFamily: "var(--font-mono)", whiteSpace: "nowrap" }}>
              Epoch {epoch} <span style={{ color: "var(--dim)", fontWeight: 400 }}>/ {history.length}</span>
            </span>
            {entry?.psnr != null && <MetricBadge color="var(--green)">PSNR {fmtMetric(entry.psnr)} dB</MetricBadge>}
            {entry?.fullPsnr != null && <MetricBadge color="var(--teal)">full {fmtMetric(entry.fullPsnr)} dB</MetricBadge>}
            {entry?.ssim != null && <MetricBadge color="var(--blue)">SSIM {fmtMetric(entry.ssim, 4)}</MetricBadge>}
            {entry?.fullSsim != null && <MetricBadge color="var(--purple)">full {fmtMetric(entry.fullSsim, 4)}</MetricBadge>}
          </div>

          <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
            {FRAME_ORDER.map((k) => {
              const enabled = pathFor(entry, k) != null;
              const active = k === kind;
              return (
                <button
                  key={k}
                  aria-pressed={active}
                  onClick={() => onNavigate(epoch, k)}
                  disabled={!enabled}
                  title={`View ${FRAME_META[k].label}`}
                  style={{
                    fontSize: 10, fontFamily: "var(--font-mono)", padding: "3px 10px", borderRadius: 20,
                    cursor: enabled ? "pointer" : "default",
                    border: active ? "1px solid var(--blue)" : "1px solid var(--border)",
                    background: active ? "var(--blue-dim)" : "var(--bg2)",
                    color: active ? "var(--blue)" : enabled ? "var(--muted)" : "var(--dim)",
                  }}
                >
                  {FRAME_META[k].label}
                </button>
              );
            })}
          </div>

          <div style={{ display: "flex", gap: 4, alignItems: "center", flexShrink: 0 }}>
            <button onClick={() => zoomAroundPoint(1)} disabled={zoom === 1}
              aria-label="Reset zoom to fit" title="Fit — double-click image toggles" style={toolBtn(zoom === 1)}>
              ⊟ fit
            </button>
            <button onClick={() => zoomAroundPoint(zoom - 0.5)} disabled={zoom <= 1}
              aria-label="Zoom out" title="Zoom out" style={toolBtn(zoom <= 1)}>
              −
            </button>
            <span style={{
              fontSize: 10, fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--text)",
              minWidth: 32, textAlign: "center",
            }}>
              {zoomLabel}
            </span>
            <button onClick={() => zoomAroundPoint(zoom + 0.5)} disabled={zoom >= 10}
              aria-label="Zoom in" title="Zoom in" style={toolBtn(zoom >= 10)}>
              +
            </button>
            <button onClick={onClose} aria-label="Close (Esc)" title="Close (Esc)"
              style={{ ...toolBtn(false), color: "var(--muted)", marginLeft: 2 }}>
              ✕ close
            </button>
          </div>
        </div>

        {/* ── Image stage ── */}
        <div ref={containerRef}
          style={{
            position: "relative", flex: 1, minHeight: 0, width: "100%",
            border: "1px solid var(--border)", borderRadius: "var(--radius-lg)",
            padding: 6, overflow: "hidden", ...CHECKERBOARD_BG,
          }}
          onWheel={handleWheel}
          onDoubleClick={path ? handleDoubleClick : undefined}
        >
          <button
            onClick={() => onNavigate(history[idx - 1].epoch, kind)}
            disabled={idx <= 0}
            aria-label="Previous epoch (Left arrow)"
            title="Previous epoch (←)"
            style={navArrowStyle("left", idx <= 0)}
          >‹</button>

          {path ? (
            <img
              key={path}
              src={convertFileSrc(path)} alt={FRAME_META[kind].label}
              onLoad={handleImgLoad}
              onMouseDown={handleMouseDown}
              style={{
                display: "block",
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${effectiveZoom})`,
                transformOrigin: "0 0",
                cursor: dragging ? "grabbing" : isZoomed ? "grab" : "default",
                userSelect: "none",
              }}
            />
          ) : (
            <div style={{
              width: "100%", height: "100%",
              display: "flex", alignItems: "center", justifyContent: "center",
              color: "var(--dim)", fontFamily: "var(--font-mono)", fontSize: 11,
            }}>
              no {FRAME_META[kind].label} frame for this epoch
            </div>
          )}

          {path && !loaded && (
            <div style={{
              position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
              background: "rgba(13,15,17,0.35)", zIndex: 1, pointerEvents: "none",
            }}>
              <div style={{
                width: 18, height: 18, borderRadius: "50%",
                border: "2px solid var(--border2)", borderTopColor: "var(--text)",
                animation: "spin 0.8s linear infinite",
              }} />
            </div>
          )}

          <button
            onClick={() => onNavigate(history[idx + 1].epoch, kind)}
            disabled={idx >= history.length - 1}
            aria-label="Next epoch (Right arrow)"
            title="Next epoch (→)"
            style={navArrowStyle("right", idx >= history.length - 1)}
          >›</button>
        </div>

        <div style={{
          fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)",
          textAlign: "center", flexShrink: 0, letterSpacing: "0.03em",
        }}>
          ←→ epoch · ↑↓ frame · scroll zoom · drag pan · Esc close
        </div>
      </div>
    </div>
  );
}

function navArrowStyle(side: "left" | "right", disabled: boolean): CSSProperties {
  return {
    position: "absolute", [side]: 4, top: "50%", transform: "translateY(-50%)",
    width: 32, height: 32, borderRadius: "50%", border: "1px solid var(--border2)",
    background: "rgba(20,23,25,0.85)", color: disabled ? "var(--dim)" : "var(--text)",
    fontSize: 18, lineHeight: 1,
    cursor: disabled ? "default" : "pointer",
    opacity: disabled ? 0.4 : 1, zIndex: 2, transition: "opacity 0.12s ease",
  };
}

function FrameCell({ label, path, onExpand }: { label: string; path: string | null; onExpand: () => void }) {
  const [hover, setHover] = useState(false);
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={path ? onExpand : undefined}
      onKeyDown={path ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onExpand(); } } : undefined}
      role="button"
      tabIndex={path ? 0 : -1}
      aria-label={path ? `Open ${label} frame preview` : undefined}
      style={{
        background: "var(--bg2)", border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)", overflow: "hidden",
        display: "flex", alignItems: "center", justifyContent: "center",
        position: "relative", minHeight: 60, cursor: path ? "zoom-in" : "default",
        transition: "border-color 0.15s ease", outline: "none",
      }}
    >
      {path ? (
        <img
          src={convertFileSrc(path)}
          alt={label}
          loading="lazy"
          style={{ width: "100%", height: "100%", objectFit: "contain" }}
        />
      ) : (
        <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>—</span>
      )}
      <span style={{
        position: "absolute", top: 5, left: 6, fontSize: 9, fontWeight: 600,
        color: "var(--text)", fontFamily: "var(--font-mono)", letterSpacing: "0.04em",
        background: "rgba(13,15,17,0.75)", padding: "2px 6px", borderRadius: 20,
        backdropFilter: "blur(2px)",
      }}>
        {label}
      </span>
      {path && (
        <span style={{
          position: "absolute", top: 5, right: 6, width: 18, height: 18, borderRadius: "50%",
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11,
          color: "var(--text)", background: "rgba(13,15,17,0.75)", opacity: hover ? 1 : 0,
          transition: "opacity 0.12s ease", pointerEvents: "none",
        }}>
          ⤢
        </span>
      )}
    </div>
  );
}

export function ValidationPanel() {
  const latestFrames = useTrainingStore((s) => s.validationFrames);
  const history = useTrainingStore((s) => s.validationHistory);
  const validationRunning = useTrainingStore((s) => s.validationRunning);
  const [pinnedEpoch, setPinnedEpoch] = useState<number | null>(null);
  const [lightbox, setLightbox] = useState<{ epoch: number; kind: FrameKind } | null>(null);
  const lightboxTriggerRef = useRef<HTMLElement | null>(null);

  const latestEpoch = history.length > 0 ? history[history.length - 1].epoch : null;
  const isLive = pinnedEpoch == null;
  const selectedEpoch = pinnedEpoch ?? latestEpoch;

  const openLightbox = (kind: FrameKind) => {
    if (selectedEpoch == null) return;
    lightboxTriggerRef.current = document.activeElement as HTMLElement | null;
    setLightbox({ epoch: selectedEpoch, kind });
  };

  const closeLightbox = () => {
    setLightbox(null);
    lightboxTriggerRef.current?.focus?.();
    lightboxTriggerRef.current = null;
  };

  const selectedEntry = useMemo(
    () => history.find((e) => e.epoch === selectedEpoch) ?? null,
    [history, selectedEpoch],
  );

  const activeFrames: ValidationFrames | null = selectedEntry ?? (isLive ? latestFrames : null);

  const cells = FRAME_ORDER.map((kind) => ({
    kind, label: FRAME_META[kind].label, path: pathFor(activeFrames, kind),
  }));

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <PanelHeader
        label="Validation Frames"
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {selectedEntry?.psnr != null && (
              <span style={{ fontSize: 9.5, color: "var(--green)", fontFamily: "var(--font-mono)" }}>
                {fmtMetric(selectedEntry.psnr)} dB
              </span>
            )}
            {selectedEpoch != null && (
              <span style={{
                fontSize: 9.5, fontFamily: "var(--font-mono)", color: isLive ? "var(--green)" : "var(--muted)",
                background: isLive ? "var(--green-dim)" : "var(--bg2)", padding: "2px 8px", borderRadius: 20,
                border: isLive ? "1px solid rgba(77,186,127,0.3)" : "1px solid var(--border)",
              }}>
                {isLive && validationRunning
                  ? `● validating · e${selectedEpoch}`
                  : isLive
                    ? `● live · e${selectedEpoch}`
                    : `epoch ${selectedEpoch}`}
              </span>
            )}
          </div>
        }
      />

      <Filmstrip
        history={history}
        selectedEpoch={selectedEpoch}
        isLive={isLive}
        onSelect={(epoch) => setPinnedEpoch(epoch === latestEpoch ? null : epoch)}
        onResumeLive={() => setPinnedEpoch(null)}
      />

      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr",
        gap: 6, flex: 1, minHeight: 0, padding: "0 14px 14px",
      }}>
        {cells.map(({ kind, label, path }) => (
          <FrameCell key={kind} label={label} path={path} onExpand={() => openLightbox(kind)} />
        ))}
      </div>

      {lightbox && history.length > 0 && (
        <Lightbox
          history={history}
          epoch={lightbox.epoch}
          kind={lightbox.kind}
          onClose={closeLightbox}
          onNavigate={(epoch, kind) => {
            setLightbox({ epoch, kind });
            if (epoch !== latestEpoch) setPinnedEpoch(epoch);
            else setPinnedEpoch(null);
          }}
        />
      )}
    </div>
  );
}
