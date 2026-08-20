import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import type { ValidationHistoryEntry } from "../../store/trainingStore";
import { CHECKERBOARD_BG } from "../../lib/checkerboardBg";
import { fmt } from "../../lib/format";
import { FRAME_ORDER, FRAME_META, pathFor, type FrameKind } from "./frameMeta";

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

export function Lightbox({
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
            {entry?.psnr != null && <MetricBadge color="var(--green)">PSNR {fmt(entry.psnr)} dB</MetricBadge>}
            {entry?.fullPsnr != null && <MetricBadge color="var(--teal)">full {fmt(entry.fullPsnr)} dB</MetricBadge>}
            {entry?.ssim != null && <MetricBadge color="var(--blue)">SSIM {fmt(entry.ssim, 4)}</MetricBadge>}
            {entry?.fullSsim != null && <MetricBadge color="var(--purple)">full {fmt(entry.fullSsim, 4)}</MetricBadge>}
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