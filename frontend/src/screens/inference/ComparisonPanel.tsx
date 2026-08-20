import { useRef, useCallback, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useInferenceStore } from "../../store/inferenceStore";
import { CHECKERBOARD_BG } from "../../lib/checkerboardBg";

export function ComparisonPanel({ splitterPct, onSplitterPctChange }: { splitterPct: number; onSplitterPctChange: (pct: number) => void }) {
  const result = useInferenceStore((s) => s.result);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const lrSrc = result?.success && result.preview_input_path ? convertFileSrc(result.preview_input_path) : null;
  const srSrc = result?.success && result.preview_output_path ? convertFileSrc(result.preview_output_path) : null;

  const lrLabel = result?.success && result.input_resolution ? `${result.input_resolution.width}×${result.input_resolution.height}` : null;
  const srLabel = result?.success && result.output_resolution ? `${result.output_resolution.width}×${result.output_resolution.height}` : null;

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    onSplitterPctChange(Math.max(2, Math.min(98, (x / rect.width) * 100)));
  }, [onSplitterPctChange]);

  const handleMouseUp = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  const startDrag = () => {
    dragging.current = true;
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <div
      ref={containerRef}
      onMouseDown={startDrag}
      className="si-compare"
      style={CHECKERBOARD_BG}
    >
      {/* Base layer — SR */}
      <div className="si-layer">
        {srSrc ? (
          <img src={srSrc} alt="Output (SR)" className="si-img" draggable={false} />
        ) : (
          <div style={{ width: "100%", height: "100%", ...CHECKERBOARD_BG }} />
        )}
      </div>

      {/* Top layer — LR (upscaled to match SR scale), clipped at the handle */}
      <div className="si-layer" style={{ clipPath: `polygon(0 0, ${splitterPct}% 0, ${splitterPct}% 100%, 0 100%)` }}>
        {lrSrc ? (
          <img src={lrSrc} alt="Input (LR)" className="si-img" draggable={false} />
        ) : (
          <div style={{ width: "100%", height: "100%", ...CHECKERBOARD_BG }} />
        )}
      </div>

      {/* Badges */}
      {lrLabel && (
        <div className="si-badge" style={{ left: 12, background: "color-mix(in srgb, var(--amber) 85%, transparent)" }}>
          LR {lrLabel}
        </div>
      )}
      {srLabel && (
        <div className="si-badge" style={{ right: 12, background: "color-mix(in srgb, var(--green) 85%, transparent)" }}>
          SR {srLabel}
        </div>
      )}

      {/* Handle line + knob */}
      <div className="si-handle" style={{ left: `${splitterPct}%` }} />
      <div className="si-knob" style={{ left: `${splitterPct}%` }}>
        <span className="si-knob-icon">◂▸</span>
      </div>

      {/* Placeholder */}
      {!result && (
        <div className="si-placeholder">
          <span className="si-placeholder-text">
            Run inference to see comparison
          </span>
        </div>
      )}

      <input
        type="range"
        min={2}
        max={98}
        value={splitterPct}
        onChange={(e) => onSplitterPctChange(Number(e.target.value))}
        className="si-range"
      />
    </div>
  );
}