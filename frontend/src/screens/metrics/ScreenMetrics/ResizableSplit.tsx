import { useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

export function ResizableSplit({ top, bottom, defaultRatio = 0.5, minPx = 80, minTopPx }: {
  top: ReactNode;
  bottom: ReactNode;
  defaultRatio?: number;
  minPx?: number;
  minTopPx?: number;
}) {
  const [ratio, setRatio] = useState(defaultRatio);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startY: number; startRatio: number } | null>(null);

  const onMouseDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragRef.current = { startY: e.clientY, startRatio: ratio };
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const delta = ev.clientY - dragRef.current.startY;
      let r = dragRef.current.startRatio + delta / rect.height;
      r = Math.max(minPx / rect.height, Math.min(1 - minPx / rect.height, r));
      setRatio(r);
    };
    const onUp = () => {
      dragRef.current = null;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  return (
    <div ref={containerRef} style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ flex: ratio, minHeight: minTopPx ?? 0, overflow: "hidden" }}>
        {top}
      </div>
      <div
        onMouseDown={onMouseDown}
        style={{
          flexShrink: 0, height: 8, cursor: "row-resize", position: "relative",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        <div style={{
          width: 28, height: 3, borderRadius: 2, background: "var(--border2)",
          transition: "background 0.15s ease",
        }} />
      </div>
      <div style={{ flex: 1 - ratio, minHeight: 0, overflow: "hidden" }}>
        {bottom}
      </div>
    </div>
  );
}
