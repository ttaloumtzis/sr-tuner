export interface StackedBarSegment {
  label: string;
  value: number;
  color: string;
}

interface StackedBarProps {
  segments: StackedBarSegment[];
  height?: number;
}

/** Horizontal segmented bar — used for the VRAM breakdown so the split reads at a glance. */
export function StackedBar({ segments, height = 8 }: StackedBarProps) {
  const total = segments.reduce((sum, s) => sum + s.value, 0);
  if (total <= 0) {
    return (
      <div style={{ height, borderRadius: height / 2, background: "var(--bg3)" }} />
    );
  }
  return (
    <div
      style={{
        display: "flex",
        height,
        borderRadius: height / 2,
        overflow: "hidden",
        background: "var(--bg3)",
      }}
    >
      {segments.map((s, i) => {
        const pct = (s.value / total) * 100;
        if (pct <= 0) return null;
        return (
          <div
            key={i}
            title={`${s.label}: ${s.value.toFixed(2)} GB`}
            style={{
              width: `${pct}%`,
              background: s.color,
              transition: "width 0.3s ease",
            }}
          />
        );
      })}
    </div>
  );
}
