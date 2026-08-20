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
      <div className="stacked-bar" style={{ height, borderRadius: height / 2 }} />
    );
  }
  return (
    <div
      className="stacked-bar"
      style={{ height, borderRadius: height / 2 }}
    >
      {segments.map((s, i) => {
        const pct = (s.value / total) * 100;
        if (pct <= 0) return null;
        return (
          <div
            key={i}
            title={`${s.label}: ${s.value.toFixed(2)} GB`}
            className="stacked-bar-seg"
            style={{
              width: `${pct}%`,
              background: s.color,
            }}
          />
        );
      })}
    </div>
  );
}