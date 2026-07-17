interface PBarProps {
  value: number;
  max?: number;
  color?: string;
  height?: number;
}

export function PBar({ value, max = 100, color = "var(--green)", height = 6 }: PBarProps) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div
      style={{
        background: "var(--bg3)",
        borderRadius: 2,
        height,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${pct}%`,
          height: "100%",
          background: color,
          borderRadius: 2,
          transition: "width 0.3s ease",
        }}
      />
    </div>
  );
}
