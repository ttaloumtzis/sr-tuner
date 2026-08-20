interface PBarProps {
  value: number;
  max?: number;
  color?: string;
  height?: number;
}

export function PBar({ value, max = 100, color = "var(--green)", height = 6 }: PBarProps) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="progress-bar" style={{ height }}>
      <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}