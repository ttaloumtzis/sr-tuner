interface RadialGaugeProps {
  label: string;
  value: string;
  pct: number | null;
  color: string;
  size?: number;
  strokeWidth?: number;
  sub?: string;
}

export function RadialGauge({
  label, value, pct, color, size = 72, strokeWidth = 7, sub,
}: RadialGaugeProps) {
  const hasData = pct != null && Number.isFinite(pct);
  const clamped = Math.max(0, Math.min(100, hasData ? (pct as number) : 0));
  const r = (size - strokeWidth) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - clamped / 100);
  const critical = hasData && clamped >= 92;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 7, minWidth: size }}>
      <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: "block", transform: "rotate(-90deg)" }}>
          <circle cx={c} cy={c} r={r} fill="none" stroke="var(--bg3)" strokeWidth={strokeWidth} />
          <circle
            cx={c} cy={c} r={r} fill="none"
            stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={hasData ? offset : circumference}
            opacity={hasData ? 1 : 0.28}
            style={{ transition: "stroke-dashoffset 0.35s ease, opacity 0.2s ease" }}
          />
          {critical && (
            <circle cx={c} cy={c} r={r} fill="none" stroke={color} strokeWidth={strokeWidth + 3} opacity={0.18} />
          )}
        </svg>
        <div style={{
          position: "absolute", inset: 0, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
        }}>
          <span style={{
            fontSize: Math.round(size * 0.205), fontWeight: 700, lineHeight: 1,
            color: hasData ? "var(--text)" : "var(--dim)", fontFamily: "var(--font-mono)",
          }}>
            {value}
          </span>
        </div>
      </div>
      <div style={{ textAlign: "center", lineHeight: 1.3 }}>
        <div style={{ fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--font-mono)", letterSpacing: "0.05em" }}>
          {label}
        </div>
        {sub && (
          <div style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)", marginTop: 1 }}>
            {sub}
          </div>
        )}
      </div>
    </div>
  );
}

function tempColor(temp: number | null): string {
  if (temp == null) return "var(--dim)";
  if (temp >= 88) return "var(--red)";
  if (temp >= 72) return "var(--amber)";
  return "var(--green)";
}

export function TempBadge({ temp, size = 72 }: { temp: number | null; size?: number }) {
  const color = tempColor(temp);
  const hot = temp != null && temp >= 88;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 7, minWidth: size }}>
      <div style={{
        width: size, height: size, borderRadius: 15, flexShrink: 0,
        background: `color-mix(in srgb, ${color} 15%, var(--bg2))`,
        border: `1.5px solid color-mix(in srgb, ${color} 42%, var(--border))`,
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        transition: "background 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease",
        boxShadow: hot ? `0 0 14px color-mix(in srgb, ${color} 45%, transparent)` : "none",
      }}>
        <span style={{ fontSize: Math.round(size * 0.26), fontWeight: 700, lineHeight: 1, color, fontFamily: "var(--font-mono)" }}>
          {temp != null ? Math.round(temp) : "—"}
        </span>
        <span style={{ fontSize: 9, color, opacity: 0.85, fontFamily: "var(--font-mono)", marginTop: 2 }}>
          {temp != null ? "°C" : "no data"}
        </span>
      </div>
      <div style={{ fontSize: 9.5, color: "var(--muted)", fontFamily: "var(--font-mono)", letterSpacing: "0.05em" }}>
        GPU TEMP
      </div>
    </div>
  );
}
