interface InfoRowProps {
  label: string;
  value: string | number | null;
  color?: string;
  mono?: boolean;
  dec?: number;
  baseline?: boolean;
  border?: boolean;
  emphasis?: boolean;
  labelSize?: number;
  labelMono?: boolean;
  ellipsis?: boolean;
}

export function InfoRow({
  label,
  value,
  color,
  mono = true,
  dec = 2,
  baseline,
  border = true,
  emphasis,
  labelSize = 10,
  labelMono,
  ellipsis,
}: InfoRowProps) {
  const text =
    value == null ? "—" : typeof value === "number" ? value.toFixed(dec) : value;
  return (
    <div
      className="info-row"
      style={{
        alignItems: baseline ? "baseline" : undefined,
        borderBottom: border ? "1px solid var(--border)" : undefined,
        padding: border ? "4px 0" : undefined,
        gap: ellipsis ? 4 : undefined,
      }}
    >
      <span
        className="info-row-label"
        style={{
          fontSize: labelSize,
          fontFamily: labelMono ? "var(--font-mono)" : "var(--font-sans)",
          flexShrink: ellipsis ? 0 : undefined,
        }}
      >
        {label}
      </span>
      <span
        className="info-row-value"
        style={{
          fontSize: emphasis ? 12 : 11,
          color: value == null ? "var(--dim)" : (color ?? "var(--text)"),
          fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
          fontWeight:
            value == null ? 400 : emphasis ? 600 : baseline && !ellipsis ? 500 : 400,
          textAlign: ellipsis ? "right" : undefined,
          overflow: ellipsis ? "hidden" : undefined,
          textOverflow: ellipsis ? "ellipsis" : undefined,
          whiteSpace: ellipsis ? "nowrap" : undefined,
        }}
      >
        {text}
      </span>
    </div>
  );
}