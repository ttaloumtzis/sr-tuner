interface InfoRowProps {
  label: string;
  value: string | number;
  color?: string;
  mono?: boolean;
}

export function InfoRow({ label, value, color, mono }: InfoRowProps) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "4px 0",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <span style={{ fontSize: 11, color: "var(--muted)" }}>{label}</span>
      <span
        style={{
          fontSize: 11,
          color: color ?? "var(--text)",
          fontFamily: mono ? "var(--font-mono)" : undefined,
        }}
      >
        {value}
      </span>
    </div>
  );
}
