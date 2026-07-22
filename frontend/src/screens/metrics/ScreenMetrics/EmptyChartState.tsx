export function EmptyChartState({ label }: { label: string }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      height: "100%", flexDirection: "column", gap: 4,
    }}>
      <div style={{
        width: 26, height: 26, borderRadius: "50%", border: "1.5px dashed var(--border2)",
      }} />
      <span style={{ fontSize: 10.5, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
        awaiting {label}…
      </span>
    </div>
  );
}
