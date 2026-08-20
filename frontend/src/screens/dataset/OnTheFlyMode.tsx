export function OnTheFlyMode() {
  return (
    <div style={{ background: "color-mix(in srgb, var(--amber) 10%, var(--bg2))", border: "1px solid color-mix(in srgb, var(--amber) 40%, transparent)", borderRadius: "var(--radius-md)", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, textAlign: "center" }}>
      <span style={{ fontSize: 14, color: "var(--amber)", fontWeight: 600 }}>On-the-fly</span>
      <span style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>Decode video directly during training — ~90% less disk usage.<br />Coming soon.</span>
    </div>
  );
}