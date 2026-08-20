import { useDatasetStore } from "../../store/datasetStore";

export function ScaleBar() {
  const s = useDatasetStore();
  const presets = [1, 2, 4, 8];

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 10px", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", flexWrap: "wrap" }}>
      <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Scale</span>
      {presets.map((p) => (
        <button key={p} onClick={() => { s.setScale(p); }}
          style={{ background: s.scale === p ? "var(--green)" : "var(--bg3)", border: `1px solid ${s.scale === p ? "var(--green)" : "var(--border)"}`, color: s.scale === p ? "#0d0f11" : "var(--muted)", fontSize: 11, fontWeight: 600, padding: "2px 9px", borderRadius: "var(--radius-sm)", cursor: "pointer", transition: "var(--transition-fast)" }}>
          ×{p}
        </button>
      ))}
    </div>
  );
}