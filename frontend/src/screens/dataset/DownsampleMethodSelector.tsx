import { useDatasetStore, type DownscaleKernel } from "../../store/datasetStore";

export function DownsampleMethodSelector() {
  const s = useDatasetStore();
  const options: { id: DownscaleKernel; label: string }[] = [
    { id: "area", label: "Area" }, { id: "bicubic", label: "Bicubic" }, { id: "bilinear", label: "Bilinear" },
    { id: "lanczos", label: "Lanczos" }, { id: "nearest", label: "Nearest" },
  ];
  return (
    <div style={{ padding: "7px 10px", background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1 }}>Downsample Method</span>
        {options.map((opt) => (
          <button key={opt.id} onClick={() => s.setKernel(opt.id)}
            style={{ background: s.kernel === opt.id ? "var(--green)" : "var(--bg3)", border: `1px solid ${s.kernel === opt.id ? "var(--green)" : "var(--border)"}`, color: s.kernel === opt.id ? "#0d0f11" : "var(--muted)", fontSize: 11, fontWeight: s.kernel === opt.id ? 600 : 400, padding: "3px 11px", borderRadius: 10, cursor: "pointer", transition: "var(--transition-fast)" }}>
            {opt.label}
          </button>
        ))}
      </div>
      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "var(--muted)", cursor: "pointer" }}>
        <input type="checkbox" checked={s.antialias} onChange={(e) => s.setAntialias(e.target.checked)} style={{ accentColor: "var(--green)" }} />
        Antialias pre-filter
      </label>
    </div>
  );
}