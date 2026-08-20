import { useState } from "react";

export function TypeCard({ label, description, active, disabled, onClick }: {
  label: string; description: string; active: boolean; disabled?: boolean; onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <div onClick={disabled ? undefined : onClick} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}
      style={{ background: active ? "var(--greenDim)" : hovered && !disabled ? "var(--bg2)" : "var(--bg1)", border: `1px solid ${active ? "var(--green)" : hovered && !disabled ? "var(--muted)" : "var(--border)"}`, borderRadius: "var(--radius-md)", padding: "12px 14px", cursor: disabled ? "not-allowed" : "pointer", transition: "var(--transition-fast)", display: "flex", flexDirection: "column", gap: 4, opacity: disabled ? 0.5 : 1 }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: active ? "var(--green)" : "var(--text)" }}>{label}{disabled && <span style={{ color: "var(--dim)", fontWeight: 400, marginLeft: 6 }}>(soon)</span>}</span>
      <span style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>{description}</span>
    </div>
  );
}