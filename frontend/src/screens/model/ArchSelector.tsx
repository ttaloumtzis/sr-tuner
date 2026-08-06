import { useState } from "react";
import { getAllArchitectures, type ArchitectureDef, type Architecture } from "../../lib/architectures";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      <span style={{ fontSize: 10, color: "var(--muted)" }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)" }}>{value}</span>
    </div>
  );
}

function ArchCard({ def, active, onClick }: { def: ArchitectureDef; active: boolean; onClick: () => void }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex", flexDirection: "column", gap: 6,
        padding: 12, borderRadius: "var(--radius-md)", cursor: "pointer",
        background: active ? "var(--green-dim)" : hovered ? "var(--bg3)" : "var(--bg2)",
        border: `1px solid ${active ? "var(--green)" : hovered ? "var(--border2)" : "var(--border)"}`,
        transition: "var(--transition-fast)",
        textAlign: "left", width: "100%",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: active ? "var(--green)" : "var(--bg3)",
          border: "1px solid var(--border)",
          flexShrink: 0,
        }} />
        <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>{def.displayName}</span>
      </div>
      <span style={{ fontSize: 10, color: "var(--green)", fontWeight: 600, letterSpacing: "0.3px" }}>{def.tag}</span>
      <span style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.4 }}>{def.description}</span>
      <div style={{ display: "flex", gap: 12, marginTop: 2 }} title="Reference figures at default settings — actual values depend on the config you choose below">
        <Stat label="VRAM" value={def.reference.vram} />
        <Stat label="Params" value={def.reference.params} />
      </div>
    </button>
  );
}

interface ArchSelectorProps {
  selected: Architecture;
  onSelect: (a: Architecture) => void;
}

export function ArchSelector({ selected, onSelect }: ArchSelectorProps) {
  const all = getAllArchitectures();
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {all.map((def) => (
        <ArchCard key={def.id} def={def} active={selected === def.id} onClick={() => onSelect(def.id as Architecture)} />
      ))}
    </div>
  );
}