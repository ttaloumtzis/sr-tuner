import { useState } from "react";

// ── Action card ────────────────────────────────────────────────────────────

export function ActionCard({
  icon,
  title,
  subtitle,
  onClick,
  active,
}: {
  icon: string;
  title: string;
  subtitle: string;
  onClick: () => void;
  active?: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  const lit = hovered || active;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: lit ? "var(--bg2)" : "var(--bg1)",
        border: `1px solid ${lit ? "var(--green)66" : "var(--border)"}`,
        borderRadius: "var(--radius-md)",
        padding: "18px 20px",
        cursor: "pointer",
        transition: "var(--transition-normal)",
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <span style={{ fontSize: 20 }}>{icon}</span>
      <span
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: lit ? "var(--green)" : "var(--text)",
          transition: "color 0.12s",
        }}
      >
        {title}
      </span>
      <span style={{ fontSize: 11, color: "var(--muted)" }}>{subtitle}</span>
    </div>
  );
}