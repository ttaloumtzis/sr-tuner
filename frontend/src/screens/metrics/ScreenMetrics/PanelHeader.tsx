import type { ReactNode } from "react";

export function PanelHeader({ label, right }: { label: string; right?: ReactNode }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "9px 14px 7px", flexShrink: 0,
    }}>
      <span style={{
        fontSize: 10, letterSpacing: "0.06em", color: "var(--muted)",
        fontFamily: "var(--font-mono)", fontWeight: 600, textTransform: "uppercase",
      }}>
        {label}
      </span>
      {right}
    </div>
  );
}
