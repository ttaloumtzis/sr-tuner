import { ReactNode, useState, CSSProperties } from "react";
import { IconChevron } from "./icons";

interface CollapsibleSectionProps {
  title: string;
  icon?: ReactNode;
  subtitle?: string;
  badge?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
  style?: CSSProperties;
}

export function CollapsibleSection({
  title,
  icon,
  subtitle,
  badge,
  defaultOpen = true,
  children,
  style,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div
      style={{
        background: "var(--bg1)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
        ...style,
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          all: "unset",
          boxSizing: "border-box",
          width: "100%",
          padding: "7px 10px",
          display: "flex",
          alignItems: "center",
          gap: 8,
          cursor: "pointer",
          borderBottom: open ? "1px solid var(--border)" : "none",
        }}
      >
        {icon && (
          <span style={{ color: "var(--muted)", display: "flex", flexShrink: 0 }}>{icon}</span>
        )}
        <span style={{ fontSize: 11, color: "var(--text)", fontWeight: 500 }}>{title}</span>
        {subtitle && (
          <span style={{ fontSize: 10, color: "var(--dim)" }}>{subtitle}</span>
        )}
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {badge}
          <IconChevron
            size={11}
            color="var(--muted)"
            style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}
          />
        </span>
      </button>
      {open && <div style={{ padding: 10 }}>{children}</div>}
    </div>
  );
}
