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
  contentStyle?: CSSProperties;
}

export function CollapsibleSection({
  title,
  icon,
  subtitle,
  badge,
  defaultOpen = true,
  children,
  style,
  contentStyle,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="collapsible" style={style}>
      <button
        onClick={() => setOpen((o) => !o)}
        className={`collapsible-header${open ? " collapsible-header-open" : ""}`}
      >
        {icon && (
          <span className="collapsible-icon">{icon}</span>
        )}
        <span className="collapsible-title">{title}</span>
        {subtitle && (
          <span className="collapsible-subtitle">{subtitle}</span>
        )}
        <span className="collapsible-right">
          {badge}
          <IconChevron
            size={11}
            color="var(--muted)"
            style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}
          />
        </span>
      </button>
      {open && <div className="collapsible-body" style={contentStyle}>{children}</div>}
    </div>
  );
}