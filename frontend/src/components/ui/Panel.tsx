import { ReactNode, CSSProperties } from "react";

interface PanelProps {
  title?: string;
  icon?: ReactNode;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  style?: CSSProperties;
  noPadding?: boolean;
}

export function Panel({ title, icon, subtitle, actions, children, style, noPadding }: PanelProps) {
  return (
    <div
      style={{
        background: "var(--bg1)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        ...style,
      }}
    >
      {title && (
        <div
          style={{
            padding: "6px 10px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            gap: 7,
            justifyContent: "space-between",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
            {icon && <span style={{ color: "var(--muted)", display: "flex", flexShrink: 0 }}>{icon}</span>}
            <span style={{ fontSize: 11, color: "var(--text)", fontWeight: 500, whiteSpace: "nowrap" }}>
              {title}
            </span>
            {subtitle && (
              <span style={{ fontSize: 10, color: "var(--dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {subtitle}
              </span>
            )}
          </div>
          {actions && (
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexShrink: 0 }}>
              {actions}
            </div>
          )}
        </div>
      )}
      <div
        style={{
          padding: noPadding ? 0 : 10,
          flex: 1,
          overflow: "auto",
          minHeight: 0,
        }}
      >
        {children}
      </div>
    </div>
  );
}
