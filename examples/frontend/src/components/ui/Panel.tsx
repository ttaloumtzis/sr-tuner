import { ReactNode, CSSProperties } from "react";

interface PanelProps {
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
  style?: CSSProperties;
  noPadding?: boolean;
}

export function Panel({ title, actions, children, style, noPadding }: PanelProps) {
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
            padding: "7px 12px",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexShrink: 0,
          }}
        >
          <span style={{ fontSize: 11, color: "var(--text)", fontWeight: 500 }}>
            {title}
          </span>
          {actions && (
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              {actions}
            </div>
          )}
        </div>
      )}
      <div
        style={{
          padding: noPadding ? 0 : 12,
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
