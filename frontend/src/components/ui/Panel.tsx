import { ReactNode, CSSProperties } from "react";

interface PanelProps {
  title?: string;
  icon?: ReactNode;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  style?: CSSProperties;
  noPadding?: boolean;
  shrink?: boolean;
  grow?: boolean;
}

export function Panel({ title, icon, subtitle, actions, children, style, noPadding, shrink, grow }: PanelProps) {
  return (
    <div
      className={["panel", shrink && "panel-shrink", grow && "panel-grow"].filter(Boolean).join(" ")}
      style={style}
    >
      {title && (
        <div className="panel-header">
          <div className="panel-header-inner">
            {icon && <span className="panel-icon">{icon}</span>}
            <span className="panel-title">{title}</span>
            {subtitle && <span className="panel-subtitle">{subtitle}</span>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </div>
      )}
      <div className={noPadding ? "panel-body panel-body-nopad" : "panel-body"}>
        {children}
      </div>
    </div>
  );
}