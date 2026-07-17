import React from "react";

export interface ErrorDialogAction {
  label: string;
  onClick: () => void;
  variant?: "primary" | "ghost";
}

interface ErrorDialogProps {
  open: boolean;
  title: string;
  detail: string;
  suggestions?: string[];
  actions?: ErrorDialogAction[];
  onClose?: () => void;
}

export function ErrorDialog({
  open,
  title,
  detail,
  suggestions,
  actions,
  onClose,
}: ErrorDialogProps) {
  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="err-dialog-title"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 10000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0,0,0,0.6)",
        backdropFilter: "blur(2px)",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <div
        style={{
          background: "var(--bg2)",
          border: "1px solid var(--border2)",
          borderTop: "3px solid var(--red)",
          borderRadius: "var(--radius-lg)",
          padding: "24px 28px",
          width: 480,
          maxWidth: "90vw",
          boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
        }}
      >
        <h2
          id="err-dialog-title"
          style={{
            margin: "0 0 10px",
            color: "var(--red)",
            fontSize: 14,
            fontFamily: "var(--font-sans)",
            fontWeight: 600,
          }}
        >
          {title}
        </h2>

        <p
          style={{
            margin: "0 0 16px",
            color: "var(--muted)",
            fontSize: 12,
            fontFamily: "var(--font-mono)",
            lineHeight: 1.6,
            wordBreak: "break-word",
          }}
        >
          {detail}
        </p>

        {suggestions && suggestions.length > 0 && (
          <div
            style={{
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              padding: "10px 14px",
              marginBottom: 20,
            }}
          >
            <p
              style={{
                margin: "0 0 6px",
                color: "var(--muted)",
                fontSize: 11,
                fontFamily: "var(--font-sans)",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              Suggestions
            </p>
            <ul style={{ margin: 0, padding: "0 0 0 16px" }}>
              {suggestions.map((s, i) => (
                <li
                  key={i}
                  style={{
                    color: "var(--text)",
                    fontSize: 12,
                    fontFamily: "var(--font-sans)",
                    lineHeight: 1.6,
                  }}
                >
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          {onClose && (
            <DialogBtn label="Dismiss" variant="ghost" onClick={onClose} />
          )}
          {actions?.map((a, i) => (
            <DialogBtn key={i} label={a.label} variant={a.variant} onClick={a.onClick} />
          ))}
        </div>
      </div>
    </div>
  );
}

function DialogBtn({
  label,
  variant = "ghost",
  onClick,
}: {
  label: string;
  variant?: "primary" | "ghost";
  onClick: () => void;
}) {
  const [hovered, setHovered] = React.useState(false);
  const isPrimary = variant === "primary";

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: isPrimary
          ? hovered ? "#c04444" : "var(--red)"
          : hovered ? "var(--bg3)" : "transparent",
        border: isPrimary ? "none" : "1px solid var(--border)",
        color: isPrimary ? "#fff" : hovered ? "var(--text)" : "var(--muted)",
        fontSize: 12,
        fontFamily: "var(--font-sans)",
        fontWeight: isPrimary ? 600 : 400,
        padding: "6px 16px",
        borderRadius: "var(--radius-sm)",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
    >
      {label}
    </button>
  );
}
