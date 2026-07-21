import { ReactNode } from "react";
import { IconAlert, IconCheck } from "./icons";

type AlertTone = "amber" | "red" | "green" | "muted";

const TONE_MAP: Record<AlertTone, { fg: string; bg: string; border: string }> = {
  amber: { fg: "#f5a623", bg: "rgba(245,166,35,0.1)", border: "rgba(245,166,35,0.4)" },
  red:   { fg: "#f87171", bg: "rgba(239,68,68,0.1)",  border: "rgba(239,68,68,0.4)"  },
  green: { fg: "var(--green)", bg: "var(--green-dim)", border: "var(--green)" },
  muted: { fg: "var(--muted)", bg: "var(--bg2)",      border: "var(--border)"       },
};

interface InlineAlertProps {
  tone?: AlertTone;
  icon?: boolean;
  children: ReactNode;
}

/** Small inline notice used for warnings, errors, and informational asides
 *  throughout Training Setup. Consolidates what used to be four near-identical
 *  inline-styled `<div>`s into one consistent, reusable look. */
export function InlineAlert({ tone = "amber", icon = true, children }: InlineAlertProps) {
  const c = TONE_MAP[tone];
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 6,
        padding: "6px 8px",
        background: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: "var(--radius-sm)",
        fontSize: 10,
        lineHeight: 1.4,
        color: c.fg,
      }}
    >
      {icon && tone !== "muted" && (
        tone === "green"
          ? <IconCheck size={12} color={c.fg} strokeWidth={3} style={{ flexShrink: 0, marginTop: 1 }} />
          : <IconAlert size={12} color={c.fg} style={{ flexShrink: 0, marginTop: 1 }} />
      )}
      <div style={{ minWidth: 0 }}>{children}</div>
    </div>
  );
}