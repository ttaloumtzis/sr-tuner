import { useState, CSSProperties } from "react";

interface BtnProps {
  children: React.ReactNode;
  variant?: "ghost" | "solid";
  color?: string;
  onClick?: () => void;
  small?: boolean;
  full?: boolean;
  disabled?: boolean;
  title?: string;
  style?: CSSProperties;
  type?: "button" | "submit" | "reset";
}

export function Btn({
  children,
  variant = "ghost",
  color,
  onClick,
  small,
  full,
  disabled,
  title,
  style,
  type = "button",
}: BtnProps) {
  const [hovered, setHovered] = useState(false);

  const solidColor = color ?? "var(--green)";
  const bg =
    variant === "solid"
      ? disabled
        ? "var(--bg2)"
        : hovered
        ? solidColor + "dd"
        : solidColor
      : disabled
      ? "var(--bg2)"
      : hovered
      ? "var(--bg2)"
      : "var(--bg3)";

  const border =
    variant === "solid"
      ? "transparent"
      : color
      ? color + "66"
      : "var(--border)";

  const textColor =
    disabled
      ? "var(--dim)"
      : variant === "solid"
      ? "#0d0f11"
      : hovered
      ? color ?? "var(--text)"
      : color ?? "var(--muted)";

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: bg,
        border: `1px solid ${disabled ? "var(--border)" : border}`,
        color: textColor,
        fontSize: small ? 10 : 12,
        fontWeight: variant === "solid" ? 600 : 400,
        padding: small ? "3px 9px" : full ? "8px" : "5px 14px",
        borderRadius: "var(--radius-sm)",
        cursor: disabled ? "default" : "pointer",
        fontFamily: "var(--font-sans)",
        width: full ? "100%" : undefined,
        whiteSpace: "nowrap",
        opacity: disabled ? 0.5 : 1,
        transition: "var(--transition-fast)",
        flexShrink: 0,
        ...style,
      }}
    >
      {children}
    </button>
  );
}
