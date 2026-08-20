import { CSSProperties } from "react";

interface BtnProps {
  children: React.ReactNode;
  variant?: "ghost" | "solid";
  color?: string;
  onClick?: () => void;
  small?: boolean;
  full?: boolean;
  centered?: boolean;
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
  centered,
  disabled,
  title,
  style,
  type = "button",
}: BtnProps) {
  const cls = [
    "btn",
    variant === "solid" ? "btn-solid" : color ? "btn-colored" : null,
    small && "btn-small",
    full && "btn-full",
    centered && "btn-center",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cls}
      style={
        {
          "--btn-color": color ?? "var(--green)",
          ...style,
        } as CSSProperties
      }
    >
      {children}
    </button>
  );
}