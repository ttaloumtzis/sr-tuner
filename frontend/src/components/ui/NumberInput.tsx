import { CSSProperties } from "react";

interface NumberInputProps {
  value: number | string;
  onChange: (v: number, raw?: string) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  title?: string;
  compact?: boolean;
  style?: CSSProperties;
  onBlur?: () => void;
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
  title,
  compact,
  style,
  onBlur,
}: NumberInputProps) {
  return (
    <input
      type="number"
      value={
        typeof value === "string"
          ? value
          : Number.isFinite(value)
          ? value
          : ""
      }
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      title={title}
      onBlur={onBlur}
      onChange={(e) => {
        const n = Number(e.target.value);
        if (min !== undefined && max !== undefined) {
          onChange(Number.isNaN(n) ? 0 : Math.max(min, Math.min(max, n)), e.target.value);
        } else {
          onChange(n, e.target.value);
        }
      }}
      className="input"
      style={{
        fontFamily: "var(--font-mono)",
        padding: compact ? "4px 7px" : "5px 8px",
        fontSize: compact ? 11.5 : 12,
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? "not-allowed" : undefined,
        boxSizing: "border-box",
        ...style,
      }}
    />
  );
}