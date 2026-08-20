import { CSSProperties } from "react";

interface TextInputProps {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
  title?: string;
  mono?: boolean;
  compact?: boolean;
  style?: CSSProperties;
}

export function TextInput({
  value,
  onChange,
  placeholder,
  disabled,
  title,
  mono,
  compact,
  style,
}: TextInputProps) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      title={title}
      onChange={(e) => onChange(e.target.value)}
      className="input"
      style={{
        fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
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