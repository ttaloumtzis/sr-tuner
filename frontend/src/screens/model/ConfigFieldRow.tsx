import { useId } from "react";

export interface SliderField {
  type: "slider";
  kind: "int" | "float";
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface DropdownField {
  type: "dropdown";
  key: string;
  label: string;
  options: (number | string)[];
  default: number | string;
}

export interface TextField {
  type: "text";
  key: string;
  label: string;
  default?: unknown;
}

export type ConfigField = SliderField | DropdownField | TextField;

interface ConfigFieldRowProps {
  field: ConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}

export function ConfigFieldRow({ field, value, onChange }: ConfigFieldRowProps) {
  if (field.type === "text") {
    const textValue = String(value ?? "");
    const placeholder =
      field.key === "rgb_mean"
        ? "0.4488, 0.4371, 0.4040"
        : field.key === "depths"
          ? "6,6,6,6,6,6"
          : "6,6,6,6,6,6";
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>{field.label}</span>
        <input
          type="text"
          value={textValue}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          style={{
            background: "var(--bg2)", border: "1px solid var(--border)",
            borderRadius: 6, padding: "4px 8px", fontSize: 12, color: "var(--text)",
            outline: "none", fontFamily: "var(--font-mono)",
          }}
        />
      </label>
    );
  }
  if (field.type === "dropdown") {
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>{field.label}</span>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {field.options.map((opt) => {
            const active = value === opt;
            return (
              <button
                key={String(opt)}
                onClick={() => onChange(opt)}
                style={{
                  background: active ? "var(--green)" : "var(--bg3)",
                  border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
                  color: active ? "#0d0f11" : "var(--muted)",
                  fontSize: 11, fontWeight: active ? 600 : 400,
                  padding: "4px 12px", borderRadius: 8,
                  cursor: "pointer", transition: "var(--transition-fast)",
                  whiteSpace: "nowrap",
                }}
              >
                {String(opt)}
              </button>
            );
          })}
        </div>
      </label>
    );
  }
  if (field.type === "slider") {
    const id = useId();
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>{field.label}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            id={id}
            type="range"
            min={field.min}
            max={field.max}
            step={field.step}
            value={Number(value)}
            onChange={(e) => {
              const raw = parseFloat(e.target.value);
              onChange(field.kind === "int" ? Math.round(raw) : raw);
            }}
            style={{ flex: 1, accentColor: "var(--green)", height: 4, cursor: "pointer" }}
          />
          <span style={{
            fontSize: 12, fontWeight: 600, color: "var(--text)",
            minWidth: 32, textAlign: "right", fontFamily: "var(--font-mono)",
          }}>
            {field.kind === "int" ? Number(value).toFixed(0) : Number(value).toFixed(1)}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--muted)" }}>
          <span>{field.min}</span>
          <span>{field.max}</span>
        </div>
      </label>
    );
  }
  return null;
}

export function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>{label}</span>
      <span style={{
        fontSize: 12, color: "var(--muted)", fontFamily: "var(--font-mono)",
        background: "var(--bg2)", border: "1px solid var(--border)",
        borderRadius: 6, padding: "4px 8px",
      }}>
        {value}
      </span>
    </div>
  );
}
