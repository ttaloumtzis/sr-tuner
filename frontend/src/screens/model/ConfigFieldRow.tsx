import { useId } from "react";
import { Tooltip } from "../../components/ui/Tooltip";
import type { ConfigField } from "../../lib/architectures";

interface ConfigFieldRowProps {
  field: ConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}

function FieldLabel({ text, hint }: { text: string; hint?: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>
      {text}
      {hint && <Tooltip text={hint} />}
    </span>
  );
}

export function ConfigFieldRow({ field, value, onChange }: ConfigFieldRowProps) {
  const sliderId = useId();

  const inputStyle = {
    background: "var(--bg3)", border: "1px solid var(--border)",
    borderRadius: "var(--radius-sm)", padding: "4px 8px", fontSize: 12, color: "var(--text)",
    outline: "none", fontFamily: "var(--font-mono)" as const,
    transition: "border-color 0.15s",
  };

  if (field.type === "text") {
    const textValue = String(value ?? "");
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <FieldLabel text={field.label} hint={field.hint} />
        <input
          type="text"
          value={textValue}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
          style={inputStyle}
          onFocus={(e) => (e.currentTarget.style.borderColor = "var(--green)")}
          onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
        />
      </label>
    );
  }
  if (field.type === "dropdown") {
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <FieldLabel text={field.label} hint={field.hint} />
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
                  padding: "4px 12px", borderRadius: "var(--radius-sm)",
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
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <FieldLabel text={field.label} hint={field.hint} />
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            id={sliderId}
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

export function CodeRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>{label}</span>
      <span style={{
        fontSize: 12, color: "var(--muted)", fontFamily: "var(--font-mono)",
        background: "var(--bg3)", border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)", padding: "4px 8px",
      }}>
        {value}
      </span>
    </div>
  );
}