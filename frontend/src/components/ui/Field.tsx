import { ReactNode } from "react";

interface FieldProps {
  label: ReactNode;
  hint?: string;
  children: ReactNode;
}

export function Field({ label, hint, children }: FieldProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <label
          style={{
            fontSize: 9.5,
            color: "var(--muted)",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            fontFamily: "var(--font-sans)",
          }}
        >
          {label}
        </label>
        {hint && (
          <span
            style={{
              fontSize: 10,
              color: "var(--dim)",
              fontStyle: "italic",
            }}
          >
            {hint}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}
