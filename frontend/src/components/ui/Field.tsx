import { ReactNode } from "react";

interface FieldProps {
  label: ReactNode;
  hint?: string;
  children: ReactNode;
}

export function Field({ label, hint, children }: FieldProps) {
  return (
    <div className="field">
      <div className="field-label">
        <label className="field-label-text">
          {label}
        </label>
        {hint && (
          <span className="field-hint">
            {hint}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}