interface ToggleProps {
  on: boolean;
  onChange: () => void;
  disabled?: boolean;
}

export function Toggle({ on, onChange, disabled }: ToggleProps) {
  return (
    <div
      role="switch"
      aria-checked={on}
      onClick={disabled ? undefined : onChange}
      style={{
        width: 32,
        height: 18,
        borderRadius: 10,
        position: "relative",
        cursor: disabled ? "default" : "pointer",
        flexShrink: 0,
        opacity: disabled ? 0.5 : 1,
        background: on ? "var(--green)" : "var(--bg3)",
        border: `1px solid ${on ? "var(--green)" : "var(--border)"}`,
        transition: "background 0.2s, border-color 0.2s",
      }}
    >
      <div
        style={{
          width: 12,
          height: 12,
          borderRadius: "50%",
          background: on ? "#0d0f11" : "var(--muted)",
          position: "absolute",
          top: 2,
          left: on ? 16 : 2,
          /* spring easing: overshoot then settle */
          transition:
            "left 0.2s cubic-bezier(0.34,1.56,0.64,1), background 0.2s",
          boxShadow: on ? "0 1px 3px rgba(0,0,0,0.4)" : "none",
        }}
      />
    </div>
  );
}
