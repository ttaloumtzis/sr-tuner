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
      className={`toggle ${on ? "toggle-on" : "toggle-off"}`}
      style={disabled ? { cursor: "default", opacity: 0.5 } : undefined}
    >
      <div className={`toggle-knob ${on ? "toggle-knob-on" : "toggle-knob-off"}`} />
    </div>
  );
}