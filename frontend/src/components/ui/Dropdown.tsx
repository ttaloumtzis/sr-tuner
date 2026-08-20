import { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom";

export interface DropdownOption {
  value: string;
  label: string;
}

interface DropdownProps {
  value?: string;
  defaultValue?: string;
  options: (string | DropdownOption)[];
  onChange?: (value: string) => void;
  placeholder?: string;
  mono?: boolean;
}

function resolveOption(opt: string | DropdownOption): DropdownOption {
  return typeof opt === "string" ? { value: opt, label: opt } : opt;
}

export function Dropdown({
  value: controlledValue,
  defaultValue,
  options,
  onChange,
  placeholder = "Select…",
  mono,
}: DropdownProps) {
  const [internalValue, setInternalValue] = useState(
    controlledValue ?? defaultValue ?? ""
  );
  const [open, setOpen] = useState(false);
  const [menuStyle, setMenuStyle] = useState<React.CSSProperties>({});
  const triggerRef = useRef<HTMLDivElement>(null);

  const value = controlledValue !== undefined ? controlledValue : internalValue;
  const resolved = options.map(resolveOption);
  const selected = resolved.find((o) => o.value === value);

  useEffect(() => {
    if (controlledValue !== undefined) setInternalValue(controlledValue);
  }, [controlledValue]);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (triggerRef.current && !triggerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("click", handleClick);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("click", handleClick);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const handleOpen = () => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setMenuStyle({
      position: "fixed",
      top: rect.bottom + 2,
      left: rect.left,
      width: rect.width,
      zIndex: 2000,
    });
    setOpen((o) => !o);
  };

  const select = (v: string) => {
    setInternalValue(v);
    setOpen(false);
    onChange?.(v);
  };

  const menu = open
    ? ReactDOM.createPortal(
        <div className="dropdown-menu" style={menuStyle}>
          {resolved.map((opt) => {
            const active = opt.value === value;
            return (
              <div
                key={opt.value}
                onClick={() => select(opt.value)}
                className={
                  active ? "dropdown-option dropdown-option-active" : "dropdown-option"
                }
                style={{
                  fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
                }}
                onMouseEnter={(e) => {
                  if (!active)
                    (e.currentTarget as HTMLDivElement).style.background =
                      "var(--bg3)";
                }}
                onMouseLeave={(e) => {
                  if (!active)
                    (e.currentTarget as HTMLDivElement).style.background =
                      "transparent";
                }}
              >
                {active && (
                  <span className="dropdown-check">✓</span>
                )}
                {opt.label}
              </div>
            );
          })}
        </div>,
        document.body
      )
    : null;

  return (
    <>
      <div
        ref={triggerRef}
        className={`dropdown${open ? " dropdown-open" : ""}`}
        onClick={handleOpen}
        style={{
          fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
          color: selected ? "var(--text)" : "var(--dim)",
        }}
      >
        <span className="dropdown-value">
          {selected?.label ?? placeholder}
        </span>
        <span
          className="dropdown-caret"
          style={{ transform: open ? "rotate(180deg)" : "none" }}
        >
          ▾
        </span>
      </div>
      {menu}
    </>
  );
}