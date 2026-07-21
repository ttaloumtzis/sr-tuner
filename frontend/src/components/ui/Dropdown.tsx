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

  const inputBase: React.CSSProperties = {
    background: "var(--bg3)",
    border: `1px solid ${open ? "var(--green)" : "var(--border)"}`,
    borderRadius: "var(--radius-sm)",
    padding: "5px 8px",
    fontSize: 12,
    color: selected ? "var(--text)" : "var(--dim)",
    width: "100%",
    outline: "none",
    transition: "border-color 0.15s",
    fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
    cursor: "pointer",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    userSelect: "none",
  };

  const menu = open
    ? ReactDOM.createPortal(
        <div
          style={{
            ...menuStyle,
            background: "var(--bg2)",
            border: "1px solid var(--border2)",
            borderRadius: "var(--radius-sm)",
            boxShadow: "0 8px 28px rgba(0,0,0,0.5)",
            maxHeight: 200,
            overflowY: "auto",
          }}
        >
          {resolved.map((opt) => {
            const active = opt.value === value;
            return (
              <div
                key={opt.value}
                onClick={() => select(opt.value)}
                style={{
                  padding: "6px 9px",
                  fontSize: 12,
                  cursor: "pointer",
                  color: active ? "var(--green)" : "var(--text)",
                  background: active ? "var(--green-dim)" : "transparent",
                  transition: "background 0.1s",
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
                  <span style={{ marginRight: 6, fontSize: 10 }}>✓</span>
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
      <div ref={triggerRef} style={inputBase} onClick={handleOpen}>
        <span
          style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            flex: 1,
          }}
        >
          {selected?.label ?? placeholder}
        </span>
        <span
          style={{
            color: "var(--muted)",
            marginLeft: 8,
            flexShrink: 0,
            display: "inline-block",
            transition: "transform 0.15s",
            transform: open ? "rotate(180deg)" : "none",
            fontSize: 12,
          }}
        >
          ▾
        </span>
      </div>
      {menu}
    </>
  );
}
