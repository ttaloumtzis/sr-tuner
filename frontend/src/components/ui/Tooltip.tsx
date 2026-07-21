import { useState, useRef, ReactNode } from "react";
import ReactDOM from "react-dom";
import { IconInfo } from "./icons";

interface TooltipProps {
  text: ReactNode;
  children?: ReactNode;
}

/**
 * Small "(i)" affordance that reveals a hint on hover/focus.
 * Rendered through a portal so it never gets clipped by a Panel's overflow.
 */
export function Tooltip({ text, children }: TooltipProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const ref = useRef<HTMLSpanElement>(null);

  const TOOLTIP_WIDTH = 220;
  const show = () => {
    const rect = ref.current?.getBoundingClientRect();
    if (rect) {
      const margin = 8;
      let left = rect.left + rect.width / 2;
      const half = TOOLTIP_WIDTH / 2;
      left = Math.max(half + margin, Math.min(left, window.innerWidth - half - margin));
      setPos({ top: rect.top - 6, left });
    }
    setOpen(true);
  };

  return (
    <span
      ref={ref}
      onMouseEnter={show}
      onMouseLeave={() => setOpen(false)}
      style={{ display: "inline-flex", alignItems: "center", cursor: "help", color: "var(--dim)" }}
    >
      {children ?? <IconInfo size={11} />}
      {open &&
        ReactDOM.createPortal(
          <div
            style={{
              position: "fixed",
              top: pos.top,
              left: pos.left,
              transform: "translate(-50%, -100%)",
              background: "var(--bg2)",
              border: "1px solid var(--border2)",
              borderRadius: "var(--radius-sm)",
              padding: "6px 9px",
              fontSize: 10.5,
              lineHeight: 1.5,
              color: "var(--muted)",
              maxWidth: TOOLTIP_WIDTH,
              boxShadow: "0 8px 20px rgba(0,0,0,0.45)",
              zIndex: 3000,
              pointerEvents: "none",
            }}
          >
            {text}
          </div>,
          document.body
        )}
    </span>
  );
}
