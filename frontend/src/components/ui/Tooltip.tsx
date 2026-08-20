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

  const TOOLTIP_WIDTH = 280;
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
      className="tooltip-trigger"
    >
      {children ?? <IconInfo size={11} />}
      {open &&
        ReactDOM.createPortal(
          <div
            className="tooltip-bubble"
            style={{
              top: pos.top,
              left: pos.left,
              transform: "translate(-50%, -100%)",
              maxWidth: TOOLTIP_WIDTH,
            }}
          >
            {text}
          </div>,
          document.body
        )}
    </span>
  );
}