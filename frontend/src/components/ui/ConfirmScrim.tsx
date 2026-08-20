import { useEffect } from "react";
import type { ReactNode } from "react";
import { Btn } from "./Btn";

interface ConfirmScrimProps {
  title: string;
  message: ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
  danger?: boolean;
  escToClose?: boolean;
  width?: number;
  maxWidth?: number;
  zIndex?: number;
}

export function ConfirmScrim({
  title,
  message,
  confirmLabel,
  onConfirm,
  onCancel,
  danger,
  escToClose,
  width,
  maxWidth,
  zIndex,
}: ConfirmScrimProps) {
  useEffect(() => {
    if (!escToClose) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [escToClose, onCancel]);

  return (
    <div className="scrim" style={zIndex != null ? { zIndex } : undefined}>
      <div className="scrim-box" style={{ width, maxWidth: maxWidth ?? undefined }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text)" }}>{title}</div>
        <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>{message}</div>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <Btn onClick={onCancel}>Cancel</Btn>
          <Btn variant="solid" color={danger ? "var(--red)" : undefined} onClick={onConfirm}>
            {confirmLabel}
          </Btn>
        </div>
      </div>
    </div>
  );
}