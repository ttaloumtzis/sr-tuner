import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

export type ToastKind = "success" | "warning" | "error" | "info";

export interface Toast {
  id: string;
  kind: ToastKind;
  message: string;
  durationMs: number;
}

interface ToastContextValue {
  show: (kind: ToastKind, message: string, durationMs?: number) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const KIND_COLOR: Record<ToastKind, string> = {
  success: "var(--green)",
  warning: "var(--amber)",
  error:   "var(--red)",
  info:    "var(--blue)",
};

const KIND_BG: Record<ToastKind, string> = {
  success: "var(--green-dim)",
  warning: "var(--amber-dim)",
  error:   "#3d1a1a",
  info:    "var(--blue-dim)",
};

const DEFAULT_DISMISS_MS = 3500;

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDismiss, toast.durationMs);
    return () => clearTimeout(t);
  }, [onDismiss, toast.durationMs]);

  return (
    <div
      role="alert"
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 10,
        background: KIND_BG[toast.kind],
        border: `1px solid ${KIND_COLOR[toast.kind]}44`,
        borderLeft: `3px solid ${KIND_COLOR[toast.kind]}`,
        borderRadius: "var(--radius-md)",
        padding: "10px 14px",
        minWidth: 260,
        maxWidth: 380,
        boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
        animation: "toast-in 0.18s ease",
      }}
    >
      <span
        style={{
          color: "var(--text)",
          fontSize: 12,
          lineHeight: 1.5,
          flex: 1,
          fontFamily: "var(--font-sans)",
        }}
      >
        {toast.message}
      </span>
      <button
        onClick={onDismiss}
        aria-label="Dismiss"
        style={{
          background: "none",
          border: "none",
          color: "var(--muted)",
          cursor: "pointer",
          fontSize: 16,
          lineHeight: 1,
          padding: 0,
          flexShrink: 0,
        }}
      >
        ×
      </button>
    </div>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const counterRef = useRef(0);

  const show = useCallback((kind: ToastKind, message: string, durationMs = DEFAULT_DISMISS_MS) => {
    const id = `toast-${++counterRef.current}`;
    setToasts((prev) => [...prev, { id, kind, message, durationMs }]);
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ show }}>
      <style>{`
        @keyframes toast-in {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
      {children}
      <div
        aria-live="polite"
        style={{
          position: "fixed",
          bottom: 16,
          right: 16,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          zIndex: 9999,
          pointerEvents: "none",
        }}
      >
        {toasts.map((t) => (
          <div key={t.id} style={{ pointerEvents: "auto" }}>
            <ToastItem toast={t} onDismiss={() => dismiss(t.id)} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
