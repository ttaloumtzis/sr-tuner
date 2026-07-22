interface SubTabPillProps {
  label: string;
  active: boolean;
  onClick: () => void;
}

/** Small pill-style tab used for sub-navigation within a screen (Create/View, Templates/Advanced, etc).
 *  Previously duplicated verbatim in ScreenModelConfig, ScreenModelCreate, and ScreenDatasetSetup. */
export function SubTabPill({ label, active, onClick }: SubTabPillProps) {
  return (
    <button
      onClick={onClick}
      role="tab"
      aria-selected={active}
      style={{
        background: active ? "var(--green)" : "var(--bg3)",
        border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
        color: active ? "#0d0f11" : "var(--muted)",
        fontSize: 11,
        fontWeight: active ? 600 : 400,
        padding: "4px 16px",
        borderRadius: 12,
        cursor: "pointer",
        transition: "var(--transition-fast)",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}
