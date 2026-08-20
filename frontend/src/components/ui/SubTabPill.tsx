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
      className={`subtab-pill ${active ? "subtab-pill-active" : "subtab-pill-inactive"}`}
    >
      {label}
    </button>
  );
}