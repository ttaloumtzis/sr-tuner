import { useModelStore } from "../../store/modelStore";
import { ScreenModelCreate } from "./ScreenModelCreate";
import { ScreenModelView } from "./ScreenModelView";

function SubTabPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} role="tab" aria-selected={active}
      style={{
        background: active ? "var(--green)" : "var(--bg3)",
        border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
        color: active ? "#0d0f11" : "var(--muted)",
        fontSize: 11, fontWeight: active ? 600 : 400,
        padding: "4px 16px", borderRadius: 12,
        cursor: "pointer", transition: "var(--transition-fast)",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}

export function ScreenModelConfig() {
  const subTab = useModelStore((s) => s.subTab);
  const setSubTab = useModelStore((s) => s.setSubTab);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ display: "flex", gap: 6, padding: "8px 16px 0", flexShrink: 0 }}>
        <SubTabPill label="Create Model" active={subTab === "create"} onClick={() => setSubTab("create")} />
        <SubTabPill label="Model View" active={subTab === "view"} onClick={() => setSubTab("view")} />
      </div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, padding: "16px" }}>
        {subTab === "create" && <ScreenModelCreate />}
        {subTab === "view" && <ScreenModelView />}
      </div>
    </div>
  );
}
