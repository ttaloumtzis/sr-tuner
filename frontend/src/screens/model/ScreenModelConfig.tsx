import { useModelStore } from "../../store/modelStore";
import { SubTabPill } from "../../components/ui/SubTabPill";
import { ScreenModelCreate } from "./ScreenModelCreate";
import { ScreenModelView } from "./ScreenModelView";

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
