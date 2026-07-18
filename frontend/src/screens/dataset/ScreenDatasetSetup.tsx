import { useDatasetStore, type DatasetSubTab } from "../../store/datasetStore";
import { useDatasetSSE } from "../../hooks/useDatasetSSE";
import { JobOverlay } from "../../components/dataset/JobOverlay";
import { ScreenDatasetCreate } from "./ScreenDatasetCreate";
import { ScreenBrowseDatasets } from "./ScreenBrowseDatasets";
import { ScreenMergeDatasets } from "./ScreenMergeDatasets";

function SubTabPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} role="tab" aria-selected={active}
      style={{ background: active ? "var(--green)" : "var(--bg3)", border: `1px solid ${active ? "var(--green)" : "var(--border)"}`, color: active ? "#0d0f11" : "var(--muted)", fontSize: 11, fontWeight: active ? 600 : 400, padding: "4px 16px", borderRadius: 12, cursor: "pointer", transition: "var(--transition-fast)", whiteSpace: "nowrap" }}>
      {label}
    </button>
  );
}

export function ScreenDatasetSetup() {
  const subTab = useDatasetStore((s) => s.subTab);
  const setSubTab = useDatasetStore((s) => s.setSubTab);
  useDatasetSSE();

  const tabs: { id: DatasetSubTab; label: string }[] = [
    { id: "create", label: "Create Dataset" },
    { id: "browse", label: "Browse Datasets" },
    { id: "merge", label: "Merge Datasets" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ display: "flex", gap: 6, padding: "8px 16px 0", flexShrink: 0 }}>
        {tabs.map((t) => (
          <SubTabPill key={t.id} label={t.label} active={subTab === t.id} onClick={() => setSubTab(t.id)} />
        ))}
      </div>
      <JobOverlay />
      <div style={{ flex: 1, overflow: "auto", padding: 16, position: "relative" }}>
        {subTab === "create" && <ScreenDatasetCreate />}
        {subTab === "browse" && <ScreenBrowseDatasets />}
        {subTab === "merge" && <ScreenMergeDatasets />}
      </div>
    </div>
  );
}