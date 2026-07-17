import { useEffect } from "react";
import { SRProjManager } from "./SRProjManager";
import { type TabId } from "./srproj";

export function useUiStatePersist(
  activeRunId: string | null,
  activeTab: TabId | null
) {
  useEffect(() => {
    SRProjManager.setActiveRun(activeRunId);
  }, [activeRunId]);

  useEffect(() => {
    SRProjManager.setActiveTab(activeTab);
  }, [activeTab]);
}
