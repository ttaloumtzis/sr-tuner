import { describe, it, expect, beforeEach } from "vitest";
import { useUiStore } from "../uiStore";
import { resetAllStores } from "../../test-utils/resetStores";

describe("uiStore", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("setActiveTab updates activeTab", () => {
    useUiStore.getState().setActiveTab("metrics");
    expect(useUiStore.getState().activeTab).toBe("metrics");
  });

  it("setActiveTab cycles through all valid tabs", () => {
    const tabs = ["dataset", "model", "training", "metrics", "checkpoints", "history", "inference"] as const;
    for (const tab of tabs) {
      useUiStore.getState().setActiveTab(tab);
      expect(useUiStore.getState().activeTab).toBe(tab);
    }
  });

  it("setDisplayedRunId updates displayedRunId", () => {
    useUiStore.getState().setDisplayedRunId("run-xyz");
    expect(useUiStore.getState().displayedRunId).toBe("run-xyz");

    useUiStore.getState().setDisplayedRunId(null);
    expect(useUiStore.getState().displayedRunId).toBeNull();
  });

  it("setLastHeartbeat updates lastHeartbeat timestamp", () => {
    const ts = Date.now();
    useUiStore.getState().setLastHeartbeat(ts);
    expect(useUiStore.getState().lastHeartbeat).toBe(ts);
  });

  it("togglePanel inverts panel expanded state", () => {
    expect(useUiStore.getState().expandedPanels["panel-a"]).toBeUndefined();

    useUiStore.getState().togglePanel("panel-a");
    expect(useUiStore.getState().expandedPanels["panel-a"]).toBe(true);

    useUiStore.getState().togglePanel("panel-a");
    expect(useUiStore.getState().expandedPanels["panel-a"]).toBe(false);
  });

  it("markComparisonHistoryPending adds run id to pending set", () => {
    useUiStore.getState().markComparisonHistoryPending("run-1");
    expect(useUiStore.getState().comparisonHistoriesPending.has("run-1")).toBe(true);
  });

  it("markComparisonHistoryReceived removes run id from pending set", () => {
    useUiStore.getState().markComparisonHistoryPending("run-1");
    useUiStore.getState().markComparisonHistoryReceived("run-1");
    expect(useUiStore.getState().comparisonHistoriesPending.has("run-1")).toBe(false);
  });
});
