import { create } from "zustand";
import type { DeviceInfo, ErrorMessage } from "../lib/ipc-types";

export type TabId =
  | "dataset"
  | "model"
  | "training"
  | "metrics"
  | "checkpoints"
  | "history"
  | "inference";

interface UiState {
  activeTab: TabId;
  displayedRunId: string | null;
  lastHeartbeat: number;
  deviceName: string | null;
  detectedDevices: DeviceInfo[];
  lastIpcError: ErrorMessage | null;
  expandedPanels: Record<string, boolean>;
  toasts: { id: string; message: string; type: "info" | "success" | "error" }[];
  // §22.2 — run IDs currently selected for comparison; histories requested on panel open
  comparisonRunIds: string[];
  // §22.2 — run IDs whose run.history.request is in-flight (awaiting response)
  comparisonHistoriesPending: Set<string>;
  setActiveTab: (tab: TabId) => void;
  setDisplayedRunId: (id: string | null) => void;
  setLastHeartbeat: (ts: number) => void;
  setDeviceName: (name: string | null) => void;
  setDetectedDevices: (devices: DeviceInfo[]) => void;
  setLastIpcError: (err: ErrorMessage | null) => void;
  setExpandedPanels: (panels: Record<string, boolean>) => void;
  togglePanel: (panelId: string) => void;
  addToast: (message: string, type?: "info" | "success" | "error") => void;
  removeToast: (id: string) => void;
  setComparisonRunIds: (ids: string[]) => void;
  markComparisonHistoryPending: (runId: string) => void;
  markComparisonHistoryReceived: (runId: string) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeTab: "dataset",
  displayedRunId: null,
  lastHeartbeat: 0,
  deviceName: null,
  detectedDevices: [],
  lastIpcError: null,
  expandedPanels: {},
  toasts: [],
  comparisonRunIds: [],
  comparisonHistoriesPending: new Set(),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setDisplayedRunId: (id) => set({ displayedRunId: id }),
  setLastHeartbeat: (ts) => set({ lastHeartbeat: ts }),
  setDeviceName: (name) => set({ deviceName: name }),
  setDetectedDevices: (devices) => set({ detectedDevices: devices }),
  setLastIpcError: (err) => set({ lastIpcError: err }),
  setExpandedPanels: (panels) => set({ expandedPanels: panels }),
  togglePanel: (panelId) =>
    set((s) => ({
      expandedPanels: { ...s.expandedPanels, [panelId]: !s.expandedPanels[panelId] },
    })),
  addToast: (message, type = "info") =>
    set((s) => ({
      toasts: [...s.toasts, { id: crypto.randomUUID(), message, type }],
    })),
  removeToast: (id) =>
    set((s) => ({
      toasts: s.toasts.filter((t) => t.id !== id),
    })),
  setComparisonRunIds: (ids) => set({ comparisonRunIds: ids }),
  markComparisonHistoryPending: (runId) =>
    set((s) => ({
      comparisonHistoriesPending: new Set([...s.comparisonHistoriesPending, runId]),
    })),
  markComparisonHistoryReceived: (runId) =>
    set((s) => {
      const next = new Set(s.comparisonHistoriesPending);
      next.delete(runId);
      return { comparisonHistoriesPending: next };
    }),
}));
