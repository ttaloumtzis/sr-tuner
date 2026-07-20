import { create } from "zustand";
import type { SSEEvent } from "../lib/api-types";

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
  isServerConnected: boolean;
  lastApiError: SSEEvent | null;
  expandedPanels: Record<string, boolean>;
  toasts: { id: string; message: string; type: "info" | "success" | "error" }[];
  comparisonRunIds: string[];
  workspaceReady: boolean;
  workspaceError: string | null;
  setActiveTab: (tab: TabId) => void;
  setDisplayedRunId: (id: string | null) => void;
  setServerConnected: (connected: boolean) => void;
  setLastApiError: (err: SSEEvent | null) => void;
  setExpandedPanels: (panels: Record<string, boolean>) => void;
  togglePanel: (panelId: string) => void;
  addToast: (message: string, type?: "info" | "success" | "error") => void;
  removeToast: (id: string) => void;
  setComparisonRunIds: (ids: string[]) => void;
  setWorkspaceReady: (ready: boolean) => void;
  setWorkspaceError: (error: string | null) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeTab: "dataset",
  displayedRunId: null,
  isServerConnected: false,
  lastApiError: null,
  expandedPanels: {},
  toasts: [],
  comparisonRunIds: [],
  workspaceReady: false,
  workspaceError: null,
  setActiveTab: (tab) => set({ activeTab: tab }),
  setDisplayedRunId: (id) => set({ displayedRunId: id }),
  setServerConnected: (connected) => set({ isServerConnected: connected }),
  setLastApiError: (err) => set({ lastApiError: err }),
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
  setWorkspaceReady: (ready) => set({ workspaceReady: ready }),
  setWorkspaceError: (error) => set({ workspaceError: error }),
}));