import { create } from "zustand";
import type { SSEEvent, SystemInfo } from "../lib/api-types";

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

  // Wizard state
  showWizard: boolean;
  wizardStep: number;
  systemInfo: SystemInfo | null;
  selectedBackend: string;
  selectedEnvType: "venv" | "sidecar";
  installationLog: string[];
  installProgress: number;
  installError: string | null;
  installationDone: boolean;

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
  setShowWizard: (v: boolean) => void;
  setWizardStep: (v: number) => void;
  setSystemInfo: (v: SystemInfo | null) => void;
  setSelectedBackend: (v: string) => void;
  setSelectedEnvType: (v: "venv" | "sidecar") => void;
  appendInstallLog: (line: string) => void;
  setInstallProgress: (v: number) => void;
  setInstallError: (v: string | null) => void;
  setInstallationDone: (v: boolean) => void;
  resetWizard: () => void;
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

  showWizard: false,
  wizardStep: 0,
  systemInfo: null,
  selectedBackend: "",
  selectedEnvType: "venv",
  installationLog: [],
  installProgress: 0,
  installError: null,
  installationDone: false,

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

  setShowWizard: (v) => set({ showWizard: v }),
  setWizardStep: (v) => set({ wizardStep: v }),
  setSystemInfo: (v) => set({ systemInfo: v, selectedBackend: v?.default_backend ?? "" }),
  setSelectedBackend: (v) => set({ selectedBackend: v }),
  setSelectedEnvType: (v) => set({ selectedEnvType: v }),
  appendInstallLog: (line) =>
    set((s) => ({ installationLog: [...s.installationLog, line] })),
  setInstallProgress: (v) => set({ installProgress: v }),
  setInstallError: (v) => set({ installError: v }),
  setInstallationDone: (v) => set({ installationDone: v }),
  resetWizard: () =>
    set({
      showWizard: false,
      wizardStep: 0,
      systemInfo: null,
      selectedBackend: "",
      selectedEnvType: "venv",
      installationLog: [],
      installProgress: 0,
      installError: null,
      installationDone: false,
    }),
}));