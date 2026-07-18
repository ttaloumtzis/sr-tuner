import { useEffect } from "react";
import { ToastProvider } from "./components/shell/ToastProvider";
import { TitleBar } from "./components/shell/TitleBar";
import { LandingTitleBar } from "./components/shell/LandingTitleBar";
import { TabBar } from "./components/shell/TabBar";
import { StatusBar } from "./components/shell/StatusBar";
import { ErrorRouter } from "./components/shell/ErrorRouter";
import { ConnectionErrorDialog } from "./components/shell/ConnectionErrorDialog";
import { useProjectStore } from "./store/projectStore";
import { useUiStore } from "./store/uiStore";
import { useSSEConnection } from "./hooks/useSSEConnection";
import { initApiUrl } from "./lib/api";
import { ProjectScreen } from "./screens/ProjectScreen";
import { ScreenDatasetSetup } from "./screens/dataset/ScreenDatasetSetup";
import { ScreenModelConfig } from "./screens/model/ScreenModelConfig";
import { ScreenTrainingSetup } from "./screens/training/ScreenTrainingSetup";
import { ScreenMetrics } from "./screens/metrics/ScreenMetrics";
import { ScreenCheckpoints } from "./screens/checkpoints/ScreenCheckpoints";
import { ScreenInference } from "./screens/inference/ScreenInference";

function TabContent() {
  const activeTab = useUiStore((s) => s.activeTab);

  return (
    <div style={{ flex: 1, overflow: "auto", background: "var(--bg0)", display: "flex", flexDirection: "column" }}>
      {activeTab === "dataset" && <ScreenDatasetSetup />}
      {activeTab === "model" && <ScreenModelConfig />}
      {activeTab === "training" && <ScreenTrainingSetup />}
      {activeTab === "metrics" && <ScreenMetrics />}
      {activeTab === "checkpoints" && <ScreenCheckpoints />}
      {activeTab === "inference" && <ScreenInference />}
      {activeTab !== "dataset" && activeTab !== "model" && activeTab !== "training" && activeTab !== "metrics" && activeTab !== "checkpoints" && activeTab !== "inference" && (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--dim)", fontSize: 12, fontFamily: "var(--font-mono)" }}>
          {activeTab}
        </div>
      )}
    </div>
  );
}

function ProjectLayout() {
  const { isConnected: _, showDialog, retry } = useSSEConnection();

  const handleExit = () => {
    try {
      // @ts-ignore
      window.__TAURI__?.invoke("stop_python_server");
    } catch { /* browser mode */ }
    window.close();
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%", overflow: "hidden" }}>
      <TitleBar />
      <TabBar />
      <TabContent />
      <StatusBar />
      <ConnectionErrorDialog open={showDialog} onRetry={retry} onExit={handleExit} />
      <ErrorRouter />
    </div>
  );
}

function LandingLayout() {
  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%", overflow: "hidden" }}>
      <LandingTitleBar />
      <ProjectScreen />
    </div>
  );
}

export default function App() {
  const project = useProjectStore((s) => s.project);

  useEffect(() => {
    initApiUrl();
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("start_python_server");
      } catch {
        // running in browser or server already running
      }
    })();
  }, []);

  return (
    <ToastProvider>
      {project ? <ProjectLayout /> : <LandingLayout />}
    </ToastProvider>
  );
}