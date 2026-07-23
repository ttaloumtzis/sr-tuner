import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ToastProvider } from "./components/shell/ToastProvider";

import { TabBar } from "./components/shell/TabBar";
import { StatusBar } from "./components/shell/StatusBar";
import { ErrorRouter } from "./components/shell/ErrorRouter";
import { ConnectionErrorDialog } from "./components/shell/ConnectionErrorDialog";
import { useProjectStore } from "./store/projectStore";
import { useUiStore } from "./store/uiStore";
import { useSSEConnection } from "./hooks/useSSEConnection";
import { useTrainingSSE } from "./hooks/useTrainingSSE";
import { initApiUrl, initWorkspace } from "./lib/api";
import { ProjectScreen } from "./screens/ProjectScreen";
import { SetupWizard } from "./screens/setup/SetupWizard";
import { parentFromProjFile } from "./lib/path";
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
  useTrainingSSE();

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%", overflow: "hidden" }}>
      <TabBar />
      <TabContent />
      <StatusBar />
      <ErrorRouter />
    </div>
  );
}

function LoadingWorkspace({ name }: { name: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%", alignItems: "center", justifyContent: "center", background: "var(--bg0)", gap: 12 }}>
      <div style={{ color: "var(--dim)", fontSize: 11, fontFamily: "var(--font-mono)" }}>
        Loading project <span style={{ color: "var(--text)" }}>{name}</span>…
      </div>
    </div>
  );
}

function WorkspaceErrorView({ name, error, onRetry }: { name: string; error: string; onRetry: () => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%", alignItems: "center", justifyContent: "center", background: "var(--bg0)", gap: 12, padding: 24 }}>
      <div style={{ color: "var(--red)", fontSize: 12, fontFamily: "var(--font-mono)" }}>
        Failed to open workspace for {name}
      </div>
      <div style={{ color: "var(--muted)", fontSize: 10, maxWidth: 400, textAlign: "center", wordBreak: "break-word" }}>
        {error}
      </div>
      <button onClick={onRetry} style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text)", cursor: "pointer", fontSize: 11, padding: "4px 12px", fontFamily: "var(--font-mono)" }}>
        Retry
      </button>
    </div>
  );
}

function LandingLayout() {
  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100%", height: "100%", overflow: "hidden" }}>
      <ProjectScreen />
    </div>
  );
}

const WORKSPACE_MAX_RETRIES = 5;

export default function App() {
  const project = useProjectStore((s) => s.project);
  const isServerConnected = useUiStore((s) => s.isServerConnected);
  const workspaceReady = useUiStore((s) => s.workspaceReady);
  const workspaceError = useUiStore((s) => s.workspaceError);
  const showWizard = useUiStore((s) => s.showWizard);
  const setShowWizard = useUiStore((s) => s.setShowWizard);
  const setWorkspaceReady = useUiStore((s) => s.setWorkspaceReady);
  const setWorkspaceError = useUiStore((s) => s.setWorkspaceError);

  useEffect(() => {
    initApiUrl();
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const firstRun = await invoke<boolean>("check_first_run");
        if (firstRun) {
          setShowWizard(true);
          return; // wizard will start server after completion
        }
        await invoke("start_python_server");
      } catch (err) {
        console.error("Python server failed to start:", err);
      }
    })();
  }, [setShowWizard]);

  useEffect(() => {
    if (!project) {
      setWorkspaceReady(false);
      setWorkspaceError(null);
    }
  }, [project, setWorkspaceReady, setWorkspaceError]);

  useEffect(() => {
    if (!project || !isServerConnected || workspaceReady) return;

    const projectDir = parentFromProjFile(project.filePath);
    let cancelled = false;

    (async () => {
      for (let retries = 0; retries < WORKSPACE_MAX_RETRIES && !cancelled; retries++) {
        try {
          await initWorkspace(projectDir);
          if (!cancelled) {
            setWorkspaceReady(true);
            setWorkspaceError(null);
          }
          return;
        } catch (err) {
          if (cancelled) return;
          if (retries >= WORKSPACE_MAX_RETRIES - 1) {
            const msg = err instanceof Error ? err.message : String(err);
            setWorkspaceError(msg);
            console.warn("Workspace init failed after", WORKSPACE_MAX_RETRIES, "retries:", err);
            return;
          }
          await new Promise(r => setTimeout(r, 1000 * (retries + 1)));
        }
      }
    })();

    return () => { cancelled = true; };
  }, [project, isServerConnected, workspaceReady, setWorkspaceReady, setWorkspaceError]);

  const { showDialog, retry } = useSSEConnection();

  const handleExit = () => {
    try {
      // @ts-ignore
      window.__TAURI__?.invoke("stop_python_server");
    } catch { /* browser mode */ }
    window.close();
  };

  const handleRetryWorkspace = () => {
    setWorkspaceError(null);
    setWorkspaceReady(false);
  };

  const handleWizardComplete = () => {
    invoke("start_python_server").catch((err) =>
      console.error("Python server failed to start after wizard:", err)
    );
  };

  if (showWizard) {
    return (
      <ToastProvider>
        <SetupWizard onComplete={handleWizardComplete} />
      </ToastProvider>
    );
  }

  let content;
  if (project && !workspaceReady && !workspaceError) {
    content = <LoadingWorkspace name={project.name} />;
  } else if (project && workspaceError && !workspaceReady) {
    content = <WorkspaceErrorView name={project.name} error={workspaceError} onRetry={handleRetryWorkspace} />;
  } else if (project && workspaceReady) {
    content = <ProjectLayout />;
  } else {
    content = <LandingLayout />;
  }

  return (
    <ToastProvider>
      {content}
      <ConnectionErrorDialog open={showDialog} onRetry={retry} onExit={handleExit} />
    </ToastProvider>
  );
}