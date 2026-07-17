import { useState } from "react";
import { ToastProvider } from "./components/shell/ToastProvider";
import { TitleBar } from "./components/shell/TitleBar";
import { TabBar } from "./components/shell/TabBar";
import { StatusBar } from "./components/shell/StatusBar";
import { SidecarExitDialog } from "./components/shell/SidecarExitDialog";
import { CrashRecoveryDialog } from "./components/shell/CrashRecoveryDialog";
import { ErrorRouter } from "./components/shell/ErrorRouter";
import { useProjectStore } from "./store/projectStore";
import { useUiStore } from "./store/uiStore";
import { useSidecarLifecycle } from "./hooks/useSidecarLifecycle";
import { useProjectRestoration } from "./hooks/useProjectRestoration";
import { ProjectScreen } from "./screens/ProjectScreen";
import { ScreenDatasetSetup } from "./screens/dataset/ScreenDatasetSetup";
import { ScreenModelConfig } from "./screens/model/ScreenModelConfig";
import { ScreenTrainingSetup } from "./screens/training/ScreenTrainingSetup";
import { ScreenMetrics } from "./screens/metrics/ScreenMetrics";
import { ScreenCheckpoints } from "./screens/checkpoints/ScreenCheckpoints";
import { ScreenInference } from "./screens/inference/ScreenInference";
import { ScreenOnboarding } from "./screens/onboarding/ScreenOnboarding";

function TabContent() {
  const activeTab = useUiStore((s) => s.activeTab);

  return (
    <div
      style={{
        flex: 1,
        overflow: "auto",
        background: "var(--bg0)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {activeTab === "dataset" && <ScreenDatasetSetup />}
      {activeTab === "model" && <ScreenModelConfig />}
      {activeTab === "training" && <ScreenTrainingSetup />}
      {activeTab === "metrics" && <ScreenMetrics />}
      {activeTab === "checkpoints" && <ScreenCheckpoints />}
      {activeTab === "inference" && <ScreenInference />}
      {activeTab !== "dataset" &&
        activeTab !== "model" &&
        activeTab !== "training" &&
        activeTab !== "metrics" &&
        activeTab !== "checkpoints" &&
        activeTab !== "inference" && (
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--dim)",
              fontSize: 12,
              fontFamily: "var(--font-mono)",
            }}
          >
            {activeTab}
          </div>
        )}
    </div>
  );
}

// §19.2, 19.3, 19.4 — Lifecycle effects + exit dialog live inside the project layout
// so the sidecar is already running when these hooks activate.
// §21 — Project restoration runs once per project open inside the same layout.
function ProjectLayout() {
  const { exitDialog, closeSidecarExitDialog } = useSidecarLifecycle();
  const { crashRecovery, handleResumeCrashedRun, handleAbandonCrashedRun } =
    useProjectRestoration();

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100%",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <TitleBar />
      <TabBar />
      <TabContent />
      <StatusBar />

      {/* §19.3 — Sidecar unexpected exit dialog */}
      <SidecarExitDialog
        open={exitDialog.open}
        lastEpoch={exitDialog.lastEpoch}
        runId={exitDialog.runId}
        onClose={closeSidecarExitDialog}
      />

      {/* §21.5 — Crash recovery dialog (project reopen with in-progress run) */}
      <CrashRecoveryDialog
        open={crashRecovery.open}
        runName={crashRecovery.runName}
        lastEpoch={crashRecovery.lastEpoch}
        lastCheckpointPath={crashRecovery.lastCheckpointPath}
        onResume={handleResumeCrashedRun}
        onAbandon={handleAbandonCrashedRun}
      />

      {/* §20.9 — Global IPC error router */}
      <ErrorRouter />
    </div>
  );
}

function LandingLayout() {
  return <ProjectScreen />;
}

export default function App() {
  const project = useProjectStore((s) => s.project);
  const [onboardingDone, setOnboardingDone] = useState(false);

  return (
    <ToastProvider>
      {!onboardingDone ? (
        <ScreenOnboarding onComplete={() => setOnboardingDone(true)} />
      ) : project ? (
        <ProjectLayout />
      ) : (
        <LandingLayout />
      )}
    </ToastProvider>
  );
}
