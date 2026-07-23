import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useUiStore } from "../../store/uiStore";
import type { SystemInfo } from "../../lib/api-types";
import { SetupStepInstall } from "./SetupStepInstall";
import { SetupStepDone } from "./SetupStepDone";
import { SetupWizardLinux } from "./SetupWizardLinux";
import { SetupWizardMacos } from "./SetupWizardMacos";
import { SetupWizardWindows } from "./SetupWizardWindows";

export function SetupWizard({ onComplete }: { onComplete: () => void }) {
  const {
    systemInfo, wizardStep, showWizard,
    setWizardStep, setSystemInfo, setInstallError,
    setInstallationDone, resetWizard,
  } = useUiStore();

  // Step 0: probe system on mount
  const [probing, setProbing] = useState(true);
  const [probeError, setProbeError] = useState<string | null>(null);

  useEffect(() => {
    if (!showWizard) return;
    setProbing(true);
    setProbeError(null);
    invoke<SystemInfo>("probe_system")
      .then((info) => {
        setSystemInfo(info);
        setWizardStep(1);
      })
      .catch((err) => {
        setProbeError(String(err));
      })
      .finally(() => setProbing(false));
  }, [showWizard, setSystemInfo, setWizardStep]);

  if (!showWizard || !systemInfo) {
    if (probing) {
      return (
        <div style={containerStyle}>
          <div style={cardStyle}>
            <div style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)", textAlign: "center" }}>
              {probeError ? `Probe failed: ${probeError}` : "Detecting system..."}
            </div>
          </div>
        </div>
      );
    }
    return null;
  }

  const handleInstallStart = async (backend: string, envType: "venv" | "sidecar") => {
    setWizardStep(4);
    setInstallError(null);
    setInstallationDone(false);
    try {
      await invoke("install_env", { backend, envType });
    } catch (err) {
      setInstallError(String(err));
    }
  };

  const handleCancel = () => {
    invoke("cancel_install").catch(() => {});
    setWizardStep(2);
  };

  const handleLaunch = () => {
    resetWizard();
    onComplete();
  };

  const wizardProps = {
    systemInfo,
    onStart: handleInstallStart,
    onBack: () => setWizardStep(Math.max(1, wizardStep - 1)),
    onNext: () => setWizardStep(wizardStep + 1),
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <div style={{ fontSize: 16, fontWeight: 700, color: "var(--green)", fontFamily: "var(--font-mono)", letterSpacing: "0.04em", marginBottom: 4 }}>
          SR TUNER SETUP
        </div>
        <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 20 }}>
          {wizardStep === 1 ? "System Detection" :
           wizardStep === 2 ? "Select Backend" :
           wizardStep === 3 ? "Select Environment Type" :
           wizardStep === 4 ? "Installing..." :
           wizardStep === 5 ? "Complete" : ""}
        </div>

        {(wizardStep === 1 || wizardStep === 2 || wizardStep === 3) && systemInfo.os === "linux" && (
          <SetupWizardLinux step={wizardStep} {...wizardProps} />
        )}
        {(wizardStep === 1 || wizardStep === 2) && systemInfo.os === "macos" && (
          <SetupWizardMacos step={wizardStep} {...wizardProps} />
        )}
        {(wizardStep === 1 || wizardStep === 2) && systemInfo.os === "windows" && (
          <SetupWizardWindows step={wizardStep} {...wizardProps} />
        )}

        {wizardStep === 4 && <SetupStepInstall onCancel={handleCancel} onProceed={handleLaunch} />}
        {wizardStep === 5 && <SetupStepDone onLaunch={handleLaunch} />}
      </div>
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  width: "100%",
  height: "100%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "var(--bg0)",
};

const cardStyle: React.CSSProperties = {
  width: "100%",
  maxWidth: 520,
  background: "var(--bg1)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-lg)",
  padding: "28px 32px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
};
