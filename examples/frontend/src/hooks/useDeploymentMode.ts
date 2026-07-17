import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";

export type DeploymentMode = "bundled" | "dev";

export const ONBOARDING_HEADER: Record<DeploymentMode, string> = {
  bundled: "Checking system requirements",
  dev: "Checking development environment",
};

export function useDeploymentMode(): DeploymentMode | null {
  const [mode, setMode] = useState<DeploymentMode | null>(null);

  useEffect(() => {
    invoke<DeploymentMode>("get_deployment_mode").then(setMode);
  }, []);

  return mode;
}
