import { useState, useEffect, useCallback } from "react";
import { useUiStore } from "../store/uiStore";
import { useProjectStore } from "../store/projectStore";
import { getBaseUrl } from "../lib/api";

const MAX_RETRIES = 8;
const BASE_DELAY = 3000;

export function useSSEConnection() {
  const isConnected = useUiStore((s) => s.isServerConnected);
  const workspaceReady = useUiStore((s) => s.workspaceReady);
  const setConnected = useUiStore((s) => s.setServerConnected);
  const setWorkspaceReady = useUiStore((s) => s.setWorkspaceReady);
  const setWorkspaceError = useUiStore((s) => s.setWorkspaceError);
  const [retryCount, setRetryCount] = useState(0);
  const [showDialog, setShowDialog] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${getBaseUrl()}/api/health`);
      if (res.ok) {
        setConnected(true);
        setRetryCount(0);
        setShowDialog(false);

        // Check if workspace is still initialised on the backend
        const data = await res.json();
        const project = useProjectStore.getState().project;
        if (project && !data.workspace && workspaceReady) {
          const { parentFromProjFile } = await import("../lib/path");
          const projectDir = parentFromProjFile(project.filePath);
          try {
            const initRes = await fetch(`${getBaseUrl()}/api/workspace/init`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ path: projectDir }),
            });
            if (initRes.ok) {
              setWorkspaceReady(true);
              setWorkspaceError(null);
            } else {
              setWorkspaceReady(false);
              setWorkspaceError("Workspace re-initialisation failed");
            }
          } catch {
            setWorkspaceReady(false);
            setWorkspaceError("Connection lost during workspace re-initialisation");
          }
        }
        return true;
      }
    } catch {
      // server not reachable
    }
    return false;
  }, [setConnected, setWorkspaceReady, setWorkspaceError, workspaceReady]);

  useEffect(() => {
    if (isConnected) return;

    let checking = false;
    const interval = setInterval(async () => {
      if (checking) return;
      checking = true;
      const ok = await checkHealth();
      checking = false;
      if (!ok) {
        setRetryCount((r) => {
          const next = r + 1;
          if (next >= MAX_RETRIES) setShowDialog(true);
          return next;
        });
      }
    }, BASE_DELAY * Math.min(2 ** retryCount, 30000));

    return () => clearInterval(interval);
  }, [isConnected, retryCount, checkHealth]);

  return { isConnected, showDialog, retry: () => { setShowDialog(false); setRetryCount(0); checkHealth(); } };
}