import { useState, useEffect, useCallback } from "react";
import { useUiStore } from "../store/uiStore";
import { getBaseUrl } from "../lib/api";

const MAX_RETRIES = 8;
const BASE_DELAY = 3000;

export function useSSEConnection() {
  const isConnected = useUiStore((s) => s.isServerConnected);
  const setConnected = useUiStore((s) => s.setServerConnected);
  const [retryCount, setRetryCount] = useState(0);
  const [showDialog, setShowDialog] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${getBaseUrl()}/api/health`);
      if (res.ok) {
        setConnected(true);
        setRetryCount(0);
        setShowDialog(false);
        return true;
      }
    } catch {
      // server not reachable
    }
    return false;
  }, [setConnected]);

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