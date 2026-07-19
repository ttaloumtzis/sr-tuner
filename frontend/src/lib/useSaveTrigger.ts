import { useEffect, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { SRProjManager } from "./SRProjManager";
import { cancelJob } from "./api";
import { useTrainingStore } from "../store/trainingStore";

export function useSaveTrigger() {
  const [saving, setSaving] = useState(false);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerSave = async () => {
    if (saving) return;
    setSaving(true);
    try {
      await SRProjManager.save();
    } finally {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(() => setSaving(false), 2000);
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        triggerSave();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  // Wire app-close event to cancel training, save, then destroy
  useEffect(() => {
    const appWindow = getCurrentWindow();
    let unlisten: (() => void) | undefined;

    appWindow
      .onCloseRequested(async (evt) => {
        evt.preventDefault();

        const state = useTrainingStore.getState();
        if (state.activeTrainingRunId && state.status === "running") {
          try {
            await cancelJob(state.activeTrainingRunId);
          } catch {
            // backend may already be unreachable
          }
        }

        await SRProjManager.save();

        try {
          // @ts-ignore
          window.__TAURI__?.invoke("stop_python_server");
        } catch {
          // browser mode
        }

        appWindow.destroy();
      })
      .then((fn) => {
        unlisten = fn;
      });

    return () => {
      unlisten?.();
    };
  }, []);

  return { saving, triggerSave };
}
