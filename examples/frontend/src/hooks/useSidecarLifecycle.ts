// §19.2 — Frozen-sidecar detection: reset timer on heartbeat; at 2× interval → disconnected
// §19.3 — Show error dialog when sidecar exits unexpectedly mid-training
// §19.4 — Cleanup on app close: send training.stop, await checkpoint, then terminate

import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useTrainingStore } from "../store/trainingStore";
import { useUiStore } from "../store/uiStore";
import { sendToSidecar, killSidecar } from "../lib/ipc";

// Heartbeat intervals match sidecar emission schedule (§16.12)
const IDLE_HEARTBEAT_MS = 10_000;    // sidecar emits every 10 s when idle
const TRAINING_HEARTBEAT_MS = 30_000; // sidecar emits every 30 s during training
const TIMEOUT_MULTIPLIER = 2;         // "2× expected interval" per spec

export interface SidecarExitState {
  open: boolean;
  lastEpoch: number;
  runId: string | null;
}

export function useSidecarLifecycle() {
  const [exitDialog, setExitDialog] = useState<SidecarExitState>({
    open: false,
    lastEpoch: 0,
    runId: null,
  });

  // §19.2 — Frozen-sidecar detection
  useEffect(() => {
    const interval = setInterval(() => {
      const lastHeartbeat = useUiStore.getState().lastHeartbeat;
      if (lastHeartbeat === 0) return; // sidecar not yet started

      const { status } = useTrainingStore.getState();
      if (status === "disconnected" || status === "idle" || status === "done") return;

      const expectedMs =
        status === "running" || status === "paused"
          ? TRAINING_HEARTBEAT_MS
          : IDLE_HEARTBEAT_MS;
      const timeoutMs = expectedMs * TIMEOUT_MULTIPLIER;

      if (Date.now() - lastHeartbeat > timeoutMs) {
        useTrainingStore.setState({ status: "disconnected" });
      }
    }, 5_000);

    return () => clearInterval(interval);
  }, []);

  // §19.3 — Listen for unexpected sidecar process termination
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    listen<number | null>("sidecar-terminated", () => {
      const { status, epoch, activeTrainingRunId } = useTrainingStore.getState();
      if (status === "running" || status === "paused" || status === "disconnected") {
        setExitDialog({ open: true, lastEpoch: epoch, runId: activeTrainingRunId });
        useTrainingStore.setState({ status: "idle", activeTrainingRunId: null });
      }
    }).then((fn) => { unlisten = fn; });

    return () => { unlisten?.(); };
  }, []);

  // §19.4 — Cleanup on app close: stop training then terminate
  useEffect(() => {
    const win = getCurrentWindow();
    let unlisten: (() => void) | null = null;

    win.onCloseRequested(async (event) => {
      const { status, activeTrainingRunId } = useTrainingStore.getState();
      if (status !== "running" && status !== "paused") return;

      event.preventDefault();

      // Send training.stop with save_checkpoint: true (best-effort)
      try {
        if (activeTrainingRunId) {
          await sendToSidecar({
            type: "training.stop",
            run_id: activeTrainingRunId,
            save_checkpoint: true,
          });
        }
      } catch {
        // Best-effort; proceed to kill
      }

      // §19.4a [Gap K] — hard 5-second timeout: if checkpoint.saved is not
      // received in time, forcibly terminate rather than hanging indefinitely.
      const deadline = Date.now() + 5_000;
      const poll = setInterval(async () => {
        const s = useTrainingStore.getState().status;
        if (s === "idle" || s === "done" || Date.now() >= deadline) {
          clearInterval(poll);
          try { await killSidecar(); } catch { /* ignore */ }
          await win.destroy();
        }
      }, 500);
    }).then((fn) => { unlisten = fn; });

    return () => { unlisten?.(); };
  }, []);

  function closeSidecarExitDialog() {
    setExitDialog((prev) => ({ ...prev, open: false }));
  }

  return { exitDialog, closeSidecarExitDialog };
}
