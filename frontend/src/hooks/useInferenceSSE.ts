import { useEffect, useRef } from "react";
import { useInferenceStore } from "../store/inferenceStore";
import { getBaseUrl, getJobStatus } from "../lib/api";
import type { InferenceResult } from "../lib/api-types";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;

/**
 * Subscribes to `/api/events?job_id=<id>` for an inference job.
 *
 * Handles tile progress events, the rich `done` result payload, and error
 * events. On a dropped connection it falls back to `getJobStatus` to apply
 * the terminal state so a refresh/subtab-switch can't strand the UI.
 */
export function useInferenceSSE() {
  const jobId = useInferenceStore((s) => s.activeJobId);
  const esRef = useRef<EventSource | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const syncFromServer = async (id: string): Promise<boolean> => {
    try {
      const job = await getJobStatus(id);
      const state = useInferenceStore.getState();
      if (state.activeJobId !== id) return true; // stale job — stop reconnecting
      if (job.status === "completed") {
        const payload = (job.result ?? {}) as Record<string, unknown>;
        useInferenceStore.getState().setResult(payload as unknown as InferenceResult);
        useInferenceStore.getState().setStatus("done");
        return true;
      }
      if (job.status === "failed" || job.status === "cancelled") {
        useInferenceStore.getState().setErrorMsg(job.error || "Inference failed");
        useInferenceStore.getState().setStatus("error");
        return true;
      }
      return false; // still running — keep retrying
    } catch {
      return false;
    }
  };

  useEffect(() => {
    if (!jobId) return;

    // Terminal state may already exist server-side (subtab switch, refresh).
    syncFromServer(jobId);

    const connect = () => {
      const baseUrl = getBaseUrl();
      const es = new EventSource(`${baseUrl}/api/events?job_id=${jobId}`);
      esRef.current = es;

      es.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data) as Record<string, unknown>;
          const type = event.type as string;
          const store = useInferenceStore.getState();

          switch (type) {
            case "progress_start": {
              const total = (event.total as number | null) ?? null;
              store.setTileProgress(0, total ?? 0);
              store.setStatus("running");
              break;
            }
            case "progress_update": {
              const n = (event.n as number) ?? 1;
              const s = useInferenceStore.getState();
              const total = s.tilesTotal;
              store.setTileProgress(s.tilesDone + n, total);
              break;
            }
            case "progress_end": {
              store.setTileProgress(
                useInferenceStore.getState().tilesTotal,
                useInferenceStore.getState().tilesTotal,
              );
              break;
            }
            case "done": {
              store.setResult(event as unknown as InferenceResult);
              store.setStatus("done");
              reconnectAttempts.current = 0;
              break;
            }
            case "error": {
              store.setErrorMsg((event.message as string) || "Unknown inference error");
              store.setStatus("error");
              break;
            }
          }
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = async () => {
        const state = useInferenceStore.getState();
        if (state.status === "done" || state.status === "error") return;
        es.close();
        esRef.current = null;

        const terminal = await syncFromServer(jobId);
        if (terminal) return;

        const delay = Math.min(RECONNECT_BASE_MS * 2 ** reconnectAttempts.current, RECONNECT_MAX_MS);
        reconnectAttempts.current += 1;
        if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
        reconnectTimer.current = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
      reconnectAttempts.current = 0;
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    };
  }, [jobId]);

  return esRef;
}
