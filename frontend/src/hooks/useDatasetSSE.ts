import { useEffect, useRef } from "react";
import { useDatasetStore } from "../store/datasetStore";
import { getBaseUrl, getJobStatus } from "../lib/api";
import type { HealthReport } from "../lib/api-types";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;

export function useDatasetSSE() {
  const jobId = useDatasetStore((s) => s.jobId);
  const setJobStatus = useDatasetStore((s) => s.setJobStatus);
  const setJobError = useDatasetStore((s) => s.setJobError);
  const startProgressStep = useDatasetStore((s) => s.startProgressStep);
  const updateProgressStep = useDatasetStore((s) => s.updateProgressStep);
  const finishProgressStep = useDatasetStore((s) => s.finishProgressStep);
  const setMergeResults = useDatasetStore((s) => s.setMergeResults);
  const setValidationResult = useDatasetStore((s) => s.setValidationResult);
  const setJobHealthReport = useDatasetStore((s) => s.setJobHealthReport);
  const esRef = useRef<EventSource | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Poll the backend job registry for the terminal state of a job whose
   * SSE stream is no longer delivering events (dropped connection, page
   * refresh, subtab switch). Returns true if a terminal state was applied.
   */
  const syncFromServer = async (id: string): Promise<boolean> => {
    try {
      const job = await getJobStatus(id);
      const state = useDatasetStore.getState();
      if (state.jobId !== id) return true; // stale job — stop reconnecting
      if (job.status === "completed") {
        setJobStatus("done");
        if (job.result?.results) {
          setMergeResults(job.result.results as { scale: number; output_path: string; source_datasets: string[] }[]);
        }
        if (job.result?.validation) {
          setValidationResult(job.result.validation as { valid: boolean; problems: string[]; num_pairs: number });
        }
        return true;
      }
      if (job.status === "failed" || job.status === "cancelled") {
        setJobStatus("error");
        setJobError(job.error || "Job failed");
        return true;
      }
      return false; // still pending/running — keep retrying
    } catch {
      return false; // server unreachable or job gone — retry the stream
    }
  };

  useEffect(() => {
    if (!jobId) {
      return;
    }

    // If the job already finished server-side (e.g. we re-entered this screen
    // after a subtab switch), apply the terminal state right away instead of
    // waiting for events that will never come.
    syncFromServer(jobId);

    const connect = () => {
      const baseUrl = getBaseUrl();
      const es = new EventSource(`${baseUrl}/api/events?job_id=${jobId}`);
      esRef.current = es;

      let stepId = 0;
      let stepStartTime = performance.now();
      let stepCurrent = 0;

      es.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data) as Record<string, unknown>;
          const type = event.type as string;

          switch (type) {
            case "progress_start": {
              const total = (event.total as number | null) ?? null;
              const desc = (event.desc as string) || "";
              startProgressStep(desc, total);
              stepId = useDatasetStore.getState().progressSteps.length - 1;
              stepStartTime = performance.now();
              stepCurrent = 0;
              setJobStatus("running");
              break;
            }
            case "progress_update": {
              const n = (event.n as number) ?? 1;
              stepCurrent += n;
              const elapsed = (performance.now() - stepStartTime) / 1000;
              const fps = elapsed > 0 ? stepCurrent / elapsed : 0;
              const total = useDatasetStore.getState().progressSteps[stepId]?.total ?? null;
              const etaSec = total != null && fps > 0 ? (total - stepCurrent) / fps : null;
              updateProgressStep(stepId, stepCurrent, fps, etaSec);
              break;
            }
            case "progress_end": {
              finishProgressStep(stepId);
              break;
            }
            case "done": {
              setJobStatus("done");
              reconnectAttempts.current = 0;
              const results = (event as Record<string, unknown>).results;
              if (results) setMergeResults(results as { scale: number; output_path: string; source_datasets: string[] }[]);
              const validation = (event as Record<string, unknown>).validation;
              if (validation) setValidationResult(validation as { valid: boolean; problems: string[]; num_pairs: number });
              const report = (event as Record<string, unknown>).report;
              if (report) setJobHealthReport(report as HealthReport);
              break;
            }
            case "error": {
              setJobStatus("error");
              setJobError((event.message as string) || "Unknown error");
              break;
            }
          }
        } catch {
          // ignore malformed events
        }
      };

      es.onerror = async () => {
        const state = useDatasetStore.getState();
        if (state.jobStatus === "done" || state.jobStatus === "error") return;
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
