import { useEffect, useRef } from "react";
import { useDatasetStore } from "../store/datasetStore";
import { getBaseUrl } from "../lib/api";

export function useDatasetSSE() {
  const jobId = useDatasetStore((s) => s.jobId);
  const setJobStatus = useDatasetStore((s) => s.setJobStatus);
  const setJobError = useDatasetStore((s) => s.setJobError);
  const startProgressStep = useDatasetStore((s) => s.startProgressStep);
  const updateProgressStep = useDatasetStore((s) => s.updateProgressStep);
  const finishProgressStep = useDatasetStore((s) => s.finishProgressStep);
  const clearJob = useDatasetStore((s) => s.clearJob);
  const setMergeResults = useDatasetStore((s) => s.setMergeResults);
  const setValidationResult = useDatasetStore((s) => s.setValidationResult);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId) {
      return;
    }

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
            const results = (event as Record<string, unknown>).results;
            if (results) setMergeResults(results as { scale: number; output_path: string; source_datasets: string[] }[]);
            const validation = (event as Record<string, unknown>).validation;
            if (validation) setValidationResult(validation as { valid: boolean; problems: string[]; num_pairs: number });
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

    es.onerror = () => {
      const state = useDatasetStore.getState();
      if (state.jobStatus === "done" || state.jobStatus === "error") return;
      if (es.readyState === EventSource.CLOSED) {
        clearJob();
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [jobId]);

  return esRef;
}
