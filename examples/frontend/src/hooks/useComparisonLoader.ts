// §22.2 — On comparison panel open, send run.history.request for each selected
// run whose history is not already cached.  Charts must wait until all
// responses are received before rendering.

import { useEffect } from "react";
import { useUiStore } from "../store/uiStore";
import { useTrainingStore } from "../store/trainingStore";
import { useProjectStore } from "../store/projectStore";
import { sendToSidecar } from "../lib/ipc";

/**
 * Call inside the RunComparisonPanel (or any component that owns the
 * comparison view).  Returns `{ allLoaded }` — charts should only render
 * when this is true.
 */
export function useComparisonLoader(): { allLoaded: boolean } {
  const comparisonRunIds   = useUiStore((s) => s.comparisonRunIds);
  const pending            = useUiStore((s) => s.comparisonHistoriesPending);
  const markPending        = useUiStore((s) => s.markComparisonHistoryPending);
  const runHistories       = useTrainingStore((s) => s.runHistories);
  const project            = useProjectStore((s) => s.project);

  useEffect(() => {
    if (!project || comparisonRunIds.length === 0) return;

    for (const runId of comparisonRunIds) {
      // Skip runs whose history is already cached or in-flight.
      if (runHistories[runId] !== undefined || pending.has(runId)) continue;

      const run = project.runs.find((r) => r.run_id === runId);
      if (!run) continue;

      markPending(runId);
      sendToSidecar({
        type: "run.history.request",
        run_id: runId,
        log_dir: run.paths.log_dir,
      }).catch(() => {
        // If the send fails the panel will stay in loading state.
        // The user can close and re-open to retry.
      });
    }
  // Re-run whenever the set of selected runs changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comparisonRunIds, project]);

  const allLoaded =
    comparisonRunIds.length > 0 &&
    comparisonRunIds.every((id) => runHistories[id] !== undefined);

  return { allLoaded };
}
