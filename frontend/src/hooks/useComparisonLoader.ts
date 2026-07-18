// §22.2 — On comparison panel open, send run.history.request for each selected
// run whose history is not already cached.  Charts must wait until all
// responses are received before rendering.

import { useEffect } from "react";
import { useUiStore } from "../store/uiStore";
import { useTrainingStore } from "../store/trainingStore";
import { getJobStatus } from "../lib/api";

export function useComparisonLoader(): { allLoaded: boolean } {
  const comparisonRunIds   = useUiStore((s) => s.comparisonRunIds);
  const runHistories       = useTrainingStore((s) => s.runHistories);
  const addToast           = useUiStore((s) => s.addToast);

  useEffect(() => {
    if (comparisonRunIds.length === 0) return;

    for (const runId of comparisonRunIds) {
      if (runHistories[runId] !== undefined) continue;
      getJobStatus(runId).then((status) => {
        if (status.result) {
          useTrainingStore.setState((s) => ({
            runHistories: { ...s.runHistories, [runId]: { gLossHistory: [], psnrHistory: [], ssimHistory: [], dLossHistory: [], totalLossHistory: [] } },
          }));
        }
      }).catch(() => addToast("Failed to load run history", "error"));
    }
  }, [comparisonRunIds, runHistories, addToast]);

  const allLoaded = comparisonRunIds.length > 0 && comparisonRunIds.every((id) => runHistories[id] !== undefined);
  return { allLoaded };
}
