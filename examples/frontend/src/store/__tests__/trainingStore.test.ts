import { describe, it, expect, beforeEach } from "vitest";
import { useTrainingStore } from "../trainingStore";
import { resetAllStores } from "../../test-utils/resetStores";

describe("trainingStore", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("setting status to running and activeTrainingRunId simulates startTraining", () => {
    useTrainingStore.setState({ status: "running", activeTrainingRunId: "run-abc" });

    const state = useTrainingStore.getState();
    expect(state.status).toBe("running");
    expect(state.activeTrainingRunId).toBe("run-abc");
  });

  it("appending to lossHistory does not mutate prior array", () => {
    const initial = [1.0, 0.9];
    useTrainingStore.setState({ lossHistory: initial });

    const before = useTrainingStore.getState().lossHistory;
    const snapshot = [...before];

    useTrainingStore.setState((s) => ({ lossHistory: [...s.lossHistory, 0.8] }));

    expect(before).toEqual(snapshot);
    expect(useTrainingStore.getState().lossHistory).toEqual([1.0, 0.9, 0.8]);
  });

  it("setting status to paused simulates pauseTraining", () => {
    useTrainingStore.setState({ status: "running" });
    useTrainingStore.setState({ status: "paused" });
    expect(useTrainingStore.getState().status).toBe("paused");
  });

  it("setting status to running from paused simulates resumeTraining", () => {
    useTrainingStore.setState({ status: "paused" });
    useTrainingStore.setState({ status: "running" });
    expect(useTrainingStore.getState().status).toBe("running");
  });

  it("clearing status and activeTrainingRunId simulates stopTraining", () => {
    useTrainingStore.setState({ status: "running", activeTrainingRunId: "run-abc" });
    useTrainingStore.setState({ status: "idle", activeTrainingRunId: null });

    const state = useTrainingStore.getState();
    expect(state.status).toBe("idle");
    expect(state.activeTrainingRunId).toBeNull();
  });

  it("runHistories accumulates per-run history without cross-run interference", () => {
    useTrainingStore.setState({
      runHistories: {
        "run-1": { gLossHistory: [1.0], dLossHistory: [null], totalLossHistory: [1.0], psnrHistory: [], ssimHistory: [] },
      },
    });

    useTrainingStore.setState((s) => ({
      runHistories: {
        ...s.runHistories,
        "run-2": { gLossHistory: [2.0], dLossHistory: [null], totalLossHistory: [2.0], psnrHistory: [], ssimHistory: [] },
      },
    }));

    const state = useTrainingStore.getState();
    expect(state.runHistories["run-1"].gLossHistory).toEqual([1.0]);
    expect(state.runHistories["run-2"].gLossHistory).toEqual([2.0]);
  });
});
