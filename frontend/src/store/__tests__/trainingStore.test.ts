import { beforeEach, describe, expect, it } from "vitest";
import { useTrainingStore } from "../trainingStore";

describe("trainingStore stage lifecycle", () => {
  beforeEach(() => {
    useTrainingStore.getState().reset();
  });

  it("tracks the current run stage", () => {
    const s = useTrainingStore.getState();
    expect(s.stage).toBeNull();

    s.setStage("warmup");
    expect(useTrainingStore.getState().stage).toBe("warmup");

    useTrainingStore.getState().setStage("training");
    expect(useTrainingStore.getState().stage).toBe("training");
  });

  it("tracks the pre-training dataset-scan progress", () => {
    const s = useTrainingStore.getState();
    s.setStage("preparing");
    s.setPreparingProgress({ done: 0, total: 120 });
    s.setPreparingProgress({ done: 40, total: 120 });
    expect(useTrainingStore.getState().preparingProgress).toEqual({ done: 40, total: 120 });
  });

  it("clears stage and progress on reset", () => {
    const s = useTrainingStore.getState();
    s.setStage("saving");
    s.setPreparingProgress({ done: 120, total: 120 });
    s.reset();
    expect(useTrainingStore.getState().stage).toBeNull();
    expect(useTrainingStore.getState().preparingProgress).toBeNull();
  });
});
