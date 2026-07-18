import { describe, it, expect, beforeEach } from "vitest";
import { useDatasetStore } from "../datasetStore";
import { resetAllStores } from "../../test-utils/resetStores";

describe("datasetStore", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("setStrategy updates validation strategy", () => {
    useDatasetStore.getState().setStrategy("separate_folder");
    expect(useDatasetStore.getState().strategy).toBe("separate_folder");
  });

  it('strategy "none" does not auto-clear validationPath but path can be set to null', () => {
    useDatasetStore.getState().setStrategy("none");
    useDatasetStore.getState().setValidationPath(null);

    const state = useDatasetStore.getState();
    expect(state.strategy).toBe("none");
    expect(state.validationPath).toBeNull();
  });

  it('strategy "none" with validationPath null is valid initial state', () => {
    useDatasetStore.setState({ strategy: "none", validationPath: null });
    const state = useDatasetStore.getState();
    expect(state.strategy).toBe("none");
    expect(state.validationPath).toBeNull();
  });

  it("setKernel updates kernel field", () => {
    useDatasetStore.getState().setKernel("bilinear");
    expect(useDatasetStore.getState().kernel).toBe("bilinear");
  });

  it("setHrPath updates hrPath", () => {
    useDatasetStore.getState().setHrPath("/data/hr");
    expect(useDatasetStore.getState().hrPath).toBe("/data/hr");
  });

  it("setValidationPath updates validationPath", () => {
    useDatasetStore.getState().setValidationPath("/data/val");
    expect(useDatasetStore.getState().validationPath).toBe("/data/val");
  });

  it("setValidationPath can be set to null", () => {
    useDatasetStore.getState().setValidationPath("/data/val");
    useDatasetStore.getState().setValidationPath(null);
    expect(useDatasetStore.getState().validationPath).toBeNull();
  });

  it("setValidationSplitRatio updates validationSplitRatio", () => {
    useDatasetStore.getState().setValidationSplitRatio(0.2);
    expect(useDatasetStore.getState().validationSplitRatio).toBeCloseTo(0.2);
  });

  it("initial state has auto_split strategy", () => {
    expect(useDatasetStore.getState().strategy).toBe("auto_split");
  });

  it("initial state has validationSplitRatio of 0.1", () => {
    expect(useDatasetStore.getState().validationSplitRatio).toBe(0.1);
  });
});
