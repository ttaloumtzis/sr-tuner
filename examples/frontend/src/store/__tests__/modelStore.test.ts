import { describe, it, expect, beforeEach } from "vitest";
import { useModelStore } from "../modelStore";
import { resetAllStores } from "../../test-utils/resetStores";

describe("modelStore", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("setArchitecture updates architecture", () => {
    useModelStore.getState().setArchitecture("SwinIR");
    expect(useModelStore.getState().architecture).toBe("SwinIR");
  });

  it("setArchitecture works for all valid architectures", () => {
    const archs = ["Real-ESRGAN", "SwinIR", "HAT", "EDSR"] as const;
    for (const arch of archs) {
      useModelStore.getState().setArchitecture(arch);
      expect(useModelStore.getState().architecture).toBe(arch);
    }
  });

  it("setPretrainedPath stores path", () => {
    useModelStore.getState().setPretrainedPath("/models/pretrained.pth");
    expect(useModelStore.getState().pretrainedPath).toBe("/models/pretrained.pth");
  });

  it("setPretrainedPath with null clears path", () => {
    useModelStore.getState().setPretrainedPath("/models/pretrained.pth");
    useModelStore.getState().setPretrainedPath(null);
    expect(useModelStore.getState().pretrainedPath).toBeNull();
  });

  it("setHyperparameters merges partial updates immutably", () => {
    const originalHp = useModelStore.getState().hyperparameters;
    useModelStore.getState().setHyperparameters({ batchSize: 8 });

    const updatedHp = useModelStore.getState().hyperparameters;
    expect(updatedHp.batchSize).toBe(8);
    expect(updatedHp.learningRate).toBe(originalHp.learningRate);
    expect(originalHp.batchSize).toBe(16);
  });

  it("setLossWeights merges partial updates", () => {
    useModelStore.getState().setLossWeights({ pixel: 2.0 });
    const lw = useModelStore.getState().lossWeights;
    expect(lw.pixel).toBe(2.0);
    expect(lw.perceptual).toBe(1.0);
  });

  it("setAugmentations merges partial updates", () => {
    useModelStore.getState().setAugmentations({ mixup: true });
    expect(useModelStore.getState().augmentations.mixup).toBe(true);
    expect(useModelStore.getState().augmentations.horizontal_flip).toBe(true);
  });

  it("resetHyperparameters restores default values", () => {
    useModelStore.getState().setHyperparameters({ batchSize: 32, learningRate: 1e-3 });
    useModelStore.getState().resetHyperparameters();

    const hp = useModelStore.getState().hyperparameters;
    expect(hp.batchSize).toBe(16);
    expect(hp.learningRate).toBe(1e-4);
  });
});
