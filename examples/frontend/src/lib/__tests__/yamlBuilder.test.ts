import { describe, it, expect } from "vitest";
import { buildYaml } from "../yamlBuilder";
import type { AugmentationConfig } from "../srproj";
import type { Hyperparameters, LossWeights } from "../../store/modelStore";

const BASE_HP: Hyperparameters = {
  scale: 4,
  lrScheduler: "cosine",
  optimizer: "Adam",
  learningRate: 1e-4,
  batchSize: 4,
  patchSize: 192,
  totalIter: 300000,
};

const BASE_LW: LossWeights = {
  pixel: 1.0,
  perceptual: 0.1,
  adversarial: 0.005,
};

const BASE_AUG: AugmentationConfig = {
  horizontal_flip: true,
  vertical_flip: false,
  rotation_90: false,
  mixup: true,
  color_jitter: false,
  random_degradation: false,
  gaussian_blur: false,
  noise_injection: false,
};

describe("buildYaml", () => {
  it("contains model_type RealESRGANer for Real-ESRGAN", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("model_type: RealESRGANer");
  });

  it("contains scale 4", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("scale: 4");
  });

  it("contains optimizer type Adam", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("type: Adam");
  });

  it("contains pixel loss weight 1.0", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("loss_weight: !!float 1.0");
  });

  it("contains perceptual loss weight 0.1", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("loss_weight: !!float 0.1");
  });

  it("contains adversarial loss weight 0.0", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("loss_weight: !!float 0.0");
  });

  it("enabled augmentations appear in yaml", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("- horizontal_flip");
    expect(yaml).toContain("- mixup");
  });

  it("disabled augmentations do not appear in yaml", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).not.toContain("- color_jitter");
    expect(yaml).not.toContain("- vertical_flip");
  });

  it("toggling all augmentations off produces [] entry", () => {
    const noAug: AugmentationConfig = {
      horizontal_flip: false, vertical_flip: false, rotation_90: false,
      mixup: false, color_jitter: false, random_degradation: false,
      gaussian_blur: false, noise_injection: false,
    };
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, noAug, null);
    expect(yaml).toContain("[]");
  });

  it("includes pretrained_network_g when path provided", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, "/models/pretrained.pth");
    expect(yaml).toContain("pretrained_network_g: /models/pretrained.pth");
  });

  it("excludes pretrained_network_g when path is null", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).not.toContain("pretrained_network_g");
  });

  it("uses CosineAnnealingRestartLR for cosine scheduler", () => {
    const yaml = buildYaml("Real-ESRGAN", BASE_HP, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("CosineAnnealingRestartLR");
  });

  it("uses MultiStepLR for non-cosine scheduler", () => {
    const hp = { ...BASE_HP, lrScheduler: "step" };
    const yaml = buildYaml("Real-ESRGAN", hp, BASE_LW, BASE_AUG, null);
    expect(yaml).toContain("MultiStepLR");
  });
});
