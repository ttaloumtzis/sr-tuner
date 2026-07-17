import type { Architecture, AugmentationConfig } from "./srproj";
import type { Hyperparameters, LossWeights } from "../store/modelStore";

const archMap: Record<Architecture, string> = {
  "Real-ESRGAN": "RealESRGANer",
  SwinIR: "SwinIR",
  HAT: "HAT",
  EDSR: "EDSR",
};

export function buildYaml(
  arch: Architecture,
  hp: Hyperparameters,
  lw: LossWeights,
  aug: AugmentationConfig,
  pretrainedPath: string | null,
): string {
  const enabledAugs =
    Object.entries(aug)
      .filter(([, v]) => v)
      .map(([k]) => `\n    - ${k}`)
      .join("") || "\n    []";

  const pretrainedLine = pretrainedPath
    ? `\n  pretrained_network_g: ${pretrainedPath}`
    : "";

  return `# SR Tuner — Generated BasicSR Config
name: sr_tuner_run
model_type: ${archMap[arch]}

scale: ${hp.scale}

# Networks
network_g:
  type: ${archMap[arch]}
  scale: ${hp.scale}${pretrainedLine}

# Training
train:
  optim_g:
    type: ${hp.optimizer}
    lr: !!float ${hp.learningRate.toExponential(1)}

  scheduler:
    type: ${hp.lrScheduler === "cosine" ? "CosineAnnealingRestartLR" : "MultiStepLR"}

  total_iter: ${hp.totalIter}

  pixel_opt:
    type: L1Loss
    loss_weight: !!float ${lw.pixel.toFixed(1)}

  perceptual_opt:
    type: PerceptualLoss
    loss_weight: !!float ${lw.perceptual.toFixed(1)}

  gan_opt:
    type: GANLoss
    loss_weight: !!float ${lw.adversarial.toFixed(1)}

# Datasets
datasets:
  train:
    batch_size_per_gpu: ${hp.batchSize}
    gt_size: ${hp.patchSize}
    augmentations:${enabledAugs}
`;
}
