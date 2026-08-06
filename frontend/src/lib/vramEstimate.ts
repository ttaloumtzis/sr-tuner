export interface VramBreakdown {
  totalGb: number;
  weightsGb: number;
  gradsGb: number;
  adamGb: number;
  activationsGb: number;
  inputGb: number;
  upsamplerGb: number;
  overheadGb: number;
}

interface ArchProfile {
  profile: "transformer" | "conv";
  paramRef: number;
  widthKey: string;
  widthRef: number;
  depthKey: string;
  depthRef: number;
  headsKey?: string;
  headsRef?: number;
  mlpRatioKey?: string;
  mlpRatioRef?: number;
  outChKey?: string;
}

/**
 * Per-architecture reference profile. `*Ref` values define the config at which
 * `ACTIVATION_REF_GB` was calibrated; scaling factors extrapolate from there.
 * `paramRef` matches the real parameter count at the reference config (measured,
 * not theoretical). Unknown architectures fall back to the conv profile using
 * `paramCount` only.
 */
const PROFILES: Record<string, ArchProfile> = {
  swinir: {
    profile: "transformer",
    paramRef: 14_130_000,
    widthKey: "embed_dim", widthRef: 180,
    depthKey: "depths", depthRef: 36,
    headsKey: "num_heads", headsRef: 6,
    mlpRatioKey: "mlp_ratio", mlpRatioRef: 2.0,
    outChKey: "num_out_ch",
  },
  rrdb_esrgan: {
    profile: "conv",
    paramRef: 2_800_000,
    widthKey: "num_feat", widthRef: 64,
    depthKey: "num_block", depthRef: 23,
    outChKey: "num_out_ch",
  },
};

const DEFAULT_PROFILE: ArchProfile = {
  profile: "conv",
  paramRef: 2_800_000,
  widthKey: "num_feat", widthRef: 64,
  depthKey: "num_block", depthRef: 23,
};

/**
 * Reference activation memory (GB) for the MODEL's own activations at the
 * calibration config: batchSize=4, patchSize=64, fp32, gradient checkpointing
 * OFF, default loss. The VGG19 perceptual branch of the default loss is NOT
 * folded in — it is modeled separately by ``VGG_REF_GB`` below.
 *
 * Calibrated 2026-08-06 on ROCm 6.3 (16 GB RX 7000-class card) against real
 * `scripts/calibrate_vram.py` probe runs (peak RESERVED memory). Constants
 * are card/stack-specific; re-run the calibrate script to re-fit.
 */
const ACTIVATION_REF_GB: Record<string, number> = {
  transformer: 6.24,
  conv: 0.54,
};

/**
 * VGG19 perceptual branch memory (GB) at the calibration config
 * (batchSize=4, patchSize=64, scale=4 → HR 256², fp32, default composite
 * loss). Unlike the model's activations this branch is independent of model
 * width and is NOT reduced by gradient checkpointing, so it scales only with
 * batch size × HR patch area. The probe always measures the default L1+VGG
 * loss; the activation refs above are net of this term.
 */
const VGG_REF_GB = 0.26;

const BYTES_FP32 = 4;

/** Activations scale linearly with batch size (measured batch-exponent ≈ 1.0). */
const BATCH_EXP = 1.0;
/** Patch area is the dominant driver of activation memory. */
const PATCH_EXP = 2.0;
/** Transformer width/depth/heads/MLP scaling is sub-linear (fixed conv floors). */
const WIDTH_EXP = 0.7;
/** Gradient checkpointing cuts activations ~5x (measured 6.5 GB → 1.3 GB). */
const CHECKPOINT_FACTOR = 0.2;
/** torch/HIP context + MIOpen/cuDNN workspace + allocator fragmentation. */
const CONTEXT_FLOOR_GB = 0.6;

export interface VramEstimateOptions {
  arch: string;
  batchSize: number;
  patchSize: number;
  fp16: boolean;
  scale?: number;
  config?: Record<string, unknown>;
  gradientCheckpointing?: boolean;
}

export function parseDepths(depths: unknown): number[] {
  if (Array.isArray(depths)) return depths.map((d) => Number(d)).filter((n) => !isNaN(n));
  if (typeof depths === "string") {
    return depths.split(",").map((v) => parseFloat(v.trim())).filter((n) => !isNaN(n));
  }
  return [];
}

function num(v: unknown, fallback: number): number {
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return isNaN(n) ? fallback : n;
}

/** Depth configs are arrays/strings (e.g. [6,6,6,6,6,6]); sum them. */
function depthOf(v: unknown, fallback: number): number {
  const summed = parseDepths(v).reduce((a, b) => a + b, 0);
  return summed > 0 ? summed : num(v, fallback);
}

/** Heads configs are uniform arrays; take the first element. */
function firstOf(v: unknown, fallback: number): number {
  const list = parseDepths(v);
  return list.length > 0 ? list[0] : num(v, fallback);
}

function estimateParamCount(arch: string, config: Record<string, unknown>): number {
  const p = PROFILES[arch] ?? DEFAULT_PROFILE;
  const width = num(config[p.widthKey], p.widthRef);
  const depth = depthOf(config[p.depthKey], p.depthRef);
  const factor = (width / p.widthRef) ** 2 * (depth / p.depthRef);
  return Math.round(p.paramRef * factor);
}

export function estimateVramBreakdown(opts: VramEstimateOptions): VramBreakdown {
  const { arch, batchSize, patchSize, fp16 } = opts;
  if (!batchSize || !patchSize || !arch) {
    return {
      totalGb: 0, weightsGb: 0, gradsGb: 0, adamGb: 0,
      activationsGb: 0, inputGb: 0, upsamplerGb: 0, overheadGb: 0,
    };
  }

  const config = opts.config ?? {};
  const p = PROFILES[arch] ?? DEFAULT_PROFILE;
  const fp16Factor = fp16 ? 0.5 : 1.0;
  const bytesPerElem = fp16 ? 2 : BYTES_FP32;
  const scale = opts.scale ?? 4;
  const outCh = num(config[p.outChKey ?? "num_out_ch"], 3);

  const paramCount = estimateParamCount(arch, config);

  const weightsGb = (paramCount * 4) / 1e9;
  const gradsGb = (paramCount * 4 * fp16Factor) / 1e9;
  const adamGb = (paramCount * 8) / 1e9;
  const inputGb = (batchSize * 3 * patchSize ** 2 * bytesPerElem) / 1e9;

  // Spatial + width/depth scaling from the calibration point (B=4, patch=64).
  let activationFactor =
    (batchSize / 4) ** BATCH_EXP *
    (patchSize / 64) ** PATCH_EXP;
  if (p.profile === "transformer") {
    const width = num(config[p.widthKey], p.widthRef);
    const depth = depthOf(config[p.depthKey], p.depthRef);
    const heads = firstOf(config[p.headsKey ?? ""], p.headsRef ?? 6);
    const mlp = num(config[p.mlpRatioKey ?? ""], p.mlpRatioRef ?? 2.0);
    const ws = (width / p.widthRef) * (heads / (p.headsRef ?? 6)) * (mlp / (p.mlpRatioRef ?? 2.0)) * (depth / p.depthRef);
    activationFactor *= ws ** WIDTH_EXP;
  } else {
    const width = num(config[p.widthKey], p.widthRef);
    const depth = depthOf(config[p.depthKey], p.depthRef);
    activationFactor *= (width / p.widthRef) * (depth / p.depthRef);
  }

  let modelActivationsGb = ACTIVATION_REF_GB[p.profile] * activationFactor * fp16Factor;
  if (opts.gradientCheckpointing) {
    modelActivationsGb *= CHECKPOINT_FACTOR;
  }

  // VGG19 perceptual branch (default loss): width-independent, not
  // checkpointed, scales with batch × HR patch area. Kept separate from the
  // model activations so width/checkpoint scaling apply only to the model.
  const vggHrScale = (patchSize * scale) / 256;
  const vggGb = VGG_REF_GB * (batchSize / 4) * vggHrScale ** 2 * fp16Factor;

  const activationsGb = modelActivationsGb + vggGb;

  // Upsampler intermediates + HR output/loss. PixelShuffle convs emit
  // `4*embed_dim`-channel maps at LR resolution per ×2 stage; the output and
  // the loss tensor are at `patch*scale` resolution.
  const upsamplerStages = scale > 1 ? Math.log2(scale) : 0;
  const upsampleWidth = num(config[p.widthKey], p.widthRef);
  const upsamplerGb =
    (batchSize * patchSize ** 2 * Math.max(upsamplerStages, 0) * upsampleWidth * 4 * bytesPerElem) / 1e9 +
    (batchSize * outCh * (patchSize * scale) ** 2 * bytesPerElem) / 1e9;

  // Fixed torch/HIP context + workspace floor. Components below sum to the
  // total by construction; `overheadGb` is the flat CUDA-context term.
  const overheadGb = CONTEXT_FLOOR_GB;
  const totalGb = weightsGb + gradsGb + adamGb + activationsGb + inputGb + upsamplerGb + overheadGb;

  return {
    totalGb,
    weightsGb,
    gradsGb,
    adamGb,
    activationsGb,
    inputGb,
    upsamplerGb,
    overheadGb,
  };
}

export function estimateVram(opts: VramEstimateOptions): number {
  return estimateVramBreakdown(opts).totalGb;
}
