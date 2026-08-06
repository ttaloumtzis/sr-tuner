import { describe, it, expect } from "vitest";
import { estimateVramBreakdown } from "../vramEstimate";

const SWINIR_SMALL = {
  embed_dim: 60,
  depths: [2, 2, 2, 2],
  num_heads: [2, 2, 2, 2],
  window_size: 8,
  mlp_ratio: 2.0,
  upsampler: "pixelshuffle",
  scale: 4,
};

const SWINIR_MEDIUM = {
  embed_dim: 180,
  depths: [6, 6, 6, 6, 6, 6],
  num_heads: [6, 6, 6, 6, 6, 6],
  window_size: 8,
  mlp_ratio: 2.0,
  upsampler: "pixelshuffle",
  scale: 4,
};

const RRDB = { num_feat: 64, num_block: 23, num_grow_ch: 32, scale: 4 };

describe("estimateVramBreakdown", () => {
  it("components sum exactly to the total for a matrix of configs", () => {
    const cases: Parameters<typeof estimateVramBreakdown>[0][] = [
      { arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: SWINIR_MEDIUM },
      { arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: SWINIR_MEDIUM, gradientCheckpointing: true },
      { arch: "swinir", batchSize: 4, patchSize: 128, fp16: true, scale: 4, config: SWINIR_SMALL },
      { arch: "rrdb_esrgan", batchSize: 8, patchSize: 128, fp16: false, scale: 4, config: RRDB },
      { arch: "rrdb_esrgan", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: RRDB, gradientCheckpointing: true },
      { arch: "some_unknown_arch", batchSize: 2, patchSize: 32, fp16: false, scale: 2, config: {} },
    ];
    for (const opts of cases) {
      const b = estimateVramBreakdown(opts);
      const sum = b.weightsGb + b.gradsGb + b.adamGb + b.activationsGb + b.inputGb + b.upsamplerGb + b.overheadGb;
      expect(Math.abs(sum - b.totalGb)).toBeLessThan(1e-9);
      expect(b.totalGb).toBeGreaterThan(0);
    }
  });

  it("returns all-zero breakdown when inputs are missing", () => {
    const b = estimateVramBreakdown({ arch: "", batchSize: 0, patchSize: 64, fp16: false });
    expect(b.totalGb).toBe(0);
  });

  it("regression: predicted peak reserved within tolerance of measured probes (ROCm 6.3, 16 GB)", () => {
    const tol = 1.0;
    const cases: { opts: Parameters<typeof estimateVramBreakdown>[0]; measured: number }[] = [
      { opts: { arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: SWINIR_SMALL }, measured: 1.115 },
      { opts: { arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: SWINIR_SMALL, gradientCheckpointing: true }, measured: 0.77 },
      { opts: { arch: "swinir", batchSize: 4, patchSize: 128, fp16: false, scale: 4, config: SWINIR_SMALL, gradientCheckpointing: true }, measured: 2.154 },
      { opts: { arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: { ...SWINIR_MEDIUM, embed_dim: 120, depths: [4, 4, 4, 4], num_heads: [4, 4, 4, 4] } }, measured: 2.748 },
      { opts: { arch: "swinir", batchSize: 8, patchSize: 64, fp16: false, scale: 4, config: { ...SWINIR_MEDIUM, embed_dim: 120, depths: [4, 4, 4, 4], num_heads: [4, 4, 4, 4] } }, measured: 4.77 },
      { opts: { arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: SWINIR_MEDIUM }, measured: 7.092 },
      { opts: { arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: SWINIR_MEDIUM, gradientCheckpointing: true }, measured: 1.777 },
      { opts: { arch: "rrdb_esrgan", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: RRDB }, measured: 1.055 },
      { opts: { arch: "rrdb_esrgan", batchSize: 4, patchSize: 128, fp16: false, scale: 4, config: RRDB }, measured: 3.43 },
      { opts: { arch: "rrdb_esrgan", batchSize: 8, patchSize: 128, fp16: false, scale: 4, config: RRDB }, measured: 7.477 },
    ];
    for (const { opts, measured } of cases) {
      const got = estimateVramBreakdown(opts).totalGb;
      expect(Math.abs(got - measured)).toBeLessThanOrEqual(tol);
    }
  });

  it("sums array depths (swinir depths=[6,6,6,6,6,6] ≈ reference config)", () => {
    const medium = estimateVramBreakdown({ arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4, config: SWINIR_MEDIUM });
    const shallow = estimateVramBreakdown({
      arch: "swinir", batchSize: 4, patchSize: 64, fp16: false, scale: 4,
      config: { ...SWINIR_MEDIUM, depths: [2, 2, 2, 2] },
    });
    expect(medium.activationsGb).toBeGreaterThan(3);
    expect(shallow.activationsGb).toBeLessThan(medium.activationsGb);
    expect(Math.abs(medium.totalGb - 7.262)).toBeLessThan(1.0);
  });
});
