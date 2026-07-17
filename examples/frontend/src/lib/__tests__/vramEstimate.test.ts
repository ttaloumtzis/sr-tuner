import { describe, it, expect } from "vitest";
import { estimateVram, VRAM_BASE_GB } from "../vramEstimate";

// Reference inputs: batch=4, patch=192, fp16=false
// formula: base × (batch/4) × (patch/192)² × (fp16 ? 0.5 : 1.0)

describe("estimateVram", () => {
  it("Real-ESRGAN base=8 at reference inputs equals 8", () => {
    expect(estimateVram("Real-ESRGAN", 4, 192, false)).toBeCloseTo(8);
  });

  it("SwinIR base=6 at reference inputs equals 6", () => {
    expect(estimateVram("SwinIR", 4, 192, false)).toBeCloseTo(6);
  });

  it("HAT base=10 at reference inputs equals 10", () => {
    expect(estimateVram("HAT", 4, 192, false)).toBeCloseTo(10);
  });

  it("EDSR base=4 at reference inputs equals 4", () => {
    expect(estimateVram("EDSR", 4, 192, false)).toBeCloseTo(4);
  });

  it("fp16=true halves the estimate", () => {
    const full = estimateVram("Real-ESRGAN", 4, 192, false);
    const half = estimateVram("Real-ESRGAN", 4, 192, true);
    expect(half).toBeCloseTo(full * 0.5);
  });

  it("scales linearly with batch/4", () => {
    const base = estimateVram("Real-ESRGAN", 4, 192, false);
    const doubled = estimateVram("Real-ESRGAN", 8, 192, false);
    expect(doubled).toBeCloseTo(base * 2);
  });

  it("scales quadratically with (patch/192)^2", () => {
    const base = estimateVram("Real-ESRGAN", 4, 192, false);
    const larger = estimateVram("Real-ESRGAN", 4, 384, false);
    expect(larger).toBeCloseTo(base * 4);
  });

  it("VRAM_BASE_GB table has all four architectures", () => {
    expect(VRAM_BASE_GB["Real-ESRGAN"]).toBe(8);
    expect(VRAM_BASE_GB["SwinIR"]).toBe(6);
    expect(VRAM_BASE_GB["HAT"]).toBe(10);
    expect(VRAM_BASE_GB["EDSR"]).toBe(4);
  });
});
