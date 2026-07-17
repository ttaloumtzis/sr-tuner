import type { Architecture } from "./srproj";

export const VRAM_BASE_GB: Record<Architecture, number> = {
  "Real-ESRGAN": 8,
  "SwinIR": 6,
  "HAT": 10,
  "EDSR": 4,
};

export function estimateVram(
  arch: Architecture,
  batchSize: number,
  patchSize: number,
  fp16: boolean,
): number {
  const base = VRAM_BASE_GB[arch] ?? 8;
  return base * (batchSize / 4) * Math.pow(patchSize / 192, 2) * (fp16 ? 0.5 : 1.0);
}
