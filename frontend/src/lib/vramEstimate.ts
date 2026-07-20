export interface VramBreakdown {
  totalGb: number;
  weightsGb: number;
  gradsGb: number;
  adamGb: number;
  activationsGb: number;
  inputGb: number;
  overheadGb: number;
}

const ARCH_REFS: Record<string, { totalRefGb: number; paramCount: number }> = {
  "rrdb_esrgan": { totalRefGb: 4.0, paramCount: 16_700_000 },
  "swinir": { totalRefGb: 3.5, paramCount: 11_500_000 },
};

const OVERHEAD_GB = 1.5;

function estimateParamCount(
  arch: string,
  numFeat?: number,
  numBlock?: number,
  embedDim?: number,
  depths?: number[],
): number {
  if (arch === "rrdb_esrgan") {
    const nf = numFeat ?? 64;
    const nb = numBlock ?? 23;
    const base = ARCH_REFS.rrdb_esrgan.paramCount;
    return Math.round(base * (nf / 64) ** 2 * (nb / 23));
  }
  if (arch === "swinir") {
    const ed = embedDim ?? 180;
    const ds = depths ?? [6, 6, 6, 6, 6, 6];
    const totalLayers = ds.reduce((a, b) => a + b, 0);
    const base = ARCH_REFS.swinir.paramCount;
    return Math.round(base * (ed / 180) ** 2 * (totalLayers / 36));
  }
  if (numFeat && numBlock) {
    return Math.round(10_000_000 * (numFeat / 64) ** 2 * (numBlock / 23));
  }
  if (embedDim && depths) {
    const tl = depths.reduce((a, b) => a + b, 0);
    return Math.round(8_000_000 * (embedDim / 180) ** 2 * (tl / 36));
  }
  return ARCH_REFS[arch]?.paramCount ?? 10_000_000;
}

export function estimateVramBreakdown(
  arch: string,
  batchSize: number,
  patchSize: number,
  fp16: boolean,
  _scale?: number,
  numFeat?: number,
  numBlock?: number,
  embedDim?: number,
  depths?: number[],
): VramBreakdown {
  if (!batchSize || !patchSize || !arch) {
    return { totalGb: 0, weightsGb: 0, gradsGb: 0, adamGb: 0, activationsGb: 0, inputGb: 0, overheadGb: 0 };
  }

  const paramCount = estimateParamCount(arch, numFeat, numBlock, embedDim, depths);
  const fp16Factor = fp16 ? 0.5 : 1.0;

  const weightsGb = paramCount * 4 / 1e9;
  const gradsGb = paramCount * 4 * fp16Factor / 1e9;
  const adamGb = paramCount * 8 / 1e9;
  const inputGb = batchSize * 3 * (patchSize ** 2) * (fp16 ? 2 : 4) / 1e9;

  const fixedGb = OVERHEAD_GB + weightsGb + adamGb;
  const ref = ARCH_REFS[arch];
  const totalRefGb = ref?.totalRefGb ?? fixedGb * 2.2;
  const actRefGb = Math.max(totalRefGb - fixedGb, 0.5);

  const activationsGb = actRefGb
    * (batchSize / 4)
    * ((patchSize / 64) ** 2)
    * fp16Factor;

  const totalGb = OVERHEAD_GB + weightsGb + gradsGb + adamGb + activationsGb + inputGb;

  return { totalGb, weightsGb, gradsGb, adamGb, activationsGb, inputGb, overheadGb: OVERHEAD_GB };
}

export function estimateVram(
  arch: string,
  batchSize: number,
  patchSize: number,
  fp16: boolean,
  _scale?: number,
  numFeat?: number,
  numBlock?: number,
  embedDim?: number,
  depths?: number[],
): number {
  return estimateVramBreakdown(arch, batchSize, patchSize, fp16, _scale, numFeat, numBlock, embedDim, depths).totalGb;
}
