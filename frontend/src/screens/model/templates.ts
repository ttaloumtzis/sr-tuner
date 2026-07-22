import type { Architecture } from "../../lib/srproj";

export type SwinirTemplateId = "lightning" | "light" | "medium" | "heavy" | "ultra";
export type RrdbTemplateId = "lightning" | "light" | "medium" | "heavy" | "ultra";
export type ModelTemplateId = SwinirTemplateId | RrdbTemplateId;

export interface TemplateDef {
  id: ModelTemplateId;
  name: string;
  description: string;
  paramsM: number;
  recommended?: boolean;
}

const SWINIR_TEMPLATES: TemplateDef[] = [
  { id: "lightning", name: "Lightning", description: "Fastest, minimal VRAM", paramsM: 2.1 },
  { id: "light", name: "Light", description: "Quick training, solid results", paramsM: 3.9 },
  { id: "medium", name: "Medium", description: "Balanced quality and speed", paramsM: 11.8, recommended: true },
  { id: "heavy", name: "Heavy", description: "High quality, more VRAM", paramsM: 19.2 },
  { id: "ultra", name: "Ultra", description: "Maximum quality, heavy VRAM", paramsM: 38.5 },
];

const RRDB_TEMPLATES: TemplateDef[] = [
  { id: "lightning", name: "Lightning", description: "Fastest, minimal VRAM", paramsM: 2.1 },
  { id: "light", name: "Light", description: "Quick training, solid results", paramsM: 3.9 },
  { id: "medium", name: "Medium", description: "Balanced quality and speed", paramsM: 16.7, recommended: true },
  { id: "heavy", name: "Heavy", description: "High quality, more VRAM", paramsM: 35.0 },
  { id: "ultra", name: "Ultra", description: "Maximum quality, heavy VRAM", paramsM: 60.0 },
];

const SWINIR_TEMPLATE_VALUES: Record<SwinirTemplateId, Record<string, unknown>> = {
  lightning: {
    embed_dim: 60, window_size: 6, mlp_ratio: 2.0,
    depths: "4,4,4,4", num_heads: "2,2,2,2",
    upsampler: "pixelshuffle", img_range: 1.0,
    num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040",
  },
  light: {
    embed_dim: 96, window_size: 8, mlp_ratio: 2.0,
    depths: "4,6,6,4", num_heads: "3,3,3,3",
    upsampler: "pixelshuffle", img_range: 1.0,
    num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040",
  },
  medium: {
    embed_dim: 180, window_size: 8, mlp_ratio: 2.0,
    depths: "6,6,6,6,6,6", num_heads: "6,6,6,6,6,6",
    upsampler: "pixelshuffle", img_range: 1.0,
    num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040",
  },
  heavy: {
    embed_dim: 240, window_size: 8, mlp_ratio: 2.0,
    depths: "6,6,6,6,8,8", num_heads: "8,8,8,8,8,8",
    upsampler: "pixelshuffle", img_range: 1.0,
    num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040",
  },
  ultra: {
    embed_dim: 336, window_size: 8, mlp_ratio: 2.0,
    depths: "6,8,8,8,8,6", num_heads: "12,12,12,12,12,12",
    upsampler: "pixelshuffle", img_range: 1.0,
    num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040",
  },
};

const RRDB_TEMPLATE_VALUES: Record<RrdbTemplateId, Record<string, unknown>> = {
  lightning: { num_feat: 32, num_block: 8, num_grow_ch: 16, num_in_ch: 3, num_out_ch: 3 },
  light: { num_feat: 48, num_block: 12, num_grow_ch: 24, num_in_ch: 3, num_out_ch: 3 },
  medium: { num_feat: 64, num_block: 23, num_grow_ch: 32, num_in_ch: 3, num_out_ch: 3 },
  heavy: { num_feat: 96, num_block: 32, num_grow_ch: 48, num_in_ch: 3, num_out_ch: 3 },
  ultra: { num_feat: 128, num_block: 48, num_grow_ch: 64, num_in_ch: 3, num_out_ch: 3 },
};

export function getSwinirTemplates(): TemplateDef[] {
  return SWINIR_TEMPLATES;
}

export function getRrdbTemplates(): TemplateDef[] {
  return RRDB_TEMPLATES;
}

export function getTemplateValues(arch: Architecture, id: ModelTemplateId): Record<string, unknown> {
  return arch === "swinir"
    ? { ...SWINIR_TEMPLATE_VALUES[id as SwinirTemplateId] }
    : { ...RRDB_TEMPLATE_VALUES[id as RrdbTemplateId] };
}

export function getTemplateDefaultId(_arch: Architecture): ModelTemplateId {
  return "medium";
}

export function getNumHeads(embedDim: number): number {
  const target = Math.max(2, Math.floor(embedDim / 32));
  for (let n = target; n >= 2; n--)
    if (embedDim % n === 0) return n;
  for (let n = target + 1; n <= embedDim / 2; n++)
    if (embedDim % n === 0) return n;
  return embedDim;
}

export function generateNumHeadsCsv(depthsCsv: string, numHeadsValue: number): string {
  const count = parseCSV(depthsCsv).length;
  return Array(count).fill(numHeadsValue).join(",");
}

export function parseCSV(s: string): number[] {
  return s.split(",").map((v) => parseFloat(v.trim())).filter((n) => !isNaN(n));
}

export function estimateParams(arch: Architecture, values: Record<string, unknown>): number {
  if (arch === "rrdb_esrgan") {
    const nf = (values.num_feat as number) ?? 64;
    const nb = (values.num_block as number) ?? 23;
    const ng = (values.num_grow_ch as number) ?? 32;
    return 16.7 * (nf / 64) ** 2 * (nb / 23) * Math.sqrt(ng / 32);
  }
  if (arch === "swinir") {
    const ed = (values.embed_dim as number) ?? 180;
    const depths = parseCSV(String(values.depths ?? "6,6,6,6,6,6"));
    const avgDepth = depths.length > 0 ? depths.reduce((a, b) => a + b, 0) / depths.length : 6;
    return 11.8 * (ed / 180) ** 2 * (avgDepth / 6);
  }
  return 0;
}

export function formatParamCount(paramsM: number): string {
  if (paramsM >= 1000) return `${(paramsM / 1000).toFixed(1)} B`;
  if (paramsM >= 1) return `${paramsM.toFixed(1)} M`;
  return `${(paramsM * 1000).toFixed(0)} K`;
}

export function formatWeightMB(paramsM: number): string {
  // fp32 = 4 bytes/param. paramsM is millions of params, so paramsM * 4 = MB directly
  // (1e6 params * 4 bytes = 4e6 bytes = 4 MB). Previously this divided by an extra 1024,
  // which under-reported weight size by ~1000x (e.g. a 16.7M-param model showed "0.1 MB"
  // instead of ~66.8 MB).
  return (paramsM * 4).toFixed(1);
}

export function doesConfigMatchTemplate(
  arch: Architecture,
  values: Record<string, unknown>,
): ModelTemplateId | "custom" {
  const templates = arch === "swinir" ? SWINIR_TEMPLATE_VALUES : RRDB_TEMPLATE_VALUES;
  const ids = Object.keys(templates) as ModelTemplateId[];
  for (const id of ids) {
    const tmpl = templates[id];
    let match = true;
    for (const k of Object.keys(tmpl)) {
      if (k === "rgb_mean" || k === "scale") continue;
      if (String(values[k] ?? "") !== String(tmpl[k])) {
        match = false;
        break;
      }
    }
    if (match) return id;
  }
  return "custom";
}
