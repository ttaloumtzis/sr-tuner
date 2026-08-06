export interface SliderField {
  type: "slider";
  kind: "int" | "float";
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  hint?: string;
}

export interface DropdownField {
  type: "dropdown";
  key: string;
  label: string;
  options: (number | string)[];
  default: number | string;
  hint?: string;
}

export interface TextField {
  type: "text";
  key: string;
  label: string;
  default?: unknown;
  hint?: string;
  csv?: boolean;
  placeholder?: string;
}

export type ConfigField = SliderField | DropdownField | TextField;

export interface ModelTemplate {
  id: string;
  name: string;
  description: string;
  paramsM: number;
  recommended?: boolean;
  values: Record<string, unknown>;
}

export interface ArchitectureDef {
  id: string;
  displayName: string;
  tag: string;
  description: string;
  reference: { vram: string; params: string };
  fields: ConfigField[];
  templates: ModelTemplate[];
  matchIgnoreKeys: string[];
  derivedKeys: string[];
  estimateParams: (values: Record<string, unknown>) => number;
  derive?: (changedKey: string, value: unknown, nextValues: Record<string, unknown>) => Record<string, unknown>;
  yaml: {
    csvKeys: string[];
    emptyCsvOmit: boolean;
  };
}

const HINTS: Record<string, string> = {
  scale: "Upscaling factor applied to the input image (e.g. 4x turns a 256px image into 1024px).",
  num_feat: "Width of the network — more features capture finer detail at the cost of VRAM and speed.",
  num_block: "Depth of the network — more RRDB blocks improve quality but slow down training and inference.",
  num_grow_ch: "Growth rate inside each dense block. Higher values add capacity with a smaller cost than num_feat.",
  embed_dim: "Width of the transformer's token embeddings — the SwinIR equivalent of num_feat.",
  window_size: "Size of the local attention window. Must evenly divide the input patch size.",
  mlp_ratio: "Expansion factor of the feed-forward layer inside each transformer block.",
  upsampler: "Method used to reconstruct the final high-resolution image from features.",
  img_range: "Pixel value range the network is trained to expect (usually 1.0 for [0,1]-normalized inputs).",
  num_in_ch: "Number of channels in the input image (3 for RGB, 1 for grayscale).",
  num_out_ch: "Number of channels in the output image.",
  depths: "Comma-separated transformer block count per stage, e.g. 6,6,6,6,6,6 for six stages. Each value must be a positive integer (typical 2–12). The number of values = number of stages and must match Num Heads. Non-integer entries are filtered out; empty resets to 6,6,6,6,6,6.",
  num_heads: "Comma-separated attention head count per stage (typical 2–12). Auto-derived from Embedding Dim; the number of values follows Depths. Non-integer entries filtered out.",
  rgb_mean: "Per-channel mean used to normalize inputs before training. Leave blank to use the dataset default.",
};

function hint(key: string): string | undefined {
  return HINTS[key];
}

function textCsvPlaceholder(key: string): string | undefined {
  if (key === "rgb_mean") return "0.4488, 0.4371, 0.4040";
  if (key === "depths" || key === "num_heads") return "6,6,6,6,6,6";
  return;
}

const SCALE_FIELD: DropdownField = {
  type: "dropdown", key: "scale", label: "Scale Factor",
  options: [1, 2, 4, 8], default: 4, hint: hint("scale"),
};

const RRDB_FIELDS: ConfigField[] = [
  SCALE_FIELD,
  { type: "slider", kind: "int", key: "num_feat", label: "Base Features", min: 32, max: 256, step: 8, default: 64, hint: hint("num_feat") },
  { type: "slider", kind: "int", key: "num_block", label: "RRDB Blocks", min: 4, max: 48, step: 1, default: 23, hint: hint("num_block") },
  { type: "slider", kind: "int", key: "num_grow_ch", label: "Growth Channels", min: 16, max: 128, step: 8, default: 32, hint: hint("num_grow_ch") },
  { type: "dropdown", key: "num_in_ch", label: "Input Channels", options: [1, 3], default: 3, hint: hint("num_in_ch") },
  { type: "dropdown", key: "num_out_ch", label: "Output Channels", options: [1, 3], default: 3, hint: hint("num_out_ch") },
];

const RRDB_TEMPLATES: ModelTemplate[] = [
  { id: "lightning", name: "Lightning", description: "Fastest, minimal VRAM", paramsM: 2.1, values: { num_feat: 32, num_block: 8, num_grow_ch: 16, num_in_ch: 3, num_out_ch: 3 } },
  { id: "light", name: "Light", description: "Quick training, solid results", paramsM: 3.9, values: { num_feat: 48, num_block: 12, num_grow_ch: 24, num_in_ch: 3, num_out_ch: 3 } },
  { id: "medium", name: "Medium", description: "Balanced quality and speed", paramsM: 16.7, recommended: true, values: { num_feat: 64, num_block: 23, num_grow_ch: 32, num_in_ch: 3, num_out_ch: 3 } },
  { id: "heavy", name: "Heavy", description: "High quality, more VRAM", paramsM: 35.0, values: { num_feat: 96, num_block: 32, num_grow_ch: 48, num_in_ch: 3, num_out_ch: 3 } },
  { id: "ultra", name: "Ultra", description: "Maximum quality, heavy VRAM", paramsM: 60.0, values: { num_feat: 128, num_block: 48, num_grow_ch: 64, num_in_ch: 3, num_out_ch: 3 } },
];

const SWINIR_FIELDS: ConfigField[] = [
  SCALE_FIELD,
  { type: "slider", kind: "int", key: "embed_dim", label: "Embedding Dim", min: 60, max: 384, step: 12, default: 180, hint: hint("embed_dim") },
  { type: "slider", kind: "int", key: "window_size", label: "Window Size", min: 4, max: 16, step: 2, default: 8, hint: hint("window_size") },
  { type: "slider", kind: "float", key: "mlp_ratio", label: "MLP Ratio", min: 1.0, max: 4.0, step: 0.1, default: 2.0, hint: hint("mlp_ratio") },
  { type: "text", key: "depths", label: "Depths", csv: true, placeholder: textCsvPlaceholder("depths"), hint: hint("depths") },
  { type: "text", key: "num_heads", label: "Num Heads", csv: true, placeholder: textCsvPlaceholder("num_heads"), hint: hint("num_heads") },
  { type: "dropdown", key: "upsampler", label: "Upsampler", options: ["pixelshuffle", "nearest+conv"], default: "pixelshuffle", hint: hint("upsampler") },
  { type: "slider", kind: "float", key: "img_range", label: "Image Range", min: 0.5, max: 2.0, step: 0.1, default: 1.0, hint: hint("img_range") },
  { type: "dropdown", key: "num_in_ch", label: "Input Channels", options: [1, 3], default: 3, hint: hint("num_in_ch") },
  { type: "dropdown", key: "num_out_ch", label: "Output Channels", options: [1, 3], default: 3, hint: hint("num_out_ch") },
  { type: "text", key: "rgb_mean", label: "RGB Mean", csv: true, placeholder: textCsvPlaceholder("rgb_mean"), hint: hint("rgb_mean") },
];

function getNumHeads(embedDim: number): number {
  const target = Math.max(2, Math.floor(embedDim / 32));
  for (let n = target; n >= 2; n--) if (embedDim % n === 0) return n;
  for (let n = target + 1; n <= embedDim / 2; n++) if (embedDim % n === 0) return n;
  return embedDim;
}

function generateNumHeadsCsv(depthsCsv: string, numHeadsValue: number): string {
  const count = parseCSV(depthsCsv).length;
  return Array(count).fill(numHeadsValue).join(",");
}

const _ARCHITECTURES = [
  {
    id: "rrdb_esrgan" as const,
    displayName: "RRDB-ESRGAN",
    tag: "GAN · Perceptual",
    description: "Best visual quality for textures and fine detail. Uses a discriminator network.",
    reference: { vram: "~8 GB", params: "~16M" },
    fields: RRDB_FIELDS,
    templates: RRDB_TEMPLATES,
    matchIgnoreKeys: ["scale"],
    derivedKeys: [],
    yaml: { csvKeys: [], emptyCsvOmit: true },
    estimateParams: (values: Record<string, unknown>): number => {
      const nf = (values.num_feat as number) ?? 64;
      const nb = (values.num_block as number) ?? 23;
      const ng = (values.num_grow_ch as number) ?? 32;
      return 16.7 * (nf / 64) ** 2 * (nb / 23) * Math.sqrt(ng / 32);
    },
  },
  {
    id: "swinir" as const,
    displayName: "SwinIR",
    tag: "Transformer · PSNR",
    description: "Swin Transformer backbone. Best PSNR/SSIM scores, no adversarial training.",
    reference: { vram: "~6 GB", params: "~11M" },
    fields: SWINIR_FIELDS,
    templates: [
      { id: "lightning", name: "Lightning", description: "Fastest, minimal VRAM", paramsM: 2.1, values: { embed_dim: 60, window_size: 6, mlp_ratio: 2.0, depths: "4,4,4,4", num_heads: "2,2,2,2", upsampler: "pixelshuffle", img_range: 1.0, num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040" } },
      { id: "light", name: "Light", description: "Quick training, solid results", paramsM: 3.9, values: { embed_dim: 96, window_size: 8, mlp_ratio: 2.0, depths: "4,6,6,4", num_heads: "3,3,3,3", upsampler: "pixelshuffle", img_range: 1.0, num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040" } },
      { id: "medium", name: "Medium", description: "Balanced quality and speed", paramsM: 11.8, recommended: true, values: { embed_dim: 180, window_size: 8, mlp_ratio: 2.0, depths: "6,6,6,6,6,6", num_heads: "6,6,6,6,6,6", upsampler: "pixelshuffle", img_range: 1.0, num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040" } },
      { id: "heavy", name: "Heavy", description: "High quality, more VRAM", paramsM: 19.2, values: { embed_dim: 240, window_size: 8, mlp_ratio: 2.0, depths: "6,6,6,6,8,8", num_heads: "8,8,8,8,8,8", upsampler: "pixelshuffle", img_range: 1.0, num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040" } },
      { id: "ultra", name: "Ultra", description: "Maximum quality, heavy VRAM", paramsM: 38.5, values: { embed_dim: 336, window_size: 8, mlp_ratio: 2.0, depths: "6,8,8,8,8,6", num_heads: "12,12,12,12,12,12", upsampler: "pixelshuffle", img_range: 1.0, num_in_ch: 3, num_out_ch: 3, rgb_mean: "0.4488, 0.4371, 0.4040" } },
    ],
    matchIgnoreKeys: ["scale"],
    derivedKeys: ["num_heads"],
    yaml: { csvKeys: ["depths", "num_heads", "rgb_mean"], emptyCsvOmit: true },
    estimateParams: (values: Record<string, unknown>): number => {
      const ed = (values.embed_dim as number) ?? 180;
      const depths = parseCSV(String(values.depths ?? "6,6,6,6,6,6"));
      const avgDepth = depths.length > 0 ? depths.reduce((a, b) => a + b, 0) / depths.length : 6;
      return 11.8 * (ed / 180) ** 2 * (avgDepth / 6);
    },
    derive: (changedKey: string, value: unknown, nextValues: Record<string, unknown>): Record<string, unknown> => {
      if (changedKey === "embed_dim") {
        const heads = getNumHeads(value as number);
        const depthsCsv = String(nextValues.depths ?? "6,6,6,6,6,6");
        return { num_heads: generateNumHeadsCsv(depthsCsv, heads) };
      }
      if (changedKey === "depths") {
        const heads = getNumHeads(nextValues.embed_dim as number);
        return { num_heads: generateNumHeadsCsv(String(value ?? ""), heads) };
      }
      return {};
    },
  },
] as const satisfies readonly (ArchitectureDef & { id: string })[];

export type Architecture = (typeof _ARCHITECTURES)[number]["id"];

const ARCH_MAP: Record<string, ArchitectureDef> = {};
for (const def of _ARCHITECTURES) ARCH_MAP[def.id] = def;

export function getArch(id: Architecture): ArchitectureDef {
  return ARCH_MAP[id];
}

export function getAllArchitectures(): readonly ArchitectureDef[] {
  return _ARCHITECTURES;
}

export function getDefaultTemplate(def: ArchitectureDef): ModelTemplate {
  return def.templates.find((t) => t.recommended) ?? def.templates[0];
}

export function getTemplateValues(def: ArchitectureDef, id: string): Record<string, unknown> {
  const tpl = def.templates.find((t) => t.id === id);
  return tpl ? { ...tpl.values } : {};
}

export function parseCSV(s: string): number[] {
  return s.split(",").map((v) => parseFloat(v.trim())).filter((n) => !isNaN(n));
}

export function estimateParamsFor(archId: string, values: Record<string, unknown>): number {
  const def = ARCH_MAP[archId];
  return def ? def.estimateParams(values) : 0;
}

export function formatParamCount(paramsM: number): string {
  if (paramsM >= 1000) return `${(paramsM / 1000).toFixed(1)} B`;
  if (paramsM >= 1) return `${paramsM.toFixed(1)} M`;
  return `${(paramsM * 1000).toFixed(0)} K`;
}

export function formatWeightMB(paramsM: number): string {
  return (paramsM * 4).toFixed(1);
}

export function matchTemplate(def: ArchitectureDef, values: Record<string, unknown>): string | null {
  const alwaysIgnore = new Set(["rgb_mean", ...def.matchIgnoreKeys]);
  for (const tpl of def.templates) {
    let match = true;
    for (const k of Object.keys(tpl.values)) {
      if (alwaysIgnore.has(k)) continue;
      if (String(values[k] ?? "") !== String(tpl.values[k])) {
        match = false;
        break;
      }
    }
    if (match) return tpl.id;
  }
  return null;
}

export function serializeValue(def: ArchitectureDef, key: string, raw: unknown): unknown {
  if (def.yaml.csvKeys.includes(key)) {
    const s = String(raw ?? "").trim();
    if (!s && def.yaml.emptyCsvOmit) return undefined;
    const nums = parseCSV(s);
    return nums.length > 0 ? nums : undefined;
  }
  return raw;
}

export function buildYaml(def: ArchitectureDef, values: Record<string, unknown>, name: string): string {
  const lines: string[] = [];
  lines.push(`# ${def.displayName} model configuration`);
  lines.push(`name: ${name || def.id}`);
  lines.push(`architecture: ${def.id}`);
  for (const field of def.fields) {
    const value = values[field.key] ?? (field as any).default;
    if (value === undefined || value === null) continue;
    if (def.yaml.csvKeys.includes(field.key)) {
      const nums = parseCSV(String(value));
      if (nums.length === 0 && def.yaml.emptyCsvOmit) continue;
      lines.push(`${field.key}: [${nums.join(", ")}]`);
    } else if (typeof value === "number") {
      lines.push(`${field.key}: ${value}`);
    } else {
      lines.push(`${field.key}: ${value}`);
    }
  }
  return lines.join("\n");
}