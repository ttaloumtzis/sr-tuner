import { useState } from "react";
import type { ConfigField } from "./ConfigFieldRow";
import type { Architecture } from "../../lib/srproj";

interface ArchDef {
  id: Architecture;
  tag: string;
  description: string;
  vram: string;
  params: string;
  fields: ConfigField[];
}

const RRDB_FIELDS: ConfigField[] = [
  { type: "dropdown", key: "scale", label: "Scale Factor", options: [1, 2, 4, 8], default: 4 },
  { type: "slider", kind: "int", key: "num_feat", label: "Base Features", min: 32, max: 256, step: 8, default: 64 },
  { type: "slider", kind: "int", key: "num_block", label: "RRDB Blocks", min: 4, max: 48, step: 1, default: 23 },
  { type: "slider", kind: "int", key: "num_grow_ch", label: "Growth Channels", min: 16, max: 128, step: 8, default: 32 },
  { type: "dropdown", key: "num_in_ch", label: "Input Channels", options: [1, 3], default: 3 },
  { type: "dropdown", key: "num_out_ch", label: "Output Channels", options: [1, 3], default: 3 },
];

const SWINIR_FIELDS: ConfigField[] = [
  { type: "dropdown", key: "scale", label: "Scale Factor", options: [1, 2, 4, 8], default: 4 },
  { type: "slider", kind: "int", key: "embed_dim", label: "Embedding Dim", min: 60, max: 384, step: 12, default: 180 },
  { type: "slider", kind: "int", key: "window_size", label: "Window Size", min: 4, max: 16, step: 2, default: 8 },
  { type: "slider", kind: "float", key: "mlp_ratio", label: "MLP Ratio", min: 1.0, max: 4.0, step: 0.1, default: 2.0 },
  { type: "text", key: "depths", label: "Depths" },
  { type: "text", key: "num_heads", label: "Num Heads" },
  { type: "dropdown", key: "upsampler", label: "Upsampler", options: ["pixelshuffle", "nearest+conv"], default: "pixelshuffle" },
  { type: "slider", kind: "float", key: "img_range", label: "Image Range", min: 0.5, max: 2.0, step: 0.1, default: 1.0 },
  { type: "dropdown", key: "num_in_ch", label: "Input Channels", options: [1, 3], default: 3 },
  { type: "dropdown", key: "num_out_ch", label: "Output Channels", options: [1, 3], default: 3 },
  { type: "text", key: "rgb_mean", label: "RGB Mean" },
];

const ARCH_DEFS: ArchDef[] = [
  {
    id: "rrdb_esrgan",
    tag: "GAN · Perceptual",
    description: "Best visual quality for textures and fine detail. Uses a discriminator network.",
    vram: "~8 GB",
    params: "~16M",
    fields: RRDB_FIELDS,
  },
  {
    id: "swinir",
    tag: "Transformer · PSNR",
    description: "Swin Transformer backbone. Best PSNR/SSIM scores, no adversarial training.",
    vram: "~6 GB",
    params: "~11M",
    fields: SWINIR_FIELDS,
  },
];

export type { ArchDef };
export { ARCH_DEFS, RRDB_FIELDS, SWINIR_FIELDS };

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      <span style={{ fontSize: 10, color: "var(--muted)" }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)" }}>{value}</span>
    </div>
  );
}

function ArchCard({ def, active, onClick }: { def: ArchDef; active: boolean; onClick: () => void }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex", flexDirection: "column", gap: 6,
        padding: 12, borderRadius: 8, cursor: "pointer",
        background: active ? "var(--bg3)" : hovered ? "var(--bg3)" : "var(--bg2)",
        border: `1px solid ${active ? "var(--green)" : hovered ? "var(--border2)" : "var(--border)"}`,
        transition: "var(--transition-fast)",
        textAlign: "left", width: "100%",
        opacity: active ? 1 : hovered ? 0.9 : 0.7,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: active ? "var(--green)" : "var(--bg3)",
          border: "1px solid var(--border)",
          flexShrink: 0,
        }} />
        <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>{def.id === "swinir" ? "SwinIR" : "RRDB-ESRGAN"}</span>
      </div>
      <span style={{ fontSize: 10, color: "var(--green)", fontWeight: 600, letterSpacing: "0.3px" }}>{def.tag}</span>
      <span style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.4 }}>{def.description}</span>
      <div style={{ display: "flex", gap: 12, marginTop: 2 }} title="Reference figures at default settings — actual values depend on the config you choose below">
        <Stat label="VRAM" value={def.vram} />
        <Stat label="Params" value={def.params} />
      </div>
    </button>
  );
}

interface ArchSelectorProps {
  selected: Architecture;
  onSelect: (a: Architecture) => void;
}

export function ArchSelector({ selected, onSelect }: ArchSelectorProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 220 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
        Architecture
      </span>
      {ARCH_DEFS.map((def) => (
        <ArchCard key={def.id} def={def} active={selected === def.id} onClick={() => onSelect(def.id)} />
      ))}
    </div>
  );
}
