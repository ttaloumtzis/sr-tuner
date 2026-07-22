import { useId } from "react";
import { Tooltip } from "../../components/ui/Tooltip";

export interface SliderField {
  type: "slider";
  kind: "int" | "float";
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface DropdownField {
  type: "dropdown";
  key: string;
  label: string;
  options: (number | string)[];
  default: number | string;
}

export interface TextField {
  type: "text";
  key: string;
  label: string;
  default?: unknown;
}

export type ConfigField = SliderField | DropdownField | TextField;

interface ConfigFieldRowProps {
  field: ConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}

/** Short explanations for the more jargon-heavy architecture knobs. Shown as a hover
 *  tooltip next to the field label so people tuning a model don't have to leave the app. */
const FIELD_HINTS: Partial<Record<string, string>> = {
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
  depths: "Comma-separated transformer block count per stage, e.g. 6,6,6,6,6,6 for six stages of six blocks.",
  num_heads: "Comma-separated attention head count per stage. Auto-derived from Embedding Dim.",
  rgb_mean: "Per-channel mean used to normalize inputs before training. Leave blank to use the dataset default.",
};

function FieldLabel({ text, fieldKey }: { text: string; fieldKey: string }) {
  const hint = FIELD_HINTS[fieldKey];
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>
      {text}
      {hint && <Tooltip text={hint} />}
    </span>
  );
}

export function ConfigFieldRow({ field, value, onChange }: ConfigFieldRowProps) {
  // Called unconditionally so the hook order stays stable regardless of field.type
  // (previously this was called only inside the "slider" branch, after two earlier
  // conditional returns — a rules-of-hooks violation).
  const sliderId = useId();

  if (field.type === "text") {
    const textValue = String(value ?? "");
    const placeholder =
      field.key === "rgb_mean"
        ? "0.4488, 0.4371, 0.4040"
        : field.key === "depths"
          ? "6,6,6,6,6,6"
          : "6,6,6,6,6,6";
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <FieldLabel text={field.label} fieldKey={field.key} />
        <input
          type="text"
          value={textValue}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          style={{
            background: "var(--bg2)", border: "1px solid var(--border)",
            borderRadius: 6, padding: "4px 8px", fontSize: 12, color: "var(--text)",
            outline: "none", fontFamily: "var(--font-mono)",
          }}
        />
      </label>
    );
  }
  if (field.type === "dropdown") {
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <FieldLabel text={field.label} fieldKey={field.key} />
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {field.options.map((opt) => {
            const active = value === opt;
            return (
              <button
                key={String(opt)}
                onClick={() => onChange(opt)}
                style={{
                  background: active ? "var(--green)" : "var(--bg3)",
                  border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
                  color: active ? "#0d0f11" : "var(--muted)",
                  fontSize: 11, fontWeight: active ? 600 : 400,
                  padding: "4px 12px", borderRadius: 8,
                  cursor: "pointer", transition: "var(--transition-fast)",
                  whiteSpace: "nowrap",
                }}
              >
                {String(opt)}
              </button>
            );
          })}
        </div>
      </label>
    );
  }
  if (field.type === "slider") {
    return (
      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <FieldLabel text={field.label} fieldKey={field.key} />
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            id={sliderId}
            type="range"
            min={field.min}
            max={field.max}
            step={field.step}
            value={Number(value)}
            onChange={(e) => {
              const raw = parseFloat(e.target.value);
              onChange(field.kind === "int" ? Math.round(raw) : raw);
            }}
            style={{ flex: 1, accentColor: "var(--green)", height: 4, cursor: "pointer" }}
          />
          <span style={{
            fontSize: 12, fontWeight: 600, color: "var(--text)",
            minWidth: 32, textAlign: "right", fontFamily: "var(--font-mono)",
          }}>
            {field.kind === "int" ? Number(value).toFixed(0) : Number(value).toFixed(1)}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--muted)" }}>
          <span>{field.min}</span>
          <span>{field.max}</span>
        </div>
      </label>
    );
  }
  return null;
}

/** Read-only, code-styled row used to display derived/computed values (e.g. the num_heads
 *  CSV auto-generated from Embedding Dim + Depths). Distinct from the shared `ui/InfoRow`,
 *  which is a plain label/value list row — this one has a monospace "code box" look and was
 *  previously also named `InfoRow`, which shadowed the shared component of the same name. */
export function CodeRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>{label}</span>
      <span style={{
        fontSize: 12, color: "var(--muted)", fontFamily: "var(--font-mono)",
        background: "var(--bg2)", border: "1px solid var(--border)",
        borderRadius: 6, padding: "4px 8px",
      }}>
        {value}
      </span>
    </div>
  );
}
