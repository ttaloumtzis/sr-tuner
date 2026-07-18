// §10 Model Config Screen
// Tasks: 10.1–10.9

import { useState, useEffect, useCallback } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { Toggle } from "../../components/ui/Toggle";
import { PathInput } from "../../components/ui/PathInput";
import { Dropdown } from "../../components/ui/Dropdown";
import { useModelStore } from "../../store/modelStore";
import { useProjectStore } from "../../store/projectStore";
import { useUiStore } from "../../store/uiStore";
import type { Architecture, AugmentationConfig } from "../../lib/srproj";
import type { Hyperparameters, LossWeights } from "../../store/modelStore";

// ── Architecture definitions ──────────────────────────────────────────────

interface ArchDef {
  id: Architecture;
  tag: string;
  description: string;
  vram: string;
  params: string;
  defaults: Hyperparameters;
}

const ARCH_DEFS: ArchDef[] = [
  {
    id: "rrdb_esrgan",
    tag: "GAN · Perceptual",
    description: "Best visual quality for textures and fine detail. Uses a discriminator network.",
    vram: "~8 GB",
    params: "~16M",
    defaults: {
      scale: 4,
      lrScheduler: "cosine",
      optimizer: "Adam",
      learningRate: 1e-4,
      batchSize: 4,
      patchSize: 192,
      totalIter: 300000,
    },
  },
  {
    id: "swinir",
    tag: "Transformer · PSNR",
    description: "Swin Transformer backbone. Best PSNR/SSIM scores, no adversarial training.",
    vram: "~6 GB",
    params: "~11M",
    defaults: {
      scale: 4,
      lrScheduler: "cosine",
      optimizer: "AdamW",
      learningRate: 2e-4,
      batchSize: 8,
      patchSize: 64,
      totalIter: 500000,
    },
  },
];

const AUG_DEFS: { key: keyof AugmentationConfig; label: string; hint: string }[] = [
  { key: "horizontal_flip", label: "Horizontal Flip", hint: "Mirror images left/right" },
  { key: "vertical_flip", label: "Vertical Flip", hint: "Mirror images top/bottom" },
  { key: "rotation_90", label: "Rotation 90°", hint: "Rotate by multiples of 90°" },
  { key: "mixup", label: "MixUp", hint: "Blend pairs of training samples" },
  { key: "color_jitter", label: "Color Jitter", hint: "Random brightness/contrast/saturation" },
  { key: "random_degradation", label: "Random Degradation", hint: "Simulate varied compression artifacts" },
  { key: "gaussian_blur", label: "Gaussian Blur", hint: "Apply random blur kernel" },
  { key: "noise_injection", label: "Noise Injection", hint: "Add Gaussian or Poisson noise" },
];

// ── §10.1 Layout ──────────────────────────────────────────────────────────

export function ScreenModelConfig() {
  const arch = useModelStore((s) => s.architecture);
  const hp = useModelStore((s) => s.hyperparameters);
  const lw = useModelStore((s) => s.lossWeights);
  const aug = useModelStore((s) => s.augmentations);
  const pretrainedPath = useModelStore((s) => s.pretrainedPath);
  const setArchitecture = useModelStore((s) => s.setArchitecture);
  const setHyperparameters = useModelStore((s) => s.setHyperparameters);
  const setLossWeights = useModelStore((s) => s.setLossWeights);
  const setAugmentations = useModelStore((s) => s.setAugmentations);
  const setPretrainedPath = useModelStore((s) => s.setPretrainedPath);
  const resetHyperparameters = useModelStore((s) => s.resetHyperparameters);
  const addToast = useUiStore((s) => s.addToast);

  const [yaml, setYaml] = useState("");
  const [copied, setCopied] = useState(false);

  const regenerateYaml = useCallback(() => {
    setYaml("");
  }, [arch, hp, lw, aug, pretrainedPath]);

  useEffect(() => {
    regenerateYaml();
  }, [regenerateYaml]);

  const handleArchSelect = (a: Architecture) => {
    const def = ARCH_DEFS.find((d) => d.id === a);
    if (def) {
      setArchitecture(a);
      setHyperparameters(def.defaults);
    }
  };

  const handleCopyYaml = async () => {
    try {
      await navigator.clipboard.writeText(yaml);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard not available
    }
  };

  const handleSaveModel = async () => {
    if (!modelNameInput) return;
    const proj = useProjectStore.getState().project;
    if (!proj) {
      addToast("No project open", "error");
      return;
    }
    const newModel = {
      id: crypto.randomUUID(),
      name: modelNameInput,
      architecture: arch,
      hyperparameters: {
        scale: hp.scale,
        batch_size: hp.batchSize,
        patch_size: hp.patchSize,
        learning_rate: hp.learningRate,
        optimizer: hp.optimizer,
        lr_scheduler: hp.lrScheduler,
        total_iter: hp.totalIter,
        augmentations: aug,
      },
      created_at: new Date().toISOString(),
    };
    const updated = { ...proj, models: [...(proj.models || []), newModel] };
    useProjectStore.setState({ project: updated });
    addToast(`Model "${modelNameInput}" saved`, "success");
    setModelNameInput("");
  };

  const [modelNameInput, setModelNameInput] = useState("");

  return (
    <div
      style={{
        display: "flex",
        flex: 1,
        minHeight: 0,
        overflow: "hidden",
        background: "var(--bg0)",
      }}
    >
      {/* §10.2 Architecture selector — 230px left column */}
      <ArchSelector selected={arch} onSelect={handleArchSelect} />

      {/* §10.3–10.7 Center: hyperparams + loss + aug + pretrained */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          padding: "10px 8px",
          overflow: "auto",
          minWidth: 0,
        }}
      >
        <HyperparamsPanel hp={hp} onChange={setHyperparameters} onReset={resetHyperparameters} />
        <LossWeightsPanel lw={lw} onChange={setLossWeights} />
        <AugmentationPanel aug={aug} onChange={setAugmentations} />
        <PretrainedPanel path={pretrainedPath} onChange={setPretrainedPath} />
      </div>

      {/* §10.6 YAML preview — 240px right column */}
      <div style={{ width: 240, flexShrink: 0, borderLeft: "1px solid var(--border)", display: "flex", flexDirection: "column", padding: "0 8px", gap: 8 }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            value={modelNameInput}
            onChange={(e) => setModelNameInput(e.target.value)}
            placeholder="Model name"
            style={{ flex: 1, background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 11, padding: "4px 6px", borderRadius: "var(--radius-sm)" }}
          />
          <Btn variant="solid" onClick={handleSaveModel} disabled={!modelNameInput}>Save</Btn>
        </div>
        <YamlPanel yaml={yaml} copied={copied} onCopy={handleCopyYaml} />
      </div>
    </div>
  );
}

// ── §10.2 Architecture cards ──────────────────────────────────────────────

interface ArchSelectorProps {
  selected: Architecture;
  onSelect: (a: Architecture) => void;
}

function ArchSelector({ selected, onSelect }: ArchSelectorProps) {
  return (
    <div
      style={{
        width: 230,
        flexShrink: 0,
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        gap: 0,
        overflow: "auto",
        background: "var(--bg1)",
      }}
    >
      <div
        style={{
          padding: "8px 12px 6px",
          borderBottom: "1px solid var(--border)",
          fontSize: 11,
          color: "var(--text)",
          fontWeight: 500,
          flexShrink: 0,
        }}
      >
        Architecture
      </div>
      <div style={{ display: "flex", flexDirection: "column", padding: 8, gap: 6 }}>
        {ARCH_DEFS.map((def) => (
          <ArchCard
            key={def.id}
            def={def}
            active={selected === def.id}
            onClick={() => onSelect(def.id)}
          />
        ))}
      </div>
    </div>
  );
}

interface ArchCardProps {
  def: ArchDef;
  active: boolean;
  onClick: () => void;
}

function ArchCard({ def, active, onClick }: ArchCardProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: active ? "#1a3d2a" : hovered ? "#2a2d32" : "#1a1d21",
        border: `1px solid ${active ? "#4dba7f" : hovered ? "#5d6470" : "#3a3e46"}`,
        borderRadius: 6,
        padding: "10px 12px",
        cursor: "pointer",
        transition: "0.15s",
        display: "flex",
        flexDirection: "column",
        gap: 5,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "space-between" }}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: active ? "#4dba7f" : "#dde3ea",
          }}
        >
          {def.id}
        </span>
        {active && (
          <span style={{ fontSize: 8, color: "var(--green)" }}>●</span>
        )}
      </div>
      <span
        style={{
          fontSize: 9,
          color: active ? "var(--green)" : "var(--accent)",
          fontFamily: "var(--font-mono)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        {def.tag}
      </span>
      <span style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>
        {def.description}
      </span>
      <div style={{ display: "flex", gap: 10, marginTop: 2 }}>
        <Stat label="VRAM" value={def.vram} />
        <Stat label="Params" value={def.params} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      <span style={{ fontSize: 9, color: "var(--dim)", textTransform: "uppercase" }}>{label}</span>
      <span style={{ fontSize: 10, color: "var(--text)", fontFamily: "var(--font-mono)" }}>{value}</span>
    </div>
  );
}

// ── §10.3 Hyperparameters panel ───────────────────────────────────────────

interface HyperparamsPanelProps {
  hp: Hyperparameters;
  onChange: (hp: Partial<Hyperparameters>) => void;
  onReset: () => void;
}

function HyperparamsPanel({ hp, onChange, onReset }: HyperparamsPanelProps) {
  const handleLr = (val: string) => {
    const n = parseFloat(val);
    if (!isNaN(n)) onChange({ learningRate: n });
  };

  return (
    <Panel
      title="Hyperparameters"
      actions={
        /* §10.9 Reset to defaults */
        <Btn small variant="ghost" onClick={onReset}>
          Reset to defaults
        </Btn>
      }
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "10px 16px",
        }}
      >
        <LabeledField label="Scale Factor">
          <Dropdown
            value={String(hp.scale)}
            options={["2", "4", "8"]}
            onChange={(v) => onChange({ scale: parseInt(v) })}
          />
        </LabeledField>

        <LabeledField label="LR Scheduler">
          <Dropdown
            value={hp.lrScheduler}
            options={[
              { value: "cosine", label: "Cosine Annealing" },
              { value: "multistep", label: "MultiStep LR" },
              { value: "plateau", label: "Reduce on Plateau" },
            ]}
            onChange={(v) => onChange({ lrScheduler: v })}
          />
        </LabeledField>

        <LabeledField label="Optimizer">
          <Dropdown
            value={hp.optimizer}
            options={["Adam", "AdamW", "SGD"]}
            onChange={(v) => onChange({ optimizer: v })}
          />
        </LabeledField>

        <LabeledField label="Learning Rate">
          <input
            type="number"
            step="0.00001"
            min="0.000001"
            max="0.01"
            value={hp.learningRate}
            onChange={(e) => handleLr(e.target.value)}
            style={numInputStyle}
          />
        </LabeledField>

        <LabeledField label="Total Iterations">
          <input
            type="number"
            step="10000"
            min="10000"
            value={hp.totalIter}
            onChange={(e) => onChange({ totalIter: parseInt(e.target.value) || hp.totalIter })}
            style={numInputStyle}
          />
        </LabeledField>

        <LabeledField label="Batch Size">
          <input
            type="number"
            step="1"
            min="1"
            max="64"
            value={hp.batchSize}
            onChange={(e) => onChange({ batchSize: parseInt(e.target.value) || hp.batchSize })}
            style={numInputStyle}
          />
        </LabeledField>

        <LabeledField label="Patch Size (px)">
          <input
            type="number"
            step="8"
            min="32"
            max="512"
            value={hp.patchSize}
            onChange={(e) => onChange({ patchSize: parseInt(e.target.value) || hp.patchSize })}
            style={numInputStyle}
          />
        </LabeledField>
      </div>
    </Panel>
  );
}

// ── §10.4 Loss weights ────────────────────────────────────────────────────

interface LossWeightsPanelProps {
  lw: LossWeights;
  onChange: (lw: Partial<LossWeights>) => void;
}

function LossWeightsPanel({ lw, onChange }: LossWeightsPanelProps) {
  return (
    <Panel title="Loss Weights">
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 12,
        }}
      >
        <LossField
          label="Pixel L1"
          value={lw.pixel}
          onChange={(v) => onChange({ pixel: v })}
          color="var(--text)"
        />
        <LossField
          label="Perceptual"
          value={lw.perceptual}
          onChange={(v) => onChange({ perceptual: v })}
          color="var(--blue)"
        />
        <LossField
          label="Adversarial"
          value={lw.adversarial}
          onChange={(v) => onChange({ adversarial: v })}
          color="var(--purple, #a855f7)"
        />
      </div>
    </Panel>
  );
}

interface LossFieldProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  color: string;
}

function LossField({ label, value, onChange, color }: LossFieldProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 10, color, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.04em" }}>
        {label}
      </span>
      <input
        type="number"
        step="0.1"
        min="0"
        max="10"
        value={value}
        onChange={(e) => {
          const n = parseFloat(e.target.value);
          if (!isNaN(n)) onChange(n);
        }}
        style={{ ...numInputStyle, borderColor: color === "var(--text)" ? undefined : color }}
      />
    </div>
  );
}

// ── §10.5 Augmentation toggles ────────────────────────────────────────────

interface AugmentationPanelProps {
  aug: AugmentationConfig;
  onChange: (aug: Partial<AugmentationConfig>) => void;
}

function AugmentationPanel({ aug, onChange }: AugmentationPanelProps) {
  return (
    <Panel title="Data Augmentation">
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "6px 20px",
        }}
      >
        {AUG_DEFS.map(({ key, label, hint }) => (
          <div
            key={key}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              minWidth: "calc(50% - 10px)",
              padding: "4px 0",
            }}
          >
            <Toggle
              on={aug[key]}
              onChange={() => onChange({ [key]: !aug[key] })}
            />
            <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <span style={{ fontSize: 11, color: "var(--text)" }}>{label}</span>
              <span style={{ fontSize: 9, color: "var(--dim)" }}>{hint}</span>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

// ── §10.7 Pretrained weights panel ────────────────────────────────────────

interface PretrainedPanelProps {
  path: string | null;
  onChange: (path: string | null) => void;
}

function PretrainedPanel({ path, onChange }: PretrainedPanelProps) {
  return (
    <Panel title="Pretrained Weights">
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.4 }}>
          Optional: initialize from a pre-trained .pth checkpoint. Compatibility is validated at training start.
        </span>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <PathInput
              value={path ?? ""}
              onChange={onChange}
              browseTitle="Select Pretrained Weights"
              mono
              placeholder="None — train from scratch"
            />
          </div>
          {path && (
            <Btn
              variant="ghost"
              small
              onClick={() => onChange(null)}
              style={{ flexShrink: 0 }}
            >
              Clear
            </Btn>
          )}
        </div>
      </div>
    </Panel>
  );
}

// ── §10.6 YAML preview panel ──────────────────────────────────────────────

interface YamlPanelProps {
  yaml: string;
  copied: boolean;
  onCopy: () => void;
}

function YamlPanel({ yaml, copied, onCopy }: YamlPanelProps) {
  return (
    <div
      style={{
        width: 240,
        flexShrink: 0,
        borderLeft: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        background: "var(--bg1)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "7px 12px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: 11, color: "var(--text)", fontWeight: 500 }}>
          YAML Preview
        </span>
        <Btn small variant="ghost" onClick={onCopy}>
          {copied ? "Copied!" : "Copy"}
        </Btn>
      </div>
      <pre
        style={{
          flex: 1,
          overflow: "auto",
          margin: 0,
          padding: "10px 12px",
          fontSize: 9.5,
          lineHeight: 1.6,
          fontFamily: "var(--font-mono)",
          color: "var(--muted)",
          whiteSpace: "pre",
          background: "transparent",
        }}
        dangerouslySetInnerHTML={{ __html: syntaxHighlight(yaml) }}
      />
    </div>
  );
}

// Simple YAML syntax highlighter (no external library)
function syntaxHighlight(yaml: string): string {
  return yaml
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .split("\n")
    .map((line) => {
      if (/^\s*#/.test(line)) {
        return `<span style="color:var(--dim)">${line}</span>`;
      }
      const keyMatch = line.match(/^(\s*)([^:]+)(:)(.*)$/);
      if (keyMatch) {
        const [, indent, key, colon, rest] = keyMatch;
        const coloredKey = `<span style="color:var(--green)">${key}</span>`;
        const coloredRest = rest
          ? `<span style="color:var(--text)">${rest}</span>`
          : "";
        return `${indent}${coloredKey}${colon}${coloredRest}`;
      }
      if (/^\s*- /.test(line)) {
        return `<span style="color:var(--text)">${line}</span>`;
      }
      return line;
    })
    .join("\n");
}

// ── Helpers ────────────────────────────────────────────────────────────────

interface LabeledFieldProps {
  label: string;
  children: React.ReactNode;
}

function LabeledField({ label, children }: LabeledFieldProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span
        style={{
          fontSize: 10,
          color: "var(--muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          fontWeight: 500,
        }}
      >
        {label}
      </span>
      {children}
    </div>
  );
}

const numInputStyle: React.CSSProperties = {
  background: "var(--bg3)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)",
  padding: "6px 10px",
  fontSize: 12,
  color: "var(--text)",
  width: "100%",
  outline: "none",
  fontFamily: "var(--font-mono)",
  boxSizing: "border-box",
};
