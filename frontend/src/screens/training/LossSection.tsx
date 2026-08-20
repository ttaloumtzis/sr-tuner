import { useState, useRef, useEffect, type CSSProperties } from "react";
import ReactDOM from "react-dom";
import { LOSS_TYPE_OPTIONS, type LossType } from "../../lib/api-types";
import { Field } from "../../components/ui/Field";
import { Dropdown } from "../../components/ui/Dropdown";
import { GroupLabel } from "../../components/ui/GroupLabel";
import { LabelWithHint } from "../../components/ui/LabelWithHint";
import { NumberInput } from "../../components/ui/NumberInput";
import { useRunConfigStore } from "../../store/runConfigStore";

const VGG_LAYERS = [
  "relu1_1","relu1_2","relu2_1","relu2_2",
  "relu3_1","relu3_2","relu3_3","relu3_4",
  "relu4_1","relu4_2","relu4_3","relu4_4",
  "relu5_1","relu5_2","relu5_3","relu5_4",
];

interface WeightInputProps {
  value: number;
  onChange: (v: number) => void;
}
function WeightInput({ value, onChange }: WeightInputProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (Number.isFinite(value) ? String(value) : "");
  const sliderVal = Math.min(1, Math.max(0.01, Number.isFinite(value) ? value : 0.01));
  return (
    <div className="ts-row-8">
      <input
        type="range"
        min={0.01}
        max={1}
        step={0.01}
        value={sliderVal}
        onChange={(e) => {
          setDraft(null);
          onChange(Number(e.target.value));
        }}
        className="ts-range"
      />
      <NumberInput
        compact
        value={shown}
        min={0}
        step={0.01}
        onChange={(n, raw) => {
          setDraft(raw ?? String(n));
          if (Number.isFinite(n)) onChange(n);
        }}
        onBlur={() => setDraft(null)}
        style={{ width: 76, flexShrink: 0 }}
      />
    </div>
  );
}

interface LayerMultiSelectProps {
  layers: string[];
  onChange: (layers: string[]) => void;
}
function LayerMultiSelect({ layers, onChange }: LayerMultiSelectProps) {
  const [open, setOpen] = useState(false);
  const [menuStyle, setMenuStyle] = useState<CSSProperties>({});
  const triggerRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (triggerRef.current && !triggerRef.current.contains(target) &&
          menuRef.current && !menuRef.current.contains(target)) {
        setOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("click", handleClick);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("click", handleClick);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const handleOpen = () => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setMenuStyle({
      position: "fixed",
      top: Math.min(rect.bottom + 2, window.innerHeight - 250),
      left: rect.left,
      width: Math.max(rect.width, 200),
      zIndex: 2000,
    });
    setOpen((o) => !o);
  };

  const toggle = (layer: string) => {
    const next = layers.includes(layer)
      ? layers.filter((l) => l !== layer)
      : [...layers, layer];
    onChange(next.length > 0 ? next : layers);
  };

  const triggerBase: CSSProperties = {
    background: "var(--bg3)",
    border: `1px solid ${open ? "var(--green)" : "var(--border)"}`,
    borderRadius: "var(--radius-sm)",
    padding: "5px 8px",
    fontSize: 12,
    color: layers.length > 0 ? "var(--text)" : "var(--dim)",
    width: "100%",
    outline: "none",
    cursor: "pointer",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    userSelect: "none",
    boxSizing: "border-box",
    transition: "border-color 0.15s",
  };

  const menu = open
    ? ReactDOM.createPortal(
        <div ref={menuRef} style={{
          ...menuStyle,
          background: "var(--bg2)",
          border: "1px solid var(--border2)",
          borderRadius: "var(--radius-sm)",
          boxShadow: "0 8px 28px rgba(0,0,0,0.5)",
          maxHeight: 220,
          overflowY: "auto",
        }}>
          {VGG_LAYERS.map((layer) => {
            const active = layers.includes(layer);
            return (
              <div
                key={layer}
                onClick={() => toggle(layer)}
                className={`ts-ms-item${active ? " ts-ms-item-active" : ""}`}
              >
                <span className={`ts-ms-check${active ? " ts-ms-check-active" : ""}`}>
                  {active ? "✓" : ""}
                </span>
                {layer}
              </div>
            );
          })}
        </div>,
        document.body,
      )
    : null;

  return (
    <>
      <div ref={triggerRef} style={triggerBase} onClick={handleOpen}>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
          {layers.length === 0
            ? "Select layers…"
            : `${layers.length} layer${layers.length === 1 ? "" : "s"}`}
        </span>
        <span style={{
          color: "var(--muted)", marginLeft: 8, flexShrink: 0,
          transition: "transform 0.15s",
          transform: open ? "rotate(180deg)" : "none",
          fontSize: 12,
        }}>▾</span>
      </div>
      {menu}
    </>
  );
}

export function LossSection() {
  const s = useRunConfigStore();

  const totalLossWeight = Object.values(s.lossConfig).reduce((sum, e) => sum + (e.weight || 0), 0);

  return (
    <div className="ts-group">
      <GroupLabel>Losses</GroupLabel>
      <div className="ts-col-8">
        {Object.entries(s.lossConfig).map(([name, entry]) => {
          const typeOpt = LOSS_TYPE_OPTIONS.find((o) => o.value === entry.type);
          return (
            <div key={name} className="ts-loss-box">
              <div className="ts-loss-header">
                <span className="ts-loss-name">{name}</span>
                <div className="progress-bar ts-loss-track">
                  <div style={{
                    height: "100%",
                    width: `${totalLossWeight > 0 ? Math.min(100, (entry.weight / totalLossWeight) * 100) : 0}%`,
                    background: "var(--green)", borderRadius: 2, transition: "width 0.15s",
                  }} />
                </div>
                <span className="ts-pct">
                  {totalLossWeight > 0 ? `${Math.round((entry.weight / totalLossWeight) * 100)}%` : "—"}
                </span>
                {!(name === "pixel" && ["l1", "l2"].includes(entry.type)) && (
                  <button
                    onClick={() => s.removeLoss(name)}
                    className="ts-icon-btn-danger"
                    title="Remove loss"
                  >✕</button>
                )}
              </div>
              <div className="ts-loss-fields">
                <div className="ts-field-150">
                  <Field label="Type">
                    <div className="ts-relative">
                      <Dropdown
                        value={entry.type}
                        options={LOSS_TYPE_OPTIONS.map(o => ({ value: o.value, label: o.label }))}
                        onChange={(newType) => {
                          const opt = LOSS_TYPE_OPTIONS.find((o) => o.value === newType);
                          const next = { ...s.lossConfig };
                          next[name] = { type: newType as LossType, weight: entry.weight };
                          if (opt?.needsLayers) {
                            next[name].layers = newType === "vgg"
                              ? ["relu5_4"]
                              : ["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_2"];
                          } else {
                            delete next[name].layers;
                          }
                          s.setLossConfig(next);
                        }}
                      />
                    </div>
                  </Field>
                </div>
                <div className="ts-field-200">
                  <Field label="Weight">
                    <WeightInput value={entry.weight} onChange={(v) => s.setLossWeight(name, v)} />
                  </Field>
                </div>
              </div>
              {typeOpt?.needsLayers && entry.layers && (
                <div className="ts-mt-6">
                  <Field label={<LabelWithHint label="Layers"
                    hint={entry.type === "vgg"
                      ? "Which VGG19 feature layers to compare. Lower layers (relu1_1–2_2) capture edges & textures; higher layers (relu4_4–5_4) capture semantic content."
                      : "Compute Gram matrices on these VGG19 layers. Lower layers (relu1_1–2_2) capture fine texture patterns; higher layers (relu4_4–5_2) capture spatial layout."}
                  />}>
                    <LayerMultiSelect
                      layers={entry.layers}
                      onChange={(layers) => {
                        const next = { ...s.lossConfig };
                        next[name] = { ...next[name], layers };
                        s.setLossConfig(next);
                      }}
                    />
                  </Field>
                </div>
              )}
            </div>
          );
        })}
        <div className="ts-add-row">
          {LOSS_TYPE_OPTIONS
            .filter((opt) => !Object.values(s.lossConfig).some((e) => e.type === opt.value))
            .map((opt) => (
              <button
                key={opt.value}
                onClick={() => {
                  s.addLoss(opt.value);
                }}
                className="ts-add-btn"
              >
                + {opt.label}
              </button>
            ))}
        </div>
      </div>
    </div>
  );
}