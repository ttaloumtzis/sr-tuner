import { useState, useMemo, useCallback } from "react";
import "./ScreenModelCreate.css";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { InfoRow } from "../../components/ui/InfoRow";
import { SubTabPill } from "../../components/ui/SubTabPill";
import { Tag } from "../../components/ui/Tag";
import { Field } from "../../components/ui/Field";
import { IconCpu, IconSliders, IconRocket } from "../../components/ui/icons";
import { useModelStore } from "../../store/modelStore";
import { createInstance } from "../../lib/api";
import { useToast } from "../../components/shell/ToastProvider";
import {
  getArch,
  getDefaultTemplate,
  getTemplateValues,
  matchTemplate,
  serializeValue,
  buildYaml,
  formatParamCount,
  formatWeightMB,
  type Architecture,
} from "../../lib/architectures";
import { ArchSelector } from "./ArchSelector";
import { ConfigFieldRow, CodeRow } from "./ConfigFieldRow";

function TemplateCard({
  tpl, active, recommended, onClick,
}: {
  tpl: { id: string; name: string; description: string; paramsM: number };
  active: boolean;
  recommended?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex", flexDirection: "column", gap: 4,
        padding: 12, borderRadius: "var(--radius-md)", cursor: "pointer",
        background: active ? "var(--green-dim)" : "var(--bg2)",
        border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
        transition: "var(--transition-fast)",
        textAlign: "left", position: "relative",
      }}
    >
      {recommended && (
        <span style={{
          position: "absolute", top: -6, right: 8,
          fontSize: 9, fontWeight: 700, textTransform: "uppercase",
          background: "var(--green)", color: "#0d0f11",
          padding: "0 6px", borderRadius: 4, letterSpacing: "0.3px",
        }}>
          Recommended
        </span>
      )}
      <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>{tpl.name}</span>
      <span style={{ fontSize: 11, color: "var(--muted)" }}>{tpl.description}</span>
      <span style={{
        fontSize: 11, fontWeight: 600, color: "var(--text)", fontFamily: "var(--font-mono)",
      }}>
        {formatParamCount(tpl.paramsM)} params
      </span>
    </button>
  );
}

function ConfigPanel({
  def, values, onChange,
}: {
  def: ReturnType<typeof getArch>;
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  const displayFields = def.fields.filter((f) => !def.derivedKeys.includes(f.key));
  const derivedFields = def.fields.filter((f) => def.derivedKeys.includes(f.key));

  return (
    <Panel title="Architecture Config" icon={<IconSliders size={13} />}>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="mc-group">
          <div className="mc-grid" style={{ "--mc-grid-min": "140px", "--mc-grid-max": "260px" } as React.CSSProperties}>
            {displayFields.map((field) => (
              <ConfigFieldRow
                key={field.key}
                field={field}
                value={values[field.key] ?? (field as any).default}
                onChange={(v) => onChange(field.key, v)}
              />
            ))}
          </div>
        </div>
        {derivedFields.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {derivedFields.map((field) => (
              <CodeRow key={field.key} label={field.label} value={String(values[field.key] ?? "")} />
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}

function YamlPanel({ yaml, copied, onCopy }: { yaml: string; copied: boolean; onCopy: () => void }) {
  return (
    <div style={{
      background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden",
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "6px 10px", borderBottom: "1px solid var(--border)",
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>YAML Preview</span>
        <Btn onClick={onCopy} variant="ghost" small>{copied ? "Copied!" : "Copy"}</Btn>
      </div>
      <pre style={{
        margin: 0, padding: 10, fontSize: 11, lineHeight: 1.5,
        fontFamily: "var(--font-mono)", overflow: "auto", maxHeight: 160,
        color: "var(--text)", whiteSpace: "pre",
      }} dangerouslySetInnerHTML={{ __html: syntaxHighlight(yaml) }} />
    </div>
  );
}

function syntaxHighlight(yaml: string): string {
  return yaml
    .split("\n")
    .map((line) => {
      if (line.startsWith("#")) {
        return `<span style="color:var(--muted);opacity:0.5">${line}</span>`;
      }
      if (line.includes(":")) {
        const [key, ...rest] = line.split(":");
        const val = rest.join(":").trim();
        return `<span style="color:var(--green)">${key}</span>: <span style="color:var(--text)">${val}</span>`;
      }
      if (line.startsWith("- ")) {
        return `<span style="color:var(--text)">${line}</span>`;
      }
      return line;
    })
    .join("\n");
}

export function ScreenModelCreate() {
  const arch = useModelStore((s) => s.architecture);
  const setArch = useModelStore((s) => s.setArchitecture);
  const setSubTab = useModelStore((s) => s.setSubTab);
  const { show } = useToast();

  const [innerTab, setInnerTab] = useState<"template" | "advanced">("template");
  const [configValues, setConfigValues] = useState<Record<string, unknown>>(() => {
    const def = getArch(arch);
    const defaultTpl = getDefaultTemplate(def);
    return { ...getTemplateValues(def, defaultTpl.id), scale: 4 };
  });
  const [modelNameInput, setModelNameInput] = useState("");
  const [copied, setCopied] = useState(false);

  const def = useMemo(() => getArch(arch), [arch]);
  const scaleOptions = useMemo(() => {
    const sf = def.fields.find((f) => f.key === "scale");
    return sf?.type === "dropdown" ? (sf.options as number[]) : [1, 2, 4, 8];
  }, [def]);

  const selectedTemplate = useMemo(() => matchTemplate(def, configValues), [def, configValues]);
  const paramsM = useMemo(() => def.estimateParams(configValues), [def, configValues]);
  const weightFp32MB = useMemo(() => formatWeightMB(paramsM), [paramsM]);
  const weightFp16MB = useMemo(() => (parseFloat(weightFp32MB) / 2).toFixed(1), [weightFp32MB]);
  const yaml = useMemo(() => buildYaml(def, configValues, modelNameInput), [def, configValues, modelNameInput]);

  const handleArchSelect = useCallback((newArch: Architecture) => {
    setArch(newArch);
    const newDef = getArch(newArch);
    const defaultTpl = getDefaultTemplate(newDef);
    setConfigValues({ ...getTemplateValues(newDef, defaultTpl.id), scale: 4 });
    setInnerTab("template");
  }, [setArch]);

  const handleTemplateSelect = useCallback((id: string) => {
    const vals = getTemplateValues(def, id);
    setConfigValues((prev) => ({ ...vals, scale: prev.scale ?? 4 }));
  }, [def]);

  const handleChange = useCallback((key: string, value: unknown) => {
    setConfigValues((prev) => {
      let next = { ...prev, [key]: value };
      const result = def.derive?.(key, value, next);
      if (result) next = { ...next, ...result };
      return next;
    });
  }, [def]);

  const handleCopyYaml = useCallback(async () => {
    await navigator.clipboard.writeText(yaml);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [yaml]);

  const handleCreateInstance = useCallback(async () => {
    if (!modelNameInput.trim()) {
      show("error", "Please enter a model name");
      return;
    }
    const config: Record<string, unknown> = {};
    for (const f of def.fields) {
      const raw = configValues[f.key] ?? (f as any).default;
      const val = serializeValue(def, f.key, raw);
      if (val !== undefined) config[f.key] = val;
    }
    try {
      await createInstance(modelNameInput.trim(), arch, config);
      show("success", `Model "${modelNameInput.trim()}" created`);
      setSubTab("view");
    } catch (e: any) {
      show("error", e?.message ?? "Failed to create model instance");
    }
  }, [modelNameInput, arch, def, configValues, show, setSubTab]);

  return (
    <div className="mc-layout">
      <div className="mc-main">
        <Panel title="Architecture" icon={<IconCpu size={13} />}>
          <ArchSelector selected={arch} onSelect={handleArchSelect} />
        </Panel>

        <div style={{ display: "flex", gap: 6 }}>
          <SubTabPill label="Templates" active={innerTab === "template"} onClick={() => setInnerTab("template")} />
          <SubTabPill label="Advanced" active={innerTab === "advanced"} onClick={() => setInnerTab("advanced")} />
          {selectedTemplate === null && innerTab === "template" && (
            <Tag color="amber">Custom config</Tag>
          )}
        </div>

        {innerTab === "template" ? (
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
            <div className="mc-group" style={{ marginBottom: 12 }}>
              <span style={{ fontSize: 9.5, color: "var(--dim)", textTransform: "uppercase", letterSpacing: "0.08em", fontFamily: "var(--font-sans)", fontWeight: 600 }}>Scale Factor</span>
              <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                {scaleOptions.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => handleChange("scale", opt)}
                    style={{
                      background: (configValues.scale as number) === opt ? "var(--green)" : "var(--bg3)",
                      border: `1px solid ${(configValues.scale as number) === opt ? "var(--green)" : "var(--border)"}`,
                      color: (configValues.scale as number) === opt ? "#0d0f11" : "var(--muted)",
                      fontSize: 11, fontWeight: (configValues.scale as number) === opt ? 600 : 400,
                      padding: "4px 12px", borderRadius: "var(--radius-sm)",
                      cursor: "pointer", transition: "var(--transition-fast)",
                    }}
                  >
                    {opt}x
                  </button>
                ))}
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: 8 }}>
              {def.templates.map((tpl) => (
                <TemplateCard
                  key={tpl.id}
                  tpl={tpl}
                  active={selectedTemplate === tpl.id}
                  recommended={tpl.recommended}
                  onClick={() => handleTemplateSelect(tpl.id)}
                />
              ))}
            </div>
          </div>
        ) : (
          <ConfigPanel
            def={def}
            values={configValues}
            onChange={handleChange}
          />
        )}
      </div>

      <div className="mc-sidebar">
        <Panel title="Model">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <Field label="Model Name">
              <input
                type="text"
                value={modelNameInput}
                placeholder="my_upscaler_v1"
                onChange={(e) => setModelNameInput(e.target.value)}
                style={{
                  background: "var(--bg3)", border: "1px solid var(--border)",
                  borderRadius: "var(--radius-sm)", padding: "6px 10px", fontSize: 13,
                  color: "var(--text)", outline: "none",
                  fontFamily: "var(--font-mono)",
                  transition: "border-color 0.15s",
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "var(--green)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
              />
            </Field>
            <Btn
              variant="solid"
              full
              onClick={handleCreateInstance}
              disabled={!modelNameInput.trim()}
              style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 7 }}
            >
              <IconRocket size={13} color="#0d0f11" /> Create Instance
            </Btn>
          </div>
        </Panel>

        <Panel title="Estimate">
          <InfoRow label="Parameters" value={formatParamCount(paramsM)} mono />
          <InfoRow label="Weights (f32)" value={`${weightFp32MB} MB`} mono />
          <InfoRow label="Weights (f16)" value={`${weightFp16MB} MB`} mono />
        </Panel>

        <YamlPanel yaml={yaml} copied={copied} onCopy={handleCopyYaml} />
      </div>
    </div>
  );
}