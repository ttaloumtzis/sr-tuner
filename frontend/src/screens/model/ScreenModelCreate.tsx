import { useState, useMemo, useCallback } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { useModelStore } from "../../store/modelStore";
import { createInstance } from "../../lib/api";
import { useToast } from "../../components/shell/ToastProvider";
import type { Architecture } from "../../lib/srproj";
import { ArchSelector, ARCH_DEFS } from "./ArchSelector";
import { ConfigFieldRow, InfoRow } from "./ConfigFieldRow";
import {
  getTemplateValues,
  getTemplateDefaultId,
  getSwinirTemplates,
  getRrdbTemplates,
  getNumHeads,
  generateNumHeadsCsv,
  estimateParams,
  formatParamCount,
  formatWeightMB,
  parseCSV,
  type ModelTemplateId,
  type TemplateDef,
} from "./templates";

function SubTabPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} role="tab" aria-selected={active}
      style={{
        background: active ? "var(--green)" : "var(--bg3)",
        border: `1px solid ${active ? "var(--green)" : "var(--border)"}`,
        color: active ? "#0d0f11" : "var(--muted)",
        fontSize: 11, fontWeight: active ? 600 : 400,
        padding: "4px 16px", borderRadius: 12,
        cursor: "pointer", transition: "var(--transition-fast)",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}

function SizeRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "2px 0" }}>
      <span style={{ fontSize: 11, color: "var(--muted)" }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)", fontFamily: "var(--font-mono)" }}>{value}</span>
    </div>
  );
}

function TemplateCard({ tpl, active, recommended, onClick }: {
  tpl: TemplateDef;
  active: boolean;
  recommended?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex", flexDirection: "column", gap: 4,
        padding: 12, borderRadius: 8, cursor: "pointer",
        background: active ? "var(--bg3)" : "var(--bg2)",
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
      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text)", fontFamily: "var(--font-mono)" }}>
        {formatParamCount(tpl.paramsM)} params
      </span>
    </button>
  );
}

function ConfigPanel({ fields, values, onChange, arch, innerTab }: {
  fields: import("./ConfigFieldRow").ConfigField[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
  arch: Architecture;
  innerTab: "template" | "advanced";
}) {
  if (innerTab === "template") return null;

  const isSwinir = arch === "swinir";
  const displayFields = isSwinir
    ? fields.filter(f => f.key !== "depths" && f.key !== "num_heads")
    : fields;

  const derivedHeads = isSwinir
    ? generateNumHeadsCsv(
        String(values.depths ?? "6,6,6,6,6,6"),
        getNumHeads(values.embed_dim as number),
      )
    : null;

  return (
    <Panel title="Architecture Config">
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {displayFields.map((field) => (
          <ConfigFieldRow
            key={field.key}
            field={field}
            value={values[field.key] ?? field.default}
            onChange={(v) => onChange(field.key, v)}
          />
        ))}
        {isSwinir && (
          <>
            <InfoRow label="Depths" value={String(values.depths ?? "")} />
            <InfoRow label="Num Heads" value={derivedHeads ?? ""} />
          </>
        )}
      </div>
    </Panel>
  );
}

function YamlPanel({ yaml, copied, onCopy }: { yaml: string; copied: boolean; onCopy: () => void }) {
  return (
    <div style={{
      background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 6, overflow: "hidden",
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

function buildYaml(values: Record<string, unknown>, name: string, arch: Architecture): string {
  const lines: string[] = [];
  lines.push(`# ${arch} model configuration`);
  lines.push(`name: ${name || arch}`);
  lines.push(`type: ${arch === "rrdb_esrgan" ? "rrdbnet" : arch}`);
  for (const [key, value] of Object.entries(values)) {
    if (key === "num_in_ch" || key === "num_out_ch") continue;
    if (key === "depths" || key === "num_heads") {
      const arr = parseCSV(String(value));
      lines.push(`${key}: [${arr.join(", ")}]`);
    } else if (key === "rgb_mean" && (value === null || value === "" || value === "null")) {
      continue;
    } else if (key === "rgb_mean") {
      const arr = Array.isArray(value) ? value : parseCSV(String(value));
      if (arr.length > 0) lines.push(`${key}: [${arr.join(", ")}]`);
    } else if (typeof value === "number") {
      lines.push(`${key}: ${value}`);
    } else {
      lines.push(`${key}: ${value}`);
    }
  }
  return lines.join("\n");
}

function serializeConfigValue(key: string, raw: unknown): unknown {
  if (key === "depths" || key === "num_heads") {
    const nums = parseCSV(String(raw ?? ""));
    return nums.length > 0 ? nums : raw;
  }
  if (key === "rgb_mean") {
    const s = String(raw ?? "").trim();
    if (!s || s.toLowerCase() === "null") return null;
    const nums = parseCSV(s);
    return nums.length > 0 ? nums : null;
  }
  return raw;
}

export function ScreenModelCreate() {
  const arch = useModelStore((s) => s.architecture);
  const setArch = useModelStore((s) => s.setArchitecture);
  const setSubTab = useModelStore((s) => s.setSubTab);
  const { show } = useToast();

  const [innerTab, setInnerTab] = useState<"template" | "advanced">("template");
  const [configValues, setConfigValues] = useState<Record<string, unknown>>(
    () => ({ ...getTemplateValues(arch, getTemplateDefaultId(arch)), scale: 4 }),
  );
  const [selectedTemplate, setSelectedTemplate] = useState<ModelTemplateId | "custom">(
    getTemplateDefaultId(arch),
  );
  const [modelNameInput, setModelNameInput] = useState("");
  const [copied, setCopied] = useState(false);

  const def = useMemo(() => ARCH_DEFS.find((d) => d.id === arch) ?? ARCH_DEFS[0], [arch]);

  const paramsM = useMemo(() => estimateParams(arch, configValues), [arch, configValues]);

  const weightFp32MB = useMemo(() => formatWeightMB(paramsM), [paramsM]);
  const weightFp16MB = useMemo(() => (parseFloat(weightFp32MB) / 2).toFixed(1), [weightFp32MB]);

  const templates = useMemo(
    () => (arch === "swinir" ? getSwinirTemplates() : getRrdbTemplates()),
    [arch],
  );

  const yaml = useMemo(() => buildYaml(configValues, modelNameInput, arch), [configValues, modelNameInput, arch]);

  const handleArchSelect = useCallback((newArch: Architecture) => {
    setArch(newArch);
    const vals = getTemplateValues(newArch, getTemplateDefaultId(newArch));
    setConfigValues({ ...vals, scale: 4 });
    setSelectedTemplate(getTemplateDefaultId(newArch));
    setInnerTab("template");
  }, [setArch]);

  const handleTemplateSelect = useCallback((id: ModelTemplateId) => {
    const vals = getTemplateValues(arch, id);
    setConfigValues((prev) => ({ ...vals, scale: prev.scale ?? 4 }));
    setSelectedTemplate(id);
  }, [arch]);

  const handleChange = useCallback((key: string, value: unknown) => {
    setConfigValues((prev) => {
      let next = { ...prev, [key]: value };
      if (arch === "swinir" && key === "embed_dim") {
        const heads = getNumHeads(value as number);
        const depthsCsv = String(prev.depths ?? "6,6,6,6,6,6");
        next.num_heads = generateNumHeadsCsv(depthsCsv, heads);
      }
      return next;
    });
  }, [arch]);

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
      config[f.key] = serializeConfigValue(f.key, configValues[f.key] ?? f.default);
    }
    try {
      await createInstance(modelNameInput.trim(), arch, config);
      show("success", `Model "${modelNameInput.trim()}" created`);
      setSubTab("view");
    } catch (e: any) {
      show("error", e?.message ?? "Failed to create model instance");
    }
  }, [modelNameInput, arch, def.fields, configValues, show, setSubTab]);

  const templateCards = (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: 8, paddingTop: 8 }}>
      {templates.map((tpl) => (
        <TemplateCard
          key={tpl.id}
          tpl={tpl}
          active={selectedTemplate === tpl.id}
          recommended={tpl.recommended}
          onClick={() => handleTemplateSelect(tpl.id)}
        />
      ))}
    </div>
  );

  return (
    <div style={{ display: "flex", gap: 16, flex: 1, minHeight: 0, overflow: "hidden" }}>
      <ArchSelector selected={arch} onSelect={handleArchSelect} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8, overflow: "hidden" }}>
        <div style={{ display: "flex", gap: 6 }}>
          <SubTabPill label="Templates" active={innerTab === "template"} onClick={() => setInnerTab("template")} />
          <SubTabPill label="Advanced" active={innerTab === "advanced"} onClick={() => setInnerTab("advanced")} />
        </div>
        <div style={{ display: "flex", gap: 16, flex: 1, minHeight: 0, overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto" }}>
            {innerTab === "template" ? (
              <div>
                <div style={{ marginBottom: 12 }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>Scale Factor</span>
                  <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                    {[1, 2, 4, 8].map((opt) => (
                      <button
                        key={opt}
                        onClick={() => handleChange("scale", opt)}
                        style={{
                          background: (configValues.scale as number) === opt ? "var(--green)" : "var(--bg3)",
                          border: `1px solid ${(configValues.scale as number) === opt ? "var(--green)" : "var(--border)"}`,
                          color: (configValues.scale as number) === opt ? "#0d0f11" : "var(--muted)",
                          fontSize: 11, fontWeight: (configValues.scale as number) === opt ? 600 : 400,
                          padding: "4px 12px", borderRadius: 8,
                          cursor: "pointer", transition: "var(--transition-fast)",
                        }}
                      >
                        {opt}x
                      </button>
                    ))}
                  </div>
                </div>
                {templateCards}
              </div>
            ) : (
              <ConfigPanel
                fields={def.fields}
                values={configValues}
                onChange={handleChange}
                arch={arch}
                innerTab={innerTab}
              />
            )}
          </div>
          <div style={{ width: 260, display: "flex", flexDirection: "column", gap: 8, flexShrink: 0, overflow: "hidden" }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)" }}>Model Name</span>
              <input
                type="text"
                value={modelNameInput}
                placeholder="my_upscaler_v1"
                onChange={(e) => setModelNameInput(e.target.value)}
                style={{
                  background: "var(--bg2)", border: "1px solid var(--border)",
                  borderRadius: 6, padding: "6px 10px", fontSize: 13,
                  color: "var(--text)", outline: "none",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </label>
            <Btn variant="solid" onClick={handleCreateInstance} disabled={!modelNameInput.trim()}>Create Instance</Btn>
            <div style={{ borderTop: "1px solid var(--border)", paddingTop: 6 }}>
              <SizeRow label="Parameters" value={formatParamCount(paramsM)} />
              <SizeRow label="Weights (f32)" value={`${weightFp32MB} MB`} />
              <SizeRow label="Weights (f16)" value={`${weightFp16MB} MB`} />
            </div>
            <YamlPanel yaml={yaml} copied={copied} onCopy={handleCopyYaml} />
          </div>
        </div>
      </div>
    </div>
  );
}
