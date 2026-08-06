import { Field } from "../../../components/ui/Field";
import { Toggle } from "../../../components/ui/Toggle";
import { Btn } from "../../../components/ui/Btn";
import { InlineAlert } from "../../../components/ui/InlineAlert";
import type { MergePlan } from "./mergePlanner";

interface MergeSettingsProps {
  plan: MergePlan;
  datasetsDir: string;
  customName: string;
  onCustomNameChange: (n: string) => void;
  keepSources: boolean;
  onKeepSourcesChange: (v: boolean) => void;
  scanning: boolean;
  isMerging: boolean;
  hasSelection: boolean;
  onPreview: () => void;
  onMerge: () => void;
}

export function MergeSettings({
  plan,
  datasetsDir,
  customName,
  onCustomNameChange,
  keepSources,
  onKeepSourcesChange,
  scanning,
  isMerging,
  hasSelection,
  onPreview,
  onMerge,
}: MergeSettingsProps) {
  const { groups, totalSelected, totalPairs } = plan;
  const multiScale = groups.length > 1;
  const showNameWarning = plan.warnings.customNameWithMultipleScales;
  const busy = scanning || isMerging;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <Field label="Output location">
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--muted)",
            background: "var(--bg3)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            padding: "5px 10px",
            width: "100%",
            maxWidth: 480,
            boxSizing: "border-box",
            overflowWrap: "anywhere",
          }}
        >
          {datasetsDir || "…"}
        </div>
        <span style={{ fontSize: 10, color: "var(--dim)", marginTop: 2 }}>
          {"Merged datasets are created inside your project's datasets folder as merged-x{N}, one folder per scale group."}
        </span>
      </Field>

      <Field label="Output name" hint="optional">
        <input
          value={customName}
          onChange={(e) => onCustomNameChange(e.target.value)}
          placeholder="Leave empty for auto-naming (merged-xN)"
          aria-label="Output name"
          style={{
            width: "100%",
            maxWidth: 480,
            background: "var(--bg3)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            color: "var(--text)",
            fontSize: 11,
            padding: "5px 10px",
            fontFamily: "var(--font-mono)",
            outline: "none",
          }}
        />
        {showNameWarning && (
          <div style={{ marginTop: 4 }}>
            <InlineAlert tone="amber">
              Custom name with multiple scale groups will raise an error. Use the
              scale filter or leave name empty.
            </InlineAlert>
          </div>
        )}
      </Field>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <Toggle on={keepSources} onChange={() => onKeepSourcesChange(!keepSources)} />
        <label
          style={{ fontSize: 11, color: "var(--muted)", cursor: "pointer" }}
          onClick={() => onKeepSourcesChange(!keepSources)}
        >
          Keep source datasets (don't delete after merge)
        </label>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 10px",
          background: "var(--bg2)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)",
          fontSize: 11,
          flexWrap: "wrap",
        }}
      >
        <span style={{ color: "var(--text)" }}>
          {totalSelected} dataset{totalSelected !== 1 ? "s" : ""}
        </span>
        <span style={{ color: "var(--dim)" }}>·</span>
        <span style={{ color: "var(--text)" }}>{totalPairs.toLocaleString()} pairs</span>
        <span style={{ color: "var(--dim)" }}>·</span>
        <span style={{ color: multiScale ? "var(--amber)" : "var(--muted)" }}>
          {groups.length} merged output{groups.length !== 1 ? "s" : ""} (one per
          scale)
        </span>
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <Btn onClick={onPreview} disabled={!hasSelection || busy}>
          Preview Merge
        </Btn>
        <Btn variant="solid" onClick={onMerge} disabled={!hasSelection || busy}>
          {isMerging ? "Merging…" : "Execute Merge"}
        </Btn>
      </div>
    </div>
  );
}
