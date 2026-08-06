import { Panel } from "../../../components/ui/Panel";
import { InlineAlert } from "../../../components/ui/InlineAlert";
import type { MergePlan } from "./mergePlanner";

export function MergePreview({ plan }: { plan: MergePlan }) {
  const existing = plan.warnings.existingTargets;

  return (
    <Panel title="Merge Preview">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {existing.length > 0 && (
          <InlineAlert tone="red">
            Output already exists: {existing.join(", ")}. The merge will fail
            unless those directories are removed or a different output is used.
          </InlineAlert>
        )}
        {plan.groups.map((g) => (
          <div
            key={g.scale}
            style={{
              background: "var(--bg2)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              padding: "8px 10px",
              display: "flex",
              flexDirection: "column",
              gap: 4,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "var(--green)" }}>
                Merge ×{g.scale}
              </span>
              <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                {g.totalPairs.toLocaleString()} pairs total
              </span>
            </div>
            <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
              → {g.outputPath}
            </span>
            <span style={{ fontSize: 10, color: "var(--muted)" }}>
              Sources: {g.sources.map((d) => d.name).join(", ")}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
