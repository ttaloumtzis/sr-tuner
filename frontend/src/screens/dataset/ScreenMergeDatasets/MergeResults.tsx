import { Panel } from "../../../components/ui/Panel";

interface MergeResultRow {
  scale: number;
  output_path: string;
  source_datasets: string[];
}

export function MergeResults({
  results,
  keepSources,
}: {
  results: MergeResultRow[];
  keepSources: boolean;
}) {
  return (
    <Panel title="Merge Results">
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {results.map((r) => (
          <div key={r.scale} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--green)" }}>
            <span>✓ Merged ×{r.scale}</span>
            <span style={{ fontFamily: "var(--font-mono)", color: "var(--muted)", fontSize: 10 }}>
              → {r.output_path}
            </span>
          </div>
        ))}
        {!keepSources && (
          <span style={{ fontSize: 10, color: "var(--amber)", fontStyle: "italic" }}>
            Source datasets have been removed.
          </span>
        )}
      </div>
    </Panel>
  );
}
