import { PBar } from "../../../components/ui/PBar";
import { useDatasetStore } from "../../../store/datasetStore";

export function MergeProgress() {
  const progressSteps = useDatasetStore((s) => s.progressSteps);
  const active = [...progressSteps].reverse().find((st) => st.status === "active");

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        padding: "8px 10px",
        background: "var(--bg2)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
        <span style={{ fontSize: 11, color: "var(--text)", fontWeight: 600 }}>
          {active ? active.desc : "Merging datasets…"}
        </span>
        {active && active.total != null && active.total > 0 && (
          <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
            {Math.round((active.current / active.total) * 100)}%
          </span>
        )}
      </div>
      <PBar
        value={active?.current ?? 0}
        max={active?.total ?? (active?.current || 1)}
        color="var(--green)"
        height={5}
      />
    </div>
  );
}
