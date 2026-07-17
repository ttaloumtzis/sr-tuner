// §23.6 — Run Comparison Panel
// Metadata table + overlaid loss and PSNR SVG line charts per run.
// Uses §22.1 downsample utility (adapted for number arrays).

import { useEffect } from "react";
import { useProjectStore } from "../../store/projectStore";
import { useTrainingStore } from "../../store/trainingStore";
import { useUiStore } from "../../store/uiStore";
import { sendToSidecar } from "../../lib/ipc";
import { RunComparisonTable } from "./RunComparisonTable";

const RUN_COLORS = [
  "var(--green)",
  "var(--blue)",
  "var(--amber)",
  "var(--purple)",
  "var(--cyan)",
  "var(--red)",
];

function downsampleArr(arr: number[], maxPoints = 200): number[] {
  if (arr.length <= maxPoints) return arr;
  const stride = Math.ceil(arr.length / maxPoints);
  const result: number[] = [];
  for (let i = 0; i < arr.length; i += stride) result.push(arr[i]);
  const last = arr[arr.length - 1];
  if (result[result.length - 1] !== last) result.push(last);
  return result;
}

// ── Overlaid SVG chart ────────────────────────────────────────────────────

interface SeriesSpec {
  label: string;
  data: number[];
  color: string;
}

function OverlaidChart({ title, seriesList }: { title: string; seriesList: SeriesSpec[] }) {
  const W = 380; const H = 140;

  const allVals = seriesList.flatMap((s) => s.data);
  const min = allVals.length ? Math.min(...allVals) : 0;
  const max = allVals.length ? Math.max(...allVals) : 1;
  const range = max - min || 1;

  function toPolylinePoints(data: number[]): string {
    return data
      .map((v, i) => {
        const x = (i / Math.max(data.length - 1, 1)) * W;
        const y = H - ((v - min) / range) * H;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{
        fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)",
        padding: "8px 10px 4px",
      }}>
        {title}
      </div>
      <div style={{ display: "flex", gap: 10, padding: "0 10px 4px", flexWrap: "wrap" }}>
        {seriesList.map((s) => (
          <span key={s.label} style={{
            fontSize: 9, color: s.color, fontFamily: "var(--font-mono)",
          }}>
            ■ {s.label}
          </span>
        ))}
      </div>
      <div style={{ padding: "0 8px" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
          {[H * 0.25, H * 0.5, H * 0.75].map((y) => (
            <line key={y} x1={0} y1={y} x2={W} y2={y}
              stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3 3" />
          ))}
          {seriesList.map((s) =>
            s.data.length > 1 ? (
              <polyline
                key={s.label}
                points={toPolylinePoints(s.data)}
                fill="none"
                stroke={s.color}
                strokeWidth={1.5}
                strokeOpacity={0.85}
              />
            ) : null
          )}
        </svg>
      </div>
    </div>
  );
}

// ── RunComparisonPanel ────────────────────────────────────────────────────

interface RunComparisonPanelProps {
  runIds: string[];
  onClose: () => void;
}

export function RunComparisonPanel({ runIds, onClose }: RunComparisonPanelProps) {
  const project          = useProjectStore((s) => s.project);
  const runHistories     = useTrainingStore((s) => s.runHistories);
  const lossHistory      = useTrainingStore((s) => s.lossHistory);
  const psnrHistory      = useTrainingStore((s) => s.psnrHistory);
  const activeRunId      = useTrainingStore((s) => s.activeTrainingRunId);
  const historiesPending = useUiStore((s) => s.comparisonHistoriesPending);
  const markPending      = useUiStore((s) => s.markComparisonHistoryPending);

  const runs = (project?.runs ?? []).filter((r) => runIds.includes(r.run_id));

  // §22.2 — request history for any selected run not yet cached
  useEffect(() => {
    for (const run of runs) {
      if (
        run.run_id !== activeRunId &&
        !runHistories[run.run_id] &&
        !historiesPending.has(run.run_id)
      ) {
        markPending(run.run_id);
        sendToSidecar({
          type: "run.history.request",
          run_id: run.run_id,
          log_dir: run.paths.log_dir,
        }).catch(() => {});
      }
    }
    // re-run only when the set of selected run IDs changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runIds.join(",")]);

  const lossSeries: SeriesSpec[] = runs.map((run, i) => {
    const history =
      run.run_id === activeRunId
        ? lossHistory
        : (runHistories[run.run_id]?.gLossHistory ?? []);
    return {
      label: run.name,
      data: downsampleArr(history),
      color: RUN_COLORS[i % RUN_COLORS.length],
    };
  });

  const psnrSeries: SeriesSpec[] = runs.map((run, i) => {
    const history =
      run.run_id === activeRunId
        ? psnrHistory
        : (runHistories[run.run_id]?.psnrHistory ?? []);
    return {
      label: run.name,
      data: downsampleArr(history),
      color: RUN_COLORS[i % RUN_COLORS.length],
    };
  });

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      width: "100%", height: "100%", overflow: "hidden", background: "var(--bg0)",
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 14px", background: "var(--bg1)",
        borderBottom: "1px solid var(--border)", flexShrink: 0,
      }}>
        <span style={{
          fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)",
        }}>
          Compare Runs ({runs.length})
        </span>
        <button
          onClick={onClose}
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--muted)", fontSize: 14, padding: "0 4px", lineHeight: 1,
          }}
        >
          ✕
        </button>
      </div>

      {/* Scrollable content */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {/* Metadata table */}
        <div style={{ padding: "12px 14px 10px" }}>
          <RunComparisonTable runs={runs} />
        </div>

        {/* Overlaid charts */}
        <div style={{
          display: "flex", gap: 1,
          margin: "0 14px 14px",
          border: "1px solid var(--border)",
          background: "var(--border)",
          borderRadius: "var(--radius-sm)", overflow: "hidden",
        }}>
          <div style={{ flex: 1, background: "var(--bg1)", minWidth: 0 }}>
            <OverlaidChart title="GENERATOR LOSS" seriesList={lossSeries} />
          </div>
          <div style={{ flex: 1, background: "var(--bg1)", minWidth: 0 }}>
            <OverlaidChart title="PSNR (dB)" seriesList={psnrSeries} />
          </div>
        </div>
      </div>
    </div>
  );
}
