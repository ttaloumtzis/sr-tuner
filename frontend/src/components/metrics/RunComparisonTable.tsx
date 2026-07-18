// §22.3 — Comparison metadata table shown when ≥2 runs are selected.
// Columns: run name, status, total epochs, best PSNR, best loss, duration.
// Displays "—" for any run with no recorded history.

import type { SRProjRun, RunStatus } from "../../lib/srproj";

interface Props {
  runs: SRProjRun[];
}

const STATUS_COLOR: Record<RunStatus, string> = {
  configured: "var(--dim)",
  running:    "var(--green)",
  paused:     "var(--amber)",
  completed:  "var(--blue)",
  failed:     "var(--red)",
};

function fmt(n: number | null, decimals = 2): string {
  return n != null ? n.toFixed(decimals) : "—";
}

function fmtDuration(run: SRProjRun): string {
  if (!run.started_at) return "—";
  const endMs = run.completed_at
    ? new Date(run.completed_at).getTime()
    : Date.now();
  const sec = Math.floor((endMs - new Date(run.started_at).getTime()) / 1000);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

const TH_STYLE: React.CSSProperties = {
  padding: "6px 10px",
  fontSize: 10,
  fontFamily: "var(--font-mono)",
  color: "var(--muted)",
  textAlign: "left",
  fontWeight: 400,
  borderBottom: "1px solid var(--border)",
  whiteSpace: "nowrap",
};

const TD_STYLE: React.CSSProperties = {
  padding: "6px 10px",
  fontSize: 11,
  fontFamily: "var(--font-mono)",
  color: "var(--text)",
  borderBottom: "1px solid var(--border)",
  whiteSpace: "nowrap",
};

export function RunComparisonTable({ runs }: Props) {
  if (runs.length === 0) return null;

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 480 }}>
        <thead>
          <tr>
            <th style={TH_STYLE}>Run</th>
            <th style={TH_STYLE}>Status</th>
            <th style={{ ...TH_STYLE, textAlign: "right" }}>Total Epochs</th>
            <th style={{ ...TH_STYLE, textAlign: "right" }}>Best PSNR</th>
            <th style={{ ...TH_STYLE, textAlign: "right" }}>Best Loss</th>
            <th style={{ ...TH_STYLE, textAlign: "right" }}>Duration</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const hasHistory =
              run.metrics.epochs_completed > 0 ||
              run.metrics.best_psnr != null ||
              run.metrics.best_loss != null;

            return (
              <tr key={run.run_id}>
                <td style={TD_STYLE}>{run.name}</td>
                <td style={TD_STYLE}>
                  <span style={{ color: STATUS_COLOR[run.status] }}>
                    {run.status}
                  </span>
                </td>
                <td style={{ ...TD_STYLE, textAlign: "right" }}>
                  {hasHistory ? run.metrics.epochs_completed : "—"}
                </td>
                <td style={{ ...TD_STYLE, textAlign: "right", color: "var(--green)" }}>
                  {hasHistory ? fmt(run.metrics.best_psnr) : "—"}
                </td>
                <td style={{ ...TD_STYLE, textAlign: "right" }}>
                  {hasHistory ? fmt(run.metrics.best_loss, 4) : "—"}
                </td>
                <td style={{ ...TD_STYLE, textAlign: "right" }}>
                  {fmtDuration(run)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
