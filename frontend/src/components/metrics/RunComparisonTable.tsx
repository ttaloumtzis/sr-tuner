// §22.3 — Comparison metadata table shown when ≥2 runs are selected.
// Columns: run name, status, total epochs, best PSNR, best loss, duration.
// Displays "—" for any run with no recorded history.

import type { SRProjRun } from "../../lib/srproj";
import { STATUS_COLOR } from "../../lib/runStatus";
import { fmt, fmtDuration } from "../../lib/format";

interface Props {
  runs: SRProjRun[];
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
