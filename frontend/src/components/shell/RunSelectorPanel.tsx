// §23 Multi-Run Navigation
// Tasks: 23.1 (list panel), 23.3 (run selection), 23.4 (New Run button), 23.5 (multi-select)

import { useState } from "react";
import { useProjectStore } from "../../store/projectStore";
import { useUiStore } from "../../store/uiStore";
import { useTrainingStore } from "../../store/trainingStore";
import type { RunStatus } from "../../lib/srproj";

const STATUS_COLOR: Record<RunStatus, string> = {
  configured: "var(--dim)",
  running:    "var(--green)",
  paused:     "var(--amber)",
  completed:  "var(--blue)",
  failed:     "var(--red)",
};

const STATUS_LABEL: Record<RunStatus, string> = {
  configured: "cfg",
  running:    "run",
  paused:     "paus",
  completed:  "done",
  failed:     "fail",
};

interface RunSelectorPanelProps {
  /** "metrics" — selection only updates displayedRunId.
   *  "checkpoints" — selection also triggers checkpoint.list.request. */
  forTab: "metrics" | "checkpoints";
  selectedRunIds: string[];
  onSelectionChange: (ids: string[]) => void;
  onCompare: () => void;
}

export function RunSelectorPanel({
  forTab,
  selectedRunIds,
  onSelectionChange,
  onCompare,
}: RunSelectorPanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  const project             = useProjectStore((s) => s.project);
  const displayedRunId      = useUiStore((s) => s.displayedRunId);
  const setDisplayedRunId   = useUiStore((s) => s.setDisplayedRunId);
  const setActiveTab        = useUiStore((s) => s.setActiveTab);
  const activeTrainingRunId = useTrainingStore((s) => s.activeTrainingRunId);

  const runs = project?.runs ?? [];

  // §23.3 — clicking a run updates displayedRunId; Checkpoints tab also reloads the list
  const handleRunClick = (runId: string) => {
    setDisplayedRunId(runId);

    if (forTab === "checkpoints") {
      const run = runs.find((r) => r.run_id === runId);
      if (run) {
        // TODO: fetch checkpoints from API
      }
    }
  };

  const toggleSelect = (runId: string, checked: boolean) => {
    onSelectionChange(
      checked
        ? [...selectedRunIds, runId]
        : selectedRunIds.filter((id) => id !== runId),
    );
  };

  if (collapsed) {
    return (
      <div style={{
        width: 28, flexShrink: 0,
        borderRight: "1px solid var(--border)",
        background: "var(--bg1)",
        display: "flex", flexDirection: "column",
        alignItems: "center", paddingTop: 8,
      }}>
        <button
          onClick={() => setCollapsed(false)}
          title="Expand run list"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--muted)", fontSize: 12, padding: 4,
          }}
        >
          ▶
        </button>
      </div>
    );
  }

  const effectiveDisplayed = displayedRunId ?? activeTrainingRunId;

  return (
    <div style={{
      width: 180, flexShrink: 0,
      borderRight: "1px solid var(--border)",
      background: "var(--bg1)",
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "7px 10px 6px",
        borderBottom: "1px solid var(--border)", flexShrink: 0,
      }}>
        <span style={{
          fontSize: 10, color: "var(--muted)",
          fontFamily: "var(--font-mono)", letterSpacing: "0.06em",
        }}>
          RUNS
        </span>
        <button
          onClick={() => setCollapsed(true)}
          title="Collapse"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--dim)", fontSize: 10, padding: "1px 4px",
          }}
        >
          ◀
        </button>
      </div>

      {/* §23.1 — Run entries: name, status badge, best PSNR, current epoch */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {runs.length === 0 ? (
          <div style={{
            padding: "20px 10px", fontSize: 10, color: "var(--dim)",
            fontFamily: "var(--font-mono)", textAlign: "center", lineHeight: 1.7,
          }}>
            No runs yet.<br />Create one below.
          </div>
        ) : (
          runs.map((run) => {
            const isDisplayed = effectiveDisplayed === run.run_id;
            const isActive    = run.run_id === activeTrainingRunId;
            const isChecked   = selectedRunIds.includes(run.run_id);

            return (
              <div
                key={run.run_id}
                onClick={() => handleRunClick(run.run_id)}
                style={{
                  display: "flex", alignItems: "flex-start", gap: 6,
                  padding: "6px 10px",
                  borderBottom: "1px solid var(--border)",
                  background: isDisplayed ? "var(--bg2)" : "transparent",
                  cursor: "pointer",
                  transition: "var(--transition-fast)",
                }}
              >
                {/* §23.5 — multi-select checkbox */}
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleSelect(run.run_id, e.target.checked);
                  }}
                  onClick={(e) => e.stopPropagation()}
                  style={{ marginTop: 2, flexShrink: 0, cursor: "pointer" }}
                />

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
                    {isActive && (
                      <div style={{
                        width: 5, height: 5, borderRadius: "50%",
                        background: STATUS_COLOR[run.status], flexShrink: 0,
                        animation:
                          run.status === "running"
                            ? "pulse-dot 1.2s ease-in-out infinite"
                            : "none",
                      }} />
                    )}
                    <span style={{
                      fontSize: 11, color: "var(--text)", fontFamily: "var(--font-sans)",
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>
                      {run.name}
                    </span>
                  </div>

                  <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
                    <span style={{
                      fontSize: 9, padding: "1px 4px", borderRadius: 2,
                      fontFamily: "var(--font-mono)",
                      background: `${STATUS_COLOR[run.status]}22`,
                      color: STATUS_COLOR[run.status],
                      flexShrink: 0,
                    }}>
                      {STATUS_LABEL[run.status]}
                    </span>
                    {run.metrics.best_psnr != null && (
                      <span style={{
                        fontSize: 9, color: "var(--dim)",
                        fontFamily: "var(--font-mono)",
                      }}>
                        {run.metrics.best_psnr.toFixed(1)} dB
                      </span>
                    )}
                    {run.metrics.current_epoch > 0 && (
                      <span style={{
                        fontSize: 9, color: "var(--dim)",
                        fontFamily: "var(--font-mono)",
                      }}>
                        ep {run.metrics.current_epoch}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Footer — §23.4 New Run + §23.5 Compare button */}
      <div style={{
        borderTop: "1px solid var(--border)",
        padding: "8px 10px",
        display: "flex", flexDirection: "column", gap: 6,
        flexShrink: 0,
      }}>
        {/* §23.5 — show Compare button when ≥2 runs are selected */}
        {selectedRunIds.length >= 2 && (
          <button
            onClick={onCompare}
            style={{
              background: "var(--blue-dim)",
              border: "1px solid rgba(90,171,240,0.27)",
              color: "var(--blue)",
              borderRadius: "var(--radius-sm)",
              fontSize: 10, fontFamily: "var(--font-mono)",
              cursor: "pointer", padding: "4px 8px",
              width: "100%", textAlign: "center",
              transition: "var(--transition-fast)",
            }}
          >
            Compare {selectedRunIds.length} Runs
          </button>
        )}
        {/* §23.4 — New Run navigates to Training Setup; SRProjRun not created until
            project.run.started is received from the sidecar after launch */}
        <button
          onClick={() => setActiveTab("training")}
          style={{
            background: "transparent",
            border: "1px solid var(--border)",
            color: "var(--muted)",
            borderRadius: "var(--radius-sm)",
            fontSize: 10, fontFamily: "var(--font-mono)",
            cursor: "pointer", padding: "4px 8px",
            width: "100%", textAlign: "center",
            transition: "var(--transition-fast)",
          }}
        >
          + New Run
        </button>
      </div>
    </div>
  );
}
