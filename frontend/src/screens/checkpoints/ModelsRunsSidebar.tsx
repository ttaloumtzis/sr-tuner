import { useState } from "react";
import type { ModelRuns, RunInfo, RunStatus } from "../../lib/api-types";
import { StatusDot } from "../../components/ui/StatusDot";
import { buildRunDisplays, shortRunId, sortGroups } from "../../lib/runLabel";

interface SidebarProps {
  models: ModelRuns[];
  selectedInstance: string | null;
  selectedRunId: string | null;
  activeRunDirId: string | null;
  refreshing: boolean;
  onSelectRun: (instance: string, runId: string) => void;
  onDeleteRequest: (run: RunInfo) => void;
  onRefresh: () => void;
}

export function ModelsRunsSidebar({
  models, selectedInstance, selectedRunId, activeRunDirId, refreshing, onSelectRun, onDeleteRequest, onRefresh,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  if (models.length === 0) {
    return (
      <div className="sc-sidebar">
        <SidebarHeader refreshing={refreshing} onRefresh={onRefresh} />
        <div className="sc-sidebar-empty-pad">
          <span className="sc-empty-text">
            No model instances yet.
            <br />
            Create one in Model Config.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="sc-sidebar">
      <SidebarHeader refreshing={refreshing} onRefresh={onRefresh} />

      <div className="sc-scroll">
        {models.map((m) => {
          const isCollapsed = collapsed[m.name] ?? false;
          const hasRuns = m.runs.length > 0;
          return (
            <div key={m.name}>
              <button
                onClick={() => setCollapsed((c) => ({ ...c, [m.name]: !(c[m.name] ?? false) }))}
                className="sc-model-head"
              >
                <span className="sc-chevron">
                  {isCollapsed ? "▸" : "▾"}
                </span>
                <span className="sc-model-name">
                  {m.name}
                </span>
                <span className="sc-run-count">
                  {hasRuns ? `${m.runs.length} run${m.runs.length > 1 ? "s" : ""}` : "0 runs"}
                </span>
              </button>

              {!isCollapsed && hasRuns && (
                <div>
                  {(() => {
                    const displays = buildRunDisplays(m.runs);
                    const byGroup = new Map<string, RunInfo[]>();
                    for (const run of m.runs) {
                      const group = displays.get(run.run_id)?.group ?? "Unknown";
                      const list = byGroup.get(group) ?? [];
                      list.push(run);
                      byGroup.set(group, list);
                    }
                    const groups = sortGroups([...byGroup.keys()]);
                    return groups.map((group) => (
                      <div key={group}>
                        <div className="sc-group">
                          {group}
                        </div>
                        {byGroup.get(group)!.map((run) => {
                          const selected = selectedInstance === m.name && selectedRunId === run.run_id;
                          const active = activeRunDirId === run.run_id;
                          return (
                            <div
                              key={run.run_id}
                              onClick={() => onSelectRun(m.name, run.run_id)}
                              title={run.error ? `${run.status}: ${run.error}` : `${run.run_id} — ${run.status}`}
                              className={`sc-run-row${selected ? " sc-run-row-selected" : ""}`}
                            >
                              <StatusDot status={run.status} />
                              <span className={`sc-run-label${selected ? " sc-run-label-selected" : ""}`}>
                                {displays.get(run.run_id)?.label ?? shortRunId(run.run_id)}
                              </span>
                              <span className="sc-run-ckpts">
                                {run.checkpoint_count}
                              </span>
                              <button
                                onClick={(ev) => { ev.stopPropagation(); onDeleteRequest(run); }}
                                disabled={active}
                                title={active ? "Cannot delete the run that is currently training" : `Delete ${run.run_id}`}
                                className="sc-run-del"
                              >
                                ✕
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    ));
                  })()}
                </div>
              )}

              {!isCollapsed && !hasRuns && (
                <div className="sc-no-runs">
                  No runs yet — start training
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="sc-legend">
        {([
          ["finished", "finished"],
          ["failed", "failed"],
          ["stopped", "stopped"],
          ["running", "running"],
          ["interrupted", "interrupted"],
        ] as [RunStatus, string][]).map(([status, label]) => (
          <div key={status} className="sc-legend-row">
            <StatusDot status={status} />
            <span className="sc-legend-label">
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SidebarHeader({ refreshing, onRefresh }: { refreshing: boolean; onRefresh: () => void }) {
  return (
    <div className="sc-sidebar-header">
      <span className="sc-sidebar-title">
        MODELS &amp; RUNS
      </span>
      <button
        onClick={onRefresh}
        title="Refresh runs from disk"
        className="sc-refresh"
      >
        {refreshing ? "…" : "⟳"}
      </button>
    </div>
  );
}