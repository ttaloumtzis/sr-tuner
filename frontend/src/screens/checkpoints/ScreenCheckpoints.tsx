// Runs & Checkpoints Screen — 3-column layout:
//   left   — models & their runs (disk-derived status, run deletion)
//   center — checkpoints + metrics for the selected run
//   right  — selected checkpoint detail + inference stub

import { useState, useEffect, useMemo } from "react";
import { useTrainingStore } from "../../store/trainingStore";
import { useRunsStore } from "../../store/runsStore";
import { useUiStore } from "../../store/uiStore";
import { useInferenceStore } from "../../store/inferenceStore";
import { useModelStore } from "../../store/modelStore";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useToast } from "../../components/shell/ToastProvider";
import { Tag } from "../../components/ui/Tag";
import { Btn } from "../../components/ui/Btn";

import type { CheckpointEntry, ModelRuns, RunInfo, RunStatus } from "../../lib/api-types";
import type { Hyperparameters } from "../../store/modelStore";
import { buildRunDisplays, shortRunId, sortGroups } from "../../lib/runLabel";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(n: number | null | undefined, dec = 2): string {
  if (n == null) return "—";
  return n.toFixed(dec);
}

function fmtSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

const STATUS_COLOR: Record<RunStatus, string> = {
  running: "var(--green)",
  finished: "var(--green)",
  failed: "var(--red)",
  stopped: "var(--amber)",
  interrupted: "var(--dim)",
};

// ── Sorting ───────────────────────────────────────────────────────────────

type SortCol = "epoch" | "psnr" | "ssim" | "size" | "date";
type SortDir = "asc" | "desc";

function sortEntries(entries: CheckpointEntry[], col: SortCol, dir: SortDir): CheckpointEntry[] {
  const copy = [...entries];
  copy.sort((a, b) => {
    let va: number | string, vb: number | string;
    switch (col) {
      case "epoch": va = a.epoch; vb = b.epoch; break;
      case "psnr":  va = a.metrics.psnr  ?? -Infinity; vb = b.metrics.psnr  ?? -Infinity; break;
      case "ssim":  va = a.metrics.ssim  ?? -Infinity; vb = b.metrics.ssim  ?? -Infinity; break;
      case "size":  va = a.file_size_mb; vb = b.file_size_mb; break;
      case "date":  va = a.created_at; vb = b.created_at; break;
    }
    if (va < vb) return dir === "asc" ? -1 : 1;
    if (va > vb) return dir === "asc" ? 1 : -1;
    return 0;
  });
  return copy;
}

// ── Column header ─────────────────────────────────────────────────────────

interface ColHeaderProps {
  label: string;
  col: SortCol;
  current: SortCol;
  dir: SortDir;
  onSort: (col: SortCol) => void;
}

function ColHeader({ label, col, current, dir, onSort }: ColHeaderProps) {
  const active = col === current;
  return (
    <div
      onClick={() => onSort(col)}
      style={{
        cursor: "pointer", userSelect: "none",
        fontSize: 10, fontFamily: "var(--font-mono)",
        color: active ? "var(--green)" : "var(--muted)",
        display: "flex", alignItems: "center", gap: 3,
      }}
    >
      {label}
      {active && <span style={{ fontSize: 8 }}>{dir === "asc" ? "▲" : "▼"}</span>}
    </div>
  );
}

// ── Checkpoints Table ─────────────────────────────────────────────────────

interface TableProps {
  entries: CheckpointEntry[];
  bestPsnrPath: string | null;
  latestPath: string | null;
  selectedPath: string | null;
  sortCol: SortCol;
  sortDir: SortDir;
  trainingActive: boolean;
  onSort: (col: SortCol) => void;
  onSelect: (e: CheckpointEntry) => void;
  onDeleteRequest: (e: CheckpointEntry) => void;
}

function CheckpointsTable({
  entries, bestPsnrPath, latestPath, selectedPath,
  sortCol, sortDir, trainingActive, onSort, onSelect, onDeleteRequest,
}: TableProps) {
  const COL = "52px 1fr 1fr 80px 1.5fr 80px 40px";

  const cellSt: React.CSSProperties = {
    fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)",
    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
  };

  if (entries.length === 0) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 12, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
          No checkpoints saved yet
        </span>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <div style={{
        display: "grid", gridTemplateColumns: COL, gap: 0,
        padding: "6px 10px", borderBottom: "1px solid var(--border)",
        background: "var(--bg2)", flexShrink: 0,
        position: "sticky", top: 0, zIndex: 1,
      }}>
        <ColHeader label="EPOCH" col="epoch" current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="PSNR"  col="psnr"  current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="SSIM"  col="ssim"  current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="SIZE"  col="size"  current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="SAVED" col="date"  current={sortCol} dir={sortDir} onSort={onSort} />
        <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--muted)" }}>TAG</div>
        <div />
      </div>

      {/* Rows */}
      {entries.map((e) => {
        const isBest   = e.path === bestPsnrPath;
        const isLatest = e.path === latestPath;
        const isSel    = e.path === selectedPath;

        return (
          <div
            key={e.path}
            onClick={() => onSelect(e)}
            style={{
              display: "grid", gridTemplateColumns: COL, gap: 0,
              padding: "5px 10px", borderBottom: "1px solid var(--border)",
              background: isSel ? "var(--bg2)" : "transparent",
              cursor: "pointer", alignItems: "center",
              transition: "var(--transition-fast)",
            }}
          >
            <span style={{ ...cellSt, color: "var(--green)", fontWeight: 600 }}>
              {String(e.epoch).padStart(3, "0")}
            </span>
            <span style={cellSt}>{fmt(e.metrics.psnr)} dB</span>
            <span style={cellSt}>{fmt(e.metrics.ssim, 4)}</span>
            <span style={{ ...cellSt, color: "var(--dim)" }}>{fmtSize(e.file_size_mb)}</span>
            <span style={{ ...cellSt, color: "var(--dim)", fontSize: 10 }}>{fmtDate(e.created_at)}</span>
            <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
              {isBest   && <Tag color="green">best</Tag>}
              {isLatest && <Tag color="blue">latest</Tag>}
            </div>
            <div style={{ display: "flex", justifyContent: "center" }}>
              <button
                onClick={(ev) => { ev.stopPropagation(); onDeleteRequest(e); }}
                disabled={trainingActive}
                title={
                  trainingActive
                    ? "Cannot delete checkpoints while training is active"
                    : `Delete ${e.filename}`
                }
                style={{
                  background: "none", border: "none",
                  color: trainingActive ? "var(--dim)" : "var(--red)",
                  cursor: trainingActive ? "default" : "pointer",
                  fontSize: 14, lineHeight: 1, padding: "2px 4px",
                  opacity: trainingActive ? 0.4 : 1,
                  transition: "var(--transition-fast)",
                }}
              >
                ✕
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Storage Summary Panel ─────────────────────────────────────────────────

function StorageSummaryPanel({ entries }: { entries: CheckpointEntry[] }) {
  const totalMb = entries.reduce((acc, e) => acc + e.file_size_mb, 0);
  return (
    <div style={{
      borderTop: "1px solid var(--border)", padding: "7px 12px",
      display: "flex", gap: 24, alignItems: "center",
      background: "var(--bg1)", flexShrink: 0,
    }}>
      <div>
        <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>FILES </span>
        <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
          {entries.length}
        </span>
      </div>
      <div>
        <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>TOTAL </span>
        <span style={{ fontSize: 11, color: "var(--text)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
          {fmtSize(totalMb)}
        </span>
      </div>
    </div>
  );
}

// ── Delete-Disabled Banner ────────────────────────────────────────────────

function DeleteDisabledBanner() {
  return (
    <div style={{
      borderTop: "1px solid rgba(204,120,40,0.27)",
      background: "var(--amber-dim)", padding: "6px 12px",
      fontSize: 11, color: "var(--amber)", fontFamily: "var(--font-mono)", flexShrink: 0,
    }}>
      Delete disabled — training is active. Stop training to delete checkpoints.
    </div>
  );
}

// ── Detail Panel row helper ───────────────────────────────────────────────

function Row({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 4 }}>
      <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>
        {label}
      </span>
      <span style={{
        fontSize: 11, color: "var(--text)",
        fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
        textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
      }}>
        {value}
      </span>
    </div>
  );
}

// ── Detail Panel ──────────────────────────────────────────────────────────

interface DetailPanelProps {
  entry: CheckpointEntry | null;
  deleteDisabled: boolean;
  deleteDisabledTitle: string | undefined;
  onExportPth: (e: CheckpointEntry) => void;
  onExportOnnx: (e: CheckpointEntry) => void;
  onDeleteRequest: (e: CheckpointEntry) => void;
  onRunInference: () => void;
  onResume: (e: CheckpointEntry) => void;
}

function DetailPanel({
  entry, deleteDisabled, deleteDisabledTitle, onExportPth, onExportOnnx, onDeleteRequest, onRunInference, onResume,
}: DetailPanelProps) {
  return (
    <div style={{
      flex: 1, minWidth: 180, maxWidth: 300,
      borderLeft: "1px solid var(--border)",
      background: "var(--bg1)",
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      <div style={{
        fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)",
        padding: "10px 12px 6px", borderBottom: "1px solid var(--border)", flexShrink: 0,
      }}>
        CHECKPOINT DETAIL
      </div>

      {entry == null ? (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
            Select a checkpoint
          </span>
        </div>
      ) : (
        <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column" }}>
          {/* Validation preview placeholder */}
          <div style={{
            height: 110, background: "var(--bg2)",
            borderBottom: "1px solid var(--border)",
            display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0,
          }}>
            <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
              No preview
            </span>
          </div>

          {/* Metrics + file info */}
          <div style={{ padding: "10px 12px", display: "flex", flexDirection: "column", gap: 7 }}>
            <Row label="EPOCH" value={`${entry.epoch}`} />
            <Row label="PSNR"  value={`${fmt(entry.metrics.psnr)} dB`} />
            <Row label="SSIM"  value={fmt(entry.metrics.ssim, 4)} />
            <Row label="SIZE"  value={fmtSize(entry.file_size_mb)} />
            <Row label="SAVED" value={fmtDate(entry.created_at)} mono={false} />
          </div>

          <div style={{
            padding: "0 12px 10px",
            fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)",
            wordBreak: "break-all", lineHeight: 1.5,
          }}>
            {entry.filename}
          </div>

          {/* Export buttons */}
          <div style={{
            padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6,
            borderTop: "1px solid var(--border)", flexShrink: 0,
          }}>
            <Btn variant="solid" color="var(--green)" full onClick={() => onExportPth(entry)}>
              Export .pth
            </Btn>
            <Btn variant="ghost" color="var(--blue)" full onClick={() => onExportOnnx(entry)}>
              Export ONNX
            </Btn>
          </div>

          {/* Resume from checkpoint */}
          <div style={{
            padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6,
            borderTop: "1px solid var(--border)", flexShrink: 0,
          }}>
            <Btn variant="solid" color="var(--blue)" full onClick={() => onResume(entry)}>
              Resume Training →
            </Btn>
          </div>

          {/* Navigation + Delete */}
          <div style={{
            padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6,
            borderTop: "1px solid var(--border)", flexShrink: 0,
          }}>
            <Btn variant="solid" color="var(--green)" full onClick={onRunInference}>
              Use in Inference →
            </Btn>
            <Btn
              variant="ghost" color="var(--red)" full
              disabled={deleteDisabled}
              title={deleteDisabledTitle}
              onClick={() => onDeleteRequest(entry)}
            >
              Delete
            </Btn>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Delete Confirmation Scrim (checkpoint) ────────────────────────────────

interface DeleteScrimProps {
  entry: CheckpointEntry;
  onConfirm: () => void;
  onCancel: () => void;
}

function DeleteConfirmScrim({ entry, onConfirm, onCancel }: DeleteScrimProps) {
  const metricsFile = entry.filename.replace(/\.pth$/i, "_metrics.json");
  return (
    <div style={{
      position: "absolute", inset: 0,
      background: "rgba(13,15,17,0.72)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 100,
    }}>
      <div style={{
        background: "var(--bg1)", border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)", padding: "20px 24px",
        width: 320, display: "flex", flexDirection: "column", gap: 14,
      }}>
        <div style={{ fontSize: 13, color: "var(--text)", fontFamily: "var(--font-sans)", fontWeight: 600 }}>
          Delete checkpoint?
        </div>
        <div style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)", lineHeight: 1.8 }}>
          The following files will be deleted:
          <div style={{
            marginTop: 6, padding: "6px 10px",
            background: "var(--bg2)", borderRadius: "var(--radius-sm)",
            color: "var(--red)", fontSize: 10, lineHeight: 2,
          }}>
            {entry.filename}
            <br />
            {metricsFile}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <Btn onClick={onCancel}>Cancel</Btn>
          <Btn variant="solid" color="var(--red)" onClick={onConfirm}>Delete</Btn>
        </div>
      </div>
    </div>
  );
}

// ── Run Delete Confirmation Scrim ─────────────────────────────────────────

interface RunDeleteScrimProps {
  run: RunInfo;
  onConfirm: () => void;
  onCancel: () => void;
}

function RunDeleteConfirmScrim({ run, onConfirm, onCancel }: RunDeleteScrimProps) {
  return (
    <div style={{
      position: "absolute", inset: 0,
      background: "rgba(13,15,17,0.72)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 100,
    }}>
      <div style={{
        background: "var(--bg1)", border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)", padding: "20px 24px",
        width: 340, display: "flex", flexDirection: "column", gap: 14,
      }}>
        <div style={{ fontSize: 13, color: "var(--text)", fontFamily: "var(--font-sans)", fontWeight: 600 }}>
          Delete run {run.run_id}?
        </div>
        <div style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)", lineHeight: 1.8 }}>
          The run folder and everything in it will be permanently deleted:
          <div style={{
            marginTop: 6, padding: "6px 10px",
            background: "var(--bg2)", borderRadius: "var(--radius-sm)",
            color: "var(--red)", fontSize: 10, lineHeight: 2,
          }}>
            {run.checkpoint_count} checkpoint(s) · {fmtSize(run.total_size_mb)}
            {run.has_metrics && <><br />metrics.jsonl</>}
            <br />validation frames
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <Btn onClick={onCancel}>Cancel</Btn>
          <Btn variant="solid" color="var(--red)" onClick={onConfirm}>Delete</Btn>
        </div>
      </div>
    </div>
  );
}

// ── Models & Runs Sidebar ─────────────────────────────────────────────────

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

function StatusDot({ status }: { status: RunStatus }) {
  const running = status === "running";
  return (
    <span
      title={status}
      style={{
        display: "inline-block",
        width: 7, height: 7, borderRadius: "50%",
        background: STATUS_COLOR[status],
        flexShrink: 0,
        animation: running ? "tabbar-pulse 1.4s ease-in-out infinite" : undefined,
      }}
    />
  );
}

function ModelsRunsSidebar({
  models, selectedInstance, selectedRunId, activeRunDirId, refreshing, onSelectRun, onDeleteRequest, onRefresh,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  if (models.length === 0) {
    return (
      <div style={{
        width: 230, borderRight: "1px solid var(--border)", background: "var(--bg1)",
        display: "flex", flexDirection: "column", flexShrink: 0,
      }}>
        <SidebarHeader refreshing={refreshing} onRefresh={onRefresh} />
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 }}>
          <span style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)", textAlign: "center", lineHeight: 1.7 }}>
            No model instances yet.
            <br />
            Create one in Model Config.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      width: 230, borderRight: "1px solid var(--border)", background: "var(--bg1)",
      display: "flex", flexDirection: "column", flexShrink: 0, minWidth: 0,
    }}>
      <SidebarHeader refreshing={refreshing} onRefresh={onRefresh} />

      <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
        {models.map((m) => {
          const isCollapsed = collapsed[m.name] ?? false;
          const hasRuns = m.runs.length > 0;
          return (
            <div key={m.name}>
              <button
                onClick={() => setCollapsed((c) => ({ ...c, [m.name]: !(c[m.name] ?? false) }))}
                style={{
                  display: "flex", alignItems: "center", gap: 5,
                  width: "100%", padding: "6px 10px",
                  background: "none", border: "none", cursor: "pointer",
                  color: "var(--text)", fontSize: 11, fontFamily: "var(--font-sans)",
                  fontWeight: 600,
                }}
              >
                <span style={{ fontSize: 9, color: "var(--dim)", width: 10, flexShrink: 0 }}>
                  {isCollapsed ? "▸" : "▾"}
                </span>
                <span style={{ flex: 1, textAlign: "left", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {m.name}
                </span>
                <span style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
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
                        <div style={{
                          padding: "4px 10px 2px 20px",
                          fontSize: 8, letterSpacing: "0.08em",
                          color: "var(--dim)", fontFamily: "var(--font-mono)",
                          textTransform: "uppercase", background: "var(--bg0)",
                        }}>
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
                              style={{
                                display: "flex", alignItems: "center", gap: 6,
                                padding: "4px 10px 4px 25px",
                                background: selected ? "var(--bg2)" : "transparent",
                                borderLeft: selected ? "2px solid var(--green)" : "2px solid transparent",
                                cursor: "pointer",
                                transition: "var(--transition-fast)",
                              }}
                            >
                              <StatusDot status={run.status} />
                              <span style={{
                                flex: 1, fontSize: 10, fontFamily: "var(--font-mono)",
                                color: selected ? "var(--text)" : "var(--muted)",
                                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                              }}>
                                {displays.get(run.run_id)?.label ?? shortRunId(run.run_id)}
                              </span>
                              <span style={{ fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>
                                {run.checkpoint_count}
                              </span>
                              <button
                                onClick={(ev) => { ev.stopPropagation(); onDeleteRequest(run); }}
                                disabled={active}
                                title={active ? "Cannot delete the run that is currently training" : `Delete ${run.run_id}`}
                                style={{
                                  background: "none", border: "none",
                                  color: active ? "var(--dim)" : "var(--red)",
                                  cursor: active ? "default" : "pointer",
                                  fontSize: 12, lineHeight: 1, padding: "1px 3px",
                                  opacity: active ? 0.35 : 1,
                                  flexShrink: 0,
                                }}
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
                <div style={{ padding: "2px 10px 8px 25px", fontSize: 9, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
                  No runs yet — start training
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{
        borderTop: "1px solid var(--border)", padding: "8px 10px",
        display: "flex", flexDirection: "column", gap: 4, flexShrink: 0,
        background: "var(--bg2)",
      }}>
        {([
          ["finished", "finished"],
          ["failed", "failed"],
          ["stopped", "stopped"],
          ["running", "running"],
          ["interrupted", "interrupted"],
        ] as [RunStatus, string][]).map(([status, label]) => (
          <div key={status} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <StatusDot status={status} />
            <span style={{ fontSize: 9, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
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
    <div style={{
      padding: "8px 10px", borderBottom: "1px solid var(--border)",
      display: "flex", alignItems: "center", gap: 6, flexShrink: 0,
      background: "var(--bg1)",
    }}>
      <span style={{
        flex: 1, fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)",
        letterSpacing: "0.08em",
      }}>
        MODELS &amp; RUNS
      </span>
      <button
        onClick={onRefresh}
        title="Refresh runs from disk"
        style={{
          background: "none", border: "1px solid var(--border)", borderRadius: 4,
          color: "var(--muted)", cursor: "pointer", fontSize: 11, lineHeight: 1,
          padding: "3px 6px", fontFamily: "var(--font-mono)",
        }}
      >
        {refreshing ? "…" : "⟳"}
      </button>
    </div>
  );
}

// ── ScreenCheckpoints ─────────────────────────────────────────────────────

export function ScreenCheckpoints() {
  const status          = useTrainingStore((s) => s.status);
  const activeRunDirId  = useTrainingStore((s) => s.activeRunDirId);
  const setActiveTab    = useUiStore((s) => s.setActiveTab);
  const models          = useRunsStore((s) => s.models);
  const loading         = useRunsStore((s) => s.loading);
  const refreshCounter  = useRunsStore((s) => s.refreshCounter);
  const selectedInstance = useRunsStore((s) => s.selectedInstance);
  const selectedRunId   = useRunsStore((s) => s.selectedRunId);
  const checkpointsByRun = useRunsStore((s) => s.checkpointsByRun);
  const selectedCheckpointPath = useRunsStore((s) => s.selectedCheckpointPath);
  const { show } = useToast();

  const trainingActive = status === "running";
  const allEntries: CheckpointEntry[] = selectedRunId ? (checkpointsByRun[selectedRunId] ?? []) : [];

  const [sortCol, setSortCol]       = useState<SortCol>("epoch");
  const [sortDir, setSortDir]       = useState<SortDir>("desc");
  const [deletingEntry, setDeletingEntry] = useState<CheckpointEntry | null>(null);
  const [deletingRun, setDeletingRun] = useState<{ instance: string; run: RunInfo } | null>(null);

  // Initial load + background refresh every 15s.
  useEffect(() => {
    const { refresh } = useRunsStore.getState();
    refresh();
    const t = setInterval(() => useRunsStore.getState().refresh(), 15000);
    return () => clearInterval(t);
  }, []);

  // SSE-driven refresh (checkpoint_saved, run lifecycle events).
  useEffect(() => {
    if (refreshCounter > 0) useRunsStore.getState().refresh();
  }, [refreshCounter]);

  // A training run just started — surface it in the sidebar immediately.
  useEffect(() => {
    if (trainingActive && activeRunDirId) useRunsStore.getState().refresh();
  }, [trainingActive, activeRunDirId]);

  const selectedRun: RunInfo | null = useMemo(() => {
    if (!selectedInstance || !selectedRunId) return null;
    return models.find((m) => m.name === selectedInstance)?.runs.find((r) => r.run_id === selectedRunId) ?? null;
  }, [models, selectedInstance, selectedRunId]);

  const isActiveRun = selectedRun != null && activeRunDirId === selectedRun.run_id && trainingActive;

  const bestPsnrPath = useMemo(() => {
    let best: CheckpointEntry | null = null;
    for (const e of allEntries) {
      if (e.metrics.psnr != null && (best == null || e.metrics.psnr > (best.metrics.psnr ?? -Infinity))) {
        best = e;
      }
    }
    return best?.path ?? null;
  }, [allEntries]);

  const latestPath = allEntries[allEntries.length - 1]?.path ?? null;

  const sorted = useMemo(
    () => sortEntries(allEntries, sortCol, sortDir),
    [allEntries, sortCol, sortDir],
  );

  const selectedEntry = sorted.find((e) => e.path === selectedCheckpointPath) ?? null;

  // ── Handlers ─────────────────────────────────────────────────────────

  const handleSort = (col: SortCol) => {
    if (col === sortCol) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortCol(col); setSortDir("desc"); }
  };

  const handleDeleteRequest = (e: CheckpointEntry) => {
    if (trainingActive) return;
    setDeletingEntry(e);
  };

  const handleDeleteConfirm = () => {
    if (!deletingEntry) return;
    // TODO: replace with api call (per-checkpoint delete)
    if (selectedCheckpointPath === deletingEntry.path) useRunsStore.getState().selectCheckpoint(null);
    setDeletingEntry(null);
  };

  const handleExportPth = (_e: CheckpointEntry) => {
    // TODO: replace with api call
  };

  const handleExportOnnx = (_e: CheckpointEntry) => {
    // TODO: replace with api call
  };

  const handleRunDeleteConfirm = async () => {
    if (!deletingRun) return;
    const { instance, run } = deletingRun;
    setDeletingRun(null);
    const ok = await useRunsStore.getState().deleteRun(instance, run.run_id);
    if (ok) show("success", `Deleted run ${run.run_id}`, 3000);
  };

  // Resume prefill from the run's on-disk config snapshot (run_config.json)
  // + instance metadata from the models API.
  const handleResume = (e: CheckpointEntry) => {
    const tc = (selectedRun?.config?.train_cfg ?? {}) as Record<string, unknown>;
    const instName = selectedRun?.config?.instance as string | undefined;
    const inst = models.find((m) => m.name === instName);

    if (inst?.architecture) {
      useModelStore.getState().setArchitecture(
        (inst.architecture === "swinir" ? "swinir" : "rrdb_esrgan") as "rrdb_esrgan" | "swinir",
      );
    }

    const hp: Partial<Hyperparameters> = { scale: inst?.scale ?? 4 };
    if (typeof tc.batch_size === "number") hp.batchSize = tc.batch_size;
    if (typeof tc.learning_rate === "number") hp.learningRate = tc.learning_rate;
    if (typeof tc.scheduler === "string") hp.lrScheduler = tc.scheduler;
    if (typeof tc.optimizer === "string") hp.optimizer = tc.optimizer;
    if (typeof tc.patch_size === "number") hp.patchSize = tc.patch_size;
    useModelStore.getState().setHyperparameters(hp);

    useRunConfigStore.getState().setSelectedInstance(instName ?? null);
    useRunConfigStore.getState().setSchedule({
      totalEpochs: typeof tc.max_epochs === "number" ? tc.max_epochs : 100,
    });
    useRunConfigStore.getState().setResumeFrom(e.path);
    setActiveTab("training");
  };

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div style={{
      display: "flex", width: "100%", height: "100%",
      overflow: "hidden", background: "var(--bg0)",
      position: "relative",
    }}>
      {/* Left — models & runs */}
      <ModelsRunsSidebar
        models={models}
        selectedInstance={selectedInstance}
        selectedRunId={selectedRunId}
        activeRunDirId={activeRunDirId}
        refreshing={loading}
        onSelectRun={(instance, runId) => useRunsStore.getState().selectRun(instance, runId)}
        onDeleteRequest={(run) => {
          const instance = models.find((m) => m.runs.some((r) => r.run_id === run.run_id))?.name ?? "";
          setDeletingRun({ instance, run });
        }}
        onRefresh={() => useRunsStore.getState().refresh()}
      />

      {/* Center — checkpoints of the selected run */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
        <div style={{
          padding: "7px 12px", borderBottom: "1px solid var(--border)",
          display: "flex", alignItems: "center", gap: 8,
          background: "var(--bg1)", flexShrink: 0,
        }}>
          {selectedRun == null ? (
            <span style={{ fontSize: 11, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
              Select a run from the left to browse its checkpoints
            </span>
          ) : (
            <>
              <StatusDot status={selectedRun.status} />
              <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)", fontWeight: 600 }}>
                {selectedRun.run_id}
              </span>
              <Tag color={selectedRun.status === "failed" ? "red" : selectedRun.status === "running" ? "green" : selectedRun.status === "finished" ? "green" : "amber"}>
                {selectedRun.status}
              </Tag>
              <span style={{ flex: 1 }} />
              <span style={{ fontSize: 10, color: "var(--dim)", fontFamily: "var(--font-mono)" }}>
                {selectedRun.checkpoint_count} ckpt · {fmtSize(selectedRun.total_size_mb)}
              </span>
            </>
          )}
        </div>

        {selectedRun != null && (
          <>
            <CheckpointsTable
              entries={sorted}
              bestPsnrPath={bestPsnrPath}
              latestPath={latestPath}
              selectedPath={selectedCheckpointPath}
              sortCol={sortCol}
              sortDir={sortDir}
              trainingActive={isActiveRun}
              onSort={handleSort}
              onSelect={(e) => useRunsStore.getState().selectCheckpoint(e.path)}
              onDeleteRequest={handleDeleteRequest}
            />
            <StorageSummaryPanel entries={allEntries} />
            {isActiveRun && <DeleteDisabledBanner />}
          </>
        )}
      </div>

      {/* Right — checkpoint detail */}
      <DetailPanel
        entry={selectedEntry}
        deleteDisabled={isActiveRun}
        deleteDisabledTitle={isActiveRun ? "Cannot delete checkpoints of the run that is currently training" : undefined}
        onExportPth={handleExportPth}
        onExportOnnx={handleExportOnnx}
        onDeleteRequest={handleDeleteRequest}
        onResume={handleResume}
        onRunInference={() => {
          if (selectedCheckpointPath) {
            const inf = useInferenceStore.getState();
            inf.setPreselectedInstance(selectedInstance);
            inf.setPreselectedCheckpointPath(selectedCheckpointPath);
            if (selectedRunId) {
              const inst = models.find((m) => m.name === selectedInstance);
              const displays = inst ? buildRunDisplays(inst.runs) : new Map();
              inf.setCheckpointContext(selectedRunId, displays.get(selectedRunId)?.label ?? shortRunId(selectedRunId));
            }
          }
          setActiveTab("inference");
        }}
      />

      {/* Delete confirmations */}
      {deletingEntry != null && (
        <DeleteConfirmScrim
          entry={deletingEntry}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeletingEntry(null)}
        />
      )}
      {deletingRun != null && (
        <RunDeleteConfirmScrim
          run={deletingRun.run}
          onConfirm={handleRunDeleteConfirm}
          onCancel={() => setDeletingRun(null)}
        />
      )}
    </div>
  );
}
