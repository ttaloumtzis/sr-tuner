// §13 Checkpoints Screen
// Tasks: 13.1–13.8, 13.9a, 13.9b, §23.2+23.7 RunSelectorPanel sidebar

import { useState, useEffect, useMemo, useRef } from "react";
import { useTrainingStore } from "../../store/trainingStore";
import { useCheckpointStore } from "../../store/checkpointStore";
import { useUiStore } from "../../store/uiStore";
import { useInferenceStore } from "../../store/inferenceStore";
import { useProjectStore } from "../../store/projectStore";
import { useModelStore } from "../../store/modelStore";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useToast } from "../../components/shell/ToastProvider";
import { Tag } from "../../components/ui/Tag";
import { Btn } from "../../components/ui/Btn";

import type { CheckpointEntry } from "../../lib/api-types";
import { basename } from "../../lib/path";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(n: number | null | undefined, dec = 2): string {
  if (n == null) return "—";
  return n.toFixed(dec);
}

function fmtSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

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

// ── Checkpoints Table (§13.2) ─────────────────────────────────────────────

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

// ── Storage Summary Panel (§13.6) ─────────────────────────────────────────

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

// ── Delete-Disabled Banner (§13.5) ────────────────────────────────────────

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

// ── Detail Panel (§13.3) ─────────────────────────────────────────────────

interface DetailPanelProps {
  entry: CheckpointEntry | null;
  trainingActive: boolean;
  onExportPth: (e: CheckpointEntry) => void;
  onExportOnnx: (e: CheckpointEntry) => void;
  onDeleteRequest: (e: CheckpointEntry) => void;
  onRunInference: () => void;
  onResume: (e: CheckpointEntry) => void;
}

function DetailPanel({
  entry, trainingActive, onExportPth, onExportOnnx, onDeleteRequest, onRunInference, onResume,
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

          {/* Export buttons (§13.4) */}
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

          {/* §24.1 — Resume from checkpoint */}
          <div style={{
            padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6,
            borderTop: "1px solid var(--border)", flexShrink: 0,
          }}>
            <Btn variant="solid" color="var(--blue)" full onClick={() => onResume(entry)}>
              Resume Training →
            </Btn>
          </div>

          {/* Navigation + Delete (§13.9a, §13.5) */}
          <div style={{
            padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6,
            borderTop: "1px solid var(--border)", flexShrink: 0,
          }}>
            <Btn variant="ghost" color="var(--blue)" full onClick={onRunInference}>
              Run Inference →
            </Btn>
            <Btn
              variant="ghost" color="var(--red)" full
              disabled={trainingActive}
              title={trainingActive ? "Cannot delete checkpoints while training is active" : undefined}
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

// ── Delete Confirmation Scrim (§13.5) ─────────────────────────────────────

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

// ── ScreenCheckpoints — §13.1 layout + §23.2+23.7 RunSelectorPanel ───────

export function ScreenCheckpoints() {
  const status         = useTrainingStore((s) => s.status);
  const activeRunId    = useTrainingStore((s) => s.activeTrainingRunId);
  const setActiveTab   = useUiStore((s) => s.setActiveTab);
  const checkpointsByRun  = useCheckpointStore((s) => s.checkpointsByRun);
  const lastExportDone    = useCheckpointStore((s) => s.lastExportDone);
  const setLastExportDone = useCheckpointStore((s) => s.setLastExportDone);
  const project        = useProjectStore((s) => s.project);
  const { show } = useToast();

  const trainingActive = status === "running";
  const runId = activeRunId;
  const allEntries: CheckpointEntry[] = runId ? (checkpointsByRun[runId] ?? []) : [];

  const [sortCol, setSortCol]       = useState<SortCol>("epoch");
  const [sortDir, setSortDir]       = useState<SortDir>("desc");
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [deletingEntry, setDeletingEntry] = useState<CheckpointEntry | null>(null);

  // §13.8 — Auto-tag highest PSNR checkpoint as "best"
  const bestPsnrPath = useMemo(() => {
    let best: CheckpointEntry | null = null;
    for (const e of allEntries) {
      if (e.metrics.psnr != null && (best == null || e.metrics.psnr > (best.metrics.psnr ?? -Infinity))) {
        best = e;
      }
    }
    return best?.path ?? null;
  }, [allEntries]);

  // "latest" = last entry in arrival order
  const latestPath = allEntries[allEntries.length - 1]?.path ?? null;

  const sorted = useMemo(
    () => sortEntries(allEntries, sortCol, sortDir),
    [allEntries, sortCol, sortDir],
  );

  const selectedEntry = sorted.find((e) => e.path === selectedPath) ?? null;

  // §13.4 — Fire export-done toast from ipc.ts signal
  const prevExportRef = useRef<typeof lastExportDone>(null);
  useEffect(() => {
    if (lastExportDone && lastExportDone !== prevExportRef.current) {
      prevExportRef.current = lastExportDone;
      show("success", `Exported ${basename(lastExportDone.path)} (${fmtSize(lastExportDone.sizeMb)})`, 4000);
      setLastExportDone(null);
    }
  }, [lastExportDone, show, setLastExportDone]);

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
    // TODO: replace with api call
    if (selectedPath === deletingEntry.path) setSelectedPath(null);
    setDeletingEntry(null);
  };

  const handleExportPth = (_e: CheckpointEntry) => {
    // TODO: replace with api call
  };

  const handleExportOnnx = (_e: CheckpointEntry) => {
    // TODO: replace with api call
  };

  // §24.2 — Pre-fill Training Setup from run config and set resumeFrom
  const handleResume = (e: CheckpointEntry) => {
    const run = project?.runs.find((r) => r.run_id === runId);
    if (run) {
      useModelStore.getState().setArchitecture(run.architecture.type);
      useModelStore.getState().setHyperparameters({
        scale: run.architecture.upscale_factor,
        batchSize: run.training_config.batch_size,
        learningRate: run.training_config.learning_rate,
        lrScheduler: run.training_config.scheduler,
        optimizer: run.training_config.optimizer,
        patchSize: run.training_config.patch_size,
      });
      useRunConfigStore.getState().setSchedule({ totalEpochs: run.training_config.num_epochs });
    }
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
      {/* Checkpoint table */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <CheckpointsTable
              entries={sorted}
              bestPsnrPath={bestPsnrPath}
              latestPath={latestPath}
              selectedPath={selectedPath}
              sortCol={sortCol}
              sortDir={sortDir}
              trainingActive={trainingActive}
              onSort={handleSort}
              onSelect={(e) => setSelectedPath(e.path)}
              onDeleteRequest={handleDeleteRequest}
            />
            <StorageSummaryPanel entries={allEntries} />
            {trainingActive && <DeleteDisabledBanner />}
          </div>

          {/* Right detail panel — 210px (§13.3) */}
          <DetailPanel
            entry={selectedEntry}
            trainingActive={trainingActive}
            onExportPth={handleExportPth}
            onExportOnnx={handleExportOnnx}
            onDeleteRequest={handleDeleteRequest}
            onResume={handleResume}
            onRunInference={() => {
              // §13.9b: pre-select checkpoint path in inferenceStore before navigating
              if (selectedPath) {
                useInferenceStore.getState().setPreselectedCheckpointPath(selectedPath);
              }
              setActiveTab("inference");
            }}
          />

      {/* Delete confirmation scrim — position:absolute (§13.5) */}
      {deletingEntry != null && (
        <DeleteConfirmScrim
          entry={deletingEntry}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeletingEntry(null)}
        />
      )}
    </div>
  );
}
