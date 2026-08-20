import type { CheckpointEntry } from "../../lib/api-types";
import { Tag } from "../../components/ui/Tag";
import { fmt, fmtSize, fmtDate } from "../../lib/format";

export type SortCol = "epoch" | "psnr" | "ssim" | "size" | "date";
export type SortDir = "asc" | "desc";

export function sortEntries(entries: CheckpointEntry[], col: SortCol, dir: SortDir): CheckpointEntry[] {
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
      className={`sc-colhead${active ? " sc-colhead-active" : ""}`}
    >
      {label}
      {active && <span className="sc-arrow">{dir === "asc" ? "▲" : "▼"}</span>}
    </div>
  );
}

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

export function CheckpointsTable({
  entries, bestPsnrPath, latestPath, selectedPath,
  sortCol, sortDir, trainingActive, onSort, onSelect, onDeleteRequest,
}: TableProps) {
  if (entries.length === 0) {
    return (
      <div className="sc-empty">
        <span className="sc-mono-dim">
          No checkpoints saved yet
        </span>
      </div>
    );
  }

  return (
    <div className="sc-table">
      {/* Header */}
      <div className="sc-table-head">
        <ColHeader label="EPOCH" col="epoch" current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="PSNR"  col="psnr"  current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="SSIM"  col="ssim"  current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="SIZE"  col="size"  current={sortCol} dir={sortDir} onSort={onSort} />
        <ColHeader label="SAVED" col="date"  current={sortCol} dir={sortDir} onSort={onSort} />
        <div className="sc-colhead-static">TAG</div>
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
            className={`sc-table-row${isSel ? " sc-table-row-selected" : ""}`}
          >
            <span className="sc-cell sc-cell-green">
              {String(e.epoch).padStart(3, "0")}
            </span>
            <span className="sc-cell">{fmt(e.metrics.psnr)} dB</span>
            <span className="sc-cell">{fmt(e.metrics.ssim, 4)}</span>
            <span className="sc-cell sc-cell-dim">{fmtSize(e.file_size_mb)}</span>
            <span className="sc-cell sc-cell-dim-sm">{fmtDate(e.created_at)}</span>
            <div className="sc-tags">
              {isBest   && <Tag color="green">best</Tag>}
              {isLatest && <Tag color="blue">latest</Tag>}
            </div>
            <div className="sc-del-cell">
              <button
                onClick={(ev) => { ev.stopPropagation(); onDeleteRequest(e); }}
                disabled={trainingActive}
                title={
                  trainingActive
                    ? "Cannot delete checkpoints while training is active"
                    : `Delete ${e.filename}`
                }
                className="sc-del-btn"
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