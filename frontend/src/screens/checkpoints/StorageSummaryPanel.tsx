import type { CheckpointEntry } from "../../lib/api-types";
import { fmtSize } from "../../lib/format";

export function StorageSummaryPanel({ entries }: { entries: CheckpointEntry[] }) {
  const totalMb = entries.reduce((acc, e) => acc + e.file_size_mb, 0);
  return (
    <div className="sc-summary">
      <div>
        <span className="sc-summary-label">FILES </span>
        <span className="sc-summary-value">
          {entries.length}
        </span>
      </div>
      <div>
        <span className="sc-summary-label">TOTAL </span>
        <span className="sc-summary-value">
          {fmtSize(totalMb)}
        </span>
      </div>
    </div>
  );
}