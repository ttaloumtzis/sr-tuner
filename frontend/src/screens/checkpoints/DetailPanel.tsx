import type { CheckpointEntry } from "../../lib/api-types";
import { Btn } from "../../components/ui/Btn";
import { InfoRow } from "../../components/ui/InfoRow";
import { fmt, fmtSize, fmtDate } from "../../lib/format";

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

export function DetailPanel({
  entry, deleteDisabled, deleteDisabledTitle, onExportPth, onExportOnnx, onDeleteRequest, onRunInference, onResume,
}: DetailPanelProps) {
  return (
    <div className="sc-detail">
      <div className="sc-detail-header">
        CHECKPOINT DETAIL
      </div>

      {entry == null ? (
        <div className="sc-empty">
          <span className="sc-mono-dim">
            Select a checkpoint
          </span>
        </div>
      ) : (
        <div className="sc-detail-body">
          {/* Validation preview placeholder */}
          <div className="sc-preview">
            <span className="sc-mono-dim-10">
              No preview
            </span>
          </div>

          {/* Metrics + file info */}
          <div className="sc-metrics">
            <InfoRow label="EPOCH" value={`${entry.epoch}`} baseline border={false} ellipsis labelSize={10} labelMono />
            <InfoRow label="PSNR" value={`${fmt(entry.metrics.psnr)} dB`} baseline border={false} ellipsis labelSize={10} labelMono />
            <InfoRow label="SSIM" value={fmt(entry.metrics.ssim, 4)} baseline border={false} ellipsis labelSize={10} labelMono />
            <InfoRow label="SIZE" value={fmtSize(entry.file_size_mb)} baseline border={false} ellipsis labelSize={10} labelMono />
            <InfoRow label="SAVED" value={fmtDate(entry.created_at)} mono={false} baseline border={false} ellipsis labelSize={10} labelMono />
          </div>

          <div className="sc-filename">
            {entry.filename}
          </div>

          {/* Export buttons */}
          <div className="sc-btn-col">
            <Btn variant="solid" color="var(--green)" full onClick={() => onExportPth(entry)}>
              Export .pth
            </Btn>
            <Btn variant="ghost" color="var(--blue)" full onClick={() => onExportOnnx(entry)}>
              Export ONNX
            </Btn>
          </div>

          {/* Resume from checkpoint */}
          <div className="sc-btn-col">
            <Btn variant="solid" color="var(--blue)" full onClick={() => onResume(entry)}>
              Resume Training →
            </Btn>
          </div>

          {/* Navigation + Delete */}
          <div className="sc-btn-col">
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