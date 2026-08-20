import { Tag } from "../../components/ui/Tag";
import type { ModelVersion } from "../../lib/api-types";
import { fmtTimestamp } from "../../lib/format";

function CfgChip({ label, value }: { label: string; value: string }) {
  return (
    <span style={{
      fontSize: 10, background: "var(--bg3)", color: "var(--muted)",
      padding: "1px 6px", borderRadius: 4, whiteSpace: "nowrap",
    }}>
      {label}: <b style={{ color: "var(--text)", fontWeight: 600 }}>{value}</b>
    </span>
  );
}

export function VersionCard({
  version, onDelete, trainingActive,
}: {
  version: ModelVersion;
  onDelete: () => void;
  trainingActive: boolean;
}) {
  const m = version.metadata ?? {};
  const ts = (m as any).timestamp ?? (m as any).created_at;
  const runName = (m as any).run_name;
  const tc = (m as any).training_config;
  const missingWeights = version.has_weights === false;

  return (
    <div className={`mv-version-card${missingWeights ? " missing" : ""}`}>
      <div className="mv-version-card-top">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="mv-version-tag">{version.tag}</span>
          {missingWeights && <Tag color="red">missing</Tag>}
        </div>
        <div className="mv-version-card-actions">
          <span className="mv-version-date">{fmtTimestamp(ts)}</span>
          <button
            onClick={(ev) => { ev.stopPropagation(); onDelete(); }}
            disabled={trainingActive}
            title={trainingActive ? "Cannot delete versions while training is active" : `Delete ${version.tag}`}
            style={{
              background: "none", border: "none",
              color: trainingActive ? "var(--dim)" : "var(--red)",
              cursor: trainingActive ? "default" : "pointer",
              fontSize: 14, lineHeight: 1, padding: "0 4px",
              opacity: trainingActive ? 0.4 : 1,
              transition: "var(--transition-fast)",
            }}
          >
            ✕
          </button>
        </div>
      </div>
      {runName && (
        <div className="mv-version-run">
          Run: <span>{runName}</span>
        </div>
      )}
      {tc && (
        <div className="mv-version-chips">
          {tc.epochs != null && <CfgChip label="Epochs" value={String(tc.epochs)} />}
          {tc.batch_size != null && <CfgChip label="BS" value={String(tc.batch_size)} />}
          {tc.learning_rate != null && <CfgChip label="LR" value={String(tc.learning_rate)} />}
          {tc.patch_size != null && <CfgChip label="Patch" value={`${tc.patch_size}`} />}
          {tc.dtype != null && <CfgChip label="DType" value={tc.dtype} />}
          {tc.seed != null && <CfgChip label="Seed" value={String(tc.seed)} />}
        </div>
      )}
    </div>
  );
}