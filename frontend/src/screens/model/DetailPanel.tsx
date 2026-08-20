import { useMemo } from "react";
import { Panel } from "../../components/ui/Panel";
import { Btn } from "../../components/ui/Btn";
import { InfoRow } from "../../components/ui/InfoRow";
import { InlineAlert } from "../../components/ui/InlineAlert";
import type { ModelInstance, ModelVersion } from "../../lib/api-types";
import { estimateParamsFor, formatParamCount, formatWeightMB } from "../../lib/architectures";
import { VersionCard } from "./VersionCard";

export function DetailPanel({
  model, versions, loadingVersions, trainingActive, onRefresh, onDeleteInstance, onDeleteVersion,
}: {
  model: ModelInstance | null;
  versions: ModelVersion[];
  loadingVersions: boolean;
  trainingActive: boolean;
  onRefresh: () => void;
  onDeleteInstance: () => void;
  onDeleteVersion: (v: ModelVersion) => void;
}) {
  const paramsM = useMemo(() => {
    if (!model) return 0;
    const config = model.config as Record<string, unknown> | undefined;
    if (!config) return 0;
    return estimateParamsFor(model.architecture ?? "", config);
  }, [model]);

  const missingCount = versions.filter((v) => v.has_weights === false).length;

  if (!model) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>Select a model instance</span>
      </div>
    );
  }

  return (
    <div className="mv-detail">
      <Panel title={model.name} shrink>
        <InfoRow label="Architecture" value={model.architecture === "swinir" ? "SwinIR" : "RRDB-ESRGAN"} mono={false} labelSize={11} />
        <InfoRow label="Scale" value={model.scale ? `${model.scale}x` : "—"} mono={false} labelSize={11} />
        <InfoRow label="Latest Version" value={model.latest_version ?? "—"} mono labelSize={11} />
        <div style={{ marginTop: 2 }}>
          <InfoRow label="Parameters" value={formatParamCount(paramsM)} mono labelSize={11} />
          <InfoRow label="Weights (f32)" value={`${formatWeightMB(paramsM)} MB`} mono labelSize={11} />
          <InfoRow label="Weights (f16)" value={`${(parseFloat(formatWeightMB(paramsM)) / 2).toFixed(1)} MB`} mono labelSize={11} />
        </div>
      </Panel>

      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <div className="mv-versions-header">
          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
            Versions
          </span>
          <Btn small variant="ghost" onClick={onRefresh} disabled={loadingVersions}>&#x21bb;</Btn>
        </div>

        {loadingVersions ? (
          <span style={{ fontSize: 11, color: "var(--muted)" }}>Loading...</span>
        ) : versions.length === 0 ? (
          <span style={{ fontSize: 11, color: "var(--muted)" }}>No versions yet</span>
        ) : (
          <div className="mv-versions-list">
            {missingCount > 0 && (
              <InlineAlert tone="muted">
                {missingCount} version(s) missing weights — deleted or incomplete.
              </InlineAlert>
            )}
            {versions.map((v) => (
              <VersionCard
                key={v.tag}
                version={v}
                onDelete={() => onDeleteVersion(v)}
                trainingActive={trainingActive}
              />
            ))}
          </div>
        )}
      </div>

      <div style={{ flexShrink: 0, paddingTop: 4 }}>
        <Btn
          variant="ghost"
          color="var(--red)"
          full
          onClick={onDeleteInstance}
          disabled={trainingActive}
          title={trainingActive ? "Cannot delete model while training is active" : undefined}
        >
          Delete Model
        </Btn>
      </div>
    </div>
  );
}