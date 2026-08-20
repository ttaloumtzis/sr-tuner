import { useState, useEffect } from "react";
import { useInferenceStore } from "../../store/inferenceStore";
import { Field } from "../../components/ui/Field";
import { Dropdown, type DropdownOption } from "../../components/ui/Dropdown";
import { Btn } from "../../components/ui/Btn";
import { InlineAlert } from "../../components/ui/InlineAlert";
import { basename, join } from "../../lib/path";
import type { CheckpointEntry, ModelVersion, RunInfo } from "../../lib/api-types";
import { buildRunDisplays, shortRunId } from "../../lib/runLabel";
import { useCheckpointStore } from "../../store/checkpointStore";

export function ModelPanel() {
  const store = useInferenceStore();
  const [instances, setInstances] = useState<DropdownOption[]>([]);
  const [instanceMeta, setInstanceMeta] = useState<Record<string, { architecture: string | null; scale: number | null }>>({});
  const [selInstance, setSelInstance] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [runDisplays, setRunDisplays] = useState<Map<string, { group: string; label: string }>>(new Map());
  const [selRun, setSelRun] = useState<string | null>(null);
  const [runCheckpoints, setRunCheckpoints] = useState<CheckpointEntry[]>([]);
  const [sourceTab, setSourceTab] = useState<"version" | "run-checkpoint">("version");

  // Model source is derived from store state: version wins, then a run
  // checkpoint (instance + checkpoint path), then a raw file, then the tab.
  const mode: "version" | "run-checkpoint" | "raw" = store.version
    ? "version"
    : store.instance && store.modelPath
      ? "run-checkpoint"
      : store.modelPath
        ? "raw"
        : sourceTab;

  // Fetch model instances on mount, then sync store state.
  useEffect(() => {
    (async () => {
      try {
        const { listInstances } = await import("../../lib/api");
        const list = await listInstances();
        setInstances(list.map((i) => ({ value: i.name, label: i.name })));
        const meta: Record<string, { architecture: string | null; scale: number | null }> = {};
        for (const i of list) meta[i.name] = { architecture: i.architecture, scale: i.scale };
        setInstanceMeta(meta);

        // If the store already has an instance selected, seed the local state
        // so stale selections are cleared. Version-mode only: fetching versions
        // is pointless for a run-checkpoint (no version selected).
        const storedInstance = store.instance;
        if (storedInstance && list.some((i) => i.name === storedInstance) && store.version) {
          setSelInstance(storedInstance);
          try {
            const { getInstanceVersions } = await import("../../lib/api");
            const vlist = await getInstanceVersions(storedInstance);
            setVersions(vlist);
            if (vlist.length > 0 && store.version) {
              const stillExists = vlist.some((v) => v.tag === store.version);
              if (!stillExists) {
                store.setVersion(null);
                store.setModelPath(null);
              }
            }
          } catch {
            setVersions([]);
          }
        }
      } catch {
        setInstances([]);
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // §13.9b — a checkpoint preselected from the Checkpoints tab. It carries the
  // owning instance; reset all local cascade/version state so a stale instance
  // dropdown or badges can't render alongside the checkpoint file.
  useEffect(() => {
    const pre = store.preselectedCheckpointPath;
    if (pre) {
      setSelInstance(null);
      setVersions([]);
      setRuns([]);
      setRunDisplays(new Map());
      setSelRun(null);
      setRunCheckpoints([]);
      store.setInstance(store.preselectedInstance);
      store.setVersion(null);
      store.setModelPath(pre);
      store.setPreselectedInstance(null);
      store.setPreselectedCheckpointPath(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.preselectedCheckpointPath]);

  // ── Version-mode handlers ────────────────────────────────────────────────

  const handleInstanceChange = async (name: string) => {
    setSelInstance(name || null);
    setVersions([]);
    if (!name) {
      store.setInstance(null);
      store.setModelPath(null);
      store.setCheckpointContext(null, null);
      return;
    }
    store.setInstance(name);
    store.setCheckpointContext(null, null);
    try {
      const { getInstanceVersions } = await import("../../lib/api");
      const list = await getInstanceVersions(name);
      setVersions(list);
      // Auto-select the latest *available* version (has_weights !== false)
      const available = list.filter((v) => v.has_weights !== false);
      if (available.length > 0) {
        const latest = available[available.length - 1];
        store.setVersion(latest.tag);
        store.setModelPath(join(latest.path, "model.pt"));
      }
    } catch {
      setVersions([]);
    }
  };

  const handleVersionChange = (tag: string) => {
    const v = versions.find((x) => x.tag === tag);
    if (!v) {
      store.setVersion(null);
      store.setModelPath(null);
      return;
    }
    // Don't select versions missing weights
    if (v.has_weights === false) {
      store.setVersion(null);
      store.setModelPath(null);
      return;
    }
    store.setVersion(v.tag);
    store.setModelPath(join(v.path, "model.pt"));
  };

  // ── Run-checkpoint cascade handlers ─────────────────────────────────────

  const handleCheckpointInstanceChange = async (name: string) => {
    setSelInstance(name || null);
    setRuns([]);
    setSelRun(null);
    setRunCheckpoints([]);
    store.setCheckpointContext(null, null);
    if (!name) return;
    try {
      const { listRuns } = await import("../../lib/api");
      const all = await listRuns();
      const inst = all.find((m) => m.name === name);
      const instRuns = inst?.runs ?? [];
      setRuns(instRuns);
      setRunDisplays(buildRunDisplays(instRuns));
    } catch {
      setRuns([]);
      setRunDisplays(new Map());
    }
  };

  const handleRunChange = async (runId: string) => {
    setSelRun(runId || null);
    setRunCheckpoints([]);
    if (!runId || !selInstance) return;
    try {
      const { listRunCheckpoints } = await import("../../lib/api");
      const entries = await listRunCheckpoints(selInstance, runId);
      setRunCheckpoints(entries);
      // Mirror into checkpointStore — the shared checkpoints cache.
      useCheckpointStore.getState().setCheckpointsForRun(runId, entries);
    } catch {
      setRunCheckpoints([]);
    }
  };

  const handleEpochChange = (path: string) => {
    const entry = runCheckpoints.find((e) => e.path === path);
    if (!entry || !selInstance || !selRun) return;
    store.setVersion(null);
    store.setModelPath(entry.path);
    store.setInstance(selInstance);
    const label = runDisplays.get(selRun)?.label ?? shortRunId(selRun);
    store.setCheckpointContext(selRun, label);
  };

  const handleChangeCheckpoint = () => {
    store.setModelPath(null);
    store.setCheckpointContext(null, null);
    if (selInstance) handleCheckpointInstanceChange(selInstance);
  };

  // ── Source toggle ───────────────────────────────────────────────────────

  const handleSelectVersionSource = () => {
    setSourceTab("version");
    setSelInstance(null);
    setVersions([]);
    setRuns([]);
    setRunDisplays(new Map());
    setSelRun(null);
    setRunCheckpoints([]);
    store.setInstance(null);
    store.setModelPath(null);
    store.setCheckpointContext(null, null);
  };

  const handleSelectRunCheckpointSource = () => {
    setSourceTab("run-checkpoint");
    setSelInstance(null);
    setVersions([]);
    setRuns([]);
    setRunDisplays(new Map());
    setSelRun(null);
    setRunCheckpoints([]);
    store.setInstance(null);
    store.setModelPath(null);
    store.setCheckpointContext(null, null);
  };

  const missingAny = versions.some((v) => v.has_weights === false);
  const versionOptions: DropdownOption[] = versions
    .filter((v) => v.has_weights !== false)
    .map((v) => ({
      value: v.tag,
      label: v.tag,
    }));

  const runOptions: DropdownOption[] = runs.map((r) => ({
    value: r.run_id,
    label: runDisplays.get(r.run_id)?.label ?? shortRunId(r.run_id),
  }));

  const epochOptions: DropdownOption[] = runCheckpoints.map((e) => {
    const psnr = e.metrics.psnr != null ? ` · PSNR ${e.metrics.psnr.toFixed(2)}` : "";
    const ssim = e.metrics.ssim != null ? ` · SSIM ${e.metrics.ssim.toFixed(4)}` : "";
    return { value: e.path, label: `${e.filename}${psnr}${ssim}` };
  });

  const meta = selInstance ? instanceMeta[selInstance] : undefined;

  const badgeRow = meta && (
    <div className="si-badge-row">
      {meta.architecture && (
        <span className="si-badge si-badge-green">
          {meta.architecture}
        </span>
      )}
      {meta.scale != null && (
        <span className="si-badge si-badge-blue">
          {meta.scale}×
        </span>
      )}
    </div>
  );

  return (
    <div className="si-stack">
      {/* Model source toggle */}
      <div className="si-toggle-row">
        <Btn small variant={mode === "version" ? "solid" : "ghost"} color="var(--green)" onClick={handleSelectVersionSource}>
          Model version
        </Btn>
        <Btn small variant={mode === "run-checkpoint" ? "solid" : "ghost"} color="var(--green)" onClick={handleSelectRunCheckpointSource}>
          Run checkpoint
        </Btn>
      </div>

      {mode === "version" && (
        <>
          <Field label="Model">
            <Dropdown
              value={selInstance ?? ""}
              options={instances}
              onChange={handleInstanceChange}
              placeholder="Select model…"
            />
          </Field>
          {badgeRow}
          <div>
            {missingAny && (
              <div className="si-mb-6">
                <InlineAlert tone="muted">
                  Some versions are missing weights and are not selectable.
                </InlineAlert>
              </div>
            )}
            <Field label="Version">
              <Dropdown
                value={store.version ?? ""}
                options={versionOptions}
                onChange={handleVersionChange}
                placeholder={selInstance ? (versionOptions.length ? "Select version…" : "No versions yet — save a checkpoint first") : "Select a model first"}
                mono
              />
            </Field>
          </div>
        </>
      )}

      {mode === "run-checkpoint" && (store.instance && store.modelPath ? (
        // A checkpoint is selected — show the owning instance + run/epoch context.
        <div className="si-stack-6">
          <div className="si-badge-row-wrap">
            <span className="si-badge si-badge-green">
              {store.instance}
            </span>
            {meta && meta.architecture && (
              <span className="si-badge si-badge-green">
                {meta.architecture}
              </span>
            )}
            {meta && meta.scale != null && (
              <span className="si-badge si-badge-blue">
                {meta.scale}×
              </span>
            )}
          </div>
          <div className="si-path-box" title={store.modelPath ?? ""}>
            {store.checkpointRunLabel ? `${store.checkpointRunLabel} · ${basename(store.modelPath ?? "")}` : basename(store.modelPath ?? "")}
          </div>
          <div>
            <Btn small onClick={handleChangeCheckpoint}>Change checkpoint</Btn>
          </div>
        </div>
      ) : (
        // Cascade: instance → run → epoch.
        <>
          <Field label="Model Instance">
            <Dropdown
              value={selInstance ?? ""}
              options={instances}
              onChange={handleCheckpointInstanceChange}
              placeholder="Select instance…"
            />
          </Field>
          {badgeRow}
          {selInstance && (
            <Field label="Run">
              {runs.length > 0 ? (
                <Dropdown
                  value={selRun ?? ""}
                  options={runOptions}
                  onChange={handleRunChange}
                  placeholder="Select run…"
                  mono
                />
              ) : (
                <div className="si-hint">No runs yet — start training</div>
              )}
            </Field>
          )}
          {selRun && (
            <Field label="Epoch">
              {runCheckpoints.length > 0 ? (
                <Dropdown
                  value={store.modelPath ?? ""}
                  options={epochOptions}
                  onChange={handleEpochChange}
                  placeholder="Select checkpoint…"
                  mono
                />
              ) : (
                <div className="si-hint">No checkpoints saved yet</div>
              )}
            </Field>
          )}
        </>
      ))}

      {mode === "raw" && (
        <div className="si-stack-4">
          <span className="si-file-label">Model file</span>
          <div className="si-path-box" title={store.modelPath ?? ""}>
            {basename(store.modelPath ?? "")}
          </div>
          <div>
            <Btn small onClick={() => { store.setModelPath(null); store.setCheckpointContext(null, null); }}>Clear</Btn>
          </div>
        </div>
      )}
    </div>
  );
}