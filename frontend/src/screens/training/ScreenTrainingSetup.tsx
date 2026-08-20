import { useState, useCallback, useEffect, useRef } from "react";
import "./ScreenTrainingSetup.css";
import { Panel } from "../../components/ui/Panel";
import { IconSliders, IconCpu, IconDatabase } from "../../components/ui/icons";
import { useRunConfigStore } from "../../store/runConfigStore";
import { useUiStore } from "../../store/uiStore";
import { ModelDataSection, type InstanceOption, type DatasetOption } from "./ModelDataSection";
import { HyperparamsSection } from "./HyperparamsSection";
import { LossSection } from "./LossSection";
import { AdvancedSection } from "./AdvancedSection";
import { TrainingSidebar } from "./TrainingSidebar";

export function ScreenTrainingSetup() {
  const workspaceReady = useUiStore((s) => s.workspaceReady);

  const [instances, setInstances] = useState<InstanceOption[]>([]);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [customConfigPath, setCustomConfigPath] = useState("");
  const [gpuTotalVramGb, setGpuTotalVramGb] = useState<number | null>(null);
  const [instancesError, setInstancesError] = useState<string | null>(null);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [fetchKey, setFetchKey] = useState(0);

  const refreshLists = useCallback(() => setFetchKey((k) => k + 1), []);

  useEffect(() => {
    (async () => {
      try {
        const { getEnv } = await import("../../lib/api");
        const env = await getEnv();
        if (env.vram_total_mb) {
          setGpuTotalVramGb(env.vram_total_mb / 1024);
        }
      } catch { console.warn("getEnv failed in training setup"); }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const { listInstances } = await import("../../lib/api");
        const list = await listInstances();
        setInstances(list.map((i: { name: string; architecture: string | null }) => ({
          value: i.name,
          label: `${i.name}${i.architecture ? ` (${i.architecture})` : ""}`,
        })));
        setInstancesError(null);
        const storedInstance = useRunConfigStore.getState().selectedInstance;
        if (storedInstance && !list.some((i: { name: string }) => i.name === storedInstance)) {
          useRunConfigStore.getState().setSelectedInstance(null);
        }
      } catch (e) {
        setInstances([]);
        setInstancesError(e instanceof Error ? e.message : String(e));
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceReady, fetchKey]);

  useEffect(() => {
    (async () => {
      try {
        const { listDatasets } = await import("../../lib/api");
        const list = await listDatasets();
        setDatasets(list.map((d: { name: string; path: string; num_pairs: number; scale: number }) => ({
          value: d.name,
          label: `${d.name} (${d.scale}× · ${d.num_pairs} pairs)`,
          path: d.path,
          pairs: d.num_pairs,
          scale: d.scale,
        })));
        setDatasetsError(null);
        const storedDataset = useRunConfigStore.getState().selectedDataset;
        if (storedDataset) {
          const match = list.find((d: { name: string }) => d.name === storedDataset);
          if (match) {
            useRunConfigStore.getState().setSelectedDatasetPath(match.path);
            useRunConfigStore.getState().setSelectedDatasetPairs(match.num_pairs);
          } else {
            useRunConfigStore.getState().setSelectedDataset(null);
            useRunConfigStore.getState().setSelectedDatasetPath(null);
            useRunConfigStore.getState().setSelectedDatasetPairs(null);
          }
        }
      } catch (e) {
        setDatasets([]);
        setDatasetsError(e instanceof Error ? e.message : String(e));
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceReady, fetchKey]);

  // Re-hydrate instance details when selectedInstance changes
  const prevInstanceRef = useRef(useRunConfigStore.getState().selectedInstance);
  const selInst = useRunConfigStore((st) => st.selectedInstance);
  useEffect(() => {
    const name = selInst;
    const changed = name !== prevInstanceRef.current;
    prevInstanceRef.current = name;
    const st = useRunConfigStore.getState();
    if (!name) {
      st.setInstanceArchitecture(null);
      st.setInstanceScale(null);
      st.setInstanceConfig(null);
      st.setInstanceVersions([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const { getInstance, getInstanceVersions } = await import("../../lib/api");
        const inst = await getInstance(name);
        if (cancelled) return;
        const cur = useRunConfigStore.getState();
        cur.setInstanceArchitecture(inst.architecture);
        cur.setInstanceScale(inst.scale ?? null);
        cur.setInstanceConfig(inst.config ?? null);
        const versions = await getInstanceVersions(name);
        if (cancelled) return;
        const available = versions.filter((v: { has_weights?: boolean }) => v.has_weights !== false);
        const versionList = available.map((v: { tag: string }) => ({ tag: v.tag, path: "" }));
        const cur2 = useRunConfigStore.getState();
        cur2.setInstanceVersions(versionList);
        if (changed && cur2.resumeFrom === null && versionList.length > 0) {
          cur2.setResumeFrom("latest");
        }
      } catch {
        if (cancelled) return;
        const cur = useRunConfigStore.getState();
        cur.setInstanceArchitecture(null);
        cur.setInstanceScale(null);
        cur.setInstanceConfig(null);
        cur.setInstanceVersions([]);
        cur.setSelectedInstance(null);
      }
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selInst]);

  return (
    <div className="ts-layout">
      <div className="ts-main">
        {/* Model & Data — unified panel: instance, dataset and run basics */}
        <Panel title="Model & Data" icon={<><IconCpu size={13} /><IconDatabase size={13} /></>}>
          <ModelDataSection
            instances={instances}
            instancesError={instancesError}
            datasets={datasets}
            datasetsError={datasetsError}
            onRefresh={refreshLists}
          />
        </Panel>

        {/* Hyperparameters */}
        <Panel title="Hyperparameters" icon={<IconSliders size={13} />}>
          <HyperparamsSection />
          <LossSection />
        </Panel>

        {/* Advanced — collapsed by default to keep the common path uncluttered */}
        <AdvancedSection customConfigPath={customConfigPath} onCustomConfigPath={setCustomConfigPath} />
      </div>

      {/* Sidebar */}
      <TrainingSidebar gpuTotalVramGb={gpuTotalVramGb} customConfigPath={customConfigPath} />
    </div>
  );
}