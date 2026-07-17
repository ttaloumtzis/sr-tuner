import { useProjectStore } from "../store/projectStore";
import { useTrainingStore } from "../store/trainingStore";
import { useDatasetStore } from "../store/datasetStore";
import { useModelStore } from "../store/modelStore";
import { useRunConfigStore } from "../store/runConfigStore";
import { useUiStore } from "../store/uiStore";
import { useInferenceStore } from "../store/inferenceStore";
import { useCheckpointStore } from "../store/checkpointStore";

export function resetAllStores(): void {
  useProjectStore.setState(useProjectStore.getInitialState(), true);
  useTrainingStore.setState(useTrainingStore.getInitialState(), true);
  useDatasetStore.setState(useDatasetStore.getInitialState(), true);
  useModelStore.setState(useModelStore.getInitialState(), true);
  useRunConfigStore.setState(useRunConfigStore.getInitialState(), true);
  useUiStore.setState(useUiStore.getInitialState(), true);
  useInferenceStore.setState(useInferenceStore.getInitialState(), true);
  useCheckpointStore.setState(useCheckpointStore.getInitialState(), true);
}
