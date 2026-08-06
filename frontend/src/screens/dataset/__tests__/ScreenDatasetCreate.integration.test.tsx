import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act, waitFor, fireEvent } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useDatasetStore } from "../../../store/datasetStore";
import { useProjectStore, type SRProject } from "../../../store/projectStore";
import { ToastProvider } from "../../../components/shell/ToastProvider";

const mockInvoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => mockInvoke(...args),
  convertFileSrc: (p: string) => p,
}));

const mockInspectDataset = vi.fn();
const mockFinalizeDataset = vi.fn();
const mockBuildDataset = vi.fn();
const mockDialogOpen = vi.fn();

vi.mock("../../../lib/api", () => ({
  inspectDataset: (...args: unknown[]) => mockInspectDataset(...args),
  finalizeDataset: (...args: unknown[]) => mockFinalizeDataset(...args),
  buildDataset: (...args: unknown[]) => mockBuildDataset(...args),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: (...args: unknown[]) => mockDialogOpen(...args),
}));

function makeProject(): SRProject {
  return {
    version: "1.0.0",
    name: "test",
    created_at: "2026-01-01T00:00:00Z",
    last_modified_at: "2026-01-01T00:00:00Z",
    default_dataset: {
      training_path: "",
      validation_path: "",
      validation_strategy: "auto_split",
      validation_split_ratio: 0.1,
      dataset_type: "video_extract",
    },
    default_model: { architecture: "rrdb_esrgan", upscale_factor: 4 },
    models: [],
    runs: [],
    ui_state: { last_active_run_id: null, last_active_tab: null, expanded_panels: {} },
    metadata: { app_version: "1.0.0", notes: null, tags: [] },
    filePath: "/proj/test.srproj",
  };
}

async function renderCreate() {
  const { ScreenDatasetCreate } = await import("../ScreenDatasetCreate");
  return render(
    <ToastProvider>
      <ScreenDatasetCreate />
    </ToastProvider>,
  );
}

describe("ScreenDatasetCreate", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
    mockInvoke.mockReset();
    mockInvoke.mockResolvedValue(false);
    mockInspectDataset.mockResolvedValue({
      hr_count: 10,
      lr_count: 10,
      pair_count: 10,
      hr_size: { width: 1024, height: 1024 },
      lr_size: { width: 512, height: 512 },
      scale_ratio: 2,
      scale_exact: true,
      scale_w: 2,
      scale_h: 2,
      has_manifest: false,
      warnings: [],
    });
    mockFinalizeDataset.mockResolvedValue({ num_pairs: 10, scale: 2, path: "/proj/datasets/foo" });
    mockBuildDataset.mockResolvedValue({ job_id: "build.1", status: "accepted" });
  });

  it("keeps On-the-fly mode disabled and ignores clicks on the card", async () => {
    await renderCreate();
    expect(useDatasetStore.getState().mode).toBe("image_folder");
    fireEvent.click(screen.getByText(/On-the-fly/));
    expect(useDatasetStore.getState().mode).toBe("image_folder");
  });

  it("selects a single video file via the dialog and shows one file row", async () => {
    mockDialogOpen.mockResolvedValue("/videos/test.mp4");
    act(() => {
      useDatasetStore.getState().setMode("video_extract");
      useProjectStore.getState().setProject(makeProject());
    });
    await renderCreate();

    fireEvent.click(screen.getByRole("button", { name: /browse files/i }));
    await waitFor(() => {
      expect(useDatasetStore.getState().videoFile).toEqual({
        name: "test.mp4",
        path: "/videos/test.mp4",
      });
    });
    expect(mockDialogOpen).toHaveBeenCalledWith(expect.objectContaining({ multiple: false }));
    expect(screen.getByText("test.mp4")).toBeTruthy();
  });

  it("builds the dataset with the correct payload and starts the build job", async () => {
    act(() => {
      useDatasetStore.getState().setMode("video_extract");
      useProjectStore.getState().setProject(makeProject());
      useDatasetStore.getState().setVideoFile({ name: "test.mp4", path: "/videos/test.mp4" });
      useDatasetStore.getState().setScale(2);
      useDatasetStore.getState().setFrameRate(24);
      useDatasetStore.getState().setMotionBlurEnabled(false);
    });
    await renderCreate();

    fireEvent.click(screen.getByRole("button", { name: /start extraction/i }));
    await waitFor(() => expect(mockBuildDataset).toHaveBeenCalled());

    const arg = mockBuildDataset.mock.calls[0][0];
    expect(arg.input).toBe("/videos/test.mp4");
    expect(arg.config_overrides.scale).toBe(2);
    expect(arg.config_overrides.frame_rate).toBe(24);
    expect(arg.config_overrides.frame_format).toBeUndefined();
    expect(arg.config_overrides.degradation.blur.motion.enabled).toBe(false);

    expect(useDatasetStore.getState().jobType).toBe("build");
    expect(useDatasetStore.getState().jobStatus).toBe("running");
    expect(useDatasetStore.getState().jobId).toBe("build.1");
    expect(useDatasetStore.getState().videoFile).toBeNull();
  });

  it("inspects an existing folder and imports it into the project", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
      useDatasetStore.getState().setRootPath("/data/foo");
    });
    await renderCreate();

    const importBtn = await screen.findByRole("button", { name: /import into project/i });
    expect(mockInspectDataset).toHaveBeenCalledWith({ path: "/data/foo" });
    fireEvent.click(importBtn);
    await waitFor(() =>
      expect(mockFinalizeDataset).toHaveBeenCalledWith({ path: "/proj/datasets/foo", scale: 2 }),
    );
    expect(useDatasetStore.getState().rootPath).toBe("");
  });
});
