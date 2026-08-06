import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act, waitFor, fireEvent } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useDatasetStore } from "../../../store/datasetStore";
import { useProjectStore, type SRProject } from "../../../store/projectStore";
import { ToastProvider } from "../../../components/shell/ToastProvider";
import type { ScannedDataset } from "../../../lib/scanDatasets";

const mockInvoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => mockInvoke(...args),
  convertFileSrc: (p: string) => p,
}));

const mockScan = vi.fn();
const mockMerge = vi.fn();

vi.mock("../../../lib/scanDatasets", () => ({
  scanDatasets: (...args: unknown[]) => mockScan(...args),
}));
vi.mock("../../../lib/api", () => ({
  mergeDatasets: (...args: unknown[]) => mockMerge(...args),
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

function scanned(name: string, scale: number, pairCount: number): ScannedDataset {
  return {
    name,
    path: `/proj/datasets/${name}`,
    scale,
    pairCount,
    hasManifest: true,
    hasHr: true,
    hasLr: true,
    isMerged: false,
  };
}

async function renderMerge() {
  const { ScreenMergeDatasets } = await import("../ScreenMergeDatasets");
  return render(
    <ToastProvider>
      <ScreenMergeDatasets />
    </ToastProvider>,
  );
}

describe("ScreenMergeDatasets", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
    mockInvoke.mockReset();
    mockInvoke.mockResolvedValue(true);
    mockScan.mockResolvedValue([
      scanned("a", 4, 100),
      scanned("b", 4, 50),
      scanned("c", 2, 25),
    ]);
    mockMerge.mockResolvedValue({ job_id: "merge.1", status: "accepted" });
  });

  it("groups datasets by scale in the merge preview", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    fireEvent.click(screen.getByRole("button", { name: /preview merge/i }));
    await screen.findByText(/Merge ×4/i);
    expect(screen.getByText("150 pairs total")).toBeTruthy();
    expect(screen.getByText(/Merge ×2/i)).toBeTruthy();
    expect(screen.getByText("25 pairs total")).toBeTruthy();
  });

  it("warns when a custom name is combined with multiple scale groups", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    fireEvent.change(screen.getByPlaceholderText(/Leave empty for auto-naming/i), {
      target: { value: "mymer" },
    });
    await screen.findByText(/Custom name with multiple scale groups will raise an error/i);
  });

  it("executes a merge with the selected dataset paths", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    fireEvent.click(screen.getByRole("button", { name: /execute merge/i }));
    await waitFor(() => expect(mockMerge).toHaveBeenCalled());
    const arg = mockMerge.mock.calls[0][0];
    expect(arg.input).toBe("/proj/datasets");
    expect(arg.input_datasets).toEqual([
      "/proj/datasets/a",
      "/proj/datasets/b",
      "/proj/datasets/c",
    ]);
    expect(useDatasetStore.getState().jobType).toBe("merge");
    expect(useDatasetStore.getState().jobStatus).toBe("running");
  });

  it("shows the filtered list when a scale filter is selected", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");
    act(() => {
      useDatasetStore.getState().setMergeScaleFilter(2);
    });
    await waitFor(() => {
      expect(screen.queryByText("a")).toBeNull();
      expect(screen.getByText("c")).toBeTruthy();
    });
  });

  it("can clear and re-select all datasets", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    fireEvent.click(screen.getByRole("button", { name: /^clear$/i }));
    expect(screen.getByRole("button", { name: /execute merge/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /preview merge/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /^select all$/i }));
    expect(screen.getByRole("button", { name: /execute merge/i })).toBeEnabled();
  });

  it("filters datasets by search query", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    fireEvent.change(screen.getByLabelText("Search datasets"), {
      target: { value: "c" },
    });
    expect(screen.queryByText("a")).toBeNull();
    expect(screen.getByText("c")).toBeTruthy();
  });

  it("shows progress while a merge is running", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    act(() => {
      useDatasetStore.getState().setJobType("merge");
      useDatasetStore.getState().setJobStatus("running");
      useDatasetStore.getState().startProgressStep("Merging scale 4 datasets", 2);
      useDatasetStore.getState().updateProgressStep(0, 1);
    });
    expect(screen.getByText("Merging scale 4 datasets")).toBeTruthy();
    expect(screen.getByText("50%")).toBeTruthy();
  });

  it("surfaces merge job errors", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    act(() => {
      useDatasetStore.getState().setJobType("merge");
      useDatasetStore.getState().setJobStatus("error");
      useDatasetStore.getState().setJobError("boom");
    });
    expect(screen.getByText(/Merge failed: boom/i)).toBeTruthy();
  });

  it("rescans the dataset list after a merge finishes", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");
    const callsBefore = mockScan.mock.calls.length;

    act(() => {
      useDatasetStore.getState().setJobType("merge");
      useDatasetStore.getState().setJobStatus("done");
    });
    await waitFor(() => {
      expect(mockScan.mock.calls.length).toBeGreaterThan(callsBefore);
    });
  });

  it("shows merge results and the source-removal note", async () => {
    act(() => {
      useProjectStore.getState().setProject(makeProject());
    });
    await renderMerge();
    await screen.findByText("a");

    act(() => {
      useDatasetStore.getState().setMergeResults([
        {
          scale: 4,
          output_path: "/proj/datasets/merged-x4",
          source_datasets: ["/proj/datasets/a", "/proj/datasets/b"],
        },
      ]);
    });
    expect(screen.getByText(/Merged ×4/i)).toBeTruthy();
    expect(screen.getByText(/Source datasets have been removed/i)).toBeTruthy();
  });
});
