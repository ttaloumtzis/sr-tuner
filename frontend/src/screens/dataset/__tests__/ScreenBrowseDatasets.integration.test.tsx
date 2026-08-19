import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act, waitFor, fireEvent } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useDatasetStore } from "../../../store/datasetStore";
import { ToastProvider } from "../../../components/shell/ToastProvider";

const mockListDatasets = vi.fn();
const mockStartValidate = vi.fn();
const mockGetDatasetHealth = vi.fn();
const mockHealthCheck = vi.fn();
const mockPrune = vi.fn();
const mockDelete = vi.fn();

vi.mock("../../../lib/api", () => ({
  listDatasets: (...args: unknown[]) => mockListDatasets(...args),
  startValidateDataset: (...args: unknown[]) => mockStartValidate(...args),
  getDatasetHealth: (...args: unknown[]) => mockGetDatasetHealth(...args),
  healthCheck: (...args: unknown[]) => mockHealthCheck(...args),
  pruneDatasetFiles: (...args: unknown[]) => mockPrune(...args),
  deleteDataset: (...args: unknown[]) => mockDelete(...args),
}));
vi.mock("../../../lib/scanDatasets", () => ({
  getDatasetPairUrls: vi.fn().mockResolvedValue(
    Array.from({ length: 10 }, (_, i) => ({ hr: `hr${i}.png`, lr: `lr${i}.png` })),
  ),
  getDatasetPairInfo: vi.fn().mockResolvedValue(null),
}));

async function renderBrowse() {
  const { ScreenBrowseDatasets } = await import("../ScreenBrowseDatasets");
  return render(
    <ToastProvider>
      <ScreenBrowseDatasets />
    </ToastProvider>,
  );
}

describe("ScreenBrowseDatasets", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
    mockListDatasets.mockResolvedValue([
      { name: "alpha", path: "/data/alpha", scale: 4, num_pairs: 10 },
      { name: "beta", path: "/data/beta", scale: 2, num_pairs: 5 },
    ]);
    mockGetDatasetHealth.mockResolvedValue({
      total_pairs: 10,
      total_hr_images: 10,
      total_lr_images: 10,
      resolutions: { "1024x1024": 10 },
      aspect_ratios: { "1.0": 10 },
      channels: { "RGB (3 channels)": 10 },
      computed_threshold: 18,
      black_frames: ["000001.png", "000002.png"],
      suspicious_frames: [],
      scale_mismatches: [],
      frame_means: {},
      unreadable: [],
    });
    mockStartValidate.mockResolvedValue({ job_id: "dataset.validate_1", status: "accepted" });
  });

  it("renders the dataset list and selects the first dataset", async () => {
    await renderBrowse();
    expect((await screen.findAllByText("alpha")).length).toBeGreaterThan(0);
    expect(screen.getByText("beta")).toBeTruthy();
    expect(screen.getAllByText("10 pairs").length).toBeGreaterThan(0);
  });

  it("Validate starts an async job instead of blocking", async () => {
    await renderBrowse();
    await screen.findAllByText("alpha");

    fireEvent.click(screen.getByRole("button", { name: /validate/i }));
    await waitFor(() => expect(mockStartValidate).toHaveBeenCalledWith({ path: "/data/alpha" }));
    expect(useDatasetStore.getState().jobType).toBe("validate");
    expect(useDatasetStore.getState().jobStatus).toBe("running");
    expect(useDatasetStore.getState().jobDatasetPath).toBe("/data/alpha");
  });

  it("shows a success toast when the validation result arrives via the store", async () => {
    await renderBrowse();
    await screen.findAllByText("alpha");

    act(() => {
      useDatasetStore
        .getState()
        .setValidationResult({ valid: true, problems: [], num_pairs: 10 });
    });
    await screen.findByText(/Dataset validated — 10 pairs, no problems/i);
  });

  it("only offers the slider and split view modes, no fake overlay", async () => {
    await renderBrowse();
    await screen.findAllByText("alpha");

    expect(screen.getByRole("button", { name: /split slider/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /side-by-side/i })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /overlay/i })).toBeNull();
  });

  it("prunes black frames through the async prune job", async () => {
    mockPrune.mockResolvedValue({ job_id: "dataset.prune_1", status: "accepted" });
    await renderBrowse();
    await screen.findAllByText("alpha");
    await screen.findByText("000001.png");

    fireEvent.click(screen.getByLabelText("000001.png"));
    fireEvent.click(screen.getByRole("button", { name: /prune selected/i }));
    await waitFor(() => expect(mockPrune).toHaveBeenCalled());
    expect(mockPrune.mock.calls[0][0]).toEqual({
      path: "/data/alpha",
      files: ["HR/000001.png"],
    });
  });

  it("deletes a dataset after confirmation", async () => {
    mockDelete.mockResolvedValue({ deleted: "alpha" });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    await renderBrowse();
    await screen.findAllByText("alpha");

    fireEvent.click(screen.getByRole("button", { name: /delete/i }));
    await waitFor(() => expect(mockDelete).toHaveBeenCalledWith("alpha"));
  });
});
