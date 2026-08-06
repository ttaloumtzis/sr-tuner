import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act, fireEvent } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useRunConfigStore } from "../../../store/runConfigStore";
import { estimateVramBreakdown } from "../../../lib/vramEstimate";

vi.mock("@tauri-apps/plugin-fs", () => ({ readDir: vi.fn().mockResolvedValue([]) }));
vi.mock("@tauri-apps/api/path", () => ({
  homeDir: vi.fn().mockResolvedValue("/home/user"),
  join: vi.fn().mockImplementation((...parts: string[]) => Promise.resolve(parts.join("/"))),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn().mockResolvedValue(null) }));

const mockListInstances = vi.fn().mockResolvedValue([{ name: "my-model", path: "/models/my-model", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: null, config: {} }]);
const mockListDatasets = vi.fn().mockResolvedValue([{ name: "ds_4x", path: "/datasets/ds_4x", scale: 4, num_pairs: 100 }]);
const mockGetInstance = vi.fn().mockResolvedValue({ name: "my-model", path: "/models/my-model", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: null, config: { scale: 4 } });
const mockGetInstanceVersions = vi.fn().mockResolvedValue([{ tag: "v1", path: "/models/my-model/versions/v1", has_weights: true }]);
const mockEstimateTrainingVram = vi.fn().mockResolvedValue({ peak_allocated_mb: 4096, peak_reserved_mb: 5120 });
const mockStartTraining = vi.fn().mockResolvedValue({ job_id: "test-job", status: "accepted" });
const mockGetEnv = vi.fn().mockResolvedValue({ vram_total_mb: 24576 });

vi.mock("../../../lib/api", () => ({
  listInstances: (...args: unknown[]) => mockListInstances(...args),
  listDatasets: (...args: unknown[]) => mockListDatasets(...args),
  getInstance: (...args: unknown[]) => mockGetInstance(...args),
  getInstanceVersions: (...args: unknown[]) => mockGetInstanceVersions(...args),
  estimateTrainingVram: (...args: unknown[]) => mockEstimateTrainingVram(...args),
  startTraining: (...args: unknown[]) => mockStartTraining(...args),
  getEnv: (...args: unknown[]) => mockGetEnv(...args),
}));

describe("Training Setup (26.9)", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
  });

  async function renderSetup() {
    const { ScreenTrainingSetup } = await import("../ScreenTrainingSetup");
    return render(<ScreenTrainingSetup />);
  }

  function getLaunchButton() {
    return screen.queryByRole("button", { name: /launch/i });
  }

  it("Launch button is disabled before prerequisites are met", async () => {
    await renderSetup();
    await act(async () => {});
    expect(getLaunchButton()).toBeDisabled();
  });

  it("Launch button becomes enabled when all prerequisites are met", async () => {
    await renderSetup();
    await act(async () => {});

    act(() => {
      useRunConfigStore.getState().setSelectedInstance("my-model");
      useRunConfigStore.getState().setInstanceArchitecture("rrdb_esrgan");
      useRunConfigStore.getState().setInstanceScale(4);
      useRunConfigStore.getState().setSelectedDataset("ds_4x");
      useRunConfigStore.getState().setSelectedDatasetPath("/datasets/ds_4x");
      useRunConfigStore.getState().setSelectedDatasetPairs(100);
    });

    expect(getLaunchButton()).not.toBeDisabled();
  });

  it("Launch button is enabled with instance and dataset (no run name required)", async () => {
    await renderSetup();
    await act(async () => {});

    act(() => {
      useRunConfigStore.getState().setSelectedInstance("my-model");
      useRunConfigStore.getState().setInstanceArchitecture("rrdb_esrgan");
      useRunConfigStore.getState().setInstanceScale(4);
      useRunConfigStore.getState().setSelectedDataset("ds_4x");
      useRunConfigStore.getState().setSelectedDatasetPath("/datasets/ds_4x");
      useRunConfigStore.getState().setSelectedDatasetPairs(100);
    });

    expect(getLaunchButton()).not.toBeDisabled();
  });

  it("re-hydrates instance details when selectedInstance changes", async () => {
    await renderSetup();
    await act(async () => {});
    act(() => {
      useRunConfigStore.getState().setSelectedInstance("my-model");
    });
    // Flush microtasks so the effect's async IIFE resolves
    await act(async () => {});
    await vi.waitFor(() => {
      expect(mockGetInstance).toHaveBeenCalledWith("my-model");
      expect(mockGetInstanceVersions).toHaveBeenCalledWith("my-model");
      expect(useRunConfigStore.getState().instanceArchitecture).toBe("rrdb_esrgan");
    });
  });

  it("Measure VRAM button calls estimateTrainingVram", async () => {
    useRunConfigStore.getState().setSelectedInstance("my-model");
    useRunConfigStore.getState().setInstanceArchitecture("rrdb_esrgan");
    useRunConfigStore.getState().setInstanceConfig({ scale: 4 });
    await renderSetup();
    await act(async () => {});
    // The re-hydration effect will set arch to "rrdb_esrgan" via mockGetInstance
    await act(async () => {});
    // Ensure animation/timers settle so button reads fresh arch
    await vi.waitFor(() => {
      expect(useRunConfigStore.getState().instanceArchitecture).toBe("rrdb_esrgan");
    });

    const measureBtn = screen.getByRole("button", { name: /measure vram/i });
    expect(measureBtn).not.toBeDisabled();

    fireEvent.click(measureBtn);
    await act(async () => {});
    await vi.waitFor(() => {
      expect(mockEstimateTrainingVram).toHaveBeenCalled();
      const vm = useRunConfigStore.getState().vramMeasure;
      expect(vm.status).toBe("done");
      expect(vm.allocatedMb).toBe(4096);
    });
  });

  it("Estimate breakdown lists every component and sums to the heuristic total", async () => {
    act(() => {
      useRunConfigStore.getState().setSelectedInstance("my-model");
      useRunConfigStore.getState().setInstanceArchitecture("rrdb_esrgan");
      useRunConfigStore.getState().setInstanceScale(4);
      useRunConfigStore.getState().setInstanceConfig({ scale: 4 });
      useRunConfigStore.getState().setBatchSize(4);
      useRunConfigStore.getState().setPatchSize(64);
    });
    await renderSetup();
    await act(async () => {});
    await vi.waitFor(() => {
      expect(useRunConfigStore.getState().instanceArchitecture).toBe("rrdb_esrgan");
    });

    const expected = estimateVramBreakdown({
      arch: "rrdb_esrgan",
      batchSize: 4,
      patchSize: 64,
      fp16: false,
      scale: 4,
      config: { scale: 4 },
      gradientCheckpointing: false,
    });

    expect(screen.getByText(`${expected.totalGb.toFixed(2)} GB`)).toBeTruthy();
    expect(screen.getByText("Upsampler")).toBeTruthy();
    expect(screen.getByText("CUDA context")).toBeTruthy();
    expect(screen.getByText("Activations")).toBeTruthy();
    expect(screen.getByText("Model weights")).toBeTruthy();
    expect(screen.queryByText("Allocator overhead")).toBeNull();
  });

  it("shows a note when measured VRAM exceeds the heuristic estimate", async () => {
    act(() => {
      useRunConfigStore.getState().setSelectedInstance("my-model");
      useRunConfigStore.getState().setInstanceArchitecture("rrdb_esrgan");
      useRunConfigStore.getState().setInstanceScale(4);
      useRunConfigStore.getState().setInstanceConfig({ scale: 4 });
      useRunConfigStore.getState().setBatchSize(4);
      useRunConfigStore.getState().setPatchSize(64);
    });
    await renderSetup();
    await act(async () => {});
    await vi.waitFor(() => {
      expect(useRunConfigStore.getState().instanceArchitecture).toBe("rrdb_esrgan");
    });

    // Default mock returns peak_reserved_mb=5120 → 5.0 GB vs heuristic ~1.28 GB.
    const measureBtn = screen.getByRole("button", { name: /measure vram/i });
    await vi.waitFor(() => expect(measureBtn).not.toBeDisabled());
    fireEvent.click(measureBtn);
    await act(async () => {});
    await vi.waitFor(() => {
      expect(screen.getByText("5.0 GB (reserved)")).toBeTruthy();
    });
    expect(screen.getByText(/measured exceeds heuristic/i)).toBeTruthy();
  });

  it("loss weight uses a slider that updates the store", async () => {
    await renderSetup();
    await act(async () => {});

    const sliders = screen.getAllByRole("slider");
    expect(sliders.length).toBeGreaterThanOrEqual(1);

    fireEvent.change(sliders[0], { target: { value: "0.5" } });
    expect(useRunConfigStore.getState().lossConfig.pixel.weight).toBe(0.5);
  });

  it("Advanced section expands to reveal relocated settings", async () => {
    await renderSetup();
    await act(async () => {});

    expect(screen.queryByText("Weight Decay")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /advanced/i }));
    expect(screen.getByText("Weight Decay")).toBeTruthy();
    expect(screen.getByText("Metrics Frequency")).toBeTruthy();
    expect(screen.getByText("Checkpointing")).toBeTruthy();
  });
});