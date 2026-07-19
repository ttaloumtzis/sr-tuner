import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useRunConfigStore } from "../../../store/runConfigStore";

vi.mock("@tauri-apps/plugin-fs", () => ({ readDir: vi.fn().mockResolvedValue([]) }));
vi.mock("@tauri-apps/api/path", () => ({
  homeDir: vi.fn().mockResolvedValue("/home/user"),
  join: vi.fn().mockImplementation((...parts: string[]) => Promise.resolve(parts.join("/"))),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn().mockResolvedValue(null) }));

const mockListInstances = vi.fn().mockResolvedValue([{ name: "my-model", path: "/models/my-model", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: null, config: {} }]);
const mockListDatasets = vi.fn().mockResolvedValue([{ name: "ds_4x", path: "/datasets/ds_4x", scale: 4, num_pairs: 100 }]);
const mockGetInstance = vi.fn().mockResolvedValue({ name: "my-model", path: "/models/my-model", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: null, config: {} });
const mockGetInstanceVersions = vi.fn().mockResolvedValue([]);
const mockValidateDatasetPath = vi.fn().mockResolvedValue({ valid: true, problems: [] });
const mockStartTraining = vi.fn().mockResolvedValue({ job_id: "test-job", status: "accepted" });

vi.mock("../../../lib/api", () => ({
  listInstances: (...args: unknown[]) => mockListInstances(...args),
  listDatasets: (...args: unknown[]) => mockListDatasets(...args),
  getInstance: (...args: unknown[]) => mockGetInstance(...args),
  getInstanceVersions: (...args: unknown[]) => mockGetInstanceVersions(...args),
  validateDatasetPath: (...args: unknown[]) => mockValidateDatasetPath(...args),
  startTraining: (...args: unknown[]) => mockStartTraining(...args),
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
    return screen.queryByRole("button", { name: /launch training/i });
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
});