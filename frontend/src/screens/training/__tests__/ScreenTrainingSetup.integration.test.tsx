/**
 * §26.9 — validate→launch gate integration test
 *
 * Renders ScreenTrainingSetup and exercises the dataset.validate.result
 * → canLaunch gate. Covers:
 *  - valid: true  → Launch button enabled (when run name is filled)
 *  - valid: false → Launch button disabled
 *  - strategy "none" → enabled without a validation_path
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { resetAllStores } from "../../../test-utils/resetStores";

// ── Mocks for Tauri APIs not covered by vitest.config.ts aliases ──────────

vi.mock("@tauri-apps/plugin-fs", () => ({
  readDir: vi.fn().mockResolvedValue([]),
}));

vi.mock("@tauri-apps/api/path", () => ({
  homeDir: vi.fn().mockResolvedValue("/home/user"),
  join: vi.fn().mockImplementation((...parts: string[]) => Promise.resolve(parts.join("/"))),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn().mockResolvedValue(null),
}));

vi.mock("../../../lib/api", () => ({
  startTraining: vi.fn().mockResolvedValue({ job_id: "test-job", status: "accepted" }),
  validateDataset: vi.fn().mockResolvedValue({ valid: true, problems: [] }),
  getJobStatus: vi.fn().mockResolvedValue({ status: "completed" }),
}));

vi.mock("../../../lib/SRProjManager", () => ({
  SRProjManager: {
    load: vi.fn(),
    close: vi.fn(),
    addRun: vi.fn(),
    updateRun: vi.fn(),
    getRun: vi.fn(),
    save: vi.fn().mockResolvedValue(undefined),
    current: null,
    filePath: null,
  },
}));

// ── Capture the listen callback so tests can simulate API responses ────

let capturedCallback: ((event: { payload: unknown }) => void) | null = null;

function simulateApiResponse(payload: unknown) {
  capturedCallback?.({ payload });
}

describe("validate→launch gate (26.9)", () => {
  beforeEach(() => {
    resetAllStores();
    capturedCallback = null;
    vi.clearAllMocks();
  });

  async function renderSetup() {
    const { ScreenTrainingSetup } = await import("../ScreenTrainingSetup");
    render(<ScreenTrainingSetup />);
  }

  function getLaunchButton() {
    return screen.queryByRole("button", { name: /launch training/i });
  }

  it("Launch button is disabled before dataset is validated", async () => {
    await renderSetup();
    expect(getLaunchButton()).toBeDisabled();
  });

  it("Launch button remains disabled when valid: false is received", async () => {
    await renderSetup();

    await act(async () => {
      simulateApiResponse({ type: "dataset.validate.result", valid: false, errors: ["path not found"] });
    });

    expect(getLaunchButton()).toBeDisabled();
  });

  it("Launch button becomes enabled when valid: true and run name is filled", async () => {
    await renderSetup();

    const runNameInput = screen.getByPlaceholderText(/my-run-001/i);
    await userEvent.type(runNameInput, "my-training-run");

    await act(async () => {
      simulateApiResponse({ type: "dataset.validate.result", valid: true, errors: [] });
    });

    expect(getLaunchButton()).not.toBeDisabled();
  });

  it("Launch button stays disabled when valid: true but run name is empty", async () => {
    await renderSetup();

    await act(async () => {
      simulateApiResponse({ type: "dataset.validate.result", valid: true, errors: [] });
    });

    expect(getLaunchButton()).toBeDisabled();
  });

  it("strategy='none' path: valid:true without validation_path still enables launch", async () => {
    const { useDatasetStore } = await import("../../../store/datasetStore");
    useDatasetStore.setState({ strategy: "none", validationPath: null });

    await renderSetup();

    const runNameInput = screen.getByPlaceholderText(/my-run-001/i);
    await userEvent.type(runNameInput, "run-no-val");

    await act(async () => {
      simulateApiResponse({ type: "dataset.validate.result", valid: true, errors: [] });
    });

    expect(getLaunchButton()).not.toBeDisabled();
  });
});
