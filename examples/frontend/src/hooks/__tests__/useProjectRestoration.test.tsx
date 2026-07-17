/**
 * §26.11 — Sidecar crash recovery: useProjectRestoration hook
 *
 * Verifies that on project reopen, runs with status "running" + non-null
 * sidecar_pid are detected as crashed, marked in SRProjManager, and the
 * crash recovery dialog state is opened.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { resetAllStores } from "../../test-utils/resetStores";
import { useProjectStore } from "../../store/projectStore";

vi.mock("../../lib/SRProjManager", () => ({
  SRProjManager: {
    load: vi.fn(),
    close: vi.fn(),
    addRun: vi.fn(),
    updateRun: vi.fn(),
    getRun: vi.fn().mockReturnValue(undefined),
    setProject: vi.fn(),
    save: vi.fn().mockResolvedValue(undefined),
    current: null,
    filePath: null,
  },
}));

vi.mock("../../lib/ipc", () => ({
  sendToSidecar: vi.fn().mockResolvedValue(undefined),
  startIpcListener: vi.fn(),
  stopIpcListener: vi.fn(),
}));

function makeCrashedRun(overrides: Record<string, unknown> = {}) {
  return {
    run_id: "run-crashed",
    name: "my crashed run",
    status: "running",
    sidecar_pid: 1234,
    created_at: "2024-01-01T00:00:00.000Z",
    started_at: "2024-01-01T00:00:00.000Z",
    completed_at: null,
    architecture: { type: "EDSR", upscale_factor: 4, custom_config: {} },
    training_config: {
      num_epochs: 100,
      batch_size: 4,
      learning_rate: 1e-4,
      scheduler: "MultiStepLR",
      optimizer: "Adam",
      patch_size: 192,
      augmentations: {
        horizontal_flip: true,
        vertical_flip: false,
        rotation_90: false,
        mixup: false,
        color_jitter: false,
        random_degradation: false,
        gaussian_blur: false,
        noise_injection: false,
      },
    },
    paths: { training_data: "", validation_data: "", checkpoint_dir: "/ckpt", log_dir: "/log" },
    metrics: {
      current_epoch: 15,
      epochs_completed: 15,
      best_loss: null,
      best_loss_epoch: null,
      best_psnr: 30.5,
      best_psnr_epoch: null,
      last_loss: null,
      last_psnr: null,
      last_ssim: null,
    },
    checkpoints: {
      total_count: 1,
      last_saved_epoch: 15,
      last_saved_path: "/ckpt/epoch_15.pth",
      best_checkpoint_path: null,
    },
    sidecar_log_file: null,
    ...overrides,
  };
}

function makeProject(runs: unknown[] = []) {
  return {
    version: "1.0",
    name: "test-project",
    filePath: "/projects/test.srproj",
    runs,
    default_dataset: {
      training_path: "",
      validation_path: null,
      validation_strategy: "none",
      dataset_type: "image_folder",
    },
    default_model: { architecture: "EDSR", upscale_factor: 4 },
    ui_state: { last_active_run_id: null, last_active_tab: null, expanded_panels: {} },
  };
}

describe("useProjectRestoration crash recovery (26.11)", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
  });

  it("sets crashRecovery.open to true when run has status running and non-null sidecar_pid", async () => {
    const { useProjectRestoration } = await import("../useProjectRestoration");

    const project = makeProject([makeCrashedRun()]);

    let result: ReturnType<typeof useProjectRestoration>;
    const { result: hookResult } = renderHook(() => useProjectRestoration());
    result = hookResult.current;

    expect(result.crashRecovery.open).toBe(false);

    act(() => {
      useProjectStore.setState({ project: project as any });
    });

    result = hookResult.current;
    expect(result.crashRecovery.open).toBe(true);
    expect(result.crashRecovery.runId).toBe("run-crashed");
    expect(result.crashRecovery.runName).toBe("my crashed run");
    expect(result.crashRecovery.lastEpoch).toBe(15);
    expect(result.crashRecovery.lastCheckpointPath).toBe("/ckpt/epoch_15.pth");
  });

  it("calls SRProjManager.updateRun with status crashed for the detected run", async () => {
    const { useProjectRestoration } = await import("../useProjectRestoration");
    const { SRProjManager } = await import("../../lib/SRProjManager");

    const project = makeProject([makeCrashedRun()]);

    renderHook(() => useProjectRestoration());

    act(() => {
      useProjectStore.setState({ project: project as any });
    });

    expect(vi.mocked(SRProjManager.updateRun)).toHaveBeenCalledWith(
      "run-crashed",
      expect.objectContaining({ status: "crashed" })
    );
  });

  it("does not open crash dialog when sidecar_pid is null (clean exit)", async () => {
    const { useProjectRestoration } = await import("../useProjectRestoration");

    const cleanRun = makeCrashedRun({ sidecar_pid: null });
    const project = makeProject([cleanRun]);

    const { result: hookResult } = renderHook(() => useProjectRestoration());

    act(() => {
      useProjectStore.setState({ project: project as any });
    });

    expect(hookResult.current.crashRecovery.open).toBe(false);
  });

  it("does not open crash dialog when run status is completed", async () => {
    const { useProjectRestoration } = await import("../useProjectRestoration");

    const completedRun = makeCrashedRun({ status: "completed", sidecar_pid: 9999 });
    const project = makeProject([completedRun]);

    const { result: hookResult } = renderHook(() => useProjectRestoration());

    act(() => {
      useProjectStore.setState({ project: project as any });
    });

    expect(hookResult.current.crashRecovery.open).toBe(false);
  });

  it("handleAbandonCrashedRun closes dialog and marks run failed", async () => {
    const { useProjectRestoration } = await import("../useProjectRestoration");
    const { SRProjManager } = await import("../../lib/SRProjManager");

    const project = makeProject([makeCrashedRun()]);
    const { result: hookResult } = renderHook(() => useProjectRestoration());

    act(() => {
      useProjectStore.setState({ project: project as any });
    });

    expect(hookResult.current.crashRecovery.open).toBe(true);

    act(() => {
      hookResult.current.handleAbandonCrashedRun();
    });

    expect(hookResult.current.crashRecovery.open).toBe(false);
    expect(vi.mocked(SRProjManager.updateRun)).toHaveBeenCalledWith(
      "run-crashed",
      expect.objectContaining({ status: "failed" })
    );
  });
});
