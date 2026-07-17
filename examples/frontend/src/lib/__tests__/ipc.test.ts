import { describe, it, expect, beforeEach, vi } from "vitest";
import { resetAllStores } from "../../test-utils/resetStores";
import { useTrainingStore } from "../../store/trainingStore";
import { useUiStore } from "../../store/uiStore";

vi.mock("../SRProjManager", () => ({
  SRProjManager: {
    load: vi.fn(),
    close: vi.fn(),
    addRun: vi.fn(),
    updateRun: vi.fn(),
    getRun: vi.fn().mockReturnValue(undefined),
    setProject: vi.fn(),
    setActiveRun: vi.fn(),
    setActiveTab: vi.fn(),
    save: vi.fn().mockResolvedValue(undefined),
    current: null,
    filePath: null,
  },
}));

let dispatchCallback: ((event: { payload: unknown }) => void) | null = null;

const { listen } = await import("@tauri-apps/api/event");
vi.mocked(listen).mockImplementation((_event, cb) => {
  dispatchCallback = cb as (event: { payload: unknown }) => void;
  return Promise.resolve(() => undefined);
});

describe("IPC dispatch — field mapping (26.21)", () => {
  beforeEach(async () => {
    resetAllStores();
    dispatchCallback = null;

    const ipc = await import("../ipc");
    ipc.stopIpcListener();
    vi.clearAllMocks();
    vi.mocked(listen).mockImplementation((_event, cb) => {
      dispatchCallback = cb as (event: { payload: unknown }) => void;
      return Promise.resolve(() => undefined);
    });

    await ipc.startIpcListener();
  });

  const sendMsg = (payload: unknown) => {
    dispatchCallback?.({ payload });
  };

  it("metrics.update maps snake_case fields to camelCase store fields", () => {
    useTrainingStore.setState({ status: "running", activeTrainingRunId: "run-1" });

    sendMsg({
      type: "metrics.update",
      iter: 100,
      epoch: 1,
      g_loss: 0.42,
      d_loss: 0.21,
      psnr: 32.5,
      ssim: 0.91,
      gpu_util: 75,
      vram_gb: 6.2,
      temp_c: 68,
      cpu_util: 30,
      speed: 12.5,
    });

    const state = useTrainingStore.getState();
    expect(state.gLoss).toBeCloseTo(0.42);
    expect(state.dLoss).toBeCloseTo(0.21);
    expect(state.gpuUtil).toBe(75);
    expect(state.vram).toBeCloseTo(6.2);
    expect(state.temp).toBe(68);
    expect(state.cpuUtil).toBe(30);
    expect(state.speed).toBeCloseTo(12.5);
    expect(state.psnr).toBeCloseTo(32.5);
    expect(state.ssim).toBeCloseTo(0.91);
  });

  it("no metrics field is silently dropped", () => {
    useTrainingStore.setState({ status: "running", activeTrainingRunId: "run-1" });

    sendMsg({
      type: "metrics.update",
      iter: 50,
      epoch: 1,
      g_loss: 0.5,
      d_loss: null,
      psnr: 30.0,
      ssim: 0.88,
      gpu_util: 50,
      vram_gb: 4.0,
      temp_c: 60,
      cpu_util: 20,
      speed: 8.0,
    });

    const state = useTrainingStore.getState();
    expect(state.iter).toBe(50);
    expect(state.epoch).toBe(1);
    expect(state.dLoss).toBeNull();
  });

  it("hardware.info with rocm device type is preserved without coercion", () => {
    sendMsg({
      type: "hardware.info",
      devices: [
        { id: "cuda:0", name: "AMD RX 7900", vram_gb: 20, type: "rocm" },
        { id: "cpu", name: "CPU", vram_gb: null, type: "cpu" },
      ],
    });

    const devices = useUiStore.getState().detectedDevices;
    expect(devices).toHaveLength(2);
    const rocm = devices.find((d) => d.id === "cuda:0");
    expect(rocm?.type).toBe("rocm");
    expect(rocm?.type).not.toBe("cuda");
  });

  it("hardware.info forwards all devices", () => {
    sendMsg({
      type: "hardware.info",
      devices: [
        { id: "cuda:0", name: "RTX 4090", vram_gb: 24, type: "cuda" },
        { id: "cuda:1", name: "RTX 3080", vram_gb: 10, type: "cuda" },
        { id: "cpu", name: "CPU", vram_gb: null, type: "cpu" },
      ],
    });

    const devices = useUiStore.getState().detectedDevices;
    expect(devices).toHaveLength(3);
  });
});

// §26.31 — run.history.request frontend hydration
describe("run.history.response hydration (26.31)", () => {
  beforeEach(async () => {
    resetAllStores();
    dispatchCallback = null;

    const ipc = await import("../ipc");
    ipc.stopIpcListener();
    vi.clearAllMocks();
    vi.mocked(listen).mockImplementation((_event, cb) => {
      dispatchCallback = cb as (event: { payload: unknown }) => void;
      return Promise.resolve(() => undefined);
    });

    await ipc.startIpcListener();
  });

  const sendMsg = (payload: unknown) => {
    dispatchCallback?.({ payload });
  };

  const makeEvent = (iter: number) => ({
    iter,
    epoch: 1,
    g_loss: 0.5,
    d_loss: null,
    psnr: 30.0,
    ssim: 0.9,
    gpu_util: null,
    vram_gb: null,
    temp_c: null,
    cpu_util: null,
    speed: null,
    timestamp: "2024-01-01T00:00:00.000Z",
  });

  it("run.history.response populates lossHistory and psnrHistory", () => {
    const events = Array.from({ length: 50 }, (_, i) => makeEvent(i + 1));

    sendMsg({ type: "run.history.response", run_id: "run-1", metrics: events });

    const state = useTrainingStore.getState();
    expect(state.runHistories["run-1"].gLossHistory).toHaveLength(50);
    expect(state.runHistories["run-1"].psnrHistory.every((v) => v === 30.0)).toBe(true);
  });

  it("response for different run_id does not overwrite first run history", () => {
    const events1 = [makeEvent(1)];
    const events2 = [makeEvent(2), makeEvent(3)];

    sendMsg({ type: "run.history.response", run_id: "run-1", metrics: events1 });
    sendMsg({ type: "run.history.response", run_id: "run-2", metrics: events2 });

    const state = useTrainingStore.getState();
    expect(state.runHistories["run-1"].gLossHistory).toHaveLength(1);
    expect(state.runHistories["run-2"].gLossHistory).toHaveLength(2);
  });

  it("hydrates active run flat history arrays when run_id matches activeTrainingRunId", () => {
    useTrainingStore.setState({ activeTrainingRunId: "run-active" });

    const events = Array.from({ length: 5 }, (_, i) => makeEvent(i + 1));
    sendMsg({ type: "run.history.response", run_id: "run-active", metrics: events });

    const state = useTrainingStore.getState();
    expect(state.lossHistory).toHaveLength(5);
    expect(state.psnrHistory).toHaveLength(5);
  });
});

// ── §26.29 — project.run.started → .srproj write ─────────────────────────

describe("project.run.started integration (26.29)", () => {
  beforeEach(async () => {
    resetAllStores();
    dispatchCallback = null;

    const ipc = await import("../ipc");
    ipc.stopIpcListener();
    vi.clearAllMocks();
    vi.mocked(listen).mockImplementation((_event, cb) => {
      dispatchCallback = cb as (event: { payload: unknown }) => void;
      return Promise.resolve(() => undefined);
    });

    await ipc.startIpcListener();
  });

  const sendMsg = (payload: unknown) => dispatchCallback?.({ payload });

  it("appends new SRProjRun with status running and correct sidecar_pid", async () => {
    const { SRProjManager } = await import("../SRProjManager");

    sendMsg({
      type: "project.run.started",
      run_id: "run-42",
      total_epochs: 200,
      sidecar_pid: 9999,
    });

    const addRunCalls = vi.mocked(SRProjManager.addRun).mock.calls;
    expect(addRunCalls).toHaveLength(1);
    const addedRun = addRunCalls[0][0];
    expect(addedRun.run_id).toBe("run-42");
    expect(addedRun.status).toBe("running");
    expect(addedRun.sidecar_pid).toBe(9999);
  });

  it("trainingStore reflects running status and activeTrainingRunId", () => {
    sendMsg({
      type: "project.run.started",
      run_id: "run-42",
      total_epochs: 200,
      sidecar_pid: 9999,
    });

    const state = useTrainingStore.getState();
    expect(state.status).toBe("running");
    expect(state.activeTrainingRunId).toBe("run-42");
  });

  it("SRProjManager.save() called exactly once synchronously after addRun", async () => {
    const { SRProjManager } = await import("../SRProjManager");

    sendMsg({
      type: "project.run.started",
      run_id: "run-42",
      total_epochs: 200,
      sidecar_pid: 9999,
    });

    expect(vi.mocked(SRProjManager.save)).toHaveBeenCalledTimes(1);
  });

  it("saved run contains sidecar_pid in the manifest", async () => {
    const { SRProjManager } = await import("../SRProjManager");

    sendMsg({
      type: "project.run.started",
      run_id: "run-pid",
      total_epochs: 100,
      sidecar_pid: 1234,
    });

    const run = vi.mocked(SRProjManager.addRun).mock.calls[0][0];
    expect(run.sidecar_pid).toBe(1234);
  });
});

// ── §26.30 — checkpoint.saved → .srproj auto-save ────────────────────────

describe("checkpoint.saved integration (26.30)", () => {
  beforeEach(async () => {
    resetAllStores();
    dispatchCallback = null;

    const ipc = await import("../ipc");
    ipc.stopIpcListener();
    vi.clearAllMocks();
    vi.mocked(listen).mockImplementation((_event, cb) => {
      dispatchCallback = cb as (event: { payload: unknown }) => void;
      return Promise.resolve(() => undefined);
    });

    await ipc.startIpcListener();
  });

  const sendMsg = (payload: unknown) => dispatchCallback?.({ payload });

  it("updateRun sets current_epoch and best_psnr on first checkpoint", async () => {
    const { SRProjManager } = await import("../SRProjManager");
    vi.mocked(SRProjManager.getRun).mockReturnValue({
      run_id: "run-1",
      metrics: { best_psnr: 30.0, current_epoch: 0 },
      checkpoints: { total_count: 0, last_saved_epoch: null, last_saved_path: null, best_checkpoint_path: null },
    } as any);

    sendMsg({
      type: "checkpoint.saved",
      run_id: "run-1",
      path: "/ckpt/epoch_10.pth",
      epoch: 10,
      psnr: 32.5,
      ssim: 0.91,
      size_mb: 50,
    });

    const updateRunCalls = vi.mocked(SRProjManager.updateRun).mock.calls;
    expect(updateRunCalls).toHaveLength(1);
    const [, patch] = updateRunCalls[0];
    expect(patch.metrics?.current_epoch).toBe(10);
    expect(patch.metrics?.best_psnr).toBeCloseTo(32.5);
  });

  it("best_psnr stays at 32.5 when second checkpoint has lower psnr", async () => {
    const { SRProjManager } = await import("../SRProjManager");

    // First checkpoint: psnr 32.5
    vi.mocked(SRProjManager.getRun).mockReturnValue({
      run_id: "run-1",
      metrics: { best_psnr: null, current_epoch: 0 },
      checkpoints: { total_count: 0, last_saved_epoch: null, last_saved_path: null, best_checkpoint_path: null },
    } as any);

    sendMsg({ type: "checkpoint.saved", run_id: "run-1", path: "/e10.pth", epoch: 10, psnr: 32.5, ssim: null, size_mb: 50 });

    // Second checkpoint: psnr 28.0 (lower)
    vi.mocked(SRProjManager.getRun).mockReturnValue({
      run_id: "run-1",
      metrics: { best_psnr: 32.5, current_epoch: 10 },
      checkpoints: { total_count: 1, last_saved_epoch: 10, last_saved_path: "/e10.pth", best_checkpoint_path: null },
    } as any);

    sendMsg({ type: "checkpoint.saved", run_id: "run-1", path: "/e20.pth", epoch: 20, psnr: 28.0, ssim: null, size_mb: 50 });

    const calls = vi.mocked(SRProjManager.updateRun).mock.calls;
    const secondCall = calls[1][1];
    expect(secondCall.metrics?.best_psnr).toBeCloseTo(32.5);
    expect(secondCall.metrics?.current_epoch).toBe(20);
  });

  it("SRProjManager.save() called after each checkpoint event", async () => {
    const { SRProjManager } = await import("../SRProjManager");
    vi.mocked(SRProjManager.getRun).mockReturnValue({ run_id: "run-1", metrics: { best_psnr: null, current_epoch: 0 }, checkpoints: { total_count: 0, last_saved_epoch: null, last_saved_path: null, best_checkpoint_path: null } } as any);

    sendMsg({ type: "checkpoint.saved", run_id: "run-1", path: "/e1.pth", epoch: 1, psnr: 30.0, ssim: null, size_mb: 40 });
    sendMsg({ type: "checkpoint.saved", run_id: "run-1", path: "/e2.pth", epoch: 2, psnr: 31.0, ssim: null, size_mb: 40 });

    expect(vi.mocked(SRProjManager.save)).toHaveBeenCalledTimes(2);
  });
});

// ── §26.10 — training.complete success toast ──────────────────────────────

describe("training.complete IPC (26.10)", () => {
  beforeEach(async () => {
    resetAllStores();
    dispatchCallback = null;

    const ipc = await import("../ipc");
    ipc.stopIpcListener();
    vi.clearAllMocks();
    vi.mocked(listen).mockImplementation((_event, cb) => {
      dispatchCallback = cb as (event: { payload: unknown }) => void;
      return Promise.resolve(() => undefined);
    });

    await ipc.startIpcListener();
  });

  const sendMsg = (payload: unknown) => dispatchCallback?.({ payload });

  it("sets trainingStore.status to done and stores finalEpoch", () => {
    sendMsg({ type: "training.complete", run_id: "run-1", final_epoch: 50 });

    const state = useTrainingStore.getState();
    expect(state.status).toBe("done");
    expect(state.finalEpoch).toBe(50);
  });

  it("finalEpoch value matches the IPC message field", () => {
    sendMsg({ type: "training.complete", run_id: "run-1", final_epoch: 200 });
    expect(useTrainingStore.getState().finalEpoch).toBe(200);
  });
});
