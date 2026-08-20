import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act, fireEvent } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useInferenceStore } from "../../../store/inferenceStore";

const mockListInstances = vi.fn();
const mockGetInstanceVersions = vi.fn();
const mockListRuns = vi.fn();
const mockListRunCheckpoints = vi.fn();
const mockStartInference = vi.fn();
const mockCancelJob = vi.fn();
const mockGetJobStatus = vi.fn();
const mockInvoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  convertFileSrc: (p: string) => `file://${p}`,
  invoke: (...args: unknown[]) => mockInvoke(...args),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn().mockResolvedValue(null) }));
vi.mock("@tauri-apps/plugin-fs", () => ({ readDir: vi.fn().mockResolvedValue([]) }));
vi.mock("@tauri-apps/api/path", () => ({
  pictureDir: vi.fn().mockResolvedValue("/home/user/Pictures"),
}));

vi.mock("../../../lib/api", () => ({
  getBaseUrl: () => "http://localhost:8765",
  listInstances: (...args: unknown[]) => mockListInstances(...args),
  getInstanceVersions: (...args: unknown[]) => mockGetInstanceVersions(...args),
  listRuns: (...args: unknown[]) => mockListRuns(...args),
  listRunCheckpoints: (...args: unknown[]) => mockListRunCheckpoints(...args),
  defaultOutputDir: () => Promise.resolve("/home/user/Pictures"),
  startInference: (...args: unknown[]) => mockStartInference(...args),
  cancelJob: (...args: unknown[]) => mockCancelJob(...args),
  getJobStatus: (...args: unknown[]) => mockGetJobStatus(...args),
}));

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  emit(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  close() {
    this.closed = true;
  }
}

function seedReadyStore(modelPath = "/runs/my-model/run_x/epoch_5.pt") {
  act(() => {
    const s = useInferenceStore.getState();
    s.setInputPath("/in/foo.png");
    s.setModelPath(modelPath);
    s.setOutputDir("/out");
  });
}

describe("ScreenInference (14.x)", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource);
    mockListInstances.mockResolvedValue([{ name: "my-model", path: "/m", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: "v2", config: {} }]);
    mockGetInstanceVersions.mockResolvedValue([
      { tag: "v1", path: "/models/my-model/versions/v1" },
      { tag: "v2", path: "/models/my-model/versions/v2" },
    ]);
    mockListRuns.mockResolvedValue([
      {
        name: "my-model", architecture: "rrdb_esrgan", scale: 4,
        runs: [
          { run_id: "run_20260701_080000", status: "finished", created_at: "2026-07-01T08:00:00Z", finished_at: null, error: null, checkpoint_count: 1, total_size_mb: 1, last_epoch: 5, has_metrics: true, config: null },
        ],
      },
    ]);
    mockListRunCheckpoints.mockResolvedValue([
      { epoch: 5, filename: "epoch_5.pt", path: "/runs/my-model/run_x/epoch_5.pt", created_at: "2026-07-01T09:00:00Z", file_size_mb: 1, metrics: { psnr: 30 } },
    ]);
    mockGetJobStatus.mockResolvedValue({ job_id: "infer.1", job_type: "inference", status: "running", created_at: 0 });
  });

  it("Run Inference is disabled before prerequisites are met", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    expect(screen.getByRole("button", { name: /run inference/i })).toBeDisabled();
  });

  it("Run Inference becomes enabled when input, model, and output dir are set", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    seedReadyStore();
    await act(async () => {});
    expect(screen.getByRole("button", { name: /run inference/i })).not.toBeDisabled();
  });

  it("is enabled with a raw checkpoint path and no model instance", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    act(() => {
      const s = useInferenceStore.getState();
      s.setInputPath("/in/foo.png");
      s.setModelPath("/models/custom.pt");
      s.setOutputDir("/out");
    });
    await act(async () => {});
    expect(screen.getByRole("button", { name: /run inference/i })).not.toBeDisabled();
  });

  it("kicks off an inference job with the selected checkpoint path and shows progress + done metrics", async () => {
    mockStartInference.mockResolvedValue({ job_id: "infer.1", status: "accepted" });
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    seedReadyStore();
    await act(async () => {});

    act(() => {
      screen.getByRole("button", { name: /run inference/i }).click();
    });
    await act(async () => {});

    expect(mockStartInference).toHaveBeenCalledWith(
      expect.objectContaining({
        input: "/in/foo.png",
        output: "/out/foo_sr.png",
        model: "/runs/my-model/run_x/epoch_5.pt",
        format: "png",
        tile: 0,
      }),
    );
    expect(useInferenceStore.getState().status).toBe("running");

    act(() => {
      FakeEventSource.instances[0].emit({
        type: "done",
        success: true,
        output: "/out/foo_sr.png",
        preview_input_path: "/preview/in.png",
        preview_output_path: "/preview/out.png",
        input_resolution: { width: 512, height: 512 },
        output_resolution: { width: 1024, height: 1024 },
        inference_time_ms: 900,
        metrics: { psnr: 31.23, ssim: 0.9211, lpips: 0.05, ms_ssim: 0.88 },
      });
    });

    expect(useInferenceStore.getState().status).toBe("done");
    expect(screen.getByText("PSNR")).toBeTruthy();
    expect(screen.getByText("31.23")).toBeTruthy();
    expect(screen.getByText("2×")).toBeTruthy();
  });

  it("sends the configured tile size instead of tiling when a size is set", async () => {
    mockStartInference.mockResolvedValue({ job_id: "infer.1", status: "accepted" });
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    seedReadyStore();
    act(() => {
      useInferenceStore.getState().setTileSize(256);
    });
    await act(async () => {});

    act(() => {
      screen.getByRole("button", { name: /run inference/i }).click();
    });
    await act(async () => {});

    expect(mockStartInference).toHaveBeenCalledWith(expect.objectContaining({ tile: 256 }));
  });

  it("renders an error banner when the job reports a failure", async () => {
    mockStartInference.mockResolvedValue({ job_id: "infer.1", status: "accepted" });
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    seedReadyStore();
    await act(async () => {});

    act(() => {
      screen.getByRole("button", { name: /run inference/i }).click();
    });
    await act(async () => {});

    act(() => {
      FakeEventSource.instances[0].emit({ type: "error", message: "CUDA out of memory" });
    });

    expect(screen.getByText(/CUDA out of memory/i)).toBeTruthy();
    expect(useInferenceStore.getState().status).toBe("error");
  });

  it("uses a raw checkpoint path when one is preselected from the Checkpoints tab", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    act(() => {
      const s = useInferenceStore.getState();
      s.setPreselectedCheckpointPath("/models/pre.pt");
    });
    await act(async () => {});
    expect(useInferenceStore.getState().modelPath).toBe("/models/pre.pt");
    expect(useInferenceStore.getState().preselectedCheckpointPath).toBeNull();
  });

  it("defaults the save directory to the user's Pictures folder", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    expect(useInferenceStore.getState().outputDir).toBe("/home/user/Pictures");
  });

  it("walks the Model → Version cascade and auto-selects the latest version", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});

    act(() => {
      useInferenceStore.getState().setInputPath("/in/foo.png");
    });
    await act(async () => {});

    // Open the Model instance dropdown and pick "my-model".
    fireEvent.click(screen.getByText("Select model…"));
    fireEvent.click(screen.getByText("my-model"));
    await act(async () => {});

    // Version selection uses getInstanceVersions, not the runs/checkpoints cascade.
    expect(mockGetInstanceVersions).toHaveBeenCalledWith("my-model");
    expect(mockListRuns).not.toHaveBeenCalled();
    expect(mockListRunCheckpoints).not.toHaveBeenCalled();
    // The latest version is auto-selected and resolves to its model.pt.
    expect(useInferenceStore.getState().instance).toBe("my-model");
    expect(useInferenceStore.getState().version).toBe("v2");
    expect(useInferenceStore.getState().modelPath).toBe("/models/my-model/versions/v2/model.pt");

    // Selecting an earlier version updates the model path.
    fireEvent.click(screen.getByText("v2"));
    fireEvent.click(screen.getByText("v1"));
    await act(async () => {});
    expect(useInferenceStore.getState().version).toBe("v1");
    expect(useInferenceStore.getState().modelPath).toBe("/models/my-model/versions/v1/model.pt");
  });

  it("runs inference by instance+version when a model version is selected", async () => {
    mockStartInference.mockResolvedValue({ job_id: "infer.1", status: "accepted" });
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    act(() => {
      useInferenceStore.getState().setInputPath("/in/foo.png");
      useInferenceStore.getState().setOutputDir("/out");
      useInferenceStore.getState().setInstance("my-model");
      useInferenceStore.getState().setVersion("v2");
      useInferenceStore.getState().setModelPath("/models/my-model/versions/v2/model.pt");
    });
    await act(async () => {});
    act(() => {
      screen.getByRole("button", { name: /run inference/i }).click();
    });
    await act(async () => {});

    expect(mockStartInference).toHaveBeenCalledWith(
      expect.objectContaining({
        instance: "my-model",
        version: "v2",
        input: "/in/foo.png",
        format: "png",
      }),
    );
  });

  it("submits a checkpoint payload (model + instance, no version) when a run checkpoint is selected", async () => {
    mockStartInference.mockResolvedValue({ job_id: "infer.1", status: "accepted" });
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    act(() => {
      const s = useInferenceStore.getState();
      s.setInputPath("/in/foo.png");
      s.setOutputDir("/out");
      s.setModelPath("/runs/my-model/run_x/epoch_5.pt");
      s.setInstance("my-model");
    });
    await act(async () => {});
    act(() => {
      screen.getByRole("button", { name: /run inference/i }).click();
    });
    await act(async () => {});

    expect(mockStartInference).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "/runs/my-model/run_x/epoch_5.pt",
        instance: "my-model",
      }),
    );
    const args = mockStartInference.mock.calls[0][0] as Record<string, unknown>;
    expect(args).not.toHaveProperty("version");
  });

  it("walks the instance → run → epoch cascade for a run checkpoint", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    act(() => {
      useInferenceStore.getState().setInputPath("/in/foo.png");
    });
    await act(async () => {});

    fireEvent.click(screen.getByText("Run checkpoint"));
    await act(async () => {});

    fireEvent.click(screen.getByText("Select instance…"));
    fireEvent.click(screen.getByText("my-model"));
    await act(async () => {});
    expect(mockListRuns).toHaveBeenCalled();

    fireEvent.click(screen.getByText("Select run…"));
    fireEvent.click(screen.getByText("run_001"));
    await act(async () => {});
    expect(mockListRunCheckpoints).toHaveBeenCalledWith("my-model", "run_20260701_080000");

    fireEvent.click(screen.getByText("Select checkpoint…"));
    fireEvent.click(screen.getByText(/epoch_5\.pt/));
    await act(async () => {});

    const s = useInferenceStore.getState();
    expect(s.modelPath).toBe("/runs/my-model/run_x/epoch_5.pt");
    expect(s.instance).toBe("my-model");
    expect(s.version).toBeNull();
    expect(s.checkpointRunId).toBe("run_20260701_080000");
  });

  it("shows empty states for an instance without runs and a run without checkpoints", async () => {
    mockListInstances.mockResolvedValue([
      { name: "my-model", path: "/m", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: "v2", config: {} },
      { name: "empty-model", path: "/e", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: null, config: {} },
    ]);
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    act(() => {
      useInferenceStore.getState().setInputPath("/in/foo.png");
    });
    await act(async () => {});

    fireEvent.click(screen.getByText("Run checkpoint"));
    await act(async () => {});

    // Instance without runs → run dropdown shows an empty state.
    fireEvent.click(screen.getByText("Select instance…"));
    fireEvent.click(screen.getByText("empty-model"));
    await act(async () => {});
    expect(screen.getByText("No runs yet — start training")).toBeTruthy();

    // Instance with a run but no checkpoints → epoch dropdown shows an empty state.
    mockListRunCheckpoints.mockResolvedValue([]);
    fireEvent.click(screen.getByText("empty-model"));
    fireEvent.click(screen.getByText("my-model"));
    await act(async () => {});
    fireEvent.click(screen.getByText("Select run…"));
    fireEvent.click(screen.getByText("run_001"));
    await act(async () => {});
    expect(screen.getByText("No checkpoints saved yet")).toBeTruthy();
  });

  it("clears a stale instance/version selection when a run checkpoint is preselected with its instance", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});
    act(() => {
      const s = useInferenceStore.getState();
      s.setInstance("my-model");
      s.setVersion("v2");
      s.setModelPath("/models/my-model/versions/v2/model.pt");
    });
    await act(async () => {});
    act(() => {
      const s = useInferenceStore.getState();
      s.setPreselectedInstance("my-model");
      s.setPreselectedCheckpointPath("/runs/my-model/run_x/epoch_5.pt");
      s.setCheckpointContext("run_20260701_080000", "run_001");
    });
    await act(async () => {});

    const s = useInferenceStore.getState();
    expect(s.instance).toBe("my-model");
    expect(s.version).toBeNull();
    expect(s.modelPath).toBe("/runs/my-model/run_x/epoch_5.pt");
    expect(s.preselectedCheckpointPath).toBeNull();
    expect(s.preselectedInstance).toBeNull();
    // The version dropdown must not render alongside the checkpoint file.
    expect(screen.queryByText("Version")).toBeNull();
    expect(screen.getByText(/epoch_5\.pt/)).toBeTruthy();
  });
});
