import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act, fireEvent } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useInferenceStore } from "../../../store/inferenceStore";

const mockListInstances = vi.fn();
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
    mockListInstances.mockResolvedValue([{ name: "my-model", path: "/m", architecture: "rrdb_esrgan", scale: 4, checkpoints: [], latest_version: null, config: {} }]);
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

  it("walks the Model → Run → Checkpoint cascade and selects the latest checkpoint", async () => {
    const { ScreenInference } = await import("../ScreenInference");
    render(<ScreenInference />);
    await act(async () => {});

    act(() => {
      useInferenceStore.getState().setInputPath("/in/foo.png");
    });
    await act(async () => {});

    // Open the Model instance dropdown and pick "my-model".
    fireEvent.click(screen.getByText("Select instance…"));
    fireEvent.click(screen.getByText("my-model"));
    await act(async () => {});

    // Instance selection auto-loads the latest run's checkpoints.
    expect(mockListRuns).toHaveBeenCalled();
    expect(mockListRunCheckpoints).toHaveBeenCalledWith("my-model", "run_20260701_080000");
    // The auto-selected latest checkpoint becomes the model path.
    expect(useInferenceStore.getState().modelPath).toBe("/runs/my-model/run_x/epoch_5.pt");
  });
});
