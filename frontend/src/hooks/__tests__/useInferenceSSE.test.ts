import { describe, it, expect, beforeEach, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { resetAllStores } from "../../test-utils/resetStores";
import { useInferenceStore } from "../../store/inferenceStore";

const mockGetJobStatus = vi.fn();

vi.mock("../../lib/api", () => ({
  getBaseUrl: () => "http://localhost:8765",
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

describe("useInferenceSSE", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource);
    mockGetJobStatus.mockResolvedValue({
      job_id: "infer.1",
      job_type: "inference",
      status: "running",
      created_at: 0,
    });
  });

  it("applies a terminal state from the job registry when the stream never delivers events", async () => {
    mockGetJobStatus.mockResolvedValue({
      job_id: "infer.1",
      job_type: "inference",
      status: "completed",
      created_at: 0,
      result: { output: "/out/foo_sr.png" },
    });
    useInferenceStore.getState().setActiveJobId("infer.1");
    useInferenceStore.getState().setStatus("running");

    const { useInferenceSSE } = await import("../useInferenceSSE");
    renderHook(() => useInferenceSSE());
    await act(async () => {});

    expect(mockGetJobStatus).toHaveBeenCalled();
    expect(useInferenceStore.getState().status).toBe("done");
    expect(useInferenceStore.getState().result?.output).toBe("/out/foo_sr.png");
  });

  it("stores the rich result delivered with the done event", async () => {
    useInferenceStore.getState().setActiveJobId("infer.1");
    useInferenceStore.getState().setStatus("running");

    const { useInferenceSSE } = await import("../useInferenceSSE");
    renderHook(() => useInferenceSSE());
    await act(async () => {});

    const result = {
      success: true,
      output: "/out/foo_sr.png",
      preview_input_path: "/preview/in.png",
      preview_output_path: "/preview/out.png",
      input_resolution: { width: 512, height: 512 },
      output_resolution: { width: 1024, height: 1024 },
      inference_time_ms: 1234,
    };

    act(() => {
      FakeEventSource.instances[0].emit({ type: "done", ...result });
    });

    expect(useInferenceStore.getState().status).toBe("done");
    expect(useInferenceStore.getState().result?.output).toBe("/out/foo_sr.png");
    expect(useInferenceStore.getState().result?.preview_input_path).toBe("/preview/in.png");
  });

  it("applies the error event and stops the stream", async () => {
    useInferenceStore.getState().setActiveJobId("infer.1");
    useInferenceStore.getState().setStatus("running");

    const { useInferenceSSE } = await import("../useInferenceSSE");
    renderHook(() => useInferenceSSE());
    await act(async () => {});

    act(() => {
      FakeEventSource.instances[0].emit({ type: "error", message: "boom" });
    });

    expect(useInferenceStore.getState().status).toBe("error");
    expect(useInferenceStore.getState().errorMsg).toBe("boom");
  });

  it("polls the job registry on stream error and applies the terminal failed state", async () => {
    mockGetJobStatus.mockResolvedValue({
      job_id: "infer.1",
      job_type: "inference",
      status: "failed",
      error: "oom",
      created_at: 0,
    });
    useInferenceStore.getState().setActiveJobId("infer.1");
    useInferenceStore.getState().setStatus("running");

    const { useInferenceSSE } = await import("../useInferenceSSE");
    renderHook(() => useInferenceSSE());
    await act(async () => {});

    act(() => {
      FakeEventSource.instances[0].onerror?.();
    });
    await act(async () => {});

    expect(useInferenceStore.getState().status).toBe("error");
    expect(useInferenceStore.getState().errorMsg).toBe("oom");
  });
});
