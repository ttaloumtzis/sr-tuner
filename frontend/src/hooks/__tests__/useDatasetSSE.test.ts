import { describe, it, expect, beforeEach, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { resetAllStores } from "../../test-utils/resetStores";
import { useDatasetStore } from "../../store/datasetStore";

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

describe("useDatasetSSE", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
    FakeEventSource.instances = [];
    vi.stubGlobal("EventSource", FakeEventSource);
    mockGetJobStatus.mockResolvedValue({
      job_id: "dataset.health_1",
      job_type: "dataset.health",
      status: "running",
      created_at: 0,
    });
  });

  it("applies a terminal job state from the job registry when the stream never delivers events", async () => {
    mockGetJobStatus.mockResolvedValue({
      job_id: "dataset.health_1",
      job_type: "dataset.health",
      status: "completed",
      created_at: 0,
    });
    useDatasetStore.getState().setJobId("dataset.health_1");
    useDatasetStore.getState().setJobType("health");
    useDatasetStore.getState().setJobStatus("running");

    const { useDatasetSSE } = await import("../useDatasetSSE");
    renderHook(() => useDatasetSSE());
    await act(async () => {});

    expect(mockGetJobStatus).toHaveBeenCalled();
    expect(useDatasetStore.getState().jobStatus).toBe("done");
  });

  it("stores the health report delivered with the done event", async () => {
    useDatasetStore.getState().setJobId("dataset.health_1");
    useDatasetStore.getState().setJobType("health");

    const { useDatasetSSE } = await import("../useDatasetSSE");
    renderHook(() => useDatasetSSE());
    await act(async () => {});

    const report = {
      total_images: 2650,
      resolutions: { "1920x1080": 2650 },
      aspect_ratios: { "1.78": 2650 },
      channels: { "RGB (3 channels)": 2650 },
      computed_threshold: 18.5,
      black_frames: [],
      unreadable: [],
    };

    act(() => {
      FakeEventSource.instances[0].emit({ type: "done", elapsed_seconds: 3, report });
    });

    expect(useDatasetStore.getState().jobStatus).toBe("done");
    expect(useDatasetStore.getState().jobHealthReport).toEqual(report);
  });

  it("polls the job registry on stream error and applies the terminal state", async () => {
    mockGetJobStatus.mockResolvedValue({
      job_id: "dataset.health_1",
      job_type: "dataset.health",
      status: "failed",
      error: "boom",
      created_at: 0,
    });
    useDatasetStore.getState().setJobId("dataset.health_1");
    useDatasetStore.getState().setJobType("health");
    useDatasetStore.getState().setJobStatus("running");

    const { useDatasetSSE } = await import("../useDatasetSSE");
    renderHook(() => useDatasetSSE());
    await act(async () => {});

    act(() => {
      FakeEventSource.instances[0].onerror?.();
    });
    await act(async () => {});

    expect(useDatasetStore.getState().jobStatus).toBe("error");
    expect(useDatasetStore.getState().jobError).toBe("boom");
    expect(FakeEventSource.instances).toHaveLength(1);
  });
});
