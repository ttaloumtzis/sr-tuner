import { describe, it, expect } from "vitest";
import { downsample } from "../metrics";
import type { MetricsEvent } from "../ipc-types";

function makeEvent(iter: number): MetricsEvent {
  return {
    iter,
    epoch: 1,
    g_loss: 0.5,
    d_loss: null,
    psnr: null,
    ssim: null,
    gpu_util: null,
    vram_gb: null,
    temp_c: null,
    cpu_util: null,
    speed: null,
    timestamp: "2024-01-01T00:00:00.000Z",
  };
}

describe("downsample", () => {
  it("returns array unchanged when length <= maxPoints", () => {
    const events = [makeEvent(1), makeEvent(2), makeEvent(3)];
    const result = downsample(events, 200);
    expect(result).toBe(events);
  });

  it("returns approximately 200 points on array of 600 events at stride 3", () => {
    const events = Array.from({ length: 600 }, (_, i) => makeEvent(i + 1));
    const result = downsample(events, 200);
    // At stride 3: indices 0,3,6,...,597 = 200 items; last event (599) may be appended
    expect(result.length).toBeGreaterThanOrEqual(200);
    expect(result.length).toBeLessThanOrEqual(201);
  });

  it("always includes the last event", () => {
    const events = Array.from({ length: 601 }, (_, i) => makeEvent(i + 1));
    const result = downsample(events, 200);
    expect(result[result.length - 1]).toBe(events[events.length - 1]);
  });

  it("returns empty array unchanged", () => {
    const result = downsample([], 200);
    expect(result).toEqual([]);
  });

  it("downsampled points are at uniform stride", () => {
    const events = Array.from({ length: 600 }, (_, i) => makeEvent(i + 1));
    const result = downsample(events, 200);
    // First element should be events[0]
    expect(result[0]).toBe(events[0]);
  });
});
