import { describe, it, expect, beforeEach } from "vitest";
import { act, render, renderHook } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useTrainingStore, type HardwareData } from "../../../store/trainingStore";
import { useRollingHistory, Sparkline } from "./MetricPrimitives";

const DEFAULTS: HardwareData = {
  cpu_percent: 10,
  ram_used_gb: 4,
  ram_total_gb: 16,
  gpu_util_percent: null,
  vram_used_gb: null,
  vram_total_gb: null,
  temp_c: null,
};

function tick(overrides: Partial<HardwareData> = {}): void {
  act(() => {
    useTrainingStore.getState().updateFromHardware({ ...DEFAULTS, ...overrides });
  });
}

describe("useRollingHistory", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("reads the store-backed history for the selected metric", () => {
    const { result } = renderHook(() => useRollingHistory((s) => s.gpuUtilHistory));
    expect(result.current).toEqual([]);
    tick({ gpu_util_percent: 42 });
    expect(result.current).toEqual([42]);
  });

  it("stays empty while the value is null, even across hardware ticks", () => {
    const { result } = renderHook(() => useRollingHistory((s) => s.gpuUtilHistory));
    tick();
    tick();
    expect(result.current).toEqual([]);
  });

  it("appends a fresh sample every hardware tick even when the value is unchanged", () => {
    const { result } = renderHook(() => useRollingHistory((s) => s.gpuUtilHistory));
    tick({ gpu_util_percent: 42 });
    tick({ gpu_util_percent: 42 });
    tick({ gpu_util_percent: 42 });
    // a flat reading still advances the window — this is the time-series behavior
    expect(result.current).toEqual([42, 42, 42]);
  });

  it("appends new values as they arrive", () => {
    const { result } = renderHook(() => useRollingHistory((s) => s.gpuUtilHistory));
    tick({ gpu_util_percent: 10 });
    tick({ gpu_util_percent: 20 });
    expect(result.current).toEqual([10, 20]);
  });

  it("caps the window at MAX_HW_HISTORY", () => {
    const { result } = renderHook(() => useRollingHistory((s) => s.cpuUtilHistory));
    for (let i = 0; i < 100; i++) {
      tick({ cpu_percent: 1 });
    }
    expect(result.current).toHaveLength(90);
    expect(result.current.every((v) => v === 1)).toBe(true);
  });

  it("resumes appending once a null value turns non-null", () => {
    const { result } = renderHook(() => useRollingHistory((s) => s.gpuUtilHistory));
    tick({ gpu_util_percent: 5 });
    tick({ gpu_util_percent: null });
    tick({ gpu_util_percent: 7 });
    expect(result.current).toEqual([5, 7]);
  });

  it("persists across unmount/remount (tab-switch resilience)", () => {
    const { result, unmount } = renderHook(() => useRollingHistory((s) => s.gpuUtilHistory));
    tick({ gpu_util_percent: 10 });
    tick({ gpu_util_percent: 11 });
    expect(result.current).toEqual([10, 11]);
    unmount();
    // history keeps accumulating while the component is gone (other tab active)
    tick({ gpu_util_percent: 12 });
    const { result: remounted } = renderHook(() => useRollingHistory((s) => s.gpuUtilHistory));
    expect(remounted.current).toEqual([10, 11, 12]);
  });

  it("computes vram/ram percentage histories from used/total", () => {
    const { result: vram } = renderHook(() => useRollingHistory((s) => s.vramPctHistory));
    const { result: ram } = renderHook(() => useRollingHistory((s) => s.ramPctHistory));
    tick({ vram_used_gb: 4, vram_total_gb: 16, ram_used_gb: 8, ram_total_gb: 16 });
    expect(vram.current).toEqual([25]);
    expect(ram.current).toEqual([50]);
  });

  it("skips percentage history when total is missing", () => {
    const { result } = renderHook(() => useRollingHistory((s) => s.vramPctHistory));
    tick({ vram_used_gb: 4, vram_total_gb: null });
    tick({ vram_used_gb: 4, vram_total_gb: 0 });
    expect(result.current).toEqual([]);
  });
});

describe("Sparkline", () => {
  it("renders nothing for an empty window", () => {
    const { container } = render(<Sparkline values={[]} color="var(--green)" />);
    expect(container.querySelector("svg")).toBeNull();
  });

  it("renders a lone dot for a single sample", () => {
    const { container } = render(<Sparkline values={[42]} color="var(--green)" />);
    const circle = container.querySelector("circle");
    expect(circle).not.toBeNull();
    expect(container.querySelector("path")).toBeNull();
  });

  it("renders a path once two or more samples exist", () => {
    const { container } = render(<Sparkline values={[40, 42]} color="var(--green)" />);
    expect(container.querySelector("path")).not.toBeNull();
    expect(container.querySelector("circle")).toBeNull();
  });

  it("windows to the last `points` samples", () => {
    const many = Array.from({ length: 40 }, (_, i) => i);
    const { container } = render(<Sparkline values={many} color="var(--green)" width={150} points={10} />);
    const path = container.querySelector("path");
    expect(path).not.toBeNull();
    // 10 windowed samples → 9 cubic segments (a full 40 would yield 39).
    expect(path!.getAttribute("d")!.match(/C/g)).toHaveLength(9);
  });
});
