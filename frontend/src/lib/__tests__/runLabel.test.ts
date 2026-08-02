import { describe, it, expect } from "vitest";
import { buildRunDisplays, shortRunId, sortGroups } from "../runLabel";
import type { RunInfo } from "../api-types";

function makeRun(run_id: string, status: RunInfo["status"], created_at: string): RunInfo {
  return {
    run_id, status, created_at,
    finished_at: null, error: null, checkpoint_count: 0,
    total_size_mb: 0, last_epoch: 0, has_metrics: false, config: null,
  };
}

describe("shortRunId", () => {
  it("formats run_20260702_090000", () => {
    expect(shortRunId("run_20260702_090000")).toBe("2026-07-02 09:00");
  });

  it("appends a suffix for colliding runs", () => {
    expect(shortRunId("run_20260702_090000_3")).toBe("2026-07-02 09:00 #3");
  });

  it("returns the id unchanged when it does not match", () => {
    expect(shortRunId("other_123")).toBe("other_123");
  });
});

describe("buildRunDisplays", () => {
  it("numbers finished runs sequentially in creation order, ignoring status order", () => {
    const runs = [
      makeRun("run_20260701_080000", "running", "2026-07-01T08:00:00Z"),
      makeRun("run_20260702_090000", "finished", "2026-07-02T09:00:00Z"),
      makeRun("run_20260703_100000", "finished", "2026-07-03T10:00:00Z"),
      makeRun("run_20260704_110000", "finished", "2026-07-04T11:00:00Z"),
    ];
    const d = buildRunDisplays(runs);
    expect(d.get("run_20260702_090000")?.label).toBe("run_001");
    expect(d.get("run_20260703_100000")?.label).toBe("run_002");
    expect(d.get("run_20260704_110000")?.label).toBe("run_003");
    // non-finished keeps its timestamp label
    expect(d.get("run_20260701_080000")?.label).toBe("2026-07-01 08:00");
  });

  it("assigns date groups based on created_at", () => {
    const runs = [
      makeRun("run_today", "finished", new Date().toISOString()),
      makeRun("run_yesterday", "finished", new Date(Date.now() - 86_400_000).toISOString()),
      makeRun("run_old", "failed", "2025-01-05T10:00:00Z"),
    ];
    const d = buildRunDisplays(runs);
    expect(d.get("run_today")?.group).toBe("Today");
    expect(d.get("run_yesterday")?.group).toBe("Yesterday");
    expect(d.get("run_old")?.group).toBe("Jan 5, 2025");
  });

  it("falls back to Unknown when created_at is missing", () => {
    const d = buildRunDisplays([makeRun("run_x", "finished", "")]);
    expect(d.get("run_x")?.group).toBe("Unknown");
  });
});

describe("sortGroups", () => {
  it("orders Today → Yesterday → Unknown → dates desc", () => {
    const groups = sortGroups(["Jan 5, 2025", "Today", "Unknown", "Yesterday", "Feb 3, 2025"]);
    expect(groups).toEqual(["Today", "Yesterday", "Feb 3, 2025", "Jan 5, 2025", "Unknown"]);
  });
});
