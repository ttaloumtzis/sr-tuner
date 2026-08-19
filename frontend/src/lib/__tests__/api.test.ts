import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { finalizeDataset } from "../api";

describe("finalizeDataset", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({ path: "/ds", scale: 4, num_pairs: 3 }),
      })),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("omits config_overrides when not provided", async () => {
    await finalizeDataset({ path: "/ds", scale: 2 });

    const [path, init] = vi.mocked(fetch).mock.calls[0] as unknown as [string, RequestInit];
    expect(path).toContain("/api/datasets/finalize");
    const body = JSON.parse(String(init.body));
    expect(body.config_overrides).toBeUndefined();
  });
});
