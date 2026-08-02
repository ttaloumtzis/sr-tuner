import { describe, it, expect, vi, beforeEach } from "vitest";
import { invoke as invokeRaw, convertFileSrc as convertFileSrcRaw } from "@tauri-apps/api/core";
import { getDatasetPairUrls, __resetTauriAvailability } from "../scanDatasets";

// Runtime imports are aliased to src/__mocks__/api.ts by vitest.config.ts;
// tsc still sees the real Tauri types, so cast to the mock shape.
const invoke = invokeRaw as unknown as ReturnType<typeof vi.fn>;
const convertFileSrc = convertFileSrcRaw as unknown as ReturnType<typeof vi.fn>;

const MANIFEST = JSON.stringify({
  config: { scale: 4 },
  pairs: [
    { hr: "HR/a.png", lr: "LR/a.png" },
    { hr: "HR/b.png", lr: "LR/b.png" },
  ],
});

beforeEach(() => {
  __resetTauriAvailability();
  invoke.mockReset();
  convertFileSrc.mockReset();
});

describe("getDatasetPairUrls", () => {
  it("reads the manifest once and serves local files via the asset protocol", async () => {
    invoke.mockImplementation(async (cmd: string) => {
      if (cmd === "path_exists") return true;
      if (cmd === "read_text_file") return MANIFEST;
      throw new Error(`unexpected command: ${cmd}`);
    });
    convertFileSrc.mockImplementation((p: string) => `asset://${p}`);

    const urls = await getDatasetPairUrls("/ws/datasets/ds1", "ds1", 2);

    expect(urls).toEqual([
      { hr: "asset:///ws/datasets/ds1/HR/a.png", lr: "asset:///ws/datasets/ds1/LR/a.png" },
      { hr: "asset:///ws/datasets/ds1/HR/b.png", lr: "asset:///ws/datasets/ds1/LR/b.png" },
    ]);
    expect(invoke).toHaveBeenCalledWith("read_text_file", {
      path: "/ws/datasets/ds1/manifest.json",
    });
  });

  it("falls back to directory scan when the manifest has no pairs", async () => {
    invoke.mockImplementation(async (cmd: string, args: { path: string }) => {
      if (cmd === "path_exists") return true;
      if (cmd === "read_text_file") return JSON.stringify({ config: { scale: 4 }, pairs: [] });
      if (cmd === "list_image_files") {
        if (args.path.endsWith("/HR")) return ["/ws/datasets/ds1/HR/a.png", "/ws/datasets/ds1/HR/b.png"];
        return ["/ws/datasets/ds1/LR/a.png", "/ws/datasets/ds1/LR/b.png"];
      }
      throw new Error(`unexpected command: ${cmd}`);
    });
    convertFileSrc.mockImplementation((p: string) => `asset://${p}`);

    const urls = await getDatasetPairUrls("/ws/datasets/ds1", "ds1", 2);

    expect(urls).toHaveLength(2);
    expect(urls[0].hr).toBe("asset:///ws/datasets/ds1/HR/a.png");
    expect(urls[0].lr).toBe("asset:///ws/datasets/ds1/LR/a.png");
  });

  it("falls back to API image URLs in browser dev mode (no Tauri)", async () => {
    invoke.mockRejectedValue(new Error("not in Tauri"));

    const urls = await getDatasetPairUrls("/ws/datasets/ds1", "ds1", 3);

    expect(urls).toHaveLength(3);
    expect(urls[0].hr).toContain("/api/datasets/ds1/image?kind=hr&index=0");
    expect(urls[2].lr).toContain("/api/datasets/ds1/image?kind=lr&index=2");
    expect(convertFileSrc).not.toHaveBeenCalled();
  });

  it("falls back to API image URLs when the manifest read fails", async () => {
    invoke.mockImplementation(async (cmd: string) => {
      if (cmd === "path_exists") return true;
      throw new Error("read failed");
    });

    const urls = await getDatasetPairUrls("/ws/datasets/ds1", "ds1", 1);

    expect(urls).toHaveLength(1);
    expect(urls[0].hr).toContain("/api/datasets/ds1/image?kind=hr&index=0");
  });
});
