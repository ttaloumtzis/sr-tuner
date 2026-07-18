import { describe, it, expect, beforeEach, vi } from "vitest";
import { SRProjManager } from "../SRProjManager";

const { invoke } = await import("@tauri-apps/api/core");
const invokeMock = invoke as ReturnType<typeof vi.fn>;

vi.mock("@tauri-apps/plugin-dialog", () => ({
  message: vi.fn().mockResolvedValue(undefined),
  confirm: vi.fn().mockResolvedValue(true),
}));

const makeSRProjFile = (overrides: Record<string, unknown> = {}) => ({
  version: "1.0.0" as const,
  name: "Test Project",
  created_at: "2024-01-01T00:00:00Z",
  last_modified_at: "2024-01-01T00:00:00Z",
  default_dataset: {
    training_path: "/data/train",
    validation_path: "/data/val",
    validation_strategy: "auto_split",
    validation_split_ratio: 0.1,
    dataset_type: "image_folder",
  },
  default_model: { architecture: "rrdb_esrgan", upscale_factor: 4 },
  runs: [],
  ui_state: { last_active_run_id: null, last_active_tab: null, expanded_panels: {} },
  metadata: { app_version: "0.1.0", notes: null, tags: [] },
  ...overrides,
});

const makeRun = (runId = "run-1") => ({
  run_id: runId,
  name: `Run ${runId}`,
  status: "configured" as const,
  created_at: "2024-01-01T00:00:00Z",
  started_at: null,
  completed_at: null,
  architecture: { type: "rrdb_esrgan" as const, upscale_factor: 4, custom_config: {} },
  training_config: {
    num_epochs: 10, batch_size: 4, learning_rate: 1e-4,
    scheduler: "cosine", optimizer: "Adam", patch_size: 192,
    augmentations: {
      horizontal_flip: true, vertical_flip: false, rotation_90: false,
      mixup: false, color_jitter: false, random_degradation: false,
      gaussian_blur: false, noise_injection: false,
    },
  },
  paths: { training_data: "/d", validation_data: "/v", checkpoint_dir: "/c", log_dir: "/l" },
  metrics: {
    current_epoch: 0, epochs_completed: 0, best_loss: null, best_loss_epoch: null,
    best_psnr: null, best_psnr_epoch: null, last_loss: null, last_psnr: null, last_ssim: null,
  },
  checkpoints: { total_count: 0, last_saved_epoch: null, last_saved_path: null, best_checkpoint_path: null },
});

describe("SRProjManager", () => {
  beforeEach(() => {
    SRProjManager.close();
    vi.clearAllMocks();
  });

  it("load() parses valid .srproj JSON into SRProjFile", async () => {
    const proj = makeSRProjFile();
    invokeMock.mockResolvedValueOnce(JSON.stringify(proj));

    const result = await SRProjManager.load("/tmp/test.srproj");

    expect(result.version).toBe("1.0.0");
    expect(result.name).toBe("Test Project");
    expect(result.runs).toEqual([]);
    expect(SRProjManager.filePath).toBe("/tmp/test.srproj");
  });

  it("load() throws on malformed JSON", async () => {
    invokeMock.mockResolvedValueOnce("{ not valid json }");

    await expect(SRProjManager.load("/tmp/bad.srproj")).rejects.toThrow();
  });

  it("load() shows version warning on mismatch and proceeds when user confirms", async () => {
    const { confirm } = await import("@tauri-apps/plugin-dialog");
    (confirm as ReturnType<typeof vi.fn>).mockResolvedValueOnce(true);

    const proj = makeSRProjFile({ version: "0.9.0" });
    invokeMock.mockResolvedValueOnce(JSON.stringify(proj));

    const result = await SRProjManager.load("/tmp/old.srproj");
    expect(result).toBeDefined();
    expect(confirm).toHaveBeenCalled();
  });

  it("load() throws when user cancels version-mismatch dialog", async () => {
    const { confirm } = await import("@tauri-apps/plugin-dialog");
    (confirm as ReturnType<typeof vi.fn>).mockResolvedValueOnce(false);

    const proj = makeSRProjFile({ version: "0.9.0" });
    invokeMock.mockResolvedValueOnce(JSON.stringify(proj));

    await expect(SRProjManager.load("/tmp/old.srproj")).rejects.toThrow("User cancelled");
  });

  it("addRun() appends new SRProjRun to the current project", async () => {
    const proj = makeSRProjFile();
    invokeMock.mockResolvedValueOnce(JSON.stringify(proj));
    await SRProjManager.load("/tmp/test.srproj");

    const run = makeRun("run-1");
    SRProjManager.addRun(run);

    expect(SRProjManager.current?.runs).toHaveLength(1);
    expect(SRProjManager.current?.runs[0].run_id).toBe("run-1");
  });

  it("updateRun() mutates only the target run by run_id", async () => {
    const proj = makeSRProjFile();
    invokeMock.mockResolvedValueOnce(JSON.stringify(proj));
    await SRProjManager.load("/tmp/test.srproj");

    SRProjManager.addRun(makeRun("run-1"));
    SRProjManager.addRun(makeRun("run-2"));

    SRProjManager.updateRun("run-1", { status: "completed" });

    const runs = SRProjManager.current?.runs ?? [];
    expect(runs.find((r) => r.run_id === "run-1")?.status).toBe("completed");
    expect(runs.find((r) => r.run_id === "run-2")?.status).toBe("configured");
  });

  it("save() serializes current state to disk via invoke", async () => {
    const proj = makeSRProjFile();
    invokeMock.mockResolvedValueOnce(JSON.stringify(proj));
    await SRProjManager.load("/tmp/test.srproj");

    invokeMock.mockResolvedValueOnce(undefined);
    await SRProjManager.save();

    expect(invokeMock).toHaveBeenCalledWith(
      "write_text_file",
      expect.objectContaining({ path: "/tmp/test.srproj", contents: expect.any(String) })
    );
  });

  it("close() resets filePath and current to null", async () => {
    const proj = makeSRProjFile();
    invokeMock.mockResolvedValueOnce(JSON.stringify(proj));
    await SRProjManager.load("/tmp/test.srproj");

    SRProjManager.close();

    expect(SRProjManager.filePath).toBeNull();
    expect(SRProjManager.current).toBeNull();
  });
});
