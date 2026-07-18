import { describe, it, expect, beforeEach, vi } from "vitest";
import { useProjectStore } from "../projectStore";
import { resetAllStores } from "../../test-utils/resetStores";

vi.mock("../../lib/SRProjManager", () => ({
  SRProjManager: {
    load: vi.fn(),
    close: vi.fn(),
    addRun: vi.fn(),
    updateRun: vi.fn(),
    getRun: vi.fn(),
    setProject: vi.fn(),
    setActiveRun: vi.fn(),
    setActiveTab: vi.fn(),
    save: vi.fn().mockResolvedValue(undefined),
    current: null,
    filePath: null,
  },
}));

vi.mock("../../screens/ProjectScreen", () => ({
  addToRecent: vi.fn(),
}));

const makeSRProjFile = () => ({
  version: "1.0.0" as const,
  name: "Test Project",
  created_at: "2024-01-01T00:00:00Z",
  last_modified_at: "2024-01-01T00:00:00Z",
  default_dataset: {
    training_path: "/data/train",
    validation_path: "/data/val",
    validation_strategy: "auto_split" as const,
    validation_split_ratio: 0.1,
    dataset_type: "image_folder" as const,
  },
  default_model: { architecture: "Real-ESRGAN" as const, upscale_factor: 4 },
  runs: [],
  ui_state: { last_active_run_id: null, last_active_tab: null, expanded_panels: {} },
  metadata: { app_version: "0.1.0", notes: null, tags: [] },
});

describe("projectStore", () => {
  beforeEach(() => {
    resetAllStores();
    vi.clearAllMocks();
  });

  it("openProject sets project with filePath and runs", async () => {
    const { SRProjManager } = await import("../../lib/SRProjManager");
    const proj = makeSRProjFile();
    (SRProjManager.load as ReturnType<typeof vi.fn>).mockResolvedValue(proj);

    await useProjectStore.getState().openProject("/tmp/test.srproj");

    const state = useProjectStore.getState();
    expect(state.project).not.toBeNull();
    expect(state.project?.filePath).toBe("/tmp/test.srproj");
    expect(state.project?.runs).toEqual([]);
    expect(state.project?.name).toBe("Test Project");
  });

  it("openProject with runs array populates runs", async () => {
    const { SRProjManager } = await import("../../lib/SRProjManager");
    const run = {
      run_id: "run-1",
      name: "Run 1",
      status: "completed" as const,
      created_at: "2024-01-01T00:00:00Z",
      started_at: null,
      completed_at: null,
      architecture: { type: "Real-ESRGAN" as const, upscale_factor: 4, custom_config: {} },
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
        current_epoch: 10, epochs_completed: 10, best_loss: null, best_loss_epoch: null,
        best_psnr: null, best_psnr_epoch: null, last_loss: null, last_psnr: null, last_ssim: null,
      },
      checkpoints: { total_count: 0, last_saved_epoch: null, last_saved_path: null, best_checkpoint_path: null },
      sidecar_pid: null,
      sidecar_log_file: null,
    };
    const proj = { ...makeSRProjFile(), runs: [run] };
    (SRProjManager.load as ReturnType<typeof vi.fn>).mockResolvedValue(proj);

    await useProjectStore.getState().openProject("/tmp/test.srproj");

    expect(useProjectStore.getState().project?.runs).toHaveLength(1);
    expect(useProjectStore.getState().project?.runs[0].run_id).toBe("run-1");
  });

  it("closeProject resets project to null", async () => {
    const { SRProjManager } = await import("../../lib/SRProjManager");
    const proj = makeSRProjFile();
    (SRProjManager.load as ReturnType<typeof vi.fn>).mockResolvedValue(proj);

    await useProjectStore.getState().openProject("/tmp/test.srproj");
    expect(useProjectStore.getState().project).not.toBeNull();

    useProjectStore.getState().closeProject();
    expect(useProjectStore.getState().project).toBeNull();
  });
});
