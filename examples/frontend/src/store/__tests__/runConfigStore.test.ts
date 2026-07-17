import { describe, it, expect, beforeEach } from "vitest";
import { useRunConfigStore } from "../runConfigStore";
import { resetAllStores } from "../../test-utils/resetStores";
import type { ResumeFrom } from "../runConfigStore";

const RESUME: ResumeFrom = {
  checkpoint_path: "/checkpoints/epoch_010.pth",
  resume_epoch: 10,
  resume_optimizer_state: true,
  resume_lr_scheduler_state: true,
};

describe("runConfigStore", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("setResumeFrom stores full ResumeFrom shape", () => {
    useRunConfigStore.getState().setResumeFrom(RESUME);

    const rf = useRunConfigStore.getState().resumeFrom;
    expect(rf).not.toBeNull();
    expect(rf?.checkpoint_path).toBe("/checkpoints/epoch_010.pth");
    expect(rf?.resume_epoch).toBe(10);
    expect(rf?.resume_optimizer_state).toBe(true);
    expect(rf?.resume_lr_scheduler_state).toBe(true);
  });

  it("setResumeFrom(null) clears resumeFrom", () => {
    useRunConfigStore.getState().setResumeFrom(RESUME);
    useRunConfigStore.getState().setResumeFrom(null);
    expect(useRunConfigStore.getState().resumeFrom).toBeNull();
  });

  it("setDevice updates device string", () => {
    useRunConfigStore.getState().setDevice("cuda:1");
    expect(useRunConfigStore.getState().device).toBe("cuda:1");
  });

  it("setFp16 toggles fp16", () => {
    expect(useRunConfigStore.getState().fp16).toBe(false);
    useRunConfigStore.getState().setFp16(true);
    expect(useRunConfigStore.getState().fp16).toBe(true);
  });

  it("setTensorboard toggles tensorboard", () => {
    expect(useRunConfigStore.getState().tensorboard).toBe(false);
    useRunConfigStore.getState().setTensorboard(true);
    expect(useRunConfigStore.getState().tensorboard).toBe(true);
  });

  it("setCompile toggles compile", () => {
    expect(useRunConfigStore.getState().compile).toBe(false);
    useRunConfigStore.getState().setCompile(true);
    expect(useRunConfigStore.getState().compile).toBe(true);
  });

  it("camelCase resumeFrom serializes to snake_case resume_from shape", () => {
    useRunConfigStore.getState().setResumeFrom(RESUME);

    const rf = useRunConfigStore.getState().resumeFrom;
    // Regression guard: confirm field names are snake_case (§11.11)
    expect(Object.keys(rf!)).toContain("checkpoint_path");
    expect(Object.keys(rf!)).toContain("resume_epoch");
    expect(Object.keys(rf!)).toContain("resume_optimizer_state");
    expect(Object.keys(rf!)).toContain("resume_lr_scheduler_state");
    // camelCase must NOT appear
    expect(Object.keys(rf!)).not.toContain("checkpointPath");
    expect(Object.keys(rf!)).not.toContain("resumeEpoch");
  });
});
