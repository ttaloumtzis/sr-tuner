import { describe, it, expect, beforeEach } from "vitest";
import { useDatasetStore } from "../datasetStore";
import { resetAllStores } from "../../test-utils/resetStores";

describe("datasetStore", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("setKernel updates kernel field", () => {
    useDatasetStore.getState().setKernel("bilinear");
    expect(useDatasetStore.getState().kernel).toBe("bilinear");
  });

  it("setScale updates scale", () => {
    useDatasetStore.getState().setScale(2);
    expect(useDatasetStore.getState().scale).toBe(2);
  });

  it("setSubTab updates subTab", () => {
    useDatasetStore.getState().setSubTab("browse");
    expect(useDatasetStore.getState().subTab).toBe("browse");
  });

  it("setMode updates mode", () => {
    useDatasetStore.getState().setMode("video_extract");
    expect(useDatasetStore.getState().mode).toBe("video_extract");
  });

  it("setRootPath updates rootPath", () => {
    useDatasetStore.getState().setRootPath("/data/dataset");
    expect(useDatasetStore.getState().rootPath).toBe("/data/dataset");
  });

  it("setFrameRate updates frameRate", () => {
    useDatasetStore.getState().setFrameRate(30);
    expect(useDatasetStore.getState().frameRate).toBe(30);
  });

  it("setResizeMethod updates resizeMethod", () => {
    useDatasetStore.getState().setResizeMethod("lanczos");
    expect(useDatasetStore.getState().resizeMethod).toBe("lanczos");
  });

  it("addVideoFiles appends to videoFiles", () => {
    useDatasetStore.getState().addVideoFiles(["/videos/test.mp4"]);
    expect(useDatasetStore.getState().videoFiles).toHaveLength(1);
    expect(useDatasetStore.getState().videoFiles[0].name).toBe("test.mp4");
  });

  it("clearVideoFiles empties the list", () => {
    useDatasetStore.getState().addVideoFiles(["/videos/test.mp4"]);
    useDatasetStore.getState().clearVideoFiles();
    expect(useDatasetStore.getState().videoFiles).toHaveLength(0);
  });

  it("initial state has create subTab", () => {
    expect(useDatasetStore.getState().subTab).toBe("create");
  });

  it("initial state has scale 4", () => {
    expect(useDatasetStore.getState().scale).toBe(4);
  });

  it("setJobDatasetPath records the dataset a job runs on", () => {
    useDatasetStore.getState().setJobDatasetPath("/data/ds");
    expect(useDatasetStore.getState().jobDatasetPath).toBe("/data/ds");
  });

  it("setJobHealthReport stores the health report from the done event", () => {
    const report = {
      total_images: 10,
      resolutions: { "1920x1080": 10 },
      aspect_ratios: { "1.78": 10 },
      channels: { "RGB (3 channels)": 10 },
      computed_threshold: 18.5,
      black_frames: ["a.png"],
      unreadable: [],
    };
    useDatasetStore.getState().setJobHealthReport(report);
    expect(useDatasetStore.getState().jobHealthReport).toEqual(report);
  });

  it("setSubTab preserves an in-flight job instead of wiping it", () => {
    useDatasetStore.getState().setJobId("dataset.health_123");
    useDatasetStore.getState().setJobDatasetPath("/data/ds");
    useDatasetStore.getState().setJobType("health");
    useDatasetStore.getState().setJobStatus("running");
    useDatasetStore.getState().setSubTab("browse");
    expect(useDatasetStore.getState().jobId).toBe("dataset.health_123");
    expect(useDatasetStore.getState().jobStatus).toBe("running");
    expect(useDatasetStore.getState().jobType).toBe("health");
    expect(useDatasetStore.getState().jobDatasetPath).toBe("/data/ds");
  });

  it("clearJob clears job fields including jobDatasetPath and jobHealthReport", () => {
    useDatasetStore.getState().setJobId("dataset.health_123");
    useDatasetStore.getState().setJobDatasetPath("/data/ds");
    useDatasetStore.getState().setJobHealthReport({
      total_images: 1,
      resolutions: {},
      aspect_ratios: {},
      channels: {},
      computed_threshold: 1,
      black_frames: [],
      unreadable: [],
    });
    useDatasetStore.getState().clearJob();
    expect(useDatasetStore.getState().jobId).toBeNull();
    expect(useDatasetStore.getState().jobStatus).toBe("idle");
    expect(useDatasetStore.getState().jobType).toBeNull();
    expect(useDatasetStore.getState().jobDatasetPath).toBeNull();
    expect(useDatasetStore.getState().jobHealthReport).toBeNull();
  });
});