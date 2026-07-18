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
});