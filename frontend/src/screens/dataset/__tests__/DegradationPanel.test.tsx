import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { resetAllStores } from "../../../test-utils/resetStores";
import { useDatasetStore } from "../../../store/datasetStore";
import { DegradationPanel } from "../DegradationPanel";

describe("DegradationPanel", () => {
  beforeEach(() => {
    resetAllStores();
  });

  it("toggles the Blur section from the store default", () => {
    render(<DegradationPanel />);
    expect(useDatasetStore.getState().degBlur).toBe(true);
    fireEvent.click(screen.getByText("Blur"));
    expect(useDatasetStore.getState().degBlur).toBe(false);
  });

  it("toggles the Noise section on", () => {
    render(<DegradationPanel />);
    expect(useDatasetStore.getState().degNoise).toBe(false);
    fireEvent.click(screen.getByText("Noise"));
    expect(useDatasetStore.getState().degNoise).toBe(true);
  });

  it("toggles motion blur enabled", () => {
    render(<DegradationPanel />);
    expect(useDatasetStore.getState().motionBlurEnabled).toBe(true);
    fireEvent.click(screen.getByLabelText("Motion blur"));
    expect(useDatasetStore.getState().motionBlurEnabled).toBe(false);
  });

  it("updates the FPS setting from the numeric input", () => {
    render(<DegradationPanel />);
    fireEvent.change(screen.getByDisplayValue("10"), { target: { value: "24" } });
    expect(useDatasetStore.getState().frameRate).toBe(24);
  });

  it("does not render the removed frame format dropdown", () => {
    render(<DegradationPanel />);
    expect(screen.queryByText("Format")).toBeNull();
  });
});
