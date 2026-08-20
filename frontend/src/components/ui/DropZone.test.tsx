import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DropZone } from "./DropZone";

describe("DropZone", () => {
  it("renders the label when no file is selected", () => {
    render(<DropZone label="Drop image here" path={null} onSelect={vi.fn()} />);
    expect(screen.getByText("Drop image here")).not.toBeNull();
  });

  it("renders the selected file's basename inline in compact mode", () => {
    render(<DropZone label="Drop image here" path="/data/img.png" onSelect={vi.fn()} />);
    expect(screen.getByText("img.png")).not.toBeNull();
  });

  it("uses the name override when provided", () => {
    render(
      <DropZone label="Drop a video here" path="/data/clip.mkv" name="clip.mkv" onSelect={vi.fn()} />
    );
    expect(screen.getByText("clip.mkv")).not.toBeNull();
  });

  it("highlights the border with a visible color on drag when accent is the invisible default", () => {
    render(<DropZone label="Drop image here" path={null} onSelect={vi.fn()} />);
    const zone = screen.getByText("Drop image here").closest(".dropzone") as HTMLElement;
    fireEvent.dragOver(zone);
    expect(zone.style.border).toContain("var(--green)");
    fireEvent.dragLeave(zone);
    expect(zone.style.border).toContain("var(--border)");
  });

  it("highlights the border with its own accent on drag when the accent is visible", () => {
    render(
      <DropZone label="Drop GT here" path={null} accent="var(--blue)" onSelect={vi.fn()} />
    );
    const zone = screen.getByText("Drop GT here").closest(".dropzone") as HTMLElement;
    fireEvent.dragOver(zone);
    expect(zone.style.border).toContain("var(--blue)");
  });

  it("keeps the accent unchanged on drag when highlightOnDrag is false", () => {
    render(
      <DropZone label="Drop here" path={null} highlightOnDrag={false} onSelect={vi.fn()} />
    );
    const zone = screen.getByText("Drop here").closest(".dropzone") as HTMLElement;
    fireEvent.dragOver(zone);
    expect(zone.style.border).toContain("var(--border)");
  });

  it("calls onSelect with the dropped file path/name", () => {
    const onSelect = vi.fn();
    render(<DropZone label="Drop image here" path={null} onSelect={onSelect} />);
    const zone = screen.getByText("Drop image here").closest(".dropzone") as HTMLElement;
    fireEvent.drop(zone, { dataTransfer: { files: [new File([""], "shot.png")] } });
    expect(onSelect).toHaveBeenCalledWith("shot.png");
  });

  it("shows a removable selected row in selectedAsRow mode and clears via the ✕ button", () => {
    const onSelect = vi.fn();
    const onClear = vi.fn();
    render(
      <DropZone
        label="Drop a video file here"
        path="/data/clip.mkv"
        name="clip.mkv"
        selectedAsRow
        onSelect={onSelect}
        onClear={onClear}
      />
    );
    const row = screen.getByText("clip.mkv");
    expect(row).not.toBeNull();
    const clear = row.closest(".dropzone-row")!.querySelector("button") as HTMLElement;
    fireEvent.click(clear);
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("calls onClear from the inline Clear button", () => {
    const onClear = vi.fn();
    render(
      <DropZone label="Drop GT here" path="/data/gt.png" onSelect={vi.fn()} onClear={onClear} />
    );
    fireEvent.click(screen.getByText("Clear"));
    expect(onClear).toHaveBeenCalledTimes(1);
  });
});