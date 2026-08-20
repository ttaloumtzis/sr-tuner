import { describe, it, expect, vi } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { ConfirmScrim } from "./ConfirmScrim";

function base(overrides: Record<string, unknown> = {}) {
  return {
    title: "Delete?",
    message: "Are you sure?",
    confirmLabel: "Delete",
    onConfirm: vi.fn(),
    onCancel: vi.fn(),
    ...overrides,
  };
}

describe("ConfirmScrim", () => {
  it("renders title, message and buttons", () => {
    const props = base();
    const { getByText } = render(<ConfirmScrim {...props} />);
    expect(getByText("Delete?")).toBeTruthy();
    expect(getByText("Are you sure?")).toBeTruthy();
    expect(getByText("Cancel")).toBeTruthy();
    expect(getByText("Delete")).toBeTruthy();
  });

  it("renders ReactNode messages", () => {
    const { getByText } = render(
      <ConfirmScrim {...base({ message: <>Files: <b>a.pth</b></> })} />
    );
    expect(getByText("a.pth")).toBeTruthy();
  });

  it("calls onCancel and onConfirm", () => {
    const props = base();
    const { getByText } = render(<ConfirmScrim {...props} />);
    fireEvent.click(getByText("Cancel"));
    expect(props.onCancel).toHaveBeenCalledTimes(1);
    fireEvent.click(getByText("Delete"));
    expect(props.onConfirm).toHaveBeenCalledTimes(1);
  });

  it("closes on Escape only when escToClose is set", () => {
    const off = base();
    render(<ConfirmScrim {...off} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(off.onCancel).not.toHaveBeenCalled();

    const on = base({ escToClose: true });
    render(<ConfirmScrim {...on} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(on.onCancel).toHaveBeenCalledTimes(1);
  });

  it("applies zIndex and width", () => {
    const { container } = render(<ConfirmScrim {...base({ width: 320, zIndex: 100 })} />);
    expect(container.querySelector(".scrim")).toHaveStyle("z-index: 100");
    expect(container.querySelector(".scrim-box")).toHaveStyle("width: 320px");
  });
});