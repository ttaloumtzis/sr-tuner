import { describe, it, expect, vi } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { NumberInput } from "./NumberInput";
import { TextInput } from "./TextInput";

describe("NumberInput", () => {
  it("supports float steps (1e-5) without clamping when only min is present", () => {
    const onChange = vi.fn();
    const { container } = render(
      <NumberInput value={0} min={0} step={1e-5} onChange={onChange} />
    );
    const input = container.querySelector("input")!;
    expect(Number(input.step)).toBe(1e-5);
    fireEvent.change(input, { target: { value: "0.0001" } });
    expect(onChange).toHaveBeenCalledWith(0.0001, "0.0001");
  });

  it("supports float steps (0.01) and clamps only when both min and max are present", () => {
    const onChange = vi.fn();
    const { container } = render(
      <NumberInput value={0.5} min={0} max={1} step={0.01} onChange={onChange} />
    );
    expect(container.querySelector("input")!.getAttribute("step")).toBe("0.01");
    fireEvent.change(container.querySelector("input")!, { target: { value: "1.5" } });
    expect(onChange).toHaveBeenCalledWith(1, "1.5");
  });

  it("does not clamp when only min is present (min-only no-clamp)", () => {
    const onChange = vi.fn();
    const { container } = render(
      <NumberInput value={0} min={0} onChange={onChange} />
    );
    fireEvent.change(container.querySelector("input")!, { target: { value: "-3" } });
    expect(onChange).toHaveBeenCalledWith(-3, "-3");
  });

  it("does not clamp when only max is present", () => {
    const onChange = vi.fn();
    const { container } = render(
      <NumberInput value={0} max={10} onChange={onChange} />
    );
    fireEvent.change(container.querySelector("input")!, { target: { value: "25" } });
    expect(onChange).toHaveBeenCalledWith(25, "25");
  });

  it("maps empty/invalid input to 0 for clamped inputs", () => {
    const onChange = vi.fn();
    const { container } = render(
      <NumberInput value={4} min={0} max={10} onChange={onChange} />
    );
    fireEvent.change(container.querySelector("input")!, { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith(0, "");
  });

  it("blanks non-finite stored values via the display guard", () => {
    const { container } = render(
      <NumberInput value={Number.NaN} min={0} max={10} onChange={vi.fn()} />
    );
    expect(container.querySelector("input")!.value).toBe("");
  });

  it("renders a finite number value", () => {
    const { container } = render(
      <NumberInput value={0.001} onChange={vi.fn()} />
    );
    expect(container.querySelector("input")!.value).toBe("0.001");
  });

  it("surfaces the raw draft string and commits only finite values", () => {
    const onChange = vi.fn();
    const { container, rerender } = render(
      <NumberInput value="0.0" onChange={onChange} />
    );
    const input = container.querySelector("input")!;
    // jsdom sanitizes invalid number input to "", mirroring browsers' number
    // input sanitization — the component still reports the sanitized raw value.
    fireEvent.change(input, { target: { value: "1e-" } });
    expect(onChange).toHaveBeenCalledWith(0, "");
    rerender(<NumberInput value="0.1" onChange={onChange} />);
    expect(container.querySelector("input")!.value).toBe("0.1");
  });

  it("keeps step, disabled and title attributes", () => {
    const { container } = render(
      <NumberInput value={1} step={2} disabled title="tile overlap" onChange={vi.fn()} />
    );
    const input = container.querySelector("input")!;
    expect(input.disabled).toBe(true);
    expect(input.getAttribute("step")).toBe("2");
    expect(input.getAttribute("title")).toBe("tile overlap");
  });
});

describe("TextInput", () => {
  it("forwards the value and change events", () => {
    const onChange = vi.fn();
    const { container } = render(<TextInput value="abc" onChange={onChange} />);
    const input = container.querySelector("input")!;
    expect(input.value).toBe("abc");
    fireEvent.change(input, { target: { value: "xyz" } });
    expect(onChange).toHaveBeenCalledWith("xyz");
  });
});