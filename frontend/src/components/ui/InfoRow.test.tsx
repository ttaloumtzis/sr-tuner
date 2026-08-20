import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { InfoRow } from "./InfoRow";

type Overrides = Partial<Omit<Parameters<typeof InfoRow>[0], "label">>;

function row(extra: Overrides = {}) {
  const { container } = render(<InfoRow label="L" value="V" {...extra} />);
  const root = container.querySelector(".info-row") as HTMLElement;
  return {
    root,
    label: container.querySelector(".info-row-label") as HTMLElement,
    value: container.querySelector(".info-row-value") as HTMLElement,
  };
}

describe("InfoRow", () => {
  it("renders label and value", () => {
    const { root, label, value } = row();
    expect(root).toBeTruthy();
    expect(label.textContent).toBe("L");
    expect(value.textContent).toBe("V");
  });

  it("formats number values with dec", () => {
    const { value } = row({ value: 3.14159, dec: 2 });
    expect(value.textContent).toBe("3.14");
  });

  it("renders — for null and dims it", () => {
    const { value } = row({ value: null });
    expect(value.textContent).toBe("—");
    expect(value.style.color).toBe("var(--dim)");
    expect(value.style.fontWeight).toBe("400");
  });

  it("applies color to the value", () => {
    const { value } = row({ value: "X", color: "var(--green)" });
    expect(value.style.color).toBe("var(--green)");
  });

  it("defaults to mono and a border", () => {
    const { root, value } = row();
    expect(value.style.fontFamily).toBe("var(--font-mono)");
    expect(root.style.borderBottom).toBe("1px solid var(--border)");
  });

  it("disables mono for the value and border when asked", () => {
    const { root, value } = row({ mono: false, border: false });
    expect(value.style.fontFamily).toBe("var(--font-sans)");
    expect(root.style.borderBottom).toBe("");
  });

  it("uses baseline alignment and 500 weight when baseline without ellipsis", () => {
    const { root, value } = row({ baseline: true });
    expect(root.style.alignItems).toBe("baseline");
    expect(value.style.fontWeight).toBe("500");
  });

  it("applies emphasis size/weight", () => {
    const { value } = row({ emphasis: true });
    expect(value.style.fontSize).toBe("12px");
    expect(value.style.fontWeight).toBe("600");
  });

  it("right-aligns and ellipsizes the value when ellipsis", () => {
    const { root, value, label } = row({ ellipsis: true });
    expect(value.style.textAlign).toBe("right");
    expect(value.style.textOverflow).toBe("ellipsis");
    expect(value.style.fontWeight).toBe("400");
    expect(label.style.flexShrink).toBe("0");
    expect(root.style.gap).toBe("4px");
  });

  it("supports a mono label independently of the value", () => {
    const { label, value } = row({ mono: false, labelMono: true });
    expect(label.style.fontFamily).toBe("var(--font-mono)");
    expect(value.style.fontFamily).toBe("var(--font-sans)");
  });

  it("uses labelSize for the label", () => {
    const { label } = row({ labelSize: 11 });
    expect(label.style.fontSize).toBe("11px");
  });
});