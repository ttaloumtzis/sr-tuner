import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { GroupLabel } from "./GroupLabel";
import { LabelWithHint } from "./LabelWithHint";

describe("GroupLabel", () => {
  it("renders its children", () => {
    const { getByText } = render(<GroupLabel>Schedule</GroupLabel>);
    const el = getByText("Schedule");
    expect(el.style.textTransform).toBe("uppercase");
    expect(el.style.fontWeight).toBe("600");
  });
});

describe("LabelWithHint", () => {
  it("renders the label and a tooltip trigger", () => {
    const { container, getByText } = render(<LabelWithHint label="Patch Size" hint="Crop size in px." />);
    expect(getByText("Patch Size")).toBeTruthy();
    expect(container.querySelector(".tooltip-trigger")).toBeTruthy();
  });
});