import { describe, it, expect } from "vitest";
import { calcEta } from "../eta";

describe("calcEta", () => {
  it("returns (totalIter - currentIter) / speed", () => {
    expect(calcEta(1000, 200, 10)).toBeCloseTo(80);
  });

  it("returns null when speed is 0", () => {
    expect(calcEta(1000, 200, 0)).toBeNull();
  });

  it("returns null when speed is negative", () => {
    expect(calcEta(1000, 200, -5)).toBeNull();
  });

  it("returns null when currentIter >= totalIter", () => {
    expect(calcEta(1000, 1000, 10)).toBeNull();
    expect(calcEta(1000, 1200, 10)).toBeNull();
  });

  it("handles fractional speeds", () => {
    expect(calcEta(100, 50, 0.5)).toBeCloseTo(100);
  });
});
