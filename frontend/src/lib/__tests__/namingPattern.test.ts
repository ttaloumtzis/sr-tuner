import { describe, it, expect } from "vitest";
import { validateNamingPattern, previewFilename } from "../namingPattern";

describe("validateNamingPattern", () => {
  it("%06d passes", () => {
    expect(validateNamingPattern("%06d")).toBeNull();
  });

  it("%d passes", () => {
    expect(validateNamingPattern("%d")).toBeNull();
  });

  it("%04d passes", () => {
    expect(validateNamingPattern("%04d")).toBeNull();
  });

  it("empty string passes (treated as default)", () => {
    expect(validateNamingPattern("")).toBeNull();
  });

  it("%s fails", () => {
    expect(validateNamingPattern("%s")).not.toBeNull();
  });

  it("%06f fails", () => {
    expect(validateNamingPattern("%06f")).not.toBeNull();
  });

  it("plain text fails", () => {
    expect(validateNamingPattern("hello")).not.toBeNull();
  });
});

describe("previewFilename", () => {
  it("%06d renders 000001.png", () => {
    expect(previewFilename("%06d")).toBe("000001.png");
  });

  it("%d renders 1.png", () => {
    expect(previewFilename("%d")).toBe("1.png");
  });

  it("%04d renders 0001.png", () => {
    expect(previewFilename("%04d")).toBe("0001.png");
  });
});
