import { describe, it, expect } from "vitest";
import { basename, dirname, join, normalize, parentFromProjFile } from "../path";

// Tests run in jsdom (Linux user-agent), so SEP = "/" by default.
// The regex patterns handle both separators regardless of SEP.

describe("basename", () => {
  it("extracts filename from Unix path", () => {
    expect(basename("/a/b/c.txt")).toBe("c.txt");
  });

  it("extracts filename from Windows path", () => {
    expect(basename("C:\\a\\b\\c.txt")).toBe("c.txt");
  });

  it("handles mixed separators", () => {
    expect(basename("a/b\\c.txt")).toBe("c.txt");
  });

  it("returns the string as-is when no separator", () => {
    expect(basename("file.txt")).toBe("file.txt");
  });

  it("handles trailing separator", () => {
    expect(basename("/a/b/")).toBe("");
  });
});

describe("dirname", () => {
  it("returns parent of Unix path", () => {
    expect(dirname("/a/b/c")).toBe("/a/b");
  });

  it("returns parent of Windows path (preserves backslash)", () => {
    expect(dirname("C:\\a\\b\\c")).toBe("C:\\a\\b");
  });

  it("returns '.' for single segment", () => {
    expect(dirname("file.txt")).toBe(".");
  });

  it("handles root path", () => {
    expect(dirname("/")).toBe(".");
  });

  it("handles mixed separators", () => {
    expect(dirname("/a/b\\c")).toBe("/a/b");
  });
});

describe("join", () => {
  it("joins two path segments", () => {
    expect(join("/a/b", "c")).toBe("/a/b/c");
  });

  it("strips trailing separators from parts", () => {
    expect(join("/a/b/", "c")).toBe("/a/b/c");
  });

  it("filters empty parts", () => {
    expect(join("/a", "", "b")).toBe("/a/b");
  });

  it("returns empty string for no parts", () => {
    expect(join()).toBe("");
  });

  it("returns part unchanged when only one part", () => {
    expect(join("/a/b")).toBe("/a/b");
  });

  it("joins multiple segments", () => {
    expect(join("/root", "datasets", "HR")).toBe("/root/datasets/HR");
  });
});

describe("normalize", () => {
  it("collapses mixed separators to platform SEP", () => {
    const result = normalize("/a/b\\c/d");
    expect(result).toBe("/a/b/c/d");
  });

  it("collapses duplicate separators", () => {
    expect(normalize("/a//b///c")).toBe("/a/b/c");
  });

  it("returns clean path unchanged", () => {
    expect(normalize("/a/b/c")).toBe("/a/b/c");
  });
});

describe("parentFromProjFile", () => {
  it("strips .srproj basename from Unix path", () => {
    expect(parentFromProjFile("/home/user/proj.srproj")).toBe("/home/user");
  });

  it("strips .srproj basename from Windows path", () => {
    expect(parentFromProjFile("C:\\Users\\me\\proj.srproj")).toBe("C:\\Users\\me");
  });

  it("handles multi-segment project name", () => {
    expect(parentFromProjFile("/a/b/my.proj.srproj")).toBe("/a/b");
  });

  it("falls back to dirname when no .srproj in path", () => {
    expect(parentFromProjFile("/a/b/c")).toBe("/a/b");
  });

  it("returns '.' for bare filename with no parent", () => {
    expect(parentFromProjFile("proj.srproj")).toBe(".");
  });
});
