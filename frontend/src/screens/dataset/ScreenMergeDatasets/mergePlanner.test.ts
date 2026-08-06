import { describe, it, expect } from "vitest";
import { buildPlans, excludeMerged, groupByScale } from "./mergePlanner";
import type { ScannedDataset } from "../../../lib/scanDatasets";

function scanned(name: string, scale: number, pairCount: number): ScannedDataset {
  return {
    name,
    path: `/proj/datasets/${name}`,
    scale,
    pairCount,
    hasManifest: true,
    hasHr: true,
    hasLr: true,
    isMerged: false,
  };
}

describe("groupByScale", () => {
  it("groups datasets by scale", () => {
    const groups = groupByScale([
      scanned("a", 4, 10),
      scanned("b", 4, 20),
      scanned("c", 2, 5),
    ]);
    expect([...groups.keys()].sort()).toEqual([2, 4]);
    expect(groups.get(4)?.map((d) => d.name)).toEqual(["a", "b"]);
  });
});

describe("buildPlans", () => {
  it("creates one plan group per scale with totals and output paths", () => {
    const plan = buildPlans({
      selected: [scanned("a", 4, 100), scanned("b", 4, 50), scanned("c", 2, 25)],
      outputPath: "/proj/merged",
      customName: "",
    });
    expect(plan.totalSelected).toBe(3);
    expect(plan.totalPairs).toBe(175);
    expect(plan.groups).toHaveLength(2);
    expect(plan.groups[0]).toMatchObject({
      scale: 2,
      totalPairs: 25,
      dirName: "merged-x2",
      outputPath: "/proj/merged/merged-x2",
    });
    expect(plan.groups[1]).toMatchObject({
      scale: 4,
      totalPairs: 150,
      dirName: "merged-x4",
      outputPath: "/proj/merged/merged-x4",
    });
    expect(plan.warnings.customNameWithMultipleScales).toBe(false);
  });

  it("applies a custom name to the single scale group", () => {
    const plan = buildPlans({
      selected: [scanned("a", 4, 10), scanned("b", 4, 20)],
      outputPath: "/proj/merged",
      customName: "mymer",
    });
    expect(plan.groups).toHaveLength(1);
    expect(plan.groups[0].outputPath).toBe("/proj/merged/mymer");
    expect(plan.warnings.customNameWithMultipleScales).toBe(false);
  });

  it("warns when a custom name spans multiple scale groups", () => {
    const plan = buildPlans({
      selected: [scanned("a", 4, 10), scanned("c", 2, 5)],
      outputPath: "/proj/merged",
      customName: "mymer",
    });
    expect(plan.warnings.customNameWithMultipleScales).toBe(true);
  });

  it("trims whitespace-only custom names", () => {
    const plan = buildPlans({
      selected: [scanned("a", 4, 10)],
      outputPath: "/proj/merged",
      customName: "   ",
    });
    expect(plan.groups[0].dirName).toBe("merged-x4");
  });
});

describe("excludeMerged", () => {
  it("drops datasets whose manifest marks them as merged", () => {
    const real = scanned("a", 4, 10);
    const merged = { ...scanned("merged-x4", 4, 999), isMerged: true };
    expect(excludeMerged([real, merged]).map((d) => d.name)).toEqual(["a"]);
  });

  it("keeps all datasets when none are merged", () => {
    const list = [scanned("a", 4, 10), scanned("b", 2, 5)];
    expect(excludeMerged(list).map((d) => d.name)).toEqual(["a", "b"]);
  });
});
