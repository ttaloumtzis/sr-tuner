import type { ScannedDataset } from "../../../lib/scanDatasets";
import { join } from "../../../lib/path";

export interface MergePlanGroup {
  scale: number;
  sources: ScannedDataset[];
  totalPairs: number;
  dirName: string;
  outputPath: string;
}

export interface MergePlanWarnings {
  customNameWithMultipleScales: boolean;
  existingTargets: string[];
}

export interface MergePlan {
  groups: MergePlanGroup[];
  totalSelected: number;
  totalPairs: number;
  warnings: MergePlanWarnings;
}

export function groupByScale(
  datasets: ScannedDataset[],
): Map<number, ScannedDataset[]> {
  const groups = new Map<number, ScannedDataset[]>();
  for (const d of datasets) {
    const arr = groups.get(d.scale) ?? [];
    arr.push(d);
    groups.set(d.scale, arr);
  }
  return groups;
}

export function buildPlans(options: {
  selected: ScannedDataset[];
  outputPath: string;
  customName: string;
}): MergePlan {
  const groups = groupByScale(options.selected);
  const scales = [...groups.keys()].sort((a, b) => a - b);

  const groupPlans: MergePlanGroup[] = scales.map((scale) => {
    const sources = groups.get(scale)!;
    const totalPairs = sources.reduce((sum, d) => sum + d.pairCount, 0);
    const dirName = options.customName.trim() || `merged-x${scale}`;
    return {
      scale,
      sources,
      totalPairs,
      dirName,
      outputPath: join(options.outputPath, dirName),
    };
  });

  const customName = options.customName.trim();

  return {
    groups: groupPlans,
    totalSelected: options.selected.length,
    totalPairs: groupPlans.reduce((sum, g) => sum + g.totalPairs, 0),
    warnings: {
      customNameWithMultipleScales: customName !== "" && groupPlans.length > 1,
      existingTargets: [],
    },
  };
}

/**
 * Exclude previously-merged datasets (detected via ``config.sources`` in their
 * manifest) so a ``merged-xN`` aggregate never shows up as a merge source.
 */
export function excludeMerged(scanned: ScannedDataset[]): ScannedDataset[] {
  return scanned.filter((d) => !d.isMerged);
}
