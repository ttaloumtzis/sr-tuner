import { invoke } from "@tauri-apps/api/core";
import type { DatasetManifest } from "./api-types";
import { basename, join } from "./path";

export interface ScannedDataset {
  name: string;
  path: string;
  scale: number;
  pairCount: number;
  hasManifest: boolean;
  hasHr: boolean;
  hasLr: boolean;
}

export async function scanDatasets(parentDir: string): Promise<ScannedDataset[]> {
  if (!parentDir) return [];
  const entries: string[] = await invoke("list_dir", { path: parentDir });
  const results: ScannedDataset[] = [];

  for (const entry of entries) {
    if (!entry.endsWith("/")) continue;
    const name = entry.replace(/\/$/, "");
    const fullPath = join(parentDir, name);

    const [hasHr, hasLr, hasManifest] = await Promise.all([
      invoke<boolean>("path_exists", { path: join(fullPath, "HR") }),
      invoke<boolean>("path_exists", { path: join(fullPath, "LR") }),
      invoke<boolean>("path_exists", { path: join(fullPath, "manifest.json") }),
    ]);

    let scale = 4;
    let pairCount = 0;

    if (hasManifest) {
      const manifest = await readManifest(fullPath);
      if (manifest) {
        scale = manifest.config.scale;
        pairCount = manifest.pairs.length;
      }
    }

    if (pairCount === 0 && hasHr) {
      const hrFiles: string[] = await invoke("list_image_files", { path: join(fullPath, "HR") });
      pairCount = hrFiles.length;
    }

    results.push({ name, path: fullPath, scale, pairCount, hasManifest, hasHr, hasLr });
  }

  results.sort((a, b) => a.name.localeCompare(b.name));
  return results;
}

export async function readManifest(datasetPath: string): Promise<DatasetManifest | null> {
  try {
    const raw = await invoke<string>("read_text_file", { path: join(datasetPath, "manifest.json") });
    return JSON.parse(raw) as DatasetManifest;
  } catch {
    return null;
  }
}

export async function listDatasetPairs(datasetPath: string): Promise<{ hr: string; lr: string }[]> {
  const manifest = await readManifest(datasetPath);
  if (manifest && manifest.pairs.length > 0) {
    return manifest.pairs.map((p) => ({
      hr: join(datasetPath, p.hr),
      lr: join(datasetPath, p.lr),
    }));
  }
  const [hrFiles, lrFiles] = await Promise.all([
    invoke<string[]>("list_image_files", { path: join(datasetPath, "HR") }),
    invoke<string[]>("list_image_files", { path: join(datasetPath, "LR") }),
  ]);
  const count = Math.min(hrFiles.length, lrFiles.length);
  const pairs: { hr: string; lr: string }[] = [];
  const sortKey = (p: string) => {
    const name = basename(p);
    return name.replace(/\D/g, "").padStart(10, "0") + name;
  };
  hrFiles.sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
  lrFiles.sort((a, b) => sortKey(a).localeCompare(sortKey(b)));
  for (let i = 0; i < count; i++) {
    pairs.push({ hr: hrFiles[i], lr: lrFiles[i] });
  }
  return pairs;
}