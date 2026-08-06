import { invoke } from "@tauri-apps/api/core";
import type { DatasetManifest } from "./api-types";
import { basename, join } from "./path";

let _tauriAvailable: boolean | null = null;

export function __resetTauriAvailability(): void {
  _tauriAvailable = null;
}

async function isTauriAvailable(): Promise<boolean> {
  if (_tauriAvailable !== null) return _tauriAvailable;
  try {
    await invoke("path_exists", { path: "/" });
    _tauriAvailable = true;
  } catch {
    _tauriAvailable = false;
  }
  return _tauriAvailable;
}

export interface ScannedDataset {
  name: string;
  path: string;
  scale: number;
  pairCount: number;
  hasManifest: boolean;
  hasHr: boolean;
  hasLr: boolean;
  isMerged: boolean;
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
    let isMerged = false;

    if (hasManifest) {
      const manifest = await readManifest(fullPath);
      if (manifest) {
        scale = manifest.config.scale;
        pairCount = manifest.pairs.length;
        isMerged = (manifest.config.sources?.length ?? 0) > 0;
      }
    }

    if (pairCount === 0 && hasHr) {
      const hrFiles: string[] = await invoke("list_image_files", { path: join(fullPath, "HR") });
      pairCount = hrFiles.length;
    }

    results.push({ name, path: fullPath, scale, pairCount, hasManifest, hasHr, hasLr, isMerged });
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

/**
 * Resolve the image URLs for every pair of a dataset.
 *
 * In the Tauri app, the manifest.json is read once and images are served
 * directly from disk via the asset protocol (``convertFileSrc``), avoiding a
 * per-image HTTP round-trip to the API. In browser dev mode (no Tauri), falls
 * back to the API image endpoint.
 *
 * @param datasetPath Absolute path of the dataset folder.
 * @param datasetName Dataset name (used for API fallback URLs).
 * @param pairCount Number of pairs, from the datasets listing (browser mode).
 */
export async function getDatasetPairUrls(
  datasetPath: string,
  datasetName: string,
  pairCount: number,
): Promise<{ hr: string; lr: string }[]> {
  if (await isTauriAvailable()) {
    try {
      const pairs = await listDatasetPairs(datasetPath);
      const { convertFileSrc } = await import("@tauri-apps/api/core");
      return pairs.map((p) => ({ hr: convertFileSrc(p.hr), lr: convertFileSrc(p.lr) }));
    } catch {
      // fall through to API URLs
    }
  }
  const { getDatasetImageUrl } = await import("./api");
  return Array.from({ length: pairCount }, (_, i) => ({
    hr: getDatasetImageUrl(datasetName, "hr", i),
    lr: getDatasetImageUrl(datasetName, "lr", i),
  }));
}