// §19.8 — GPU backend download handler
// §19.9 — Sidecar variant cache (app-data dir, version + checksum verification)
//
// Architecture:
//  • The main installer ships a CPU-only minimal sidecar.
//  • On first launch the sidecar runs in SIDECAR_MODE=minimal, detects GPU vendor,
//    emits gpu.detection_needed, and exits.
//  • This module handles the download of the matching full GPU variant, verifies
//    its checksum, caches it in <appData>/sidecar-variants/, and installs it so
//    the next sidecar spawn uses the GPU-enabled binary.

import { invoke } from "@tauri-apps/api/core";
import { appDataDir } from "@tauri-apps/api/path";

export type GpuVariant = "cuda" | "rocm" | "cpu";

// §25.7 — URL of the variant manifest published alongside each release.
// The manifest lists download URLs, SHA-256 checksums, and sizes for every
// GPU variant so the frontend can verify downloads and show accurate progress.
export const VARIANT_MANIFEST_URL =
  "https://github.com/example/sr-tuner/releases/latest/download/variant-manifest.json";

// Variant metadata returned by the release manifest hosted alongside binaries.
export interface VariantManifest {
  version: string;
  variants: {
    [key in GpuVariant]?: {
      url: string;
      sha256: string;
      size_bytes: number;
    };
  };
}

export interface DownloadProgress {
  bytesDone: number;
  bytesTotal: number | null;
  variant: GpuVariant;
}

export type DownloadProgressCallback = (progress: DownloadProgress) => void;

// §19.9 — Check local cache before network download.
// Returns the cached binary path if it exists and passes the checksum check.
export async function getCachedVariantPath(
  variant: GpuVariant,
  expectedSha256: string
): Promise<string | null> {
  const cacheDir = await variantCacheDir();
  const binaryPath = `${cacheDir}/${variant}/sidecar`;

  const exists = await invoke<boolean>("path_exists", { path: binaryPath });
  if (!exists) return null;

  // Verify checksum to detect corrupt or outdated cache
  try {
    const checksum = await invoke<string>("sha256_file", { path: binaryPath });
    return checksum === expectedSha256 ? binaryPath : null;
  } catch {
    return null;
  }
}

// §19.8 — Download the matching sidecar variant from the release server.
// Streams progress via the callback. Returns the installed binary path.
export async function downloadAndInstallVariant(
  variant: GpuVariant,
  manifestUrl: string,
  onProgress: DownloadProgressCallback
): Promise<string> {
  // 1. Fetch manifest
  const manifest = await fetchVariantManifest(manifestUrl);
  const variantMeta = manifest.variants[variant];
  if (!variantMeta) {
    throw new Error(`No ${variant} variant available in manifest`);
  }

  // 2. Check cache first (§19.9)
  const cached = await getCachedVariantPath(variant, variantMeta.sha256);
  if (cached) {
    return cached;
  }

  // 3. Download via Tauri command (handles cross-platform HTTP and file writes)
  const cacheDir = await variantCacheDir();
  const destDir = `${cacheDir}/${variant}`;
  const destPath = `${destDir}/sidecar`;

  await invoke("create_dir_all", { path: destDir });

  await invoke<void>("download_file", {
    url: variantMeta.url,
    dest: destPath,
    onProgress: (done: number, total: number | null) =>
      onProgress({ bytesDone: done, bytesTotal: total, variant }),
  });

  // 4. Verify checksum
  const checksum = await invoke<string>("sha256_file", { path: destPath });
  if (checksum !== variantMeta.sha256) {
    await invoke("delete_file", { path: destPath }).catch(() => {});
    throw new Error(
      `Checksum mismatch for ${variant} variant — download may be corrupt`
    );
  }

  // 5. Make the binary executable (Linux/macOS)
  await invoke("set_executable", { path: destPath }).catch(() => {});

  return destPath;
}

async function variantCacheDir(): Promise<string> {
  const dataDir = await appDataDir();
  return `${dataDir}/sidecar-variants`;
}

async function fetchVariantManifest(url: string): Promise<VariantManifest> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch variant manifest: ${response.status}`);
  }
  return response.json() as Promise<VariantManifest>;
}
