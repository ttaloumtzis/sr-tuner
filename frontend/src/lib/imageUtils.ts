export interface PreloadedImage {
  url: string;
  w: number;
  h: number;
}

export function preloadImage(src: string, signal?: AbortSignal): Promise<PreloadedImage> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const img = new Image();
    img.onload = () => {
      resolve({ url: src, w: img.naturalWidth, h: img.naturalHeight });
    };
    img.onerror = () => {
      reject(new Error(`Failed to load image: ${src}`));
    };
    signal?.addEventListener("abort", () => {
      img.src = "";
      reject(new DOMException("Aborted", "AbortError"));
    });
    img.src = src;
  });
}

export interface PreloadedPair {
  hr: PreloadedImage;
  lr: PreloadedImage;
}

export async function preloadPair(
  hrUrl: string,
  lrUrl: string,
  signal?: AbortSignal,
): Promise<PreloadedPair> {
  const [hr, lr] = await Promise.all([
    preloadImage(hrUrl, signal),
    preloadImage(lrUrl, signal),
  ]);
  return { hr, lr };
}