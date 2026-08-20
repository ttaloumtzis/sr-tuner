import type { ValidationFrames, ValidationHistoryEntry } from "../../store/trainingStore";

export type FrameKind = "lr" | "sr" | "gt" | "diff";

export const FRAME_META: Record<FrameKind, { label: string; key: keyof ValidationFrames }> = {
  lr:   { label: "LR",   key: "lrPath" },
  sr:   { label: "SR",   key: "srPath" },
  gt:   { label: "GT",   key: "gtPath" },
  diff: { label: "Diff", key: "diffPath" },
};
export const FRAME_ORDER: FrameKind[] = ["lr", "sr", "gt", "diff"];

export function pathFor(frames: ValidationFrames | null, kind: FrameKind): string | null {
  if (!frames) return null;
  const v = frames[FRAME_META[kind].key];
  return typeof v === "string" ? v : null;
}

export function entryPsnr(entry: ValidationHistoryEntry | null): number | null {
  return entry?.fullPsnr ?? entry?.psnr ?? null;
}