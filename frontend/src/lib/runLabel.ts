import type { RunInfo } from "./api-types";

/**
 * Short human-readable form of a run directory id.
 *
 * `run_20260702_090000` → `2026-07-02 09:00` (with ` #N` suffix for
 * colliding same-second starts).
 */
export function shortRunId(runId: string): string {
  const m = runId.match(/^run_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:_(\d+))?$/);
  if (!m) return runId;
  const [, y, mo, d, h, mi] = m;
  const suffix = m[7] ? ` #${m[7]}` : "";
  return `${y}-${mo}-${d} ${h}:${mi}${suffix}`;
}

export interface RunDisplay {
  /** Date bucket label, e.g. "Today", "Yesterday", "Jul 30, 2026", "Unknown". */
  group: string;
  /** Label shown for the run: `run_00N` for finished runs, else the timestamp. */
  label: string;
}

function dateGroupLabel(iso: string | null | undefined): string {
  if (!iso) return "Unknown";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "Unknown";

  const now = new Date();
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOfDay(now) - startOfDay(date)) / 86_400_000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days > 1 && days < 7) return date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/**
 * Compute display labels + date groups for a model instance's runs.
 *
 * Runs are ordered by `created_at` ascending. Finished runs receive a
 * sequential, global-per-model label (`run_001`, `run_002`, …) in creation
 * order; every other status keeps its timestamp label so live/failed runs
 * stay identifiable.
 *
 * Returns a `Map<run_id, RunDisplay>`.
 */
export function buildRunDisplays(runs: RunInfo[]): Map<string, RunDisplay> {
  const ordered = [...runs].sort((a, b) => {
    const ta = a.created_at ? new Date(a.created_at).getTime() : Number.POSITIVE_INFINITY;
    const tb = b.created_at ? new Date(b.created_at).getTime() : Number.POSITIVE_INFINITY;
    return ta - tb;
  });

  const result = new Map<string, RunDisplay>();
  let finishedSeq = 1;
  for (const run of ordered) {
    const group = dateGroupLabel(run.created_at);
    const label =
      run.status === "finished"
        ? `run_${String(finishedSeq++).padStart(3, "0")}`
        : shortRunId(run.run_id);
    result.set(run.run_id, { group, label });
  }
  return result;
}

/** Ordering used to render date groups: Today → Yesterday → dates desc → Unknown. */
export function sortGroups(groups: string[]): string[] {
  return [...groups].sort((a, b) => {
    const rank = (g: string) => {
      if (g === "Today") return 0;
      if (g === "Yesterday") return 1;
      if (g === "Unknown") return 3;
      return 2;
    };
    const ra = rank(a);
    const rb = rank(b);
    if (ra !== rb) return ra - rb;
    if (ra === 2) {
      const da = new Date(a).getTime();
      const db = new Date(b).getTime();
      if (!Number.isNaN(da) && !Number.isNaN(db)) return db - da;
    }
    return String(a).localeCompare(String(b));
  });
}
