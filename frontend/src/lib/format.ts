import type { SRProjRun } from "./srproj";

export function fmt(n: number | null | undefined, dec = 2): string {
  if (n == null) return "—";
  return n.toFixed(dec);
}

export function fmtPct(n: number | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)}%`;
}

export function fmtGb(n: number | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)} GB`;
}

export function fmtSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function fmtTimestamp(ts: number | undefined | null): string {
  if (ts == null || Number.isNaN(ts)) return "—";
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function fmtDuration(run: SRProjRun): string {
  if (!run.started_at) return "—";
  const endMs = run.completed_at
    ? new Date(run.completed_at).getTime()
    : Date.now();
  const sec = Math.floor((endMs - new Date(run.started_at).getTime()) / 1000);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

export function formatEta(sec: number | null): string {
  if (sec == null || sec <= 0) return "";
  if (sec < 60) return `ETA ${Math.round(sec)}s`;
  if (sec < 3600) return `ETA ${Math.round(sec / 60)}m`;
  return `ETA ${(sec / 3600).toFixed(1)}h`;
}