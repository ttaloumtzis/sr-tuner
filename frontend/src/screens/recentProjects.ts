// ── Recent projects persistence ────────────────────────────────────────────

const RECENT_KEY = "sr-tuner:recent-projects";
const MAX_RECENT = 8;

export interface RecentEntry {
  name: string;
  filePath: string;
  lastOpened: string;
}

export function loadRecent(): RecentEntry[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    return raw ? (JSON.parse(raw) as RecentEntry[]) : [];
  } catch {
    return [];
  }
}

function saveRecent(entries: RecentEntry[]): void {
  localStorage.setItem(RECENT_KEY, JSON.stringify(entries.slice(0, MAX_RECENT)));
}

export function addToRecent(entry: RecentEntry): RecentEntry[] {
  const existing = loadRecent().filter((e) => e.filePath !== entry.filePath);
  const next = [entry, ...existing].slice(0, MAX_RECENT);
  saveRecent(next);
  return next;
}

export function removeRecent(filePath: string): RecentEntry[] {
  const next = loadRecent().filter((e) => e.filePath !== filePath);
  saveRecent(next);
  return next;
}