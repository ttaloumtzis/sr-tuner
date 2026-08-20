export type RunStatus = "running" | "finished" | "failed" | "stopped" | "interrupted";

export const STATUS_COLOR: Record<RunStatus, string> = {
  running: "var(--green)",
  finished: "var(--green)",
  failed: "var(--red)",
  stopped: "var(--amber)",
  interrupted: "var(--dim)",
};