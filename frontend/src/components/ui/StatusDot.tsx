import { STATUS_COLOR, type RunStatus } from "../../lib/runStatus";

interface StatusDotProps {
  status: RunStatus;
}

export function StatusDot({ status }: StatusDotProps) {
  const running = status === "running";
  return (
    <span
      title={status}
      className="status-dot"
      style={{
        background: STATUS_COLOR[status],
        animation: running ? "tabbar-pulse 1.4s ease-in-out infinite" : undefined,
      }}
    />
  );
}