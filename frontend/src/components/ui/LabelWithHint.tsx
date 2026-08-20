import { Tooltip } from "./Tooltip";

export function LabelWithHint({ label, hint }: { label: string; hint: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      {label} <Tooltip text={hint} />
    </span>
  );
}