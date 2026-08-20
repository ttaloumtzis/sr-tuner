import { useState } from "react";
import { fmtTimestamp } from "../lib/format";
import type { RecentEntry } from "./recentProjects";

function RecentRow({
  entry,
  isLast,
  onOpen,
  onRemove,
}: {
  entry: RecentEntry;
  isLast: boolean;
  onOpen: (e: RecentEntry) => void;
  onRemove: (filePath: string) => void;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "9px 14px",
        borderBottom: isLast ? "none" : "1px solid var(--border)",
        background: hovered ? "var(--bg2)" : "transparent",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
      onClick={() => onOpen(entry)}
    >
      <div
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "var(--green)",
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, flexShrink: 0 }}>
        {entry.name}
      </span>
      <span
        style={{
          fontSize: 11,
          color: "var(--muted)",
          fontFamily: "var(--font-mono)",
          flex: 1,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          minWidth: 0,
        }}
      >
        {entry.filePath}
      </span>
      <span style={{ fontSize: 11, color: "var(--dim)", flexShrink: 0 }}>
        {fmtTimestamp(new Date(entry.lastOpened).getTime() / 1000)}
      </span>
      {hovered && (
        <button
          onClick={(ev) => { ev.stopPropagation(); onRemove(entry.filePath); }}
          title="Remove from list"
          style={{
            background: "none",
            border: "none",
            color: "var(--muted)",
            cursor: "pointer",
            fontSize: 14,
            lineHeight: 1,
            padding: "0 2px",
            flexShrink: 0,
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}

export function RecentList({
  entries,
  onOpen,
  onRemove,
}: {
  entries: RecentEntry[];
  onOpen: (e: RecentEntry) => void;
  onRemove: (filePath: string) => void;
}) {
  if (entries.length === 0) {
    return (
      <div
        style={{
          background: "var(--bg1)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "20px 16px",
          textAlign: "center",
          color: "var(--dim)",
          fontSize: 12,
        }}
      >
        No recent projects
      </div>
    );
  }

  return (
    <div
      style={{
        background: "var(--bg1)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
      }}
    >
      {entries.map((entry, i) => (
        <RecentRow
          key={entry.filePath}
          entry={entry}
          isLast={i === entries.length - 1}
          onOpen={onOpen}
          onRemove={onRemove}
        />
      ))}
    </div>
  );
}